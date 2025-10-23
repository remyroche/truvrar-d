"""
EBI Metagenomics API data collector for microbiome data.

This collector retrieves microbiome data from the EBI Metagenomics API,
specifically focusing on Tuber studies and related environmental data.
"""

import json
import logging
import pandas as pd
import requests
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from urllib.parse import urljoin
import time

from .base_collector import BaseCollector

logger = logging.getLogger(__name__)


class EBIMetagenomicsCollector(BaseCollector):
    """Collector for EBI Metagenomics API data."""
    
    def __init__(self, config: Dict[str, Any], data_dir: Path):
        super().__init__(config, data_dir)
        self.base_url = "https://www.ebi.ac.uk/metagenomics/api/latest"
        self.session.headers.update({
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        })
        
    def collect(self, search_term: str = "Tuber", 
                limit: int = 1000,
                include_samples: bool = True,
                include_abundance: bool = False) -> pd.DataFrame:
        """
        Collect microbiome data from EBI Metagenomics API.
        
        Args:
            search_term: Search term for studies (default: "Tuber")
            limit: Maximum number of studies to retrieve
            include_samples: Whether to include sample-level data
            include_abundance: Whether to retrieve abundance tables
            
        Returns:
            DataFrame with microbiome and environmental data
        """
        logger.info(f"Collecting EBI Metagenomics data for search term: {search_term}")
        
        # Step 1: Get studies
        studies_df = self._get_studies(search_term, limit)
        
        if studies_df.empty:
            logger.warning("No studies found for the search term")
            return pd.DataFrame()
            
        # Step 2: Get samples for each study
        if include_samples:
            samples_df = self._get_samples(studies_df)
            if not samples_df.empty:
                # Merge studies and samples data
                merged_df = pd.merge(studies_df, samples_df, 
                                   on='study_id', how='left', suffixes=('_study', '_sample'))
            else:
                merged_df = studies_df
        else:
            merged_df = studies_df
            
        # Step 3: Get abundance data if requested
        if include_abundance and not merged_df.empty:
            abundance_df = self._get_abundance_data(merged_df)
            if not abundance_df.empty:
                merged_df = pd.merge(merged_df, abundance_df, 
                                   on='sample_id', how='left')
        
        # Step 4: Clean and standardize data
        cleaned_df = self._clean_data(merged_df)
        
        # Save data
        output_file = self.data_dir / f"ebi_metagenomics_{search_term.lower()}.csv"
        cleaned_df.to_csv(output_file, index=False)
        logger.info(f"Saved {len(cleaned_df)} records to {output_file}")
        
        return cleaned_df
    
    def _get_studies(self, search_term: str, limit: int) -> pd.DataFrame:
        """Retrieve studies from EBI Metagenomics API."""
        studies_data = []
        url = f"{self.base_url}/studies"
        params = {
            'search': search_term,
            'page_size': min(100, limit),  # API limit
            'page': 1
        }
        
        while len(studies_data) < limit:
            try:
                response = self._make_request(url, params=params)
                data = response.json()
                
                for study in data.get('data', []):
                    if len(studies_data) >= limit:
                        break
                        
                    study_info = {
                        'study_id': study.get('id'),
                        'study_accession': study.get('attributes', {}).get('accession'),
                        'study_title': study.get('attributes', {}).get('study-name'),
                        'study_description': study.get('attributes', {}).get('study-abstract'),
                        'biome': study.get('attributes', {}).get('biome', {}).get('biome-name'),
                        'biome_category': study.get('attributes', {}).get('biome', {}).get('biome-category'),
                        'latitude': study.get('attributes', {}).get('latitude'),
                        'longitude': study.get('attributes', {}).get('longitude'),
                        'environment_biome': study.get('attributes', {}).get('environment-biome'),
                        'environment_feature': study.get('attributes', {}).get('environment-feature'),
                        'environment_material': study.get('attributes', {}).get('environment-material'),
                        'host': study.get('attributes', {}).get('host'),
                        'host_taxonomy_id': study.get('attributes', {}).get('host-taxonomy-id'),
                        'sample_count': study.get('attributes', {}).get('sample-count'),
                        'run_count': study.get('attributes', {}).get('run-count'),
                        'analysis_completed': study.get('attributes', {}).get('analysis-completed'),
                        'publication_date': study.get('attributes', {}).get('publication-date'),
                        'submission_date': study.get('attributes', {}).get('submission-date'),
                        'center_name': study.get('attributes', {}).get('center-name'),
                        'source_link': f"https://www.ebi.ac.uk/metagenomics/studies/{study.get('id')}"
                    }
                    studies_data.append(study_info)
                
                # Check if there are more pages
                if 'links' in data and 'next' in data['links']:
                    url = data['links']['next']
                    params = {}  # URL already contains parameters
                else:
                    break
                    
            except Exception as e:
                logger.error(f"Error retrieving studies: {e}")
                break
                
            time.sleep(0.5)  # Rate limiting
        
        return pd.DataFrame(studies_data)
    
    def _get_samples(self, studies_df: pd.DataFrame) -> pd.DataFrame:
        """Retrieve sample data for studies."""
        samples_data = []
        
        for _, study in studies_df.iterrows():
            study_id = study['study_id']
            url = f"{self.base_url}/studies/{study_id}/samples"
            params = {'page_size': 100}
            
            try:
                response = self._make_request(url, params=params)
                data = response.json()
                
                for sample in data.get('data', []):
                    sample_info = {
                        'study_id': study_id,
                        'sample_id': sample.get('id'),
                        'sample_accession': sample.get('attributes', {}).get('accession'),
                        'sample_name': sample.get('attributes', {}).get('sample-name'),
                        'sample_description': sample.get('attributes', {}).get('sample-desc'),
                        'latitude': sample.get('attributes', {}).get('latitude'),
                        'longitude': sample.get('attributes', {}).get('longitude'),
                        'environment_biome': sample.get('attributes', {}).get('environment-biome'),
                        'environment_feature': sample.get('attributes', {}).get('environment-feature'),
                        'environment_material': sample.get('attributes', {}).get('environment-material'),
                        'host': sample.get('attributes', {}).get('host'),
                        'host_taxonomy_id': sample.get('attributes', {}).get('host-taxonomy-id'),
                        'collection_date': sample.get('attributes', {}).get('collection-date'),
                        'geographic_location': sample.get('attributes', {}).get('geographic-location'),
                        'depth': sample.get('attributes', {}).get('depth'),
                        'elevation': sample.get('attributes', {}).get('elevation'),
                        'temperature': sample.get('attributes', {}).get('temperature'),
                        'ph': sample.get('attributes', {}).get('ph'),
                        'salinity': sample.get('attributes', {}).get('salinity'),
                        'nitrogen': sample.get('attributes', {}).get('nitrogen'),
                        'phosphorus': sample.get('attributes', {}).get('phosphorus'),
                        'carbon': sample.get('attributes', {}).get('carbon'),
                        'run_count': sample.get('attributes', {}).get('run-count'),
                        'analysis_completed': sample.get('attributes', {}).get('analysis-completed')
                    }
                    samples_data.append(sample_info)
                    
            except Exception as e:
                logger.warning(f"Error retrieving samples for study {study_id}: {e}")
                continue
                
            time.sleep(0.5)  # Rate limiting
        
        return pd.DataFrame(samples_data)
    
    def _get_abundance_data(self, samples_df: pd.DataFrame) -> pd.DataFrame:
        """Retrieve abundance data for samples (if available)."""
        abundance_data = []
        
        for _, sample in samples_df.iterrows():
            sample_id = sample['sample_id']
            
            # Try to get taxonomic abundance data
            try:
                url = f"{self.base_url}/samples/{sample_id}/taxonomy"
                response = self._make_request(url)
                data = response.json()
                
                # Process taxonomic data
                taxonomy_data = self._process_taxonomy_data(data, sample_id)
                if taxonomy_data:
                    abundance_data.append(taxonomy_data)
                    
            except Exception as e:
                logger.debug(f"No taxonomy data available for sample {sample_id}: {e}")
            
            # Try to get functional abundance data
            try:
                url = f"{self.base_url}/samples/{sample_id}/go-slim"
                response = self._make_request(url)
                data = response.json()
                
                # Process functional data
                functional_data = self._process_functional_data(data, sample_id)
                if functional_data:
                    abundance_data.append(functional_data)
                    
            except Exception as e:
                logger.debug(f"No functional data available for sample {sample_id}: {e}")
                
            time.sleep(0.5)  # Rate limiting
        
        return pd.DataFrame(abundance_data)
    
    def _process_taxonomy_data(self, data: Dict, sample_id: str) -> Optional[Dict]:
        """Process taxonomic abundance data."""
        taxonomy_data = {'sample_id': sample_id}
        
        for taxon in data.get('data', []):
            attributes = taxon.get('attributes', {})
            taxon_name = attributes.get('lineage')
            abundance = attributes.get('abundance')
            
            if taxon_name and abundance is not None:
                # Store at different taxonomic levels
                if 'family' in taxon_name.lower():
                    taxonomy_data[f'taxonomy_family_{taxon_name}'] = abundance
                elif 'genus' in taxon_name.lower():
                    taxonomy_data[f'taxonomy_genus_{taxon_name}'] = abundance
                elif 'species' in taxon_name.lower():
                    taxonomy_data[f'taxonomy_species_{taxon_name}'] = abundance
        
        return taxonomy_data if len(taxonomy_data) > 1 else None
    
    def _process_functional_data(self, data: Dict, sample_id: str) -> Optional[Dict]:
        """Process functional abundance data."""
        functional_data = {'sample_id': sample_id}
        
        for go_term in data.get('data', []):
            attributes = go_term.get('attributes', {})
            go_id = attributes.get('go-id')
            abundance = attributes.get('abundance')
            
            if go_id and abundance is not None:
                functional_data[f'functional_{go_id}'] = abundance
        
        return functional_data if len(functional_data) > 1 else None
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize the collected data."""
        if df.empty:
            return df
            
        # Convert numeric columns
        numeric_columns = ['latitude', 'longitude', 'ph', 'temperature', 'depth', 
                          'elevation', 'salinity', 'nitrogen', 'phosphorus', 'carbon']
        
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Convert date columns
        date_columns = ['collection_date', 'publication_date', 'submission_date']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # Create derived fields
        if 'latitude' in df.columns and 'longitude' in df.columns:
            df['has_coordinates'] = df['latitude'].notna() & df['longitude'].notna()
        
        # Standardize species names
        if 'host' in df.columns:
            df['host_standardized'] = df['host'].str.lower().str.strip()
        
        # Create key taxa indicators
        bacterial_families = []
        fungal_guilds = []
        
        for col in df.columns:
            if 'taxonomy_family' in col:
                family_name = col.replace('taxonomy_family_', '')
                bacterial_families.append(family_name)
            elif 'taxonomy_genus' in col:
                genus_name = col.replace('taxonomy_genus_', '')
                if any(fungus in genus_name.lower() for fungus in ['tuber', 'truffle', 'mycorrhiza']):
                    fungal_guilds.append(genus_name)
        
        # Add summary fields
        df['bacterial_families'] = df[[col for col in df.columns if 'taxonomy_family' in col]].apply(
            lambda x: json.dumps(x.dropna().to_dict()), axis=1
        )
        df['fungal_guilds'] = df[[col for col in df.columns if 'taxonomy_genus' in col]].apply(
            lambda x: json.dumps(x.dropna().to_dict()), axis=1
        )
        
        # Identify key taxa
        key_taxa = []
        for col in df.columns:
            if 'taxonomy' in col:
                taxon_name = col.split('_', 2)[-1] if '_' in col else col
                if any(key in taxon_name.lower() for key in ['pseudomonas', 'bradyrhizobium', 'rhizobium']):
                    key_taxa.append(taxon_name)
        
        df['key_taxa'] = df[[col for col in df.columns if 'taxonomy' in col]].apply(
            lambda x: json.dumps([k for k, v in x.dropna().items() 
                                if any(key in k.lower() for key in ['pseudomonas', 'bradyrhizobium', 'rhizobium'])]), axis=1
        )
        
        # Add fruiting evidence indicator
        df['fruiting_evidence'] = df['study_title'].str.contains(
            'fruiting|fruit|truffle|harvest', case=False, na=False
        ) | df['sample_description'].str.contains(
            'fruiting|fruit|truffle|harvest', case=False, na=False
        )
        
        return df
