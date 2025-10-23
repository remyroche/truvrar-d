"""
Abundance table processor for microbiome data.

This module processes OTU/ASV tables from various sources and calculates
relative abundances at different taxonomic levels for microbiome analysis.
"""

import json
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
import requests
from urllib.parse import urlparse
import zipfile
import io

logger = logging.getLogger(__name__)


class AbundanceProcessor:
    """Processor for microbiome abundance tables."""
    
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.data_dir.mkdir(exist_ok=True)
        
    def process_abundance_tables(self, 
                                abundance_sources: List[Dict[str, Any]],
                                target_taxonomic_levels: List[str] = ['family', 'genus', 'species'],
                                min_abundance_threshold: float = 0.001) -> pd.DataFrame:
        """
        Process abundance tables from multiple sources.
        
        Args:
            abundance_sources: List of source configurations
            target_taxonomic_levels: Taxonomic levels to process
            min_abundance_threshold: Minimum relative abundance to keep
            
        Returns:
            Processed abundance DataFrame
        """
        logger.info(f"Processing abundance tables from {len(abundance_sources)} sources")
        
        all_abundance_data = []
        
        for source in abundance_sources:
            try:
                abundance_df = self._process_single_source(
                    source, target_taxonomic_levels, min_abundance_threshold
                )
                if not abundance_df.empty:
                    all_abundance_data.append(abundance_df)
            except Exception as e:
                logger.error(f"Error processing source {source.get('name', 'unknown')}: {e}")
                continue
        
        if not all_abundance_data:
            logger.warning("No abundance data processed successfully")
            return pd.DataFrame()
        
        # Combine all abundance data
        combined_df = pd.concat(all_abundance_data, ignore_index=True)
        
        # Save processed data
        output_file = self.data_dir / "processed_abundance_data.csv"
        combined_df.to_csv(output_file, index=False)
        logger.info(f"Saved processed abundance data to {output_file}")
        
        return combined_df
    
    def _process_single_source(self, 
                              source: Dict[str, Any],
                              target_taxonomic_levels: List[str],
                              min_abundance_threshold: float) -> pd.DataFrame:
        """Process abundance data from a single source."""
        source_type = source.get('type', 'unknown')
        source_name = source.get('name', 'unknown')
        
        logger.info(f"Processing {source_type} source: {source_name}")
        
        if source_type == 'mgnify':
            return self._process_mgnify_abundance(source, target_taxonomic_levels, min_abundance_threshold)
        elif source_type == 'dryad':
            return self._process_dryad_abundance(source, target_taxonomic_levels, min_abundance_threshold)
        elif source_type == 'supplementary':
            return self._process_supplementary_abundance(source, target_taxonomic_levels, min_abundance_threshold)
        else:
            logger.warning(f"Unknown source type: {source_type}")
            return pd.DataFrame()
    
    def _process_mgnify_abundance(self, 
                                 source: Dict[str, Any],
                                 target_taxonomic_levels: List[str],
                                 min_abundance_threshold: float) -> pd.DataFrame:
        """Process MGnify abundance data."""
        # This would integrate with MGnify API to get abundance tables
        # For now, create a mock implementation
        
        sample_id = source.get('sample_id')
        study_id = source.get('study_id')
        
        # Mock abundance data - in practice, this would come from MGnify API
        mock_abundance_data = self._create_mock_abundance_data(
            sample_id, study_id, target_taxonomic_levels
        )
        
        return mock_abundance_data
    
    def _process_dryad_abundance(self, 
                                source: Dict[str, Any],
                                target_taxonomic_levels: List[str],
                                min_abundance_threshold: float) -> pd.DataFrame:
        """Process Dryad abundance data."""
        url = source.get('url')
        if not url:
            logger.error("No URL provided for Dryad source")
            return pd.DataFrame()
        
        try:
            # Download and process Dryad data
            abundance_df = self._download_and_process_file(url, target_taxonomic_levels, min_abundance_threshold)
            return abundance_df
        except Exception as e:
            logger.error(f"Error processing Dryad data from {url}: {e}")
            return pd.DataFrame()
    
    def _process_supplementary_abundance(self, 
                                        source: Dict[str, Any],
                                        target_taxonomic_levels: List[str],
                                        min_abundance_threshold: float) -> pd.DataFrame:
        """Process supplementary file abundance data."""
        file_path = source.get('file_path')
        if not file_path:
            logger.error("No file path provided for supplementary source")
            return pd.DataFrame()
        
        try:
            abundance_df = self._process_file(file_path, target_taxonomic_levels, min_abundance_threshold)
            return abundance_df
        except Exception as e:
            logger.error(f"Error processing supplementary file {file_path}: {e}")
            return pd.DataFrame()
    
    def _download_and_process_file(self, 
                                  url: str,
                                  target_taxonomic_levels: List[str],
                                  min_abundance_threshold: float) -> pd.DataFrame:
        """Download and process a file from URL."""
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            # Determine file type and process accordingly
            if url.endswith('.zip'):
                return self._process_zip_file(response.content, target_taxonomic_levels, min_abundance_threshold)
            elif url.endswith(('.csv', '.tsv')):
                return self._process_csv_content(response.text, target_taxonomic_levels, min_abundance_threshold)
            else:
                logger.warning(f"Unsupported file type for URL: {url}")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error downloading file from {url}: {e}")
            return pd.DataFrame()
    
    def _process_file(self, 
                     file_path: str,
                     target_taxonomic_levels: List[str],
                     min_abundance_threshold: float) -> pd.DataFrame:
        """Process a local file."""
        file_path = Path(file_path)
        
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return pd.DataFrame()
        
        if file_path.suffix == '.zip':
            with open(file_path, 'rb') as f:
                return self._process_zip_file(f.read(), target_taxonomic_levels, min_abundance_threshold)
        elif file_path.suffix in ['.csv', '.tsv']:
            return self._process_csv_file(file_path, target_taxonomic_levels, min_abundance_threshold)
        else:
            logger.warning(f"Unsupported file type: {file_path.suffix}")
            return pd.DataFrame()
    
    def _process_zip_file(self, 
                         zip_content: bytes,
                         target_taxonomic_levels: List[str],
                         min_abundance_threshold: float) -> pd.DataFrame:
        """Process abundance data from a ZIP file."""
        try:
            with zipfile.ZipFile(io.BytesIO(zip_content)) as zip_file:
                # Look for abundance tables in the ZIP
                abundance_files = [f for f in zip_file.namelist() 
                                 if any(keyword in f.lower() for keyword in ['otu', 'asv', 'abundance', 'taxonomy'])]
                
                if not abundance_files:
                    logger.warning("No abundance files found in ZIP")
                    return pd.DataFrame()
                
                # Process the first abundance file found
                with zip_file.open(abundance_files[0]) as f:
                    content = f.read().decode('utf-8')
                    return self._process_csv_content(content, target_taxonomic_levels, min_abundance_threshold)
                    
        except Exception as e:
            logger.error(f"Error processing ZIP file: {e}")
            return pd.DataFrame()
    
    def _process_csv_file(self, 
                         file_path: Path,
                         target_taxonomic_levels: List[str],
                         min_abundance_threshold: float) -> pd.DataFrame:
        """Process abundance data from a CSV/TSV file."""
        try:
            # Try different separators
            for sep in ['\t', ',', ';']:
                try:
                    df = pd.read_csv(file_path, sep=sep)
                    if len(df.columns) > 1:  # Found a valid separator
                        break
                except:
                    continue
            else:
                logger.error(f"Could not parse file {file_path}")
                return pd.DataFrame()
            
            return self._process_abundance_dataframe(df, target_taxonomic_levels, min_abundance_threshold)
            
        except Exception as e:
            logger.error(f"Error processing CSV file {file_path}: {e}")
            return pd.DataFrame()
    
    def _process_csv_content(self, 
                           content: str,
                           target_taxonomic_levels: List[str],
                           min_abundance_threshold: float) -> pd.DataFrame:
        """Process abundance data from CSV content."""
        try:
            # Try different separators
            for sep in ['\t', ',', ';']:
                try:
                    df = pd.read_csv(io.StringIO(content), sep=sep)
                    if len(df.columns) > 1:  # Found a valid separator
                        break
                except:
                    continue
            else:
                logger.error("Could not parse CSV content")
                return pd.DataFrame()
            
            return self._process_abundance_dataframe(df, target_taxonomic_levels, min_abundance_threshold)
            
        except Exception as e:
            logger.error(f"Error processing CSV content: {e}")
            return pd.DataFrame()
    
    def _process_abundance_dataframe(self, 
                                   df: pd.DataFrame,
                                   target_taxonomic_levels: List[str],
                                   min_abundance_threshold: float) -> pd.DataFrame:
        """Process an abundance DataFrame."""
        # Identify taxonomic columns and abundance columns
        taxonomic_cols = self._identify_taxonomic_columns(df)
        abundance_cols = self._identify_abundance_columns(df)
        
        if not taxonomic_cols or not abundance_cols:
            logger.warning("Could not identify taxonomic or abundance columns")
            return pd.DataFrame()
        
        # Process each taxonomic level
        processed_data = []
        
        for level in target_taxonomic_levels:
            level_data = self._process_taxonomic_level(
                df, taxonomic_cols, abundance_cols, level, min_abundance_threshold
            )
            if not level_data.empty:
                processed_data.append(level_data)
        
        if not processed_data:
            logger.warning("No data processed for any taxonomic level")
            return pd.DataFrame()
        
        # Combine all levels
        combined_df = pd.concat(processed_data, ignore_index=True)
        
        # Calculate relative abundances
        combined_df = self._calculate_relative_abundances(combined_df)
        
        return combined_df
    
    def _identify_taxonomic_columns(self, df: pd.DataFrame) -> List[str]:
        """Identify taxonomic columns in the DataFrame."""
        taxonomic_keywords = ['taxonomy', 'taxon', 'lineage', 'classification', 'kingdom', 'phylum', 
                             'class', 'order', 'family', 'genus', 'species', 'otu', 'asv']
        
        taxonomic_cols = []
        for col in df.columns:
            if any(keyword in col.lower() for keyword in taxonomic_keywords):
                taxonomic_cols.append(col)
        
        return taxonomic_cols
    
    def _identify_abundance_columns(self, df: pd.DataFrame) -> List[str]:
        """Identify abundance columns in the DataFrame."""
        abundance_keywords = ['abundance', 'count', 'reads', 'frequency', 'relative', 'proportion']
        
        abundance_cols = []
        for col in df.columns:
            if any(keyword in col.lower() for keyword in abundance_keywords):
                abundance_cols.append(col)
        
        # If no abundance keywords found, assume numeric columns are abundances
        if not abundance_cols:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            abundance_cols = [col for col in numeric_cols if col not in self._identify_taxonomic_columns(df)]
        
        return abundance_cols
    
    def _process_taxonomic_level(self, 
                               df: pd.DataFrame,
                               taxonomic_cols: List[str],
                               abundance_cols: List[str],
                               level: str,
                               min_abundance_threshold: float) -> pd.DataFrame:
        """Process data for a specific taxonomic level."""
        # Find the taxonomic column for this level
        level_col = None
        for col in taxonomic_cols:
            if level in col.lower():
                level_col = col
                break
        
        if not level_col:
            logger.warning(f"No taxonomic column found for level: {level}")
            return pd.DataFrame()
        
        # Group by taxonomic level and sum abundances
        level_data = df.groupby(level_col)[abundance_cols].sum().reset_index()
        
        # Filter by minimum abundance threshold
        for col in abundance_cols:
            level_data = level_data[level_data[col] >= min_abundance_threshold]
        
        # Add taxonomic level information
        level_data['taxonomic_level'] = level
        level_data['taxon_name'] = level_data[level_col]
        
        # Rename abundance columns to be more descriptive
        level_data = level_data.rename(columns={col: f'{level}_{col}' for col in abundance_cols})
        
        return level_data
    
    def _calculate_relative_abundances(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate relative abundances from absolute counts."""
        abundance_cols = [col for col in df.columns if any(keyword in col.lower() 
                          for keyword in ['abundance', 'count', 'reads', 'frequency'])]
        
        if not abundance_cols:
            logger.warning("No abundance columns found for relative abundance calculation")
            return df
        
        # Calculate relative abundances
        for col in abundance_cols:
            total = df[col].sum()
            if total > 0:
                df[f'{col}_relative'] = df[col] / total
            else:
                df[f'{col}_relative'] = 0
        
        return df
    
    def _create_mock_abundance_data(self, 
                                   sample_id: str,
                                   study_id: str,
                                   target_taxonomic_levels: List[str]) -> pd.DataFrame:
        """Create mock abundance data for testing."""
        # This creates realistic mock data for testing
        np.random.seed(42)  # For reproducible results
        
        mock_data = []
        
        for level in target_taxonomic_levels:
            # Create mock taxa for this level
            if level == 'family':
                taxa = ['Pseudomonadaceae', 'Bradyrhizobiaceae', 'Rhizobiaceae', 
                       'Enterobacteriaceae', 'Bacillaceae', 'Actinomycetaceae']
            elif level == 'genus':
                taxa = ['Pseudomonas', 'Bradyrhizobium', 'Rhizobium', 
                       'Escherichia', 'Bacillus', 'Streptomyces']
            elif level == 'species':
                taxa = ['Pseudomonas fluorescens', 'Bradyrhizobium japonicum', 
                       'Rhizobium leguminosarum', 'Escherichia coli', 
                       'Bacillus subtilis', 'Streptomyces coelicolor']
            else:
                continue
            
            # Create mock abundances
            for taxon in taxa:
                abundance = np.random.exponential(0.1)  # Exponential distribution
                relative_abundance = abundance / sum([np.random.exponential(0.1) for _ in taxa])
                
                mock_data.append({
                    'sample_id': sample_id,
                    'study_id': study_id,
                    'taxonomic_level': level,
                    'taxon_name': taxon,
                    'abundance': abundance,
                    'relative_abundance': relative_abundance
                })
        
        return pd.DataFrame(mock_data)
    
    def create_abundance_summary(self, abundance_df: pd.DataFrame) -> Dict[str, Any]:
        """Create a summary of abundance data."""
        if abundance_df.empty:
            return {}
        
        summary = {
            'total_samples': abundance_df['sample_id'].nunique(),
            'total_studies': abundance_df['study_id'].nunique(),
            'taxonomic_levels': abundance_df['taxonomic_level'].unique().tolist(),
            'total_taxa': abundance_df['taxon_name'].nunique(),
            'abundance_statistics': {}
        }
        
        # Calculate statistics for each taxonomic level
        for level in abundance_df['taxonomic_level'].unique():
            level_data = abundance_df[abundance_df['taxonomic_level'] == level]
            summary['abundance_statistics'][level] = {
                'taxa_count': level_data['taxon_name'].nunique(),
                'mean_abundance': level_data['abundance'].mean(),
                'median_abundance': level_data['abundance'].median(),
                'mean_relative_abundance': level_data['relative_abundance'].mean(),
                'median_relative_abundance': level_data['relative_abundance'].median()
            }
        
        return summary
