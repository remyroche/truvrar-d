"""
Biodiversity data collector for non-academic sources.

This collector specializes in gathering data from biodiversity and citizen science sources including:
- GBIF
- iNaturalist
- eBird
- iNaturalist Research Grade
- Citizen science platforms
"""
import logging
import pandas as pd
import requests
import time
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path
import json
from datetime import datetime, timezone

from .base_collector import BaseCollector
from ..utils.error_handling import handle_api_errors, retry_on_failure, APIError, DataCollectionError

logger = logging.getLogger(__name__)


class BiodiversityDataCollector(BaseCollector):
    """Specialized collector for biodiversity and citizen science data sources."""
    
    def __init__(self, config: Dict[str, Any], data_dir: Path):
        super().__init__(config, data_dir)
        self.biodiversity_sources = {
            'gbif': self._collect_gbif_data,
            'inaturalist': self._collect_inaturalist_data,
            'ebird': self._collect_ebird_data,
            'citizen_science': self._collect_citizen_science_data
        }
        
    def collect_biodiversity_data(self, source: str, species: List[str], 
                                limit: int = 10000, **kwargs) -> Union[pd.DataFrame, Dict[str, Any]]:
        """
        Collect data from biodiversity sources.
        
        Args:
            source: Biodiversity source name
            species: List of species to search for
            limit: Maximum number of records to collect
            **kwargs: Additional source-specific parameters
            
        Returns:
            DataFrame with biodiversity data or harmonized results
        """
        if source not in self.biodiversity_sources:
            raise ValueError(f"Unknown biodiversity source: {source}")
        
        logger.info(f"Collecting biodiversity data from {source} for species: {species}")
        
        try:
            data = self.biodiversity_sources[source](species, limit, **kwargs)
            
            if data.empty:
                logger.warning(f"No biodiversity data found for {source}")
                return pd.DataFrame()
            
            # Add biodiversity-specific metadata
            data = self._add_biodiversity_metadata(data, source)
            
            # Validate biodiversity data
            if not self.validate_data(data):
                logger.warning(f"Biodiversity data validation failed for {source}")
                return pd.DataFrame()
            
            logger.info(f"Collected {len(data)} biodiversity records from {source}")
            return data
            
        except Exception as e:
            logger.error(f"Error collecting biodiversity data from {source}: {e}")
            raise DataCollectionError(f"Failed to collect biodiversity data: {e}")
    
    @handle_api_errors
    @retry_on_failure(max_retries=3, delay=1.0)
    def _collect_gbif_data(self, species: List[str], limit: int, 
                          **kwargs) -> pd.DataFrame:
        """Collect data from GBIF."""
        base_url = self.config["gbif"]["base_url"]
        all_records = []
        
        for species_name in species:
            logger.info(f"Collecting GBIF data for {species_name}")
            
            # Get species key
            species_key = self._get_gbif_species_key(species_name, base_url)
            if not species_key:
                logger.warning(f"Species {species_name} not found in GBIF")
                continue
            
            # Get occurrence data
            records = self._get_gbif_occurrences(
                species_key, base_url, limit, **kwargs
            )
            all_records.extend(records)
        
        df = pd.DataFrame(all_records)
        if not df.empty:
            df = self._clean_gbif_data(df)
        
        return df
    
    def _get_gbif_species_key(self, species_name: str, base_url: str) -> Optional[int]:
        """Get GBIF species key for a given species name."""
        url = f"{base_url}/species/search"
        params = {
            'q': species_name,
            'rank': 'SPECIES',
            'limit': 1
        }
        
        try:
            response = self._make_request(url, params)
            data = response.json()
            
            if data['results']:
                return data['results'][0]['key']
            return None
            
        except Exception as e:
            logger.error(f"Error getting species key for {species_name}: {e}")
            return None
    
    def _get_gbif_occurrences(self, species_key: int, base_url: str, limit: int, 
                             **kwargs) -> List[Dict]:
        """Get GBIF occurrence records for a species."""
        url = f"{base_url}/occurrence/search"
        params = {
            'taxonKey': species_key,
            'limit': limit,
            'hasCoordinate': 'true',
            'hasGeospatialIssue': 'false'
        }
        
        # Add additional parameters
        if 'country' in kwargs:
            params['country'] = kwargs['country']
        if 'year_from' in kwargs and 'year_to' in kwargs:
            params['year'] = f"{kwargs['year_from']},{kwargs['year_to']}"
        
        all_records = []
        offset = 0
        
        while offset < limit:
            params['offset'] = offset
            current_limit = min(300, limit - offset)
            params['limit'] = current_limit
            
            try:
                response = self._make_request(url, params)
                data = response.json()
                
                records = data.get('results', [])
                if not records:
                    break
                
                all_records.extend(records)
                offset += len(records)
                
                logger.info(f"Retrieved {len(records)} records (total: {len(all_records)})")
                
            except Exception as e:
                logger.error(f"Error retrieving occurrences: {e}")
                break
        
        return all_records
    
    def _clean_gbif_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize GBIF data."""
        columns_mapping = {
            'species': 'species',
            'decimalLatitude': 'latitude',
            'decimalLongitude': 'longitude',
            'eventDate': 'event_date',
            'year': 'year',
            'month': 'month',
            'day': 'day',
            'country': 'country',
            'stateProvince': 'state_province',
            'locality': 'locality',
            'coordinateUncertaintyInMeters': 'coordinate_uncertainty',
            'basisOfRecord': 'basis_of_record',
            'institutionCode': 'institution_code',
            'collectionCode': 'collection_code',
            'catalogNumber': 'catalog_number',
            'recordedBy': 'recorded_by',
            'identifiedBy': 'identified_by',
            'gbifID': 'gbif_id'
        }
        
        # Rename columns
        available_columns = {k: v for k, v in columns_mapping.items() if k in df.columns}
        df = df.rename(columns=available_columns)
        
        # Add missing columns with default values
        for col in columns_mapping.values():
            if col not in df.columns:
                df[col] = None
        
        # Convert coordinates to numeric
        df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
        df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
        
        # Remove records without coordinates
        df = df.dropna(subset=['latitude', 'longitude'])
        
        # Convert date columns
        if 'event_date' in df.columns:
            df['event_date'] = pd.to_datetime(df['event_date'], errors='coerce')
        
        # Add source information
        df['source'] = 'GBIF'
        df['data_type'] = 'biodiversity_occurrence'
        
        # Filter out records with high coordinate uncertainty (>10km)
        if 'coordinate_uncertainty' in df.columns:
            df = df[
                (df['coordinate_uncertainty'].isna()) | 
                (df['coordinate_uncertainty'] <= 10000)
            ]
        
        return df
    
    @handle_api_errors
    @retry_on_failure(max_retries=3, delay=1.0)
    def _collect_inaturalist_data(self, species: List[str], limit: int, 
                                 **kwargs) -> pd.DataFrame:
        """Collect data from iNaturalist."""
        base_url = self.config["inaturalist"]["base_url"]
        all_records = []
        
        for species_name in species:
            logger.info(f"Collecting iNaturalist data for {species_name}")
            
            # Get taxon ID
            taxon_id = self._get_inaturalist_taxon_id(species_name, base_url)
            if not taxon_id:
                logger.warning(f"Species {species_name} not found in iNaturalist")
                continue
            
            # Get observation data
            records = self._get_inaturalist_observations(
                taxon_id, base_url, limit, **kwargs
            )
            all_records.extend(records)
        
        df = pd.DataFrame(all_records)
        if not df.empty:
            df = self._clean_inaturalist_data(df)
        
        return df
    
    def _get_inaturalist_taxon_id(self, species_name: str, base_url: str) -> Optional[int]:
        """Get iNaturalist taxon ID for a given species name."""
        url = f"{base_url}/taxa"
        params = {
            'q': species_name,
            'rank': 'species',
            'per_page': 1
        }
        
        try:
            response = self._make_request(url, params)
            data = response.json()
            
            if data['results']:
                return data['results'][0]['id']
            return None
            
        except Exception as e:
            logger.error(f"Error getting taxon ID for {species_name}: {e}")
            return None
    
    def _get_inaturalist_observations(self, taxon_id: int, base_url: str, limit: int, 
                                     **kwargs) -> List[Dict]:
        """Get iNaturalist observation records for a taxon."""
        url = f"{base_url}/observations"
        params = {
            'taxon_id': taxon_id,
            'per_page': 200,
            'has_geo': 'true',
            'quality_grade': 'research,needs_id'
        }
        
        # Add additional parameters
        if 'place_id' in kwargs:
            params['place_id'] = kwargs['place_id']
        if 'year_from' in kwargs and 'year_to' in kwargs:
            params['year'] = f"{kwargs['year_from']},{kwargs['year_to']}"
        
        all_records = []
        page = 1
        
        while len(all_records) < limit:
            params['page'] = page
            
            try:
                response = self._make_request(url, params)
                data = response.json()
                
                observations = data.get('results', [])
                if not observations:
                    break
                
                for obs in observations:
                    if len(all_records) >= limit:
                        break
                    
                    record = self._extract_inaturalist_observation_data(obs)
                    if record:
                        all_records.append(record)
                
                page += 1
                
                logger.info(f"Retrieved {len(observations)} observations (total: {len(all_records)})")
                
            except Exception as e:
                logger.error(f"Error retrieving observations: {e}")
                break
        
        return all_records
    
    def _extract_inaturalist_observation_data(self, obs: Dict) -> Optional[Dict]:
        """Extract relevant data from an iNaturalist observation record."""
        try:
            # Check if observation has coordinates
            if not obs.get('geojson') or not obs['geojson'].get('coordinates'):
                return None
            
            coords = obs['geojson']['coordinates']
            if len(coords) != 2:
                return None
            
            lon, lat = coords
            
            # Extract species information
            taxon = obs.get('taxon', {})
            species_name = taxon.get('name', '')
            
            # Extract date information
            observed_on = obs.get('observed_on', '')
            if observed_on:
                try:
                    from datetime import datetime
                    date_obj = datetime.fromisoformat(observed_on.replace('Z', '+00:00'))
                    year = date_obj.year
                    month = date_obj.month
                    day = date_obj.day
                except:
                    year = month = day = None
            else:
                year = month = day = None
            
            # Extract location information
            place_guess = obs.get('place_guess', '')
            location = obs.get('location', '')
            
            # Extract quality information
            quality_grade = obs.get('quality_grade', '')
            num_identifications = obs.get('num_identification_agreements', 0)
            num_disagreements = obs.get('num_identification_disagreements', 0)
            
            # Extract observer information
            user = obs.get('user', {})
            observer = user.get('login', '')
            
            return {
                'species': species_name,
                'latitude': lat,
                'longitude': lon,
                'event_date': observed_on,
                'year': year,
                'month': month,
                'day': day,
                'place_guess': place_guess,
                'location': location,
                'quality_grade': quality_grade,
                'num_identifications': num_identifications,
                'num_disagreements': num_disagreements,
                'observer': observer,
                'inat_id': obs.get('id'),
                'url': obs.get('uri', ''),
                'source': 'iNaturalist',
                'data_type': 'citizen_science_observation'
            }
            
        except Exception as e:
            logger.error(f"Error extracting observation data: {e}")
            return None
    
    def _clean_inaturalist_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize iNaturalist data."""
        # Convert coordinates to numeric
        df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
        df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
        
        # Remove records without coordinates
        df = df.dropna(subset=['latitude', 'longitude'])
        
        # Convert date columns
        if 'event_date' in df.columns:
            df['event_date'] = pd.to_datetime(df['event_date'], errors='coerce')
        
        # Filter out low-quality records
        df = df[df['quality_grade'].isin(['research', 'needs_id'])]
        
        # Add coordinate uncertainty estimate
        df['coordinate_uncertainty'] = 100  # Assume 100m uncertainty for iNaturalist
        
        return df
    
    @handle_api_errors
    def _collect_ebird_data(self, species: List[str], limit: int, 
                           **kwargs) -> pd.DataFrame:
        """Collect data from eBird (if API available)."""
        logger.warning("eBird collection not implemented - requires API key")
        return pd.DataFrame()
    
    @handle_api_errors
    def _collect_citizen_science_data(self, species: List[str], limit: int, 
                                    **kwargs) -> pd.DataFrame:
        """Collect data from other citizen science platforms."""
        logger.warning("Citizen science collection not implemented - requires platform-specific APIs")
        return pd.DataFrame()
    
    def _add_biodiversity_metadata(self, data: pd.DataFrame, source: str) -> pd.DataFrame:
        """Add biodiversity-specific metadata."""
        data = data.copy()
        
        # Add collection metadata
        data['collection_date'] = datetime.now(timezone.utc).isoformat()
        data['collection_source'] = source
        data['data_category'] = 'biodiversity'
        
        # Add quality indicators
        data['has_coordinates'] = data['latitude'].notna() & data['longitude'].notna()
        data['has_date'] = data['event_date'].notna()
        data['has_observer'] = data.get('recorded_by', '').notna() | data.get('observer', '').notna()
        
        # Calculate biodiversity quality score
        quality_indicators = ['has_coordinates', 'has_date', 'has_observer']
        available_indicators = [col for col in quality_indicators if col in data.columns]
        if available_indicators:
            data['biodiversity_quality_score'] = data[available_indicators].sum(axis=1) / len(available_indicators)
        else:
            data['biodiversity_quality_score'] = 0.5
        
        return data
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """Validate biodiversity data."""
        if data.empty:
            return False
        
        # Check for required biodiversity columns
        required_columns = ['species', 'latitude', 'longitude']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            logger.error(f"Missing required biodiversity columns: {missing_columns}")
            return False
        
        # Check coordinate validity
        if not ((data['latitude'] >= -90) & (data['latitude'] <= 90)).all():
            logger.error("Invalid latitude values")
            return False
        
        if not ((data['longitude'] >= -180) & (data['longitude'] <= 180)).all():
            logger.error("Invalid longitude values")
            return False
        
        logger.info(f"Biodiversity data validation passed: {len(data)} records")
        return True