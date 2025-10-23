"""
Enhanced GBIF Truffle Data Downloader

This module provides a comprehensive interface for downloading truffle occurrence
data from GBIF with advanced filtering, validation, and export capabilities.
"""

import logging
import time
import pandas as pd
import requests
from typing import Dict, List, Optional, Union, Any, Tuple
from pathlib import Path
import json
from datetime import datetime, timedelta
import yaml

logger = logging.getLogger(__name__)


class GBIFTruffleDownloader:
    """
    Enhanced GBIF data downloader specifically designed for truffle occurrence data.
    
    Features:
    - Species-specific data collection
    - Geographic and temporal filtering
    - Data quality validation
    - Multiple export formats
    - Progress tracking and error handling
    """
    
    def __init__(self, config_file: Optional[Union[str, Path]] = None):
        """
        Initialize the GBIF Truffle Downloader.
        
        Args:
            config_file: Path to configuration file (YAML or JSON)
        """
        self.config = self._load_config(config_file)
        self.session = self._create_session()
        self.base_url = self.config['gbif']['base_url']
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'records_downloaded': 0,
            'start_time': None,
            'end_time': None
        }
        
    def _load_config(self, config_file: Optional[Union[str, Path]]) -> Dict[str, Any]:
        """Load configuration from file or use defaults."""
        default_config = {
            'gbif': {
                'base_url': 'https://api.gbif.org/v1',
                'timeout': 30,
                'max_retries': 3,
                'rate_limit': 1000
            },
            'download': {
                'batch_size': 300,
                'max_records': 10000,
                'coordinate_uncertainty_max': 10000,
                'include_duplicates': False,
                'has_coordinate': True,
                'has_geospatial_issue': False
            },
            'output': {
                'default_format': 'csv',
                'include_metadata': True,
                'compress_output': False
            },
            'logging': {
                'level': 'INFO',
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            }
        }
        
        if config_file and Path(config_file).exists():
            config_path = Path(config_file)
            with open(config_path, 'r') as f:
                if config_path.suffix.lower() in ['.yaml', '.yml']:
                    user_config = yaml.safe_load(f)
                else:
                    user_config = json.load(f)
            
            # Merge user config with defaults
            for key, value in user_config.items():
                if key in default_config and isinstance(value, dict):
                    default_config[key].update(value)
                else:
                    default_config[key] = value
                    
        return default_config
    
    def _create_session(self) -> requests.Session:
        """Create a requests session with proper headers and configuration."""
        session = requests.Session()
        session.headers.update({
            'User-Agent': f'TruffleDownloader/{__version__} (https://github.com/truffle-research/truffle-occurrence-downloader)',
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        })
        return session
    
    def download_species(self, 
                        species: Union[str, List[str]], 
                        countries: Optional[List[str]] = None,
                        year_from: Optional[int] = None,
                        year_to: Optional[int] = None,
                        coordinate_bounds: Optional[Dict[str, float]] = None,
                        max_records: Optional[int] = None,
                        **kwargs) -> pd.DataFrame:
        """
        Download occurrence data for specified truffle species.
        
        Args:
            species: Species name(s) to download (e.g., "Tuber melanosporum" or ["Tuber melanosporum", "Tuber magnatum"])
            countries: List of country codes to filter by (e.g., ["FR", "IT", "ES"])
            year_from: Start year for temporal filtering
            year_to: End year for temporal filtering
            coordinate_bounds: Dictionary with 'min_lat', 'max_lat', 'min_lon', 'max_lon' for geographic bounds
            max_records: Maximum number of records to download
            **kwargs: Additional filtering parameters
            
        Returns:
            DataFrame with occurrence records
        """
        if isinstance(species, str):
            species = [species]
            
        self.stats['start_time'] = datetime.now()
        logger.info(f"Starting download for {len(species)} species: {species}")
        
        all_records = []
        
        for species_name in species:
            logger.info(f"Downloading data for {species_name}")
            species_records = self._download_single_species(
                species_name, countries, year_from, year_to, 
                coordinate_bounds, max_records, **kwargs
            )
            all_records.extend(species_records)
            logger.info(f"Downloaded {len(species_records)} records for {species_name}")
        
        if not all_records:
            logger.warning("No records found for the specified criteria")
            return pd.DataFrame()
        
        # Convert to DataFrame and clean data
        df = pd.DataFrame(all_records)
        df = self._clean_and_validate_data(df)
        
        self.stats['end_time'] = datetime.now()
        self.stats['records_downloaded'] = len(df)
        
        logger.info(f"Download complete: {len(df)} total records in {self._get_duration()}")
        return df
    
    def _download_single_species(self, 
                                species_name: str,
                                countries: Optional[List[str]] = None,
                                year_from: Optional[int] = None,
                                year_to: Optional[int] = None,
                                coordinate_bounds: Optional[Dict[str, float]] = None,
                                max_records: Optional[int] = None,
                                **kwargs) -> List[Dict]:
        """Download occurrence data for a single species."""
        
        # Get species key
        species_key = self._get_species_key(species_name)
        if not species_key:
            logger.warning(f"Species {species_name} not found in GBIF")
            return []
        
        # Build search parameters
        params = self._build_search_params(
            species_key, countries, year_from, year_to, 
            coordinate_bounds, max_records, **kwargs
        )
        
        # Download records in batches
        all_records = []
        offset = 0
        batch_size = self.config['download']['batch_size']
        max_records = max_records or self.config['download']['max_records']
        
        while offset < max_records:
            params['offset'] = offset
            params['limit'] = min(batch_size, max_records - offset)
            
            try:
                records = self._make_occurrence_request(params)
                if not records:
                    break
                    
                all_records.extend(records)
                offset += len(records)
                
                logger.info(f"Downloaded {len(records)} records (total: {len(all_records)})")
                
                # Rate limiting
                time.sleep(1.0 / self.config['gbif']['rate_limit'])
                
            except Exception as e:
                logger.error(f"Error downloading records for {species_name}: {e}")
                break
        
        return all_records
    
    def _get_species_key(self, species_name: str) -> Optional[int]:
        """Get GBIF species key for a given species name."""
        url = f"{self.base_url}/species/search"
        params = {
            'q': species_name,
            'rank': 'SPECIES',
            'limit': 1
        }
        
        try:
            response = self._make_request(url, params)
            data = response.json()
            
            if data['results']:
                species_key = data['results'][0]['key']
                logger.info(f"Found species key {species_key} for {species_name}")
                return species_key
            return None
            
        except Exception as e:
            logger.error(f"Error getting species key for {species_name}: {e}")
            return None
    
    def _build_search_params(self, 
                           species_key: int,
                           countries: Optional[List[str]] = None,
                           year_from: Optional[int] = None,
                           year_to: Optional[int] = None,
                           coordinate_bounds: Optional[Dict[str, float]] = None,
                           max_records: Optional[int] = None,
                           **kwargs) -> Dict[str, Any]:
        """Build search parameters for GBIF occurrence search."""
        params = {
            'taxonKey': species_key,
            'hasCoordinate': self.config['download']['has_coordinate'],
            'hasGeospatialIssue': self.config['download']['has_geospatial_issue']
        }
        
        # Add country filter
        if countries:
            params['country'] = '|'.join(countries)
        
        # Add temporal filter
        if year_from and year_to:
            params['year'] = f"{year_from},{year_to}"
        elif year_from:
            params['year'] = f"{year_from},*"
        elif year_to:
            params['year'] = f"*,{year_to}"
        
        # Add coordinate bounds
        if coordinate_bounds:
            if 'min_lat' in coordinate_bounds:
                params['decimalLatitude'] = f"{coordinate_bounds['min_lat']},{coordinate_bounds['max_lat']}"
            if 'min_lon' in coordinate_bounds:
                params['decimalLongitude'] = f"{coordinate_bounds['min_lon']},{coordinate_bounds['max_lon']}"
        
        # Add additional filters
        for key, value in kwargs.items():
            if value is not None:
                params[key] = value
        
        return params
    
    def _make_occurrence_request(self, params: Dict[str, Any]) -> List[Dict]:
        """Make a request to GBIF occurrence search API."""
        url = f"{self.base_url}/occurrence/search"
        
        try:
            response = self._make_request(url, params)
            data = response.json()
            return data.get('results', [])
        except Exception as e:
            logger.error(f"Error making occurrence request: {e}")
            return []
    
    def _make_request(self, url: str, params: Dict[str, Any]) -> requests.Response:
        """Make HTTP request with retry logic."""
        max_retries = self.config['gbif']['max_retries']
        timeout = self.config['gbif']['timeout']
        
        for attempt in range(max_retries):
            try:
                self.stats['total_requests'] += 1
                response = self.session.get(url, params=params, timeout=timeout)
                response.raise_for_status()
                self.stats['successful_requests'] += 1
                return response
            except requests.exceptions.RequestException as e:
                self.stats['failed_requests'] += 1
                logger.warning(f"Request attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    raise
                time.sleep(2 ** attempt)  # Exponential backoff
    
    def _clean_and_validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate the downloaded data."""
        if df.empty:
            return df
        
        # Select and rename relevant columns
        column_mapping = {
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
            'gbifID': 'gbif_id',
            'datasetKey': 'dataset_key',
            'publishingOrgKey': 'publishing_org_key'
        }
        
        # Rename columns
        available_columns = {k: v for k, v in column_mapping.items() if k in df.columns}
        df = df.rename(columns=available_columns)
        
        # Add missing columns with default values
        for col in column_mapping.values():
            if col not in df.columns:
                df[col] = None
        
        # Convert coordinates to numeric
        df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
        df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
        
        # Remove records without coordinates
        initial_count = len(df)
        df = df.dropna(subset=['latitude', 'longitude'])
        logger.info(f"Removed {initial_count - len(df)} records without coordinates")
        
        # Filter by coordinate uncertainty
        if 'coordinate_uncertainty' in df.columns:
            max_uncertainty = self.config['download']['coordinate_uncertainty_max']
            before_count = len(df)
            df = df[
                (df['coordinate_uncertainty'].isna()) | 
                (df['coordinate_uncertainty'] <= max_uncertainty)
            ]
            logger.info(f"Removed {before_count - len(df)} records with high coordinate uncertainty")
        
        # Convert date columns
        if 'event_date' in df.columns:
            df['event_date'] = pd.to_datetime(df['event_date'], errors='coerce')
        
        # Add source information
        df['source'] = 'GBIF'
        df['download_date'] = datetime.now().isoformat()
        
        # Remove duplicates if configured
        if not self.config['download']['include_duplicates']:
            before_count = len(df)
            df = df.drop_duplicates(subset=['gbif_id'], keep='first')
            logger.info(f"Removed {before_count - len(df)} duplicate records")
        
        return df
    
    def get_download_stats(self) -> Dict[str, Any]:
        """Get download statistics."""
        stats = self.stats.copy()
        if stats['start_time'] and stats['end_time']:
            stats['duration'] = self._get_duration()
        return stats
    
    def _get_duration(self) -> str:
        """Get formatted duration string."""
        if self.stats['start_time'] and self.stats['end_time']:
            duration = self.stats['end_time'] - self.stats['start_time']
            return str(duration).split('.')[0]  # Remove microseconds
        return "Unknown"
    
    def search_species(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search for species in GBIF by name.
        
        Args:
            query: Species name or partial name to search for
            limit: Maximum number of results to return
            
        Returns:
            List of species information dictionaries
        """
        url = f"{self.base_url}/species/search"
        params = {
            'q': query,
            'rank': 'SPECIES',
            'limit': limit
        }
        
        try:
            response = self._make_request(url, params)
            data = response.json()
            return data.get('results', [])
        except Exception as e:
            logger.error(f"Error searching for species: {e}")
            return []
    
    def get_species_info(self, species_key: int) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a species.
        
        Args:
            species_key: GBIF species key
            
        Returns:
            Species information dictionary or None if not found
        """
        url = f"{self.base_url}/species/{species_key}"
        
        try:
            response = self._make_request(url, {})
            return response.json()
        except Exception as e:
            logger.error(f"Error getting species info for key {species_key}: {e}")
            return None