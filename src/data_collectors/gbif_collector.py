"""
GBIF data collector for truffle occurrence records.
"""
import logging
from typing import Dict, List, Optional, Any
import pandas as pd
import requests
from .base_collector import BaseCollector

logger = logging.getLogger(__name__)


class GBIFCollector(BaseCollector):
    """Collect truffle occurrence data from GBIF."""
    
    def __init__(self, config: Dict[str, Any], data_dir):
        super().__init__(config, data_dir)
        self.base_url = config["gbif"]["base_url"]
        
    def collect(self, species: List[str], limit: int = 10000, 
                country: Optional[str] = None, year_from: Optional[int] = None,
                year_to: Optional[int] = None) -> pd.DataFrame:
        """
        Collect truffle occurrence data from GBIF.
        
        Args:
            species: List of truffle species to search for
            limit: Maximum number of records to retrieve
            country: Country code to filter by (e.g., 'FR', 'IT', 'ES')
            year_from: Start year for date filter
            year_to: End year for date filter
            
        Returns:
            DataFrame with occurrence records
        """
        all_records = []
        
        for species_name in species:
            logger.info(f"Collecting GBIF data for {species_name}")
            
            # Search for species
            species_key = self._get_species_key(species_name)
            if not species_key:
                logger.warning(f"Species {species_name} not found in GBIF")
                continue
                
            # Get occurrence data
            records = self._get_occurrences(
                species_key, limit, country, year_from, year_to
            )
            all_records.extend(records)
            
        df = pd.DataFrame(all_records)
        if not df.empty:
            df = self._clean_data(df)
            
        return df
        
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
                return data['results'][0]['key']
            return None
            
        except Exception as e:
            logger.error(f"Error getting species key for {species_name}: {e}")
            return None
            
    def _get_occurrences(self, species_key: int, limit: int, 
                        country: Optional[str], year_from: Optional[int],
                        year_to: Optional[int]) -> List[Dict]:
        """Get occurrence records for a species."""
        url = f"{self.base_url}/occurrence/search"
        params = {
            'taxonKey': species_key,
            'limit': limit,
            'hasCoordinate': 'true',
            'hasGeospatialIssue': 'false'
        }
        
        if country:
            params['country'] = country
        if year_from:
            params['year'] = f"{year_from},{year_to or 2024}"
            
        all_records = []
        offset = 0
        
        while offset < limit:
            params['offset'] = offset
            current_limit = min(300, limit - offset)  # GBIF max is 300 per request
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
        
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize GBIF data."""
        # Select relevant columns
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
        
        # Filter out records with high coordinate uncertainty (>10km)
        if 'coordinate_uncertainty' in df.columns:
            df = df[
                (df['coordinate_uncertainty'].isna()) | 
                (df['coordinate_uncertainty'] <= 10000)
            ]
            
        return df
        
    def validate_data(self, data: pd.DataFrame) -> bool:
        """Validate GBIF data."""
        required_columns = ['species', 'latitude', 'longitude']
        
        if data.empty:
            logger.warning("No data to validate")
            return False
            
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            return False
            
        # Check coordinate ranges
        if not ((data['latitude'] >= -90) & (data['latitude'] <= 90)).all():
            logger.error("Invalid latitude values")
            return False
            
        if not ((data['longitude'] >= -180) & (data['longitude'] <= 180)).all():
            logger.error("Invalid longitude values")
            return False
            
        logger.info(f"Data validation passed: {len(data)} records")
        return True