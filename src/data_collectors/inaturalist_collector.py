"""
iNaturalist data collector for truffle occurrence records.
"""
import logging
from typing import Dict, List, Optional, Any
import pandas as pd
import requests
from .base_collector import BaseCollector

logger = logging.getLogger(__name__)


class iNaturalistCollector(BaseCollector):
    """Collect truffle occurrence data from iNaturalist."""
    
    def __init__(self, config: Dict[str, Any], data_dir):
        super().__init__(config, data_dir)
        self.base_url = config["inaturalist"]["base_url"]
        
    def collect(self, species: List[str], limit: int = 10000, 
                place_id: Optional[int] = None, year_from: Optional[int] = None,
                year_to: Optional[int] = None) -> pd.DataFrame:
        """
        Collect truffle occurrence data from iNaturalist.
        
        Args:
            species: List of truffle species to search for
            limit: Maximum number of records to retrieve
            place_id: iNaturalist place ID to filter by
            year_from: Start year for date filter
            year_to: End year for date filter
            
        Returns:
            DataFrame with occurrence records
        """
        all_records = []
        
        for species_name in species:
            logger.info(f"Collecting iNaturalist data for {species_name}")
            
            # Get taxon ID
            taxon_id = self._get_taxon_id(species_name)
            if not taxon_id:
                logger.warning(f"Species {species_name} not found in iNaturalist")
                continue
                
            # Get observation data
            records = self._get_observations(
                taxon_id, limit, place_id, year_from, year_to
            )
            all_records.extend(records)
            
        df = pd.DataFrame(all_records)
        if not df.empty:
            df = self._clean_data(df)
            
        return df
        
    def _get_taxon_id(self, species_name: str) -> Optional[int]:
        """Get iNaturalist taxon ID for a given species name."""
        url = f"{self.base_url}/taxa"
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
            
    def _get_observations(self, taxon_id: int, limit: int, 
                         place_id: Optional[int], year_from: Optional[int],
                         year_to: Optional[int]) -> List[Dict]:
        """Get observation records for a taxon."""
        url = f"{self.base_url}/observations"
        params = {
            'taxon_id': taxon_id,
            'per_page': 200,  # iNaturalist max per page
            'has_geo': 'true',
            'quality_grade': 'research,needs_id'
        }
        
        if place_id:
            params['place_id'] = place_id
        if year_from:
            params['year'] = f"{year_from},{year_to or 2024}"
            
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
                        
                    record = self._extract_observation_data(obs)
                    if record:
                        all_records.append(record)
                        
                page += 1
                
                logger.info(f"Retrieved {len(observations)} observations (total: {len(all_records)})")
                
            except Exception as e:
                logger.error(f"Error retrieving observations: {e}")
                break
                
        return all_records
        
    def _extract_observation_data(self, obs: Dict) -> Optional[Dict]:
        """Extract relevant data from an observation record."""
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
                'source': 'iNaturalist'
            }
            
        except Exception as e:
            logger.error(f"Error extracting observation data: {e}")
            return None
            
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
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
        
        # Add coordinate uncertainty estimate (iNaturalist doesn't provide this)
        df['coordinate_uncertainty'] = 100  # Assume 100m uncertainty for iNaturalist
        
        return df
        
    def validate_data(self, data: pd.DataFrame) -> bool:
        """Validate iNaturalist data."""
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