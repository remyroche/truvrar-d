"""
SoilGrids data collector for soil properties.
"""
import logging
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import requests
import numpy as np
from .base_collector import BaseCollector

logger = logging.getLogger(__name__)


class SoilGridsCollector(BaseCollector):
    """Collect soil property data from SoilGrids API."""
    
    def __init__(self, config: Dict[str, Any], data_dir):
        super().__init__(config, data_dir)
        self.base_url = config["soilgrids"]["base_url"]
        
    def collect(self, coordinates: List[Tuple[float, float]], 
                variables: List[str] = None) -> pd.DataFrame:
        """
        Collect soil data for given coordinates.
        
        Args:
            coordinates: List of (latitude, longitude) tuples
            variables: List of soil variables to collect
            
        Returns:
            DataFrame with soil data
        """
        if variables is None:
            variables = [
                'phh2o', 'cac03', 'soc', 'nitrogen', 'phosporus',
                'sand', 'silt', 'clay', 'bdod', 'cec', 'cfvo'
            ]
            
        all_data = []
        
        for i, (lat, lon) in enumerate(coordinates):
            logger.info(f"Collecting soil data for point {i+1}/{len(coordinates)}: ({lat}, {lon})")
            
            try:
                soil_data = self._get_soil_data(lat, lon, variables)
                if soil_data:
                    soil_data.update({
                        'latitude': lat,
                        'longitude': lon
                    })
                    all_data.append(soil_data)
                    
            except Exception as e:
                logger.error(f"Error collecting soil data for ({lat}, {lon}): {e}")
                continue
                
        df = pd.DataFrame(all_data)
        if not df.empty:
            df = self._clean_data(df)
            
        return df
        
    def _get_soil_data(self, lat: float, lon: float, 
                      variables: List[str]) -> Optional[Dict]:
        """Get soil data for a single coordinate pair."""
        url = f"{self.base_url}/query"
        params = {
            'lon': lon,
            'lat': lat,
            'attributes': ','.join(variables)
        }
        
        try:
            response = self._make_request(url, params)
            data = response.json()
            
            if 'properties' not in data:
                logger.warning(f"No soil data available for ({lat}, {lon})")
                return None
                
            soil_data = {}
            properties = data['properties']
            
            for var in variables:
                if var in properties:
                    # Extract mean value from the first depth layer (0-5cm)
                    var_data = properties[var]
                    if 'mean' in var_data and var_data['mean']:
                        soil_data[f'soil_{var}'] = var_data['mean'][0]
                    else:
                        soil_data[f'soil_{var}'] = np.nan
                else:
                    soil_data[f'soil_{var}'] = np.nan
                    
            return soil_data
            
        except Exception as e:
            logger.error(f"Error getting soil data for ({lat}, {lon}): {e}")
            return None
            
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize soil data."""
        # Rename columns to be more descriptive
        column_mapping = {
            'soil_phh2o': 'soil_pH',
            'soil_cac03': 'soil_CaCO3_pct',
            'soil_soc': 'soil_OC_pct',
            'soil_nitrogen': 'soil_N_pct',
            'soil_phosporus': 'soil_P_mgkg',
            'soil_sand': 'soil_sand_pct',
            'soil_silt': 'soil_silt_pct',
            'soil_clay': 'soil_clay_pct',
            'soil_bdod': 'soil_bulk_density',
            'soil_cec': 'soil_CEC',
            'soil_cfvo': 'soil_coarse_fragments_pct'
        }
        
        df = df.rename(columns=column_mapping)
        
        # Convert percentage values to proper scale (0-100)
        percentage_columns = [
            'soil_CaCO3_pct', 'soil_OC_pct', 'soil_N_pct', 
            'soil_sand_pct', 'soil_silt_pct', 'soil_clay_pct',
            'soil_coarse_fragments_pct'
        ]
        
        for col in percentage_columns:
            if col in df.columns:
                df[col] = df[col] * 100  # Convert from 0-1 to 0-100
                
        # Add source information
        df['source'] = 'SoilGrids'
        
        return df
        
    def validate_data(self, data: pd.DataFrame) -> bool:
        """Validate soil data."""
        if data.empty:
            logger.warning("No soil data to validate")
            return False
            
        # Check for required columns
        required_columns = ['latitude', 'longitude']
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
            
        # Check for reasonable soil values
        if 'soil_pH' in data.columns:
            ph_values = data['soil_pH'].dropna()
            if not ph_values.empty and not ((ph_values >= 3) & (ph_values <= 10)).all():
                logger.warning("Some pH values are outside expected range (3-10)")
                
        logger.info(f"Soil data validation passed: {len(data)} records")
        return True