"""
WorldClim data collector for climate variables.
"""
import logging
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import requests
import numpy as np
import rasterio
from rasterio.warp import transform
from .base_collector import BaseCollector

logger = logging.getLogger(__name__)


class WorldClimCollector(BaseCollector):
    """Collect climate data from WorldClim."""
    
    def __init__(self, config: Dict[str, Any], data_dir):
        super().__init__(config, data_dir)
        self.base_url = config["worldclim"]["base_url"]
        self.resolution = "30s"  # 30 arc-seconds resolution
        
    def collect(self, coordinates: List[Tuple[float, float]], 
                variables: List[str] = None) -> pd.DataFrame:
        """
        Collect climate data for given coordinates.
        
        Args:
            coordinates: List of (latitude, longitude) tuples
            variables: List of climate variables to collect
            
        Returns:
            DataFrame with climate data
        """
        if variables is None:
            variables = [
                'bio1', 'bio2', 'bio3', 'bio4', 'bio5', 'bio6', 'bio7',
                'bio8', 'bio9', 'bio10', 'bio11', 'bio12', 'bio13', 'bio14',
                'bio15', 'bio16', 'bio17', 'bio18', 'bio19'
            ]
            
        all_data = []
        
        for i, (lat, lon) in enumerate(coordinates):
            logger.info(f"Collecting climate data for point {i+1}/{len(coordinates)}: ({lat}, {lon})")
            
            try:
                climate_data = self._get_climate_data(lat, lon, variables)
                if climate_data:
                    climate_data.update({
                        'latitude': lat,
                        'longitude': lon
                    })
                    all_data.append(climate_data)
                    
            except Exception as e:
                logger.error(f"Error collecting climate data for ({lat}, {lon}): {e}")
                continue
                
        df = pd.DataFrame(all_data)
        if not df.empty:
            df = self._clean_data(df)
            
        return df
        
    def _get_climate_data(self, lat: float, lon: float, 
                         variables: List[str]) -> Optional[Dict]:
        """Get climate data for a single coordinate pair."""
        climate_data = {}
        
        for var in variables:
            try:
                # Download the raster file for this variable
                raster_path = self._download_raster(var)
                if not raster_path:
                    continue
                    
                # Extract value at the coordinate
                value = self._extract_raster_value(raster_path, lat, lon)
                if value is not None:
                    climate_data[f'climate_{var}'] = value
                else:
                    climate_data[f'climate_{var}'] = np.nan
                    
            except Exception as e:
                logger.error(f"Error getting {var} for ({lat}, {lon}): {e}")
                climate_data[f'climate_{var}'] = np.nan
                
        return climate_data if climate_data else None
        
    def _download_raster(self, variable: str) -> Optional[str]:
        """Download WorldClim raster file for a variable."""
        filename = f"wc2.1_{self.resolution}_{variable}.tif"
        filepath = self.data_dir / filename
        
        if filepath.exists():
            return str(filepath)
            
        url = f"{self.base_url}/{filename}"
        
        try:
            response = self._make_request(url)
            with open(filepath, 'wb') as f:
                f.write(response.content)
            logger.info(f"Downloaded {filename}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Error downloading {filename}: {e}")
            return None
            
    def _extract_raster_value(self, raster_path: str, lat: float, 
                             lon: float) -> Optional[float]:
        """Extract value from raster at given coordinates."""
        try:
            with rasterio.open(raster_path) as src:
                # Transform coordinates to raster CRS
                x, y = transform('EPSG:4326', src.crs, [lon], [lat])
                
                # Sample the raster
                values = list(src.sample([(x[0], y[0])]))
                if values and not np.isnan(values[0]):
                    return float(values[0])
                return None
                
        except Exception as e:
            logger.error(f"Error extracting raster value: {e}")
            return None
            
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize climate data."""
        # Rename columns to be more descriptive
        column_mapping = {
            'climate_bio1': 'mean_annual_temp_C',
            'climate_bio2': 'mean_diurnal_range_C',
            'climate_bio3': 'isothermality',
            'climate_bio4': 'temp_seasonality',
            'climate_bio5': 'max_temp_warmest_month_C',
            'climate_bio6': 'min_temp_coldest_month_C',
            'climate_bio7': 'temp_annual_range_C',
            'climate_bio8': 'mean_temp_wettest_quarter_C',
            'climate_bio9': 'mean_temp_driest_quarter_C',
            'climate_bio10': 'mean_temp_warmest_quarter_C',
            'climate_bio11': 'mean_temp_coldest_quarter_C',
            'climate_bio12': 'annual_precip_mm',
            'climate_bio13': 'precip_wettest_month_mm',
            'climate_bio14': 'precip_driest_month_mm',
            'climate_bio15': 'precip_seasonality',
            'climate_bio16': 'precip_wettest_quarter_mm',
            'climate_bio17': 'precip_driest_quarter_mm',
            'climate_bio18': 'precip_warmest_quarter_mm',
            'climate_bio19': 'precip_coldest_quarter_mm'
        }
        
        df = df.rename(columns=column_mapping)
        
        # Convert temperature from 0.1°C to °C
        temp_columns = [col for col in df.columns if 'temp' in col and col.endswith('_C')]
        for col in temp_columns:
            if col in df.columns:
                df[col] = df[col] / 10.0
                
        # Add source information
        df['source'] = 'WorldClim'
        
        return df
        
    def validate_data(self, data: pd.DataFrame) -> bool:
        """Validate climate data."""
        if data.empty:
            logger.warning("No climate data to validate")
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
            
        # Check for reasonable temperature values
        if 'mean_annual_temp_C' in data.columns:
            temp_values = data['mean_annual_temp_C'].dropna()
            if not temp_values.empty and not ((temp_values >= -50) & (temp_values <= 50)).all():
                logger.warning("Some temperature values are outside expected range (-50 to 50°C)")
                
        # Check for reasonable precipitation values
        if 'annual_precip_mm' in data.columns:
            precip_values = data['annual_precip_mm'].dropna()
            if not precip_values.empty and not ((precip_values >= 0) & (precip_values <= 10000)).all():
                logger.warning("Some precipitation values are outside expected range (0 to 10000mm)")
                
        logger.info(f"Climate data validation passed: {len(data)} records")
        return True