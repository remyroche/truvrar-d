"""
Data merger for combining different data sources.
"""
import logging
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np
from geopy.distance import geodesic

logger = logging.getLogger(__name__)


class DataMerger:
    """Merge data from different sources based on spatial proximity."""
    
    def __init__(self, max_distance_km: float = 1.0):
        self.max_distance_km = max_distance_km
        
    def merge_habitat_data(self, occurrence_data: pd.DataFrame,
                          soil_data: pd.DataFrame,
                          climate_data: pd.DataFrame) -> pd.DataFrame:
        """
        Merge occurrence, soil, and climate data based on spatial proximity.
        
        Args:
            occurrence_data: Truffle occurrence records
            soil_data: Soil property data
            climate_data: Climate data
            
        Returns:
            Merged DataFrame with all data
        """
        if occurrence_data.empty:
            return pd.DataFrame()
            
        logger.info("Merging habitat data from multiple sources")
        
        # Start with occurrence data as base
        merged_data = occurrence_data.copy()
        
        # Merge soil data
        if not soil_data.empty:
            merged_data = self._merge_spatial_data(merged_data, soil_data, 'soil')
            
        # Merge climate data
        if not climate_data.empty:
            merged_data = self._merge_spatial_data(merged_data, climate_data, 'climate')
            
        # Add data quality indicators
        merged_data = self._add_data_quality_indicators(merged_data)
        
        logger.info(f"Data merging complete: {len(merged_data)} records")
        return merged_data
        
    def _merge_spatial_data(self, base_data: pd.DataFrame, 
                           spatial_data: pd.DataFrame, 
                           data_type: str) -> pd.DataFrame:
        """Merge spatial data based on coordinate proximity."""
        if spatial_data.empty:
            return base_data
            
        # Create coordinate pairs for base data
        base_coords = list(zip(base_data['latitude'], base_data['longitude']))
        
        # Find nearest spatial data points
        merged_columns = []
        
        for i, (lat, lon) in enumerate(base_coords):
            # Find closest spatial data point
            closest_idx, distance = self._find_closest_point(
                lat, lon, spatial_data
            )
            
            if closest_idx is not None and distance <= self.max_distance_km:
                # Add spatial data columns
                spatial_row = spatial_data.iloc[closest_idx]
                for col in spatial_data.columns:
                    if col not in ['latitude', 'longitude', 'source']:
                        merged_columns.append({
                            'index': i,
                            'column': col,
                            'value': spatial_row[col],
                            'distance_km': distance
                        })
            else:
                # Add NaN values for missing data
                for col in spatial_data.columns:
                    if col not in ['latitude', 'longitude', 'source']:
                        merged_columns.append({
                            'index': i,
                            'column': col,
                            'value': np.nan,
                            'distance_km': np.nan
                        })
                        
        # Create DataFrame from merged columns
        merged_df = base_data.copy()
        
        for col_data in merged_columns:
            col_name = col_data['column']
            if col_name not in merged_df.columns:
                merged_df[col_name] = np.nan
                
            merged_df.loc[col_data['index'], col_name] = col_data['value']
            
        # Add distance information
        distance_col = f'{data_type}_distance_km'
        merged_df[distance_col] = np.nan
        
        for col_data in merged_columns:
            if not pd.isna(col_data['distance_km']):
                merged_df.loc[col_data['index'], distance_col] = col_data['distance_km']
                
        return merged_df
        
    def _find_closest_point(self, lat: float, lon: float, 
                           spatial_data: pd.DataFrame) -> Tuple[Optional[int], float]:
        """Find the closest point in spatial data to given coordinates."""
        if spatial_data.empty:
            return None, np.inf
            
        # Calculate distances
        distances = []
        for _, row in spatial_data.iterrows():
            distance = geodesic(
                (lat, lon), 
                (row['latitude'], row['longitude'])
            ).kilometers
            distances.append(distance)
            
        # Find minimum distance
        min_idx = np.argmin(distances)
        min_distance = distances[min_idx]
        
        return min_idx, min_distance
        
    def _add_data_quality_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add data quality indicators to the merged dataset."""
        # Count non-null values for each data source
        soil_cols = [col for col in data.columns if col.startswith('soil_')]
        climate_cols = [col for col in data.columns if col.startswith('climate_')]
        
        # Soil data completeness
        if soil_cols:
            data['soil_data_completeness'] = data[soil_cols].notna().sum(axis=1) / len(soil_cols)
        else:
            data['soil_data_completeness'] = 0
            
        # Climate data completeness
        if climate_cols:
            data['climate_data_completeness'] = data[climate_cols].notna().sum(axis=1) / len(climate_cols)
        else:
            data['climate_data_completeness'] = 0
            
        # Overall data completeness
        all_env_cols = soil_cols + climate_cols
        if all_env_cols:
            data['overall_data_completeness'] = data[all_env_cols].notna().sum(axis=1) / len(all_env_cols)
        else:
            data['overall_data_completeness'] = 0
            
        # Data source diversity
        sources = data['source'].value_counts()
        data['source_diversity'] = len(sources)
        
        return data
        
    def merge_with_existing_data(self, new_data: pd.DataFrame, 
                                existing_data: pd.DataFrame) -> pd.DataFrame:
        """Merge new data with existing dataset."""
        if existing_data.empty:
            return new_data
            
        # Combine datasets
        combined_data = pd.concat([existing_data, new_data], ignore_index=True)
        
        # Remove duplicates based on coordinates and species
        combined_data = combined_data.drop_duplicates(
            subset=['species', 'latitude', 'longitude'],
            keep='last'  # Keep the most recent data
        )
        
        return combined_data
        
    def spatial_join(self, left_data: pd.DataFrame, right_data: pd.DataFrame,
                    left_geom_col: str = 'geometry',
                    right_geom_col: str = 'geometry') -> pd.DataFrame:
        """Perform spatial join between two datasets."""
        try:
            import geopandas as gpd
            
            # Convert to GeoDataFrames
            left_gdf = gpd.GeoDataFrame(left_data) if not isinstance(left_data, gpd.GeoDataFrame) else left_data
            right_gdf = gpd.GeoDataFrame(right_data) if not isinstance(right_data, gpd.GeoDataFrame) else right_data
            
            # Ensure same CRS
            if left_gdf.crs != right_gdf.crs:
                right_gdf = right_gdf.to_crs(left_gdf.crs)
                
            # Perform spatial join
            joined_gdf = gpd.sjoin(left_gdf, right_gdf, how='left', predicate='intersects')
            
            return joined_gdf
            
        except ImportError:
            logger.warning("GeoPandas not available, falling back to coordinate-based merging")
            return self._merge_spatial_data(left_data, right_data, 'spatial')
            
    def aggregate_by_location(self, data: pd.DataFrame, 
                             group_cols: List[str] = None,
                             agg_functions: Dict[str, str] = None) -> pd.DataFrame:
        """Aggregate data by location to reduce duplicates."""
        if group_cols is None:
            group_cols = ['latitude', 'longitude', 'species']
            
        if agg_functions is None:
            # Default aggregation functions
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            agg_functions = {col: 'mean' for col in numeric_cols if col not in group_cols}
            
            # Special handling for categorical columns
            categorical_cols = data.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                if col not in group_cols:
                    agg_functions[col] = 'first'
                    
        # Group and aggregate
        aggregated_data = data.groupby(group_cols).agg(agg_functions).reset_index()
        
        return aggregated_data