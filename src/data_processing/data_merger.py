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
    def merge_microbiome_environmental_data(self, 
                                           microbiome_data: pd.DataFrame,
                                           soil_data: pd.DataFrame,
                                           climate_data: pd.DataFrame,
                                           glim_data: pd.DataFrame,
                                           abundance_data: pd.DataFrame = None) -> pd.DataFrame:
        """
        Merge microbiome data with environmental layers (soil, climate, geology).
        
        Args:
            microbiome_data: Microbiome data from EBI Metagenomics
            soil_data: Soil data from SoilGrids
            climate_data: Climate data from WorldClim
            glim_data: Geological data from GLiM
            abundance_data: Processed abundance data (optional)
            
        Returns:
            Merged DataFrame with microbiome-environmental data
        """
        if microbiome_data.empty:
            logger.warning("No microbiome data provided")
            return pd.DataFrame()
            
        logger.info("Merging microbiome data with environmental layers")
        
        # Start with microbiome data as base
        merged_data = microbiome_data.copy()
        
        # Add soil data for samples with coordinates
        if not soil_data.empty:
            merged_data = self._merge_soil_data(merged_data, soil_data)
        
        # Add climate data for samples with coordinates
        if not climate_data.empty:
            merged_data = self._merge_climate_data(merged_data, climate_data)
        
        # Add geological data for samples with coordinates
        if not glim_data.empty:
            merged_data = self._merge_geological_data(merged_data, glim_data)
        
        # Add abundance data if provided
        if abundance_data is not None and not abundance_data.empty:
            merged_data = self._merge_abundance_data(merged_data, abundance_data)
        
        # Create derived environmental variables
        merged_data = self._create_environmental_derivatives(merged_data)
        
        # Create microbiome-environmental summary
        merged_data = self._create_microbiome_summary(merged_data)
        
        logger.info(f"Merged microbiome data: {len(merged_data)} records")
        return merged_data
    
    def _merge_soil_data(self, microbiome_data: pd.DataFrame, soil_data: pd.DataFrame) -> pd.DataFrame:
        """Merge soil data with microbiome data."""
        # Find samples with coordinates
        coords_mask = microbiome_data['latitude'].notna() & microbiome_data['longitude'].notna()
        coords_data = microbiome_data[coords_mask].copy()
        
        if coords_data.empty:
            logger.warning("No samples with coordinates for soil data merging")
            return microbiome_data
        
        # Merge soil data based on coordinates
        merged_coords = self._merge_spatial_data(coords_data, soil_data, 
                                               lat_col='latitude', lon_col='longitude',
                                               buffer_distance=0.01)
        
        # Update the original dataframe
        microbiome_data.loc[coords_mask, merged_coords.columns] = merged_coords
        
        return microbiome_data
    
    def _merge_climate_data(self, microbiome_data: pd.DataFrame, climate_data: pd.DataFrame) -> pd.DataFrame:
        """Merge climate data with microbiome data."""
        # Find samples with coordinates
        coords_mask = microbiome_data['latitude'].notna() & microbiome_data['longitude'].notna()
        coords_data = microbiome_data[coords_mask].copy()
        
        if coords_data.empty:
            logger.warning("No samples with coordinates for climate data merging")
            return microbiome_data
        
        # Merge climate data based on coordinates
        merged_coords = self._merge_spatial_data(coords_data, climate_data,
                                               lat_col='latitude', lon_col='longitude',
                                               buffer_distance=0.01)
        
        # Update the original dataframe
        microbiome_data.loc[coords_mask, merged_coords.columns] = merged_coords
        
        return microbiome_data
    
    def _merge_geological_data(self, microbiome_data: pd.DataFrame, glim_data: pd.DataFrame) -> pd.DataFrame:
        """Merge geological data with microbiome data."""
        # Find samples with coordinates
        coords_mask = microbiome_data['latitude'].notna() & microbiome_data['longitude'].notna()
        coords_data = microbiome_data[coords_mask].copy()
        
        if coords_data.empty:
            logger.warning("No samples with coordinates for geological data merging")
            return microbiome_data
        
        # Merge geological data based on coordinates
        merged_coords = self._merge_spatial_data(coords_data, glim_data,
                                               lat_col='latitude', lon_col='longitude',
                                               buffer_distance=0.01)
        
        # Update the original dataframe
        microbiome_data.loc[coords_mask, merged_coords.columns] = merged_coords
        
        return microbiome_data
    
    def _merge_abundance_data(self, microbiome_data: pd.DataFrame, abundance_data: pd.DataFrame) -> pd.DataFrame:
        """Merge abundance data with microbiome data."""
        # Merge on sample_id
        if 'sample_id' in microbiome_data.columns and 'sample_id' in abundance_data.columns:
            merged_data = pd.merge(microbiome_data, abundance_data, on='sample_id', how='left')
            logger.info(f"Merged abundance data: {merged_data['sample_id'].notna().sum()} samples with abundance data")
            return merged_data
        else:
            logger.warning("Cannot merge abundance data - missing sample_id column")
            return microbiome_data
    
    def _create_environmental_derivatives(self, merged_data: pd.DataFrame) -> pd.DataFrame:
        """Create derived environmental variables."""
        # Calculate soil pH from different sources
        if 'ph' in merged_data.columns and 'soil_ph' in merged_data.columns:
            merged_data['ph_combined'] = merged_data['ph'].fillna(merged_data['soil_ph'])
        elif 'soil_ph' in merged_data.columns:
            merged_data['ph_combined'] = merged_data['soil_ph']
        elif 'ph' in merged_data.columns:
            merged_data['ph_combined'] = merged_data['ph']
        
        # Calculate temperature from different sources
        if 'temperature' in merged_data.columns and 'mean_temp' in merged_data.columns:
            merged_data['temp_combined'] = merged_data['temperature'].fillna(merged_data['mean_temp'])
        elif 'mean_temp' in merged_data.columns:
            merged_data['temp_combined'] = merged_data['mean_temp']
        elif 'temperature' in merged_data.columns:
            merged_data['temp_combined'] = merged_data['temperature']
        
        # Create environmental categories
        if 'ph_combined' in merged_data.columns:
            merged_data['ph_category'] = pd.cut(merged_data['ph_combined'], 
                                              bins=[0, 6.5, 7.5, 8.5, 14], 
                                              labels=['acidic', 'neutral', 'alkaline', 'highly_alkaline'])
        
        if 'temp_combined' in merged_data.columns:
            merged_data['temp_category'] = pd.cut(merged_data['temp_combined'],
                                                bins=[-50, 5, 15, 25, 50],
                                                labels=['cold', 'cool', 'warm', 'hot'])
        
        # Create soil texture categories
        if all(col in merged_data.columns for col in ['sand', 'silt', 'clay']):
            merged_data['soil_texture'] = self._classify_soil_texture(
                merged_data['sand'], merged_data['silt'], merged_data['clay']
            )
        
        # Create geological pH preference
        if 'glim_rock_type_code' in merged_data.columns:
            merged_data['geological_ph_preference'] = merged_data['glim_rock_type_code'].apply(
                self._get_geological_ph_preference
            )
        
        return merged_data
    
    def _classify_soil_texture(self, sand: pd.Series, silt: pd.Series, clay: pd.Series) -> pd.Series:
        """Classify soil texture based on sand, silt, clay percentages."""
        # USDA soil texture triangle classification
        def classify_texture(row):
            if pd.isna(row['sand']) or pd.isna(row['silt']) or pd.isna(row['clay']):
                return 'unknown'
            
            s, si, c = row['sand'], row['silt'], row['clay']
            
            if c >= 40:
                return 'clay'
            elif si >= 40 and c < 40:
                return 'silt'
            elif s >= 85:
                return 'sand'
            elif s >= 70 and si < 30:
                return 'loamy_sand'
            elif s >= 50 and si < 50 and c < 27:
                return 'sandy_loam'
            elif s >= 23 and si >= 28 and c < 27:
                return 'loam'
            elif s >= 20 and si >= 50 and c < 27:
                return 'silt_loam'
            elif s >= 45 and c >= 27 and c < 40:
                return 'sandy_clay_loam'
            elif s < 45 and c >= 27 and c < 40:
                return 'clay_loam'
            else:
                return 'silty_clay_loam'
        
        texture_data = pd.DataFrame({'sand': sand, 'silt': silt, 'clay': clay})
        return texture_data.apply(classify_texture, axis=1)
    
    def _get_geological_ph_preference(self, rock_code: int) -> str:
        """Get pH preference for geological rock type."""
        if pd.isna(rock_code):
            return 'unknown'
        
        # Based on GLiM rock type codes
        ph_preferences = {
            1: 'slightly_acidic',  # Igneous volcanic
            2: 'neutral_alkaline',  # Igneous plutonic
            3: 'slightly_acidic',  # Metamorphic
            4: 'alkaline',  # Sedimentary carbonate
            5: 'slightly_acidic',  # Sedimentary siliciclastic
            6: 'neutral_alkaline',  # Sedimentary mixed
            7: 'variable',  # Unconsolidated sediments
            8: 'neutral_alkaline',  # Water bodies
            9: 'slightly_acidic',  # Ice and glaciers
            10: 'unknown'  # No data
        }
        return ph_preferences.get(rock_code, 'unknown')
    
    def _create_microbiome_summary(self, merged_data: pd.DataFrame) -> pd.DataFrame:
        """Create microbiome-environmental summary variables."""
        # Count bacterial families
        if 'bacterial_families' in merged_data.columns:
            merged_data['bacterial_family_count'] = merged_data['bacterial_families'].apply(
                lambda x: len(json.loads(x)) if pd.notna(x) and x != '{}' else 0
            )
        
        # Count fungal guilds
        if 'fungal_guilds' in merged_data.columns:
            merged_data['fungal_guild_count'] = merged_data['fungal_guilds'].apply(
                lambda x: len(json.loads(x)) if pd.notna(x) and x != '{}' else 0
            )
        
        # Count key taxa
        if 'key_taxa' in merged_data.columns:
            merged_data['key_taxa_count'] = merged_data['key_taxa'].apply(
                lambda x: len(json.loads(x)) if pd.notna(x) and x != '[]' else 0
            )
        
        # Create microbiome diversity indicators
        if 'bacterial_family_count' in merged_data.columns:
            merged_data['microbiome_diversity'] = pd.cut(
                merged_data['bacterial_family_count'],
                bins=[0, 5, 10, 20, 1000],
                labels=['low', 'medium', 'high', 'very_high']
            )
        
        # Create environmental-microbiome compatibility score
        merged_data['env_microbiome_compatibility'] = self._calculate_compatibility_score(merged_data)
        
        return merged_data
    
    def _calculate_compatibility_score(self, data: pd.DataFrame) -> pd.Series:
        """Calculate environmental-microbiome compatibility score."""
        scores = []
        
        for _, row in data.iterrows():
            score = 0
            
            # pH compatibility
            if 'ph_combined' in row and 'geological_ph_preference' in row:
                ph = row['ph_combined']
                pref = row['geological_ph_preference']
                
                if not pd.isna(ph) and not pd.isna(pref):
                    if pref == 'alkaline' and ph > 7.5:
                        score += 2
                    elif pref == 'neutral_alkaline' and 6.5 <= ph <= 8.0:
                        score += 2
                    elif pref == 'slightly_acidic' and 6.0 <= ph <= 7.0:
                        score += 2
                    elif pref == 'variable':
                        score += 1
            
            # Temperature compatibility
            if 'temp_combined' in row and 'temp_category' in row:
                temp = row['temp_combined']
                category = row['temp_category']
                
                if not pd.isna(temp) and not pd.isna(category):
                    # Truffles generally prefer cool to warm temperatures
                    if category in ['cool', 'warm']:
                        score += 2
                    elif category == 'cold':
                        score += 1
            
            # Soil texture compatibility
            if 'soil_texture' in row:
                texture = row['soil_texture']
                if not pd.isna(texture):
                    # Truffles generally prefer well-drained soils
                    if texture in ['sandy_loam', 'loam', 'silt_loam']:
                        score += 2
                    elif texture in ['loamy_sand', 'sandy_clay_loam']:
                        score += 1
            
            # Microbiome diversity
            if 'microbiome_diversity' in row:
                diversity = row['microbiome_diversity']
                if not pd.isna(diversity):
                    if diversity in ['high', 'very_high']:
                        score += 2
                    elif diversity == 'medium':
                        score += 1
            
            scores.append(score)
        
        return pd.Series(scores, index=data.index)
