"""
Main habitat data processor for combining and analyzing truffle habitat data.
"""
import logging
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
from pathlib import Path
import geopandas as gpd
from shapely.geometry import Point

from ..data_collectors.unified_collector import UnifiedDataCollector
from .feature_engineering import FeatureEngineer
from .data_merger import DataMerger

logger = logging.getLogger(__name__)


class HabitatProcessor:
    """Main processor for truffle habitat data collection and analysis."""
    
    def __init__(self, config: Dict[str, Any], data_dir: Path):
        self.config = config
        self.data_dir = data_dir
        
        # Initialize unified collector
        self.unified_collector = UnifiedDataCollector(config, data_dir)
        
        # Initialize processors
        self.feature_engineer = FeatureEngineer()
        self.data_merger = DataMerger()
        
    def collect_all_data(self, species: List[str], 
                        countries: Optional[List[str]] = None,
                        year_from: Optional[int] = None,
                        year_to: Optional[int] = None) -> pd.DataFrame:
        """
        Collect all data sources for truffle habitat analysis.
        
        Args:
            species: List of truffle species to collect data for
            countries: Optional list of country codes to filter by
            year_from: Start year for data collection
            year_to: End year for data collection
            
        Returns:
            Combined DataFrame with all habitat data
        """
        logger.info("Starting comprehensive data collection for truffle habitats")
        
        # Collect occurrence data
        occurrence_data = self._collect_occurrence_data(species, countries, year_from, year_to)
        
        if occurrence_data.empty:
            logger.warning("No occurrence data collected")
            return pd.DataFrame()
            
        # Extract coordinates for environmental data collection
        coordinates = list(zip(occurrence_data['latitude'], occurrence_data['longitude']))
        
        # Collect environmental data
        soil_data = self._collect_soil_data(coordinates)
        climate_data = self._collect_climate_data(coordinates)
        
        # Merge all data
        merged_data = self.data_merger.merge_habitat_data(
            occurrence_data, soil_data, climate_data
        )
        
        # Engineer additional features
        enhanced_data = self.feature_engineer.engineer_features(merged_data)
        
        logger.info(f"Data collection complete: {len(enhanced_data)} records")
        return enhanced_data
        
    def _collect_occurrence_data(self, species: List[str], 
                               countries: Optional[List[str]],
                               year_from: Optional[int],
                               year_to: Optional[int]) -> pd.DataFrame:
        """Collect occurrence data from all sources."""
        all_occurrences = []
        
        # Collect from GBIF
        try:
            gbif_data = self.unified_collector.collect(
                source='gbif',
                species=species,
                limit=10000,
                year_from=year_from,
                year_to=year_to
            )
            if isinstance(gbif_data, dict) and 'records_df' in gbif_data:
                gbif_df = gbif_data['records_df']
            else:
                gbif_df = gbif_data
                
            if not gbif_df.empty:
                all_occurrences.append(gbif_df)
                logger.info(f"Collected {len(gbif_df)} records from GBIF")
        except Exception as e:
            logger.error(f"Error collecting GBIF data: {e}")
            
        # Collect from iNaturalist
        try:
            inat_data = self.unified_collector.collect(
                source='inaturalist',
                species=species,
                limit=10000,
                year_from=year_from,
                year_to=year_to
            )
            if isinstance(inat_data, dict) and 'records_df' in inat_data:
                inat_df = inat_data['records_df']
            else:
                inat_df = inat_data
                
            if not inat_df.empty:
                all_occurrences.append(inat_df)
                logger.info(f"Collected {len(inat_df)} records from iNaturalist")
        except Exception as e:
            logger.error(f"Error collecting iNaturalist data: {e}")
            
        # Combine occurrence data
        if all_occurrences:
            combined_occurrences = pd.concat(all_occurrences, ignore_index=True)
            # Remove duplicates based on coordinates and species
            combined_occurrences = combined_occurrences.drop_duplicates(
                subset=['species', 'latitude', 'longitude']
            )
            return combined_occurrences
        else:
            return pd.DataFrame()
            
    def _collect_soil_data(self, coordinates: List[Tuple[float, float]]) -> pd.DataFrame:
        """Collect soil data for given coordinates."""
        try:
            soil_data = self.unified_collector.collect(
                source='soilgrids',
                coordinates=coordinates
            )
            if isinstance(soil_data, dict) and 'records_df' in soil_data:
                soil_df = soil_data['records_df']
            else:
                soil_df = soil_data
            logger.info(f"Collected soil data for {len(soil_df)} points")
            return soil_df
        except Exception as e:
            logger.error(f"Error collecting soil data: {e}")
            return pd.DataFrame()
            
    def _collect_climate_data(self, coordinates: List[Tuple[float, float]]) -> pd.DataFrame:
        """Collect climate data for given coordinates."""
        try:
            climate_data = self.unified_collector.collect(
                source='worldclim',
                coordinates=coordinates
            )
            if isinstance(climate_data, dict) and 'records_df' in climate_data:
                climate_df = climate_data['records_df']
            else:
                climate_df = climate_data
            logger.info(f"Collected climate data for {len(climate_df)} points")
            return climate_df
        except Exception as e:
            logger.error(f"Error collecting climate data: {e}")
            return pd.DataFrame()
            
    def analyze_habitat_characteristics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze habitat characteristics from the collected data.
        
        Args:
            data: Combined habitat data DataFrame
            
        Returns:
            Dictionary with habitat analysis results
        """
        if data.empty:
            return {}
            
        analysis = {}
        
        # Species distribution
        analysis['species_counts'] = data['species'].value_counts().to_dict()
        
        # Geographic distribution
        analysis['geographic_bounds'] = {
            'min_lat': data['latitude'].min(),
            'max_lat': data['latitude'].max(),
            'min_lon': data['longitude'].min(),
            'max_lon': data['longitude'].max()
        }
        
        # Environmental ranges by species
        env_columns = [col for col in data.columns if any(prefix in col for prefix in ['soil_', 'climate_'])]
        
        for species in data['species'].unique():
            species_data = data[data['species'] == species]
            analysis[f'{species}_ranges'] = {}
            
            for col in env_columns:
                if col in species_data.columns:
                    values = species_data[col].dropna()
                    if not values.empty:
                        analysis[f'{species}_ranges'][col] = {
                            'min': float(values.min()),
                            'max': float(values.max()),
                            'mean': float(values.mean()),
                            'std': float(values.std())
                        }
                        
        # Fruiting season analysis
        if 'month' in data.columns:
            analysis['fruiting_seasons'] = self._analyze_fruiting_seasons(data)
            
        return analysis
        
    def _analyze_fruiting_seasons(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze fruiting seasons from occurrence data."""
        fruiting_analysis = {}
        
        for species in data['species'].unique():
            species_data = data[data['species'] == species]
            months = species_data['month'].dropna()
            
            if not months.empty:
                month_counts = months.value_counts().sort_index()
                fruiting_analysis[species] = {
                    'peak_months': month_counts.nlargest(3).index.tolist(),
                    'month_distribution': month_counts.to_dict()
                }
                
        return fruiting_analysis
        
    def export_habitat_parameters(self, data: pd.DataFrame, 
                                 output_dir: Path) -> Dict[str, str]:
        """
        Export habitat parameters for hydroponic simulation.
        
        Args:
            data: Habitat data DataFrame
            output_dir: Directory to save export files
            
        Returns:
            Dictionary with file paths of exported files
        """
        output_dir.mkdir(exist_ok=True)
        exported_files = {}
        
        # Export full dataset
        csv_path = output_dir / "truffle_habitat_data.csv"
        data.to_csv(csv_path, index=False)
        exported_files['full_dataset'] = str(csv_path)
        
        # Export by species
        for species in data['species'].unique():
            species_data = data[data['species'] == species]
            species_name = species.replace(' ', '_').lower()
            species_path = output_dir / f"{species_name}_habitat_data.csv"
            species_data.to_csv(species_path, index=False)
            exported_files[f'{species_name}_dataset'] = str(species_path)
            
        # Export summary statistics
        summary_stats = self._generate_summary_statistics(data)
        summary_path = output_dir / "habitat_summary_statistics.json"
        import json
        with open(summary_path, 'w') as f:
            json.dump(summary_stats, f, indent=2)
        exported_files['summary_statistics'] = str(summary_path)
        
        # Export as GeoJSON for mapping
        if not data.empty:
            gdf = gpd.GeoDataFrame(
                data,
                geometry=[Point(xy) for xy in zip(data['longitude'], data['latitude'])],
                crs='EPSG:4326'
            )
            geojson_path = output_dir / "truffle_habitat_data.geojson"
            gdf.to_file(geojson_path, driver='GeoJSON')
            exported_files['geojson'] = str(geojson_path)
            
        logger.info(f"Exported habitat data to {output_dir}")
        return exported_files
        
    def _generate_summary_statistics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Generate summary statistics for the habitat data."""
        summary = {
            'total_records': len(data),
            'species_count': data['species'].nunique(),
            'species_list': data['species'].unique().tolist(),
            'geographic_extent': {
                'latitude_range': [float(data['latitude'].min()), float(data['latitude'].max())],
                'longitude_range': [float(data['longitude'].min()), float(data['longitude'].max())]
            }
        }
        
        # Environmental variable summaries
        env_columns = [col for col in data.columns if any(prefix in col for prefix in ['soil_', 'climate_'])]
        summary['environmental_variables'] = {}
        
        for col in env_columns:
            if col in data.columns:
                values = data[col].dropna()
                if not values.empty:
                    summary['environmental_variables'][col] = {
                        'count': len(values),
                        'min': float(values.min()),
                        'max': float(values.max()),
                        'mean': float(values.mean()),
                        'std': float(values.std())
                    }
                    
        return summary