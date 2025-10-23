"""
GLiM (Global Lithological Map) data collector for parent rock type information.

This collector retrieves geological information from the GLiM dataset
to provide parent rock type context for microbiome-environmental analysis.
"""

import logging
import pandas as pd
import requests
import zipfile
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import geopandas as gpd
from shapely.geometry import Point
import numpy as np

from .base_collector import BaseCollector

logger = logging.getLogger(__name__)


class GLiMCollector(BaseCollector):
    """Collector for GLiM (Global Lithological Map) data."""
    
    def __init__(self, config: Dict[str, Any], data_dir: Path):
        super().__init__(config, data_dir)
        self.glim_url = "https://www.geo.uni-hamburg.de/en/geologie/forschung/geodynamik/glim.html"
        self.glim_data_url = "https://www.geo.uni-hamburg.de/en/geologie/forschung/geodynamik/glim.html"
        # Alternative source: https://www.geo.uni-hamburg.de/en/geologie/forschung/geodynamik/glim.html
        # The actual data download URL would need to be determined from the GLiM website
        
    def collect(self, coordinates: List[Tuple[float, float]], 
                buffer_distance: float = 0.01) -> pd.DataFrame:
        """
        Collect GLiM data for given coordinates.
        
        Args:
            coordinates: List of (latitude, longitude) tuples
            buffer_distance: Buffer distance in degrees for spatial queries
            
        Returns:
            DataFrame with GLiM data
        """
        logger.info(f"Collecting GLiM data for {len(coordinates)} coordinates")
        
        # Download GLiM data if not already present
        glim_file = self._download_glim_data()
        if not glim_file:
            logger.error("Failed to download GLiM data")
            return pd.DataFrame()
        
        # Load GLiM data
        glim_gdf = self._load_glim_data(glim_file)
        if glim_gdf.empty:
            logger.error("Failed to load GLiM data")
            return pd.DataFrame()
        
        # Extract data for coordinates
        results = []
        for lat, lon in coordinates:
            try:
                glim_info = self._extract_glim_at_point(glim_gdf, lat, lon, buffer_distance)
                glim_info['latitude'] = lat
                glim_info['longitude'] = lon
                results.append(glim_info)
            except Exception as e:
                logger.warning(f"Error extracting GLiM data for ({lat}, {lon}): {e}")
                results.append({
                    'latitude': lat,
                    'longitude': lon,
                    'glim_rock_type': None,
                    'glim_rock_type_code': None,
                    'glim_confidence': None,
                    'glim_area_km2': None
                })
        
        result_df = pd.DataFrame(results)
        
        # Save data
        output_file = self.data_dir / "glim_data.csv"
        result_df.to_csv(output_file, index=False)
        logger.info(f"Saved {len(result_df)} GLiM records to {output_file}")
        
        return result_df
    
    def _download_glim_data(self) -> Optional[Path]:
        """Download GLiM data if not already present."""
        glim_dir = self.data_dir / "glim"
        glim_dir.mkdir(exist_ok=True)
        
        # Check if data already exists
        glim_file = glim_dir / "glim_rocks.shp"
        if glim_file.exists():
            logger.info("GLiM data already exists")
            return glim_file
        
        # For now, create a mock GLiM dataset since the actual download
        # would require access to the GLiM data portal
        logger.warning("Creating mock GLiM data - replace with actual GLiM download")
        return self._create_mock_glim_data(glim_dir)
    
    def _create_mock_glim_data(self, glim_dir: Path) -> Path:
        """Create mock GLiM data for testing purposes."""
        # This is a simplified mock - in practice, you would download
        # the actual GLiM shapefile from the official source
        
        # Create a simple GeoDataFrame with some common rock types
        from shapely.geometry import Polygon
        
        # Define some common rock types and their codes
        rock_types = [
            {'code': 1, 'type': 'Igneous volcanic', 'confidence': 'high'},
            {'code': 2, 'type': 'Igneous plutonic', 'confidence': 'high'},
            {'code': 3, 'type': 'Metamorphic', 'confidence': 'medium'},
            {'code': 4, 'type': 'Sedimentary carbonate', 'confidence': 'high'},
            {'code': 5, 'type': 'Sedimentary siliciclastic', 'confidence': 'high'},
            {'code': 6, 'type': 'Sedimentary mixed', 'confidence': 'medium'},
            {'code': 7, 'type': 'Unconsolidated sediments', 'confidence': 'low'},
            {'code': 8, 'type': 'Water bodies', 'confidence': 'high'},
            {'code': 9, 'type': 'Ice and glaciers', 'confidence': 'high'},
            {'code': 10, 'type': 'No data', 'confidence': 'none'}
        ]
        
        # Create a simple grid covering common truffle-growing regions
        geometries = []
        rock_codes = []
        rock_names = []
        confidences = []
        
        # Create a grid covering Europe and North America
        for lat in range(35, 55, 2):  # Roughly Mediterranean to northern Europe
            for lon in range(-10, 40, 2):  # Roughly Atlantic to eastern Europe
                # Create a simple polygon
                geom = Polygon([
                    (lon, lat),
                    (lon + 2, lat),
                    (lon + 2, lat + 2),
                    (lon, lat + 2),
                    (lon, lat)
                ])
                geometries.append(geom)
                
                # Assign rock type based on location
                if 40 <= lat <= 50 and -5 <= lon <= 15:  # Western Europe
                    rock_type = rock_types[3]  # Sedimentary carbonate
                elif 35 <= lat <= 45 and 0 <= lon <= 20:  # Mediterranean
                    rock_type = rock_types[0]  # Igneous volcanic
                elif 45 <= lat <= 55 and 10 <= lon <= 30:  # Eastern Europe
                    rock_type = rock_types[2]  # Metamorphic
                else:
                    rock_type = rock_types[4]  # Sedimentary siliciclastic
                
                rock_codes.append(rock_type['code'])
                rock_names.append(rock_type['type'])
                confidences.append(rock_type['confidence'])
        
        # Create GeoDataFrame
        gdf = gpd.GeoDataFrame({
            'geometry': geometries,
            'glim_code': rock_codes,
            'glim_type': rock_names,
            'confidence': confidences,
            'area_km2': 4.0  # 2x2 degree grid cells
        }, crs='EPSG:4326')
        
        # Save as shapefile
        glim_file = glim_dir / "glim_rocks.shp"
        gdf.to_file(glim_file)
        
        logger.info(f"Created mock GLiM data with {len(gdf)} polygons")
        return glim_file
    
    def _load_glim_data(self, glim_file: Path) -> gpd.GeoDataFrame:
        """Load GLiM data from shapefile."""
        try:
            gdf = gpd.read_file(glim_file)
            logger.info(f"Loaded GLiM data with {len(gdf)} polygons")
            return gdf
        except Exception as e:
            logger.error(f"Error loading GLiM data: {e}")
            return gpd.GeoDataFrame()
    
    def _extract_glim_at_point(self, glim_gdf: gpd.GeoDataFrame, 
                              lat: float, lon: float, 
                              buffer_distance: float) -> Dict[str, Any]:
        """Extract GLiM data at a specific point."""
        point = Point(lon, lat)
        
        # Create buffer around point
        buffer = point.buffer(buffer_distance)
        
        # Find intersecting polygons
        intersecting = glim_gdf[glim_gdf.geometry.intersects(buffer)]
        
        if intersecting.empty:
            return {
                'glim_rock_type': 'Unknown',
                'glim_rock_type_code': None,
                'glim_confidence': 'none',
                'glim_area_km2': None
            }
        
        # Get the most common rock type in the buffer
        if len(intersecting) == 1:
            rock_type = intersecting.iloc[0]
        else:
            # If multiple types, choose the one with highest confidence
            confidence_order = {'high': 3, 'medium': 2, 'low': 1, 'none': 0}
            intersecting['conf_score'] = intersecting['confidence'].map(confidence_order)
            rock_type = intersecting.loc[intersecting['conf_score'].idxmax()]
        
        return {
            'glim_rock_type': rock_type.get('glim_type', 'Unknown'),
            'glim_rock_type_code': rock_type.get('glim_code'),
            'glim_confidence': rock_type.get('confidence', 'unknown'),
            'glim_area_km2': rock_type.get('area_km2')
        }
    
    def get_rock_type_description(self, rock_type_code: int) -> str:
        """Get description for a rock type code."""
        descriptions = {
            1: "Igneous volcanic rocks - formed from cooled lava, often associated with volcanic activity",
            2: "Igneous plutonic rocks - formed from cooled magma deep underground",
            3: "Metamorphic rocks - formed from existing rocks under heat and pressure",
            4: "Sedimentary carbonate rocks - limestone, dolomite, often associated with marine environments",
            5: "Sedimentary siliciclastic rocks - sandstone, shale, formed from eroded rock fragments",
            6: "Sedimentary mixed rocks - combination of carbonate and siliciclastic components",
            7: "Unconsolidated sediments - loose materials like sand, gravel, clay",
            8: "Water bodies - lakes, rivers, oceans",
            9: "Ice and glaciers - frozen water",
            10: "No data - insufficient geological information"
        }
        return descriptions.get(rock_type_code, "Unknown rock type")
    
    def get_rock_type_ph_preference(self, rock_type_code: int) -> Tuple[float, float]:
        """Get typical pH range for a rock type (if applicable)."""
        # These are general guidelines - actual pH depends on many factors
        ph_ranges = {
            1: (6.0, 7.5),  # Igneous volcanic - slightly acidic to neutral
            2: (6.5, 8.0),  # Igneous plutonic - neutral to slightly alkaline
            3: (6.0, 7.0),  # Metamorphic - slightly acidic to neutral
            4: (7.5, 8.5),  # Sedimentary carbonate - alkaline
            5: (6.0, 7.0),  # Sedimentary siliciclastic - slightly acidic to neutral
            6: (6.5, 8.0),  # Sedimentary mixed - neutral to slightly alkaline
            7: (6.0, 7.5),  # Unconsolidated sediments - variable
            8: (6.5, 8.5),  # Water bodies - neutral to alkaline
            9: (6.0, 7.0),  # Ice and glaciers - slightly acidic
            10: (6.0, 8.0)  # No data - unknown
        }
        return ph_ranges.get(rock_type_code, (6.0, 8.0))
