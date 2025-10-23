"""
Unified data collector that can handle multiple data sources through configuration.

This collector replaces individual collectors while maintaining all functionality
and providing a simplified, consistent interface for data collection.
"""

import json
import logging
import pandas as pd
import requests
import numpy as np
import time
import zipfile
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from urllib.parse import urljoin
import geopandas as gpd
from shapely.geometry import Point, Polygon
import rasterio
from rasterio.warp import transform

from .base_collector import BaseCollector

logger = logging.getLogger(__name__)


class UnifiedDataCollector(BaseCollector):
    """
    Unified data collector that can handle multiple data sources.
    
    This collector replaces individual collectors (GBIF, iNaturalist, SoilGrids, 
    WorldClim, GLiM, EBI Metagenomics) while maintaining all functionality.
    """
    
    def __init__(self, config: Dict[str, Any], data_dir: Path):
        super().__init__(config, data_dir)
        self.data_sources = {
            'gbif': self._collect_gbif_data,
            'inaturalist': self._collect_inaturalist_data,
            'soilgrids': self._collect_soilgrids_data,
            'worldclim': self._collect_worldclim_data,
            'glim': self._collect_glim_data,
            'ebi_metagenomics': self._collect_ebi_metagenomics_data
        }
        
    def collect(self, source: str, **kwargs) -> pd.DataFrame:
        """
        Collect data from specified source.
        
        Args:
            source: Data source name ('gbif', 'inaturalist', 'soilgrids', 
                   'worldclim', 'glim', 'ebi_metagenomics')
            **kwargs: Source-specific parameters
            
        Returns:
            DataFrame with collected data
        """
        if source not in self.data_sources:
            raise ValueError(f"Unknown data source: {source}. Available: {list(self.data_sources.keys())}")
            
        logger.info(f"Collecting data from {source}")
        return self.data_sources[source](**kwargs)
    
    def collect_multiple(self, sources: List[str], **kwargs) -> Dict[str, pd.DataFrame]:
        """
        Collect data from multiple sources.
        
        Args:
            sources: List of data source names
            **kwargs: Parameters to pass to all sources
            
        Returns:
            Dictionary mapping source names to DataFrames
        """
        results = {}
        for source in sources:
            try:
                results[source] = self.collect(source, **kwargs)
            except Exception as e:
                logger.error(f"Error collecting from {source}: {e}")
                results[source] = pd.DataFrame()
        return results
    
    def _collect_gbif_data(self, species: List[str], limit: int = 10000, 
                          country: Optional[str] = None, year_from: Optional[int] = None,
                          year_to: Optional[int] = None) -> pd.DataFrame:
        """Collect GBIF occurrence data."""
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
                species_key, base_url, limit, country, year_from, year_to
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
                             country: Optional[str], year_from: Optional[int],
                             year_to: Optional[int]) -> List[Dict]:
        """Get GBIF occurrence records for a species."""
        url = f"{base_url}/occurrence/search"
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
        
        # Filter out records with high coordinate uncertainty (>10km)
        if 'coordinate_uncertainty' in df.columns:
            df = df[
                (df['coordinate_uncertainty'].isna()) | 
                (df['coordinate_uncertainty'] <= 10000)
            ]
            
        return df
    
    def _collect_inaturalist_data(self, species: List[str], limit: int = 10000, 
                                 place_id: Optional[int] = None, year_from: Optional[int] = None,
                                 year_to: Optional[int] = None) -> pd.DataFrame:
        """Collect iNaturalist observation data."""
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
                taxon_id, base_url, limit, place_id, year_from, year_to
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
                                     place_id: Optional[int], year_from: Optional[int],
                                     year_to: Optional[int]) -> List[Dict]:
        """Get iNaturalist observation records for a taxon."""
        url = f"{base_url}/observations"
        params = {
            'taxon_id': taxon_id,
            'per_page': 200,
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
                'source': 'iNaturalist'
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
    
    def _collect_soilgrids_data(self, coordinates: List[Tuple[float, float]], 
                               variables: List[str] = None) -> pd.DataFrame:
        """Collect SoilGrids soil property data."""
        base_url = self.config["soilgrids"]["base_url"]
        
        if variables is None:
            variables = [
                'phh2o', 'cac03', 'soc', 'nitrogen', 'phosporus',
                'sand', 'silt', 'clay', 'bdod', 'cec', 'cfvo'
            ]
            
        all_data = []
        
        for i, (lat, lon) in enumerate(coordinates):
            logger.info(f"Collecting soil data for point {i+1}/{len(coordinates)}: ({lat}, {lon})")
            
            try:
                soil_data = self._get_soilgrids_data(lat, lon, base_url, variables)
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
            df = self._clean_soilgrids_data(df)
            
        return df
    
    def _get_soilgrids_data(self, lat: float, lon: float, base_url: str, 
                           variables: List[str]) -> Optional[Dict]:
        """Get SoilGrids data for a single coordinate pair."""
        url = f"{base_url}/query"
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
    
    def _clean_soilgrids_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize SoilGrids data."""
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
    
    def _collect_worldclim_data(self, coordinates: List[Tuple[float, float]], 
                               variables: List[str] = None) -> pd.DataFrame:
        """Collect WorldClim climate data."""
        base_url = self.config["worldclim"]["base_url"]
        resolution = "30s"  # 30 arc-seconds resolution
        
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
                climate_data = self._get_worldclim_data(lat, lon, base_url, resolution, variables)
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
            df = self._clean_worldclim_data(df)
            
        return df
    
    def _get_worldclim_data(self, lat: float, lon: float, base_url: str, 
                           resolution: str, variables: List[str]) -> Optional[Dict]:
        """Get WorldClim data for a single coordinate pair."""
        climate_data = {}
        
        for var in variables:
            try:
                # Download the raster file for this variable
                raster_path = self._download_worldclim_raster(var, base_url, resolution)
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
    
    def _download_worldclim_raster(self, variable: str, base_url: str, resolution: str) -> Optional[str]:
        """Download WorldClim raster file for a variable."""
        filename = f"wc2.1_{resolution}_{variable}.tif"
        filepath = self.data_dir / filename
        
        if filepath.exists():
            return str(filepath)
            
        url = f"{base_url}/{filename}"
        
        try:
            response = self._make_request(url)
            with open(filepath, 'wb') as f:
                f.write(response.content)
            logger.info(f"Downloaded {filename}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Error downloading {filename}: {e}")
            return None
    
    def _extract_raster_value(self, raster_path: str, lat: float, lon: float) -> Optional[float]:
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
    
    def _clean_worldclim_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize WorldClim data."""
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
    
    def _collect_glim_data(self, coordinates: List[Tuple[float, float]], 
                          buffer_distance: float = 0.01) -> pd.DataFrame:
        """Collect GLiM geological data."""
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
        
        # Create mock GLiM dataset for testing
        logger.warning("Creating mock GLiM data - replace with actual GLiM download")
        return self._create_mock_glim_data(glim_dir)
    
    def _create_mock_glim_data(self, glim_dir: Path) -> Path:
        """Create mock GLiM data for testing purposes."""
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
    
    def _collect_ebi_metagenomics_data(self, search_term: str = "Tuber", 
                                      limit: int = 1000,
                                      include_samples: bool = True,
                                      include_abundance: bool = False) -> pd.DataFrame:
        """Collect EBI Metagenomics data."""
        base_url = "https://www.ebi.ac.uk/metagenomics/api/latest"
        
        # Update session headers for EBI API
        self.session.headers.update({
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        })
        
        logger.info(f"Collecting EBI Metagenomics data for search term: {search_term}")
        
        # Step 1: Get studies
        studies_df = self._get_ebi_studies(search_term, limit, base_url)
        
        if studies_df.empty:
            logger.warning("No studies found for the search term")
            return pd.DataFrame()
            
        # Step 2: Get samples for each study
        if include_samples:
            samples_df = self._get_ebi_samples(studies_df, base_url)
            if not samples_df.empty:
                # Merge studies and samples data
                merged_df = pd.merge(studies_df, samples_df, 
                                   on='study_id', how='left', suffixes=('_study', '_sample'))
            else:
                merged_df = studies_df
        else:
            merged_df = studies_df
            
        # Step 3: Get abundance data if requested
        if include_abundance and not merged_df.empty:
            abundance_df = self._get_ebi_abundance_data(merged_df, base_url)
            if not abundance_df.empty:
                merged_df = pd.merge(merged_df, abundance_df, 
                                   on='sample_id', how='left')
        
        # Step 4: Clean and standardize data
        cleaned_df = self._clean_ebi_data(merged_df)
        
        # Save data
        output_file = self.data_dir / f"ebi_metagenomics_{search_term.lower()}.csv"
        cleaned_df.to_csv(output_file, index=False)
        logger.info(f"Saved {len(cleaned_df)} records to {output_file}")
        
        return cleaned_df
    
    def _get_ebi_studies(self, search_term: str, limit: int, base_url: str) -> pd.DataFrame:
        """Retrieve studies from EBI Metagenomics API."""
        studies_data = []
        url = f"{base_url}/studies"
        params = {
            'search': search_term,
            'page_size': min(100, limit),  # API limit
            'page': 1
        }
        
        while len(studies_data) < limit:
            try:
                response = self._make_request(url, params=params)
                data = response.json()
                
                for study in data.get('data', []):
                    if len(studies_data) >= limit:
                        break
                        
                    study_info = {
                        'study_id': study.get('id'),
                        'study_accession': study.get('attributes', {}).get('accession'),
                        'study_title': study.get('attributes', {}).get('study-name'),
                        'study_description': study.get('attributes', {}).get('study-abstract'),
                        'biome': study.get('attributes', {}).get('biome', {}).get('biome-name'),
                        'biome_category': study.get('attributes', {}).get('biome', {}).get('biome-category'),
                        'latitude': study.get('attributes', {}).get('latitude'),
                        'longitude': study.get('attributes', {}).get('longitude'),
                        'environment_biome': study.get('attributes', {}).get('environment-biome'),
                        'environment_feature': study.get('attributes', {}).get('environment-feature'),
                        'environment_material': study.get('attributes', {}).get('environment-material'),
                        'host': study.get('attributes', {}).get('host'),
                        'host_taxonomy_id': study.get('attributes', {}).get('host-taxonomy-id'),
                        'sample_count': study.get('attributes', {}).get('sample-count'),
                        'run_count': study.get('attributes', {}).get('run-count'),
                        'analysis_completed': study.get('attributes', {}).get('analysis-completed'),
                        'publication_date': study.get('attributes', {}).get('publication-date'),
                        'submission_date': study.get('attributes', {}).get('submission-date'),
                        'center_name': study.get('attributes', {}).get('center-name'),
                        'source_link': f"https://www.ebi.ac.uk/metagenomics/studies/{study.get('id')}"
                    }
                    studies_data.append(study_info)
                
                # Check if there are more pages
                if 'links' in data and 'next' in data['links']:
                    url = data['links']['next']
                    params = {}  # URL already contains parameters
                else:
                    break
                    
            except Exception as e:
                logger.error(f"Error retrieving studies: {e}")
                break
                
            time.sleep(0.5)  # Rate limiting
        
        return pd.DataFrame(studies_data)
    
    def _get_ebi_samples(self, studies_df: pd.DataFrame, base_url: str) -> pd.DataFrame:
        """Retrieve sample data for studies."""
        samples_data = []
        
        for _, study in studies_df.iterrows():
            study_id = study['study_id']
            url = f"{base_url}/studies/{study_id}/samples"
            params = {'page_size': 100}
            
            try:
                response = self._make_request(url, params=params)
                data = response.json()
                
                for sample in data.get('data', []):
                    sample_info = {
                        'study_id': study_id,
                        'sample_id': sample.get('id'),
                        'sample_accession': sample.get('attributes', {}).get('accession'),
                        'sample_name': sample.get('attributes', {}).get('sample-name'),
                        'sample_description': sample.get('attributes', {}).get('sample-desc'),
                        'latitude': sample.get('attributes', {}).get('latitude'),
                        'longitude': sample.get('attributes', {}).get('longitude'),
                        'environment_biome': sample.get('attributes', {}).get('environment-biome'),
                        'environment_feature': sample.get('attributes', {}).get('environment-feature'),
                        'environment_material': sample.get('attributes', {}).get('environment-material'),
                        'host': sample.get('attributes', {}).get('host'),
                        'host_taxonomy_id': sample.get('attributes', {}).get('host-taxonomy-id'),
                        'collection_date': sample.get('attributes', {}).get('collection-date'),
                        'geographic_location': sample.get('attributes', {}).get('geographic-location'),
                        'depth': sample.get('attributes', {}).get('depth'),
                        'elevation': sample.get('attributes', {}).get('elevation'),
                        'temperature': sample.get('attributes', {}).get('temperature'),
                        'ph': sample.get('attributes', {}).get('ph'),
                        'salinity': sample.get('attributes', {}).get('salinity'),
                        'nitrogen': sample.get('attributes', {}).get('nitrogen'),
                        'phosphorus': sample.get('attributes', {}).get('phosphorus'),
                        'carbon': sample.get('attributes', {}).get('carbon'),
                        'run_count': sample.get('attributes', {}).get('run-count'),
                        'analysis_completed': sample.get('attributes', {}).get('analysis-completed')
                    }
                    samples_data.append(sample_info)
                    
            except Exception as e:
                logger.warning(f"Error retrieving samples for study {study_id}: {e}")
                continue
                
            time.sleep(0.5)  # Rate limiting
        
        return pd.DataFrame(samples_data)
    
    def _get_ebi_abundance_data(self, samples_df: pd.DataFrame, base_url: str) -> pd.DataFrame:
        """Retrieve abundance data for samples (if available)."""
        abundance_data = []
        
        for _, sample in samples_df.iterrows():
            sample_id = sample['sample_id']
            
            # Try to get taxonomic abundance data
            try:
                url = f"{base_url}/samples/{sample_id}/taxonomy"
                response = self._make_request(url)
                data = response.json()
                
                # Process taxonomic data
                taxonomy_data = self._process_ebi_taxonomy_data(data, sample_id)
                if taxonomy_data:
                    abundance_data.append(taxonomy_data)
                    
            except Exception as e:
                logger.debug(f"No taxonomy data available for sample {sample_id}: {e}")
            
            # Try to get functional abundance data
            try:
                url = f"{base_url}/samples/{sample_id}/go-slim"
                response = self._make_request(url)
                data = response.json()
                
                # Process functional data
                functional_data = self._process_ebi_functional_data(data, sample_id)
                if functional_data:
                    abundance_data.append(functional_data)
                    
            except Exception as e:
                logger.debug(f"No functional data available for sample {sample_id}: {e}")
                
            time.sleep(0.5)  # Rate limiting
        
        return pd.DataFrame(abundance_data)
    
    def _process_ebi_taxonomy_data(self, data: Dict, sample_id: str) -> Optional[Dict]:
        """Process taxonomic abundance data."""
        taxonomy_data = {'sample_id': sample_id}
        
        for taxon in data.get('data', []):
            attributes = taxon.get('attributes', {})
            taxon_name = attributes.get('lineage')
            abundance = attributes.get('abundance')
            
            if taxon_name and abundance is not None:
                # Store at different taxonomic levels
                if 'family' in taxon_name.lower():
                    taxonomy_data[f'taxonomy_family_{taxon_name}'] = abundance
                elif 'genus' in taxon_name.lower():
                    taxonomy_data[f'taxonomy_genus_{taxon_name}'] = abundance
                elif 'species' in taxon_name.lower():
                    taxonomy_data[f'taxonomy_species_{taxon_name}'] = abundance
        
        return taxonomy_data if len(taxonomy_data) > 1 else None
    
    def _process_ebi_functional_data(self, data: Dict, sample_id: str) -> Optional[Dict]:
        """Process functional abundance data."""
        functional_data = {'sample_id': sample_id}
        
        for go_term in data.get('data', []):
            attributes = go_term.get('attributes', {})
            go_id = attributes.get('go-id')
            abundance = attributes.get('abundance')
            
            if go_id and abundance is not None:
                functional_data[f'functional_{go_id}'] = abundance
        
        return functional_data if len(functional_data) > 1 else None
    
    def _clean_ebi_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize EBI Metagenomics data."""
        if df.empty:
            return df
            
        # Convert numeric columns
        numeric_columns = ['latitude', 'longitude', 'ph', 'temperature', 'depth', 
                          'elevation', 'salinity', 'nitrogen', 'phosphorus', 'carbon']
        
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Convert date columns
        date_columns = ['collection_date', 'publication_date', 'submission_date']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # Create derived fields
        if 'latitude' in df.columns and 'longitude' in df.columns:
            df['has_coordinates'] = df['latitude'].notna() & df['longitude'].notna()
        
        # Standardize species names
        if 'host' in df.columns:
            df['host_standardized'] = df['host'].str.lower().str.strip()
        
        # Create key taxa indicators
        bacterial_families = []
        fungal_guilds = []
        
        for col in df.columns:
            if 'taxonomy_family' in col:
                family_name = col.replace('taxonomy_family_', '')
                bacterial_families.append(family_name)
            elif 'taxonomy_genus' in col:
                genus_name = col.replace('taxonomy_genus_', '')
                if any(fungus in genus_name.lower() for fungus in ['tuber', 'truffle', 'mycorrhiza']):
                    fungal_guilds.append(genus_name)
        
        # Add summary fields
        df['bacterial_families'] = df[[col for col in df.columns if 'taxonomy_family' in col]].apply(
            lambda x: json.dumps(x.dropna().to_dict()), axis=1
        )
        df['fungal_guilds'] = df[[col for col in df.columns if 'taxonomy_genus' in col]].apply(
            lambda x: json.dumps(x.dropna().to_dict()), axis=1
        )
        
        # Identify key taxa
        key_taxa = []
        for col in df.columns:
            if 'taxonomy' in col:
                taxon_name = col.split('_', 2)[-1] if '_' in col else col
                if any(key in taxon_name.lower() for key in ['pseudomonas', 'bradyrhizobium', 'rhizobium']):
                    key_taxa.append(taxon_name)
        
        df['key_taxa'] = df[[col for col in df.columns if 'taxonomy' in col]].apply(
            lambda x: json.dumps([k for k, v in x.dropna().items() 
                                if any(key in k.lower() for key in ['pseudomonas', 'bradyrhizobium', 'rhizobium'])]), axis=1
        )
        
        # Add fruiting evidence indicator
        df['fruiting_evidence'] = df['study_title'].str.contains(
            'fruiting|fruit|truffle|harvest', case=False, na=False
        ) | df['sample_description'].str.contains(
            'fruiting|fruit|truffle|harvest', case=False, na=False
        )
        
        return df
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """Validate collected data."""
        if data.empty:
            logger.warning("No data to validate")
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
            
        logger.info(f"Data validation passed: {len(data)} records")
        return True