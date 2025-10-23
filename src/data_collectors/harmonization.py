"""
Data harmonization and integration layer for unified data collector.

This module provides functions to standardize, harmonize, and integrate
data from multiple sources with proper licensing, quality scoring, and metadata.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
import json
from datetime import datetime, timezone
import hashlib

logger = logging.getLogger(__name__)


class DataHarmonizer:
    """Harmonizes and integrates data from multiple sources."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.unit_conversions = self._initialize_unit_conversions()
        self.taxonomy_mappings = self._initialize_taxonomy_mappings()
        
    def _initialize_unit_conversions(self) -> Dict[str, Dict[str, float]]:
        """Initialize unit conversion factors."""
        return {
            'temperature': {
                'celsius': 1.0,
                'fahrenheit': 1.8,
                'kelvin': 1.0
            },
            'precipitation': {
                'mm': 1.0,
                'cm': 10.0,
                'inches': 25.4
            },
            'concentration': {
                'mg_kg': 1.0,
                'ppm': 1.0,
                'percent': 10000.0,
                'g_kg': 1000.0
            },
            'ph': {
                'ph_units': 1.0
            }
        }
    
    def _initialize_taxonomy_mappings(self) -> Dict[str, str]:
        """Initialize taxonomy standardization mappings."""
        return {
            'tuber melanosporum': 'Tuber melanosporum',
            'tuber magnatum': 'Tuber magnatum',
            'tuber aestivum': 'Tuber aestivum',
            'tuber borchii': 'Tuber borchii',
            'tuber brumale': 'Tuber brumale'
        }
    
    def harmonize_data(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Harmonize data from multiple sources into a unified format.
        
        Args:
            data_dict: Dictionary mapping source names to DataFrames
            
        Returns:
            Dictionary with harmonized data, metadata, and quality scores
        """
        logger.info("Starting data harmonization process")
        
        # Initialize result structure
        result = {
            'records_df': pd.DataFrame(),
            'metadata_df': pd.DataFrame(),
            'summary_stats': {},
            'quality_scores': {},
            'licensing_info': {},
            'harmonization_log': []
        }
        
        # Process each data source
        harmonized_sources = []
        for source, df in data_dict.items():
            if df.empty:
                continue
                
            logger.info(f"Harmonizing {source} data: {len(df)} records")
            
            # Add source-specific harmonization
            harmonized_df = self._harmonize_source_data(source, df)
            
            # Add quality scoring
            quality_scores = self._calculate_quality_scores(source, harmonized_df)
            
            # Add licensing information
            licensing_info = self._extract_licensing_info(source, harmonized_df)
            
            # Store results
            harmonized_sources.append(harmonized_df)
            result['quality_scores'][source] = quality_scores
            result['licensing_info'][source] = licensing_info
            
            # Log harmonization steps
            result['harmonization_log'].append({
                'source': source,
                'original_records': len(df),
                'harmonized_records': len(harmonized_df),
                'timestamp': datetime.now(timezone.utc).isoformat()
            })
        
        # Merge all harmonized data
        if harmonized_sources:
            result['records_df'] = self._merge_harmonized_data(harmonized_sources)
            
            # Create metadata
            result['metadata_df'] = self._create_metadata_df(data_dict, result['quality_scores'])
            
            # Calculate summary statistics
            result['summary_stats'] = self._calculate_summary_stats(result['records_df'])
        
        logger.info(f"Harmonization complete: {len(result['records_df'])} total records")
        return result
    
    def _harmonize_source_data(self, source: str, df: pd.DataFrame) -> pd.DataFrame:
        """Harmonize data from a specific source."""
        harmonized_df = df.copy()
        
        # Add source identifier
        harmonized_df['data_source'] = source
        
        # Standardize coordinates
        harmonized_df = self._standardize_coordinates(harmonized_df)
        
        # Standardize species names
        harmonized_df = self._standardize_species_names(harmonized_df)
        
        # Normalize units
        harmonized_df = self._normalize_units(harmonized_df, source)
        
        # Add temporal standardization
        harmonized_df = self._standardize_temporal_data(harmonized_df)
        
        # Add derived environmental indicators
        harmonized_df = self._add_environmental_indicators(harmonized_df)
        
        # Add data quality flags
        harmonized_df = self._add_quality_flags(harmonized_df, source)
        
        return harmonized_df
    
    def _standardize_coordinates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize coordinate data across sources."""
        if 'latitude' in df.columns and 'longitude' in df.columns:
            # Ensure numeric types
            df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
            df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
            
            # Add coordinate precision
            df['coord_precision'] = df.apply(
                lambda x: self._calculate_coordinate_precision(x['latitude'], x['longitude']), 
                axis=1
            )
            
            # Add coordinate quality
            df['coord_quality'] = df.apply(
                lambda x: self._assess_coordinate_quality(x['latitude'], x['longitude']), 
                axis=1
            )
        
        return df
    
    def _calculate_coordinate_precision(self, lat: float, lon: float) -> str:
        """Calculate coordinate precision based on decimal places."""
        if pd.isna(lat) or pd.isna(lon):
            return 'unknown'
        
        lat_decimals = len(str(lat).split('.')[-1]) if '.' in str(lat) else 0
        lon_decimals = len(str(lon).split('.')[-1]) if '.' in str(lon) else 0
        min_decimals = min(lat_decimals, lon_decimals)
        
        if min_decimals >= 5:
            return 'high'  # ~1m precision
        elif min_decimals >= 3:
            return 'medium'  # ~100m precision
        elif min_decimals >= 1:
            return 'low'  # ~10km precision
        else:
            return 'very_low'  # >10km precision
    
    def _assess_coordinate_quality(self, lat: float, lon: float) -> str:
        """Assess coordinate quality based on ranges and precision."""
        if pd.isna(lat) or pd.isna(lon):
            return 'invalid'
        
        # Check valid ranges
        if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
            return 'invalid'
        
        # Check for suspicious coordinates (0,0 or repeated values)
        if (lat == 0 and lon == 0) or (lat == lon):
            return 'suspicious'
        
        # Check precision
        precision = self._calculate_coordinate_precision(lat, lon)
        if precision in ['high', 'medium']:
            return 'good'
        elif precision == 'low':
            return 'fair'
        else:
            return 'poor'
    
    def _standardize_species_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize species names using taxonomy mappings."""
        if 'species' in df.columns:
            # Apply taxonomy mappings
            df['species_standardized'] = df['species'].str.lower().str.strip().map(
                self.taxonomy_mappings
            ).fillna(df['species'])
            
            # Add taxonomic hierarchy if available
            df['genus'] = df['species_standardized'].str.split().str[0]
            df['specific_epithet'] = df['species_standardized'].str.split().str[1:].str.join(' ')
        
        return df
    
    def _normalize_units(self, df: pd.DataFrame, source: str) -> pd.DataFrame:
        """Normalize units across different data sources."""
        # Temperature normalization
        temp_cols = [col for col in df.columns if 'temp' in col.lower() and 'C' in col]
        for col in temp_cols:
            if col in df.columns:
                # WorldClim temperatures are in 0.1°C, convert to °C
                if source == 'worldclim' and 'temp' in col:
                    df[col] = df[col] / 10.0
        
        # Precipitation normalization
        precip_cols = [col for col in df.columns if 'precip' in col.lower() and 'mm' in col]
        for col in precip_cols:
            if col in df.columns:
                # Ensure mm units
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Soil concentration normalization
        soil_cols = [col for col in df.columns if 'soil_' in col and ('pct' in col or 'mg' in col)]
        for col in soil_cols:
            if col in df.columns and 'pct' in col:
                # Convert from 0-1 to 0-100 if needed
                if df[col].max() <= 1.0:
                    df[col] = df[col] * 100
        
        return df
    
    def _standardize_temporal_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize temporal data across sources."""
        date_cols = [col for col in df.columns if 'date' in col.lower()]
        
        for col in date_cols:
            if col in df.columns:
                # Convert to datetime
                df[col] = pd.to_datetime(df[col], errors='coerce')
                
                # Add temporal components
                df[f'{col}_year'] = df[col].dt.year
                df[f'{col}_month'] = df[col].dt.month
                df[f'{col}_day'] = df[col].dt.day
                df[f'{col}_dayofyear'] = df[col].dt.dayofyear
                
                # Add temporal quality
                df[f'{col}_quality'] = df[col].apply(self._assess_temporal_quality)
        
        return df
    
    def _assess_temporal_quality(self, date_val) -> str:
        """Assess temporal data quality."""
        if pd.isna(date_val):
            return 'missing'
        
        # Check if date is reasonable (not too far in future/past)
        current_year = datetime.now().year
        year = date_val.year if hasattr(date_val, 'year') else None
        
        if year:
            if year > current_year:
                return 'invalid'
            elif year < 1800:
                return 'questionable'
            elif year < 1900:
                return 'old'
            else:
                return 'good'
        
        return 'unknown'
    
    def _add_environmental_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived environmental indicators."""
        # Environmental richness index
        if all(col in df.columns for col in ['soil_pH', 'mean_annual_temp_C', 'annual_precip_mm']):
            df['env_richness_index'] = self._calculate_environmental_richness(df)
        
        # Species diversity metrics (if multiple species per location)
        if 'species_standardized' in df.columns and 'latitude' in df.columns:
            df = self._add_diversity_metrics(df)
        
        # Temporal trend flags
        if 'event_date_year' in df.columns:
            df['temporal_trend_flag'] = self._calculate_temporal_trends(df)
        
        return df
    
    def _calculate_environmental_richness(self, df: pd.DataFrame) -> pd.Series:
        """Calculate environmental richness index."""
        # Normalize environmental variables to 0-1 scale
        ph_norm = (df['soil_pH'] - 3) / (10 - 3)  # pH 3-10
        temp_norm = (df['mean_annual_temp_C'] - (-10)) / (30 - (-10))  # -10 to 30°C
        precip_norm = np.log1p(df['annual_precip_mm']) / np.log1p(3000)  # 0-3000mm
        
        # Calculate richness as weighted average
        richness = (ph_norm * 0.3 + temp_norm * 0.3 + precip_norm * 0.4)
        return richness.fillna(0)
    
    def _add_diversity_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add species diversity metrics for each location."""
        # Group by coordinates and calculate diversity
        coord_groups = df.groupby(['latitude', 'longitude'])
        
        diversity_metrics = []
        for (lat, lon), group in coord_groups:
            species_list = group['species_standardized'].dropna().unique()
            n_species = len(species_list)
            
            # Shannon diversity (simplified)
            if n_species > 1:
                shannon = -sum((count/n_species) * np.log(count/n_species) 
                             for count in group['species_standardized'].value_counts())
            else:
                shannon = 0
            
            diversity_metrics.append({
                'latitude': lat,
                'longitude': lon,
                'species_count': n_species,
                'shannon_diversity': shannon
            })
        
        if diversity_metrics:
            diversity_df = pd.DataFrame(diversity_metrics)
            df = df.merge(diversity_df, on=['latitude', 'longitude'], how='left')
        
        return df
    
    def _calculate_temporal_trends(self, df: pd.DataFrame) -> pd.Series:
        """Calculate temporal trend flags."""
        if 'event_date_year' not in df.columns:
            return pd.Series(['unknown'] * len(df), index=df.index)
        
        current_year = datetime.now().year
        years = df['event_date_year'].dropna()
        
        if len(years) < 2:
            return pd.Series(['insufficient_data'] * len(df), index=df.index)
        
        # Simple trend calculation
        recent_years = years[years >= current_year - 10]
        if len(recent_years) > 0:
            return pd.Series(['recent'] * len(df), index=df.index)
        else:
            return pd.Series(['historical'] * len(df), index=df.index)
    
    def _add_quality_flags(self, df: pd.DataFrame, source: str) -> pd.DataFrame:
        """Add data quality flags."""
        # Coordinate quality
        if 'coord_quality' in df.columns:
            df['has_good_coordinates'] = df['coord_quality'].isin(['good', 'fair'])
        else:
            df['has_good_coordinates'] = False
        
        # Temporal quality
        date_cols = [col for col in df.columns if col.endswith('_quality')]
        if date_cols:
            df['has_good_temporal'] = df[date_cols].apply(
                lambda x: x.isin(['good', 'old']).any(), axis=1
            )
        else:
            df['has_good_temporal'] = False
        
        # Source-specific quality
        if source == 'gbif':
            df['has_institution_data'] = df['institution_code'].notna()
        elif source == 'inaturalist':
            df['has_quality_grade'] = df['quality_grade'].notna()
        
        return df
    
    def _calculate_quality_scores(self, source: str, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate comprehensive quality scores for a dataset."""
        if df.empty:
            return {'overall_score': 0, 'components': {}}
        
        scores = {}
        
        # Coordinate quality (40% weight)
        if 'coord_quality' in df.columns:
            coord_scores = {'good': 1.0, 'fair': 0.7, 'poor': 0.3, 'invalid': 0.0, 'suspicious': 0.1}
            coord_quality = df['coord_quality'].map(coord_scores).fillna(0).mean()
            scores['coordinate_quality'] = coord_quality
        else:
            scores['coordinate_quality'] = 0.0
        
        # Temporal quality (20% weight)
        date_cols = [col for col in df.columns if col.endswith('_quality')]
        if date_cols:
            temp_scores = {'good': 1.0, 'old': 0.8, 'questionable': 0.4, 'invalid': 0.0, 'missing': 0.0}
            temporal_quality = df[date_cols].apply(
                lambda x: x.map(temp_scores).fillna(0).max(), axis=1
            ).mean()
            scores['temporal_quality'] = temporal_quality
        else:
            scores['temporal_quality'] = 0.0
        
        # Data completeness (20% weight)
        completeness = (1 - df.isnull().sum().sum() / (len(df) * len(df.columns)))
        scores['completeness'] = completeness
        
        # Source reliability (20% weight)
        source_reliability = {
            'gbif': 0.9, 'inaturalist': 0.8, 'soilgrids': 0.95, 
            'worldclim': 0.95, 'glim': 0.85, 'ebi_metagenomics': 0.9
        }
        scores['source_reliability'] = source_reliability.get(source, 0.5)
        
        # Calculate overall score
        overall_score = (
            scores['coordinate_quality'] * 0.4 +
            scores['temporal_quality'] * 0.2 +
            scores['completeness'] * 0.2 +
            scores['source_reliability'] * 0.2
        )
        scores['overall_score'] = overall_score
        
        return scores
    
    def _extract_licensing_info(self, source: str, df: pd.DataFrame) -> Dict[str, Any]:
        """Extract licensing information for a dataset."""
        source_config = self.config.get(source, {})
        
        return {
            'license': source_config.get('license', 'Unknown'),
            'attribution': source_config.get('attribution', 'Unknown'),
            'license_url': source_config.get('data_license_url', ''),
            'record_count': len(df),
            'extraction_date': datetime.now(timezone.utc).isoformat(),
            'api_version': source_config.get('api_version', 'Unknown')
        }
    
    def _merge_harmonized_data(self, dataframes: List[pd.DataFrame]) -> pd.DataFrame:
        """Merge harmonized data from multiple sources."""
        if not dataframes:
            return pd.DataFrame()
        
        if len(dataframes) == 1:
            return dataframes[0]
        
        # Find common columns for merging
        common_cols = set(dataframes[0].columns)
        for df in dataframes[1:]:
            common_cols = common_cols.intersection(set(df.columns))
        
        # Merge on common columns, prioritizing coordinates
        merge_cols = ['latitude', 'longitude'] if 'latitude' in common_cols else list(common_cols)
        
        result = dataframes[0]
        for df in dataframes[1:]:
            result = pd.merge(result, df, on=merge_cols, how='outer', suffixes=('', '_dup'))
        
        return result
    
    def _create_metadata_df(self, data_dict: Dict[str, pd.DataFrame], 
                           quality_scores: Dict[str, Dict]) -> pd.DataFrame:
        """Create metadata DataFrame for the collection."""
        metadata = []
        
        for source, df in data_dict.items():
            if df.empty:
                continue
            
            source_config = self.config.get(source, {})
            quality = quality_scores.get(source, {})
            
            metadata.append({
                'source': source,
                'record_count': len(df),
                'columns': len(df.columns),
                'license': source_config.get('license', 'Unknown'),
                'attribution': source_config.get('attribution', 'Unknown'),
                'quality_score': quality.get('overall_score', 0.0),
                'coordinate_coverage': (df['latitude'].notna() & df['longitude'].notna()).sum() / len(df) * 100,
                'temporal_coverage': df.select_dtypes(include=['datetime64']).notna().any(axis=1).sum() / len(df) * 100,
                'collection_date': datetime.now(timezone.utc).isoformat()
            })
        
        return pd.DataFrame(metadata)
    
    def _calculate_summary_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate summary statistics for the harmonized dataset."""
        if df.empty:
            return {}
        
        stats = {
            'total_records': len(df),
            'unique_sources': df['data_source'].nunique() if 'data_source' in df.columns else 0,
            'coordinate_coverage': (df['latitude'].notna() & df['longitude'].notna()).sum() / len(df) * 100,
            'temporal_coverage': df.select_dtypes(include=['datetime64']).notna().any(axis=1).sum() / len(df) * 100,
            'missing_data_percentage': df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100
        }
        
        # Species diversity
        if 'species_standardized' in df.columns:
            stats['unique_species'] = df['species_standardized'].nunique()
            stats['most_common_species'] = df['species_standardized'].mode().iloc[0] if not df['species_standardized'].mode().empty else None
        
        # Geographic extent
        if 'latitude' in df.columns and 'longitude' in df.columns:
            valid_coords = df.dropna(subset=['latitude', 'longitude'])
            if not valid_coords.empty:
                stats['lat_range'] = [valid_coords['latitude'].min(), valid_coords['latitude'].max()]
                stats['lon_range'] = [valid_coords['longitude'].min(), valid_coords['longitude'].max()]
        
        return stats