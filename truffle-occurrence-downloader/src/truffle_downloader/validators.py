"""
Data validation module for truffle occurrence data.

This module provides comprehensive data validation and quality checks
for downloaded truffle occurrence data.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime, timedelta
import re

logger = logging.getLogger(__name__)


class DataValidator:
    """
    Comprehensive data validator for truffle occurrence data.
    
    Provides various validation methods to ensure data quality and consistency.
    """
    
    def __init__(self):
        """Initialize the data validator."""
        self.validation_results = {}
        self.quality_metrics = {}
        
    def validate_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform comprehensive data validation.
        
        Args:
            df: DataFrame with occurrence data
            
        Returns:
            Dictionary with validation results and quality metrics
        """
        if df.empty:
            return {
                'valid': False,
                'errors': ['Empty dataset'],
                'warnings': [],
                'quality_metrics': {}
            }
        
        self.validation_results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'quality_metrics': {}
        }
        
        # Run all validation checks
        self._validate_required_columns(df)
        self._validate_coordinates(df)
        self._validate_dates(df)
        self._validate_species_names(df)
        self._validate_geographic_data(df)
        self._validate_data_completeness(df)
        self._validate_outliers(df)
        self._validate_duplicates(df)
        
        # Calculate overall quality score
        self._calculate_quality_score()
        
        return self.validation_results
    
    def _validate_required_columns(self, df: pd.DataFrame) -> None:
        """Validate that required columns are present."""
        required_columns = ['species', 'latitude', 'longitude']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            self.validation_results['errors'].append(
                f"Missing required columns: {missing_columns}"
            )
            self.validation_results['valid'] = False
    
    def _validate_coordinates(self, df: pd.DataFrame) -> None:
        """Validate coordinate data."""
        if 'latitude' not in df.columns or 'longitude' not in df.columns:
            return
        
        # Check latitude range
        invalid_lat = df[(df['latitude'] < -90) | (df['latitude'] > 90)]
        if not invalid_lat.empty:
            self.validation_results['errors'].append(
                f"Found {len(invalid_lat)} records with invalid latitude values"
            )
            self.validation_results['valid'] = False
        
        # Check longitude range
        invalid_lon = df[(df['longitude'] < -180) | (df['longitude'] > 180)]
        if not invalid_lon.empty:
            self.validation_results['errors'].append(
                f"Found {len(invalid_lon)} records with invalid longitude values"
            )
            self.validation_results['valid'] = False
        
        # Check for coordinates at (0, 0) - often indicates missing data
        zero_coords = df[(df['latitude'] == 0) & (df['longitude'] == 0)]
        if not zero_coords.empty:
            self.validation_results['warnings'].append(
                f"Found {len(zero_coords)} records with coordinates at (0, 0) - may indicate missing data"
            )
    
    def _validate_dates(self, df: pd.DataFrame) -> None:
        """Validate date data."""
        if 'event_date' not in df.columns:
            return
        
        # Check for future dates
        current_date = datetime.now()
        future_dates = df[df['event_date'] > current_date]
        if not future_dates.empty:
            self.validation_results['warnings'].append(
                f"Found {len(future_dates)} records with future dates"
            )
        
        # Check for very old dates (before 1800)
        old_date_threshold = datetime(1800, 1, 1)
        old_dates = df[df['event_date'] < old_date_threshold]
        if not old_dates.empty:
            self.validation_results['warnings'].append(
                f"Found {len(old_dates)} records with very old dates (before 1800)"
            )
        
        # Check year consistency
        if 'year' in df.columns and 'event_date' in df.columns:
            df_with_dates = df.dropna(subset=['event_date', 'year'])
            if not df_with_dates.empty:
                df_with_dates['event_year'] = pd.to_datetime(df_with_dates['event_date']).dt.year
                inconsistent_years = df_with_dates[df_with_dates['year'] != df_with_dates['event_year']]
                if not inconsistent_years.empty:
                    self.validation_results['warnings'].append(
                        f"Found {len(inconsistent_years)} records with inconsistent year values"
                    )
    
    def _validate_species_names(self, df: pd.DataFrame) -> None:
        """Validate species name data."""
        if 'species' not in df.columns:
            return
        
        # Check for empty species names
        empty_species = df[df['species'].isna() | (df['species'] == '')]
        if not empty_species.empty:
            self.validation_results['errors'].append(
                f"Found {len(empty_species)} records with empty species names"
            )
            self.validation_results['valid'] = False
        
        # Check for valid species name format (Genus species)
        if not df.empty:
            species_pattern = r'^[A-Z][a-z]+\s+[a-z]+'
            invalid_species = df[~df['species'].str.match(species_pattern, na=False)]
            if not invalid_species.empty:
                self.validation_results['warnings'].append(
                    f"Found {len(invalid_species)} records with potentially invalid species name format"
                )
    
    def _validate_geographic_data(self, df: pd.DataFrame) -> None:
        """Validate geographic data quality."""
        if 'coordinate_uncertainty' not in df.columns:
            return
        
        # Check for high coordinate uncertainty
        high_uncertainty = df[df['coordinate_uncertainty'] > 10000]  # > 10km
        if not high_uncertainty.empty:
            self.validation_results['warnings'].append(
                f"Found {len(high_uncertainty)} records with high coordinate uncertainty (>10km)"
            )
        
        # Check for missing country information
        if 'country' in df.columns:
            missing_country = df[df['country'].isna() | (df['country'] == '')]
            if not missing_country.empty:
                self.validation_results['warnings'].append(
                    f"Found {len(missing_country)} records with missing country information"
                )
    
    def _validate_data_completeness(self, df: pd.DataFrame) -> None:
        """Validate data completeness."""
        total_records = len(df)
        
        # Calculate completeness for key fields
        key_fields = ['species', 'latitude', 'longitude', 'event_date', 'country']
        completeness = {}
        
        for field in key_fields:
            if field in df.columns:
                non_null_count = df[field].notna().sum()
                completeness[field] = non_null_count / total_records
        
        self.validation_results['quality_metrics']['completeness'] = completeness
        
        # Check for very low completeness
        for field, completeness_score in completeness.items():
            if completeness_score < 0.5:  # Less than 50% complete
                self.validation_results['warnings'].append(
                    f"Low completeness for {field}: {completeness_score:.1%}"
                )
    
    def _validate_outliers(self, df: pd.DataFrame) -> None:
        """Validate for statistical outliers."""
        if 'latitude' not in df.columns or 'longitude' not in df.columns:
            return
        
        # Use IQR method to detect outliers
        numeric_columns = ['latitude', 'longitude']
        outlier_counts = {}
        
        for col in numeric_columns:
            if col in df.columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
                outlier_counts[col] = len(outliers)
        
        if any(count > 0 for count in outlier_counts.values()):
            self.validation_results['warnings'].append(
                f"Found potential outliers: {outlier_counts}"
            )
    
    def _validate_duplicates(self, df: pd.DataFrame) -> None:
        """Validate for duplicate records."""
        if df.empty:
            return
        
        # Check for exact duplicates
        exact_duplicates = df.duplicated().sum()
        if exact_duplicates > 0:
            self.validation_results['warnings'].append(
                f"Found {exact_duplicates} exact duplicate records"
            )
        
        # Check for duplicates by GBIF ID
        if 'gbif_id' in df.columns:
            gbif_duplicates = df['gbif_id'].duplicated().sum()
            if gbif_duplicates > 0:
                self.validation_results['errors'].append(
                    f"Found {gbif_duplicates} duplicate GBIF IDs"
                )
                self.validation_results['valid'] = False
        
        # Check for spatial duplicates (same coordinates)
        if 'latitude' in df.columns and 'longitude' in df.columns:
            spatial_duplicates = df.duplicated(subset=['latitude', 'longitude']).sum()
            if spatial_duplicates > 0:
                self.validation_results['warnings'].append(
                    f"Found {spatial_duplicates} records with identical coordinates"
                )
    
    def _calculate_quality_score(self) -> None:
        """Calculate overall data quality score."""
        if not self.validation_results['quality_metrics']:
            return
        
        # Base score
        score = 100
        
        # Deduct points for errors
        score -= len(self.validation_results['errors']) * 20
        
        # Deduct points for warnings
        score -= len(self.validation_results['warnings']) * 5
        
        # Deduct points for low completeness
        if 'completeness' in self.validation_results['quality_metrics']:
            avg_completeness = np.mean(list(self.validation_results['quality_metrics']['completeness'].values()))
            score -= (1 - avg_completeness) * 30
        
        # Ensure score is between 0 and 100
        score = max(0, min(100, score))
        
        self.validation_results['quality_metrics']['overall_score'] = score
        
        # Add quality rating
        if score >= 90:
            rating = "Excellent"
        elif score >= 80:
            rating = "Good"
        elif score >= 70:
            rating = "Fair"
        elif score >= 60:
            rating = "Poor"
        else:
            rating = "Very Poor"
        
        self.validation_results['quality_metrics']['rating'] = rating
    
    def get_validation_summary(self) -> str:
        """Get a human-readable validation summary."""
        if not self.validation_results:
            return "No validation performed"
        
        summary = []
        summary.append(f"Data Quality: {self.validation_results['quality_metrics'].get('rating', 'Unknown')}")
        summary.append(f"Overall Score: {self.validation_results['quality_metrics'].get('overall_score', 0):.1f}/100")
        
        if self.validation_results['errors']:
            summary.append(f"Errors: {len(self.validation_results['errors'])}")
            for error in self.validation_results['errors'][:3]:  # Show first 3 errors
                summary.append(f"  - {error}")
        
        if self.validation_results['warnings']:
            summary.append(f"Warnings: {len(self.validation_results['warnings'])}")
            for warning in self.validation_results['warnings'][:3]:  # Show first 3 warnings
                summary.append(f"  - {warning}")
        
        return "\n".join(summary)
    
    def filter_high_quality_data(self, df: pd.DataFrame, 
                                min_quality_score: float = 80.0) -> pd.DataFrame:
        """
        Filter data to keep only high-quality records.
        
        Args:
            df: DataFrame with occurrence data
            min_quality_score: Minimum quality score threshold
            
        Returns:
            Filtered DataFrame with high-quality records
        """
        if df.empty:
            return df
        
        filtered_df = df.copy()
        
        # Remove records with missing coordinates
        filtered_df = filtered_df.dropna(subset=['latitude', 'longitude'])
        
        # Remove records with invalid coordinates
        filtered_df = filtered_df[
            (filtered_df['latitude'] >= -90) & (filtered_df['latitude'] <= 90) &
            (filtered_df['longitude'] >= -180) & (filtered_df['longitude'] <= 180)
        ]
        
        # Remove records with high coordinate uncertainty
        if 'coordinate_uncertainty' in filtered_df.columns:
            filtered_df = filtered_df[
                (filtered_df['coordinate_uncertainty'].isna()) |
                (filtered_df['coordinate_uncertainty'] <= 10000)
            ]
        
        # Remove records with missing species names
        filtered_df = filtered_df.dropna(subset=['species'])
        
        # Remove exact duplicates
        filtered_df = filtered_df.drop_duplicates()
        
        logger.info(f"Filtered data: {len(df)} -> {len(filtered_df)} records")
        
        return filtered_df