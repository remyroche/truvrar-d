"""
Comprehensive data validation utilities for the GTHA project.
"""
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union
from pathlib import Path
import re

from .error_handling import ValidationError, validate_dataframe, validate_coordinates

logger = logging.getLogger(__name__)


class DataValidator:
    """Comprehensive data validation class for GTHA datasets."""
    
    def __init__(self, strict_mode: bool = True):
        self.strict_mode = strict_mode
        self.validation_rules = self._initialize_validation_rules()
        
    def _initialize_validation_rules(self) -> Dict[str, Dict[str, Any]]:
        """Initialize validation rules for different data types."""
        return {
            'coordinates': {
                'latitude': {'min': -90, 'max': 90, 'required': True},
                'longitude': {'min': -180, 'max': 180, 'required': True}
            },
            'species': {
                'required': True,
                'pattern': r'^[A-Za-z]+\s+[a-z]+$',  # Genus species format
                'min_length': 3,
                'max_length': 100
            },
            'dates': {
                'year': {'min': 1800, 'max': 2030, 'required': False},
                'month': {'min': 1, 'max': 12, 'required': False},
                'day': {'min': 1, 'max': 31, 'required': False}
            },
            'environmental': {
                'soil_pH': {'min': 0, 'max': 14, 'required': False},
                'temperature': {'min': -50, 'max': 50, 'required': False},
                'precipitation': {'min': 0, 'max': 10000, 'required': False}
            }
        }
    
    def validate_habitat_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate habitat data DataFrame comprehensively.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Dictionary with validation results and cleaned data
        """
        if df.empty:
            raise ValidationError("DataFrame is empty")
        
        validation_results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'cleaned_data': df.copy(),
            'validation_summary': {}
        }
        
        # Basic DataFrame validation
        try:
            validate_dataframe(df, min_rows=1)
        except ValidationError as e:
            validation_results['errors'].append(str(e))
            validation_results['is_valid'] = False
        
        # Validate coordinates
        coord_validation = self._validate_coordinates(df)
        validation_results['validation_summary']['coordinates'] = coord_validation
        if not coord_validation['is_valid']:
            validation_results['errors'].extend(coord_validation['errors'])
            validation_results['is_valid'] = False
        
        # Validate species names
        species_validation = self._validate_species_names(df)
        validation_results['validation_summary']['species'] = species_validation
        if not species_validation['is_valid']:
            validation_results['errors'].extend(species_validation['errors'])
            validation_results['is_valid'] = False
        
        # Validate environmental variables
        env_validation = self._validate_environmental_variables(df)
        validation_results['validation_summary']['environmental'] = env_validation
        if not env_validation['is_valid']:
            validation_results['warnings'].extend(env_validation['warnings'])
        
        # Validate temporal data
        temporal_validation = self._validate_temporal_data(df)
        validation_results['validation_summary']['temporal'] = temporal_validation
        if not temporal_validation['is_valid']:
            validation_results['warnings'].extend(temporal_validation['warnings'])
        
        # Clean data if validation passed
        if validation_results['is_valid']:
            validation_results['cleaned_data'] = self._clean_data(df)
        
        return validation_results
    
    def _validate_coordinates(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate coordinate data."""
        result = {
            'is_valid': True,
            'errors': [],
            'valid_count': 0,
            'invalid_count': 0
        }
        
        if 'latitude' not in df.columns or 'longitude' not in df.columns:
            result['errors'].append("Missing coordinate columns")
            result['is_valid'] = False
            return result
        
        valid_coords = 0
        invalid_coords = 0
        
        for idx, row in df.iterrows():
            try:
                validate_coordinates(row['latitude'], row['longitude'])
                valid_coords += 1
            except ValidationError:
                invalid_coords += 1
                if self.strict_mode:
                    result['errors'].append(f"Invalid coordinates at row {idx}")
        
        result['valid_count'] = valid_coords
        result['invalid_count'] = invalid_coords
        
        if invalid_coords > 0 and self.strict_mode:
            result['is_valid'] = False
        
        return result
    
    def _validate_species_names(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate species name data."""
        result = {
            'is_valid': True,
            'errors': [],
            'valid_count': 0,
            'invalid_count': 0,
            'suggestions': {}
        }
        
        if 'species' not in df.columns:
            result['errors'].append("Missing species column")
            result['is_valid'] = False
            return result
        
        species_pattern = re.compile(self.validation_rules['species']['pattern'])
        
        for idx, row in df.iterrows():
            species_name = str(row['species']).strip()
            
            if not species_name or species_name.lower() in ['nan', 'none', '']:
                result['invalid_count'] += 1
                if self.strict_mode:
                    result['errors'].append(f"Empty species name at row {idx}")
                continue
            
            if len(species_name) < self.validation_rules['species']['min_length']:
                result['invalid_count'] += 1
                if self.strict_mode:
                    result['errors'].append(f"Species name too short at row {idx}: {species_name}")
                continue
            
            if not species_pattern.match(species_name):
                result['invalid_count'] += 1
                suggestion = self._suggest_species_name(species_name)
                if suggestion:
                    result['suggestions'][species_name] = suggestion
                if self.strict_mode:
                    result['errors'].append(f"Invalid species name format at row {idx}: {species_name}")
            else:
                result['valid_count'] += 1
        
        if result['invalid_count'] > 0 and self.strict_mode:
            result['is_valid'] = False
        
        return result
    
    def _validate_environmental_variables(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate environmental variable data."""
        result = {
            'is_valid': True,
            'warnings': [],
            'variable_validation': {}
        }
        
        env_columns = [col for col in df.columns if any(prefix in col for prefix in ['soil_', 'climate_', 'temp_', 'precip_'])]
        
        for col in env_columns:
            col_validation = self._validate_numeric_column(df, col)
            result['variable_validation'][col] = col_validation
            
            if not col_validation['is_valid']:
                result['warnings'].extend(col_validation['warnings'])
        
        return result
    
    def _validate_temporal_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate temporal data."""
        result = {
            'is_valid': True,
            'warnings': [],
            'temporal_coverage': 0
        }
        
        date_columns = [col for col in df.columns if 'date' in col.lower() or col in ['year', 'month', 'day']]
        
        if not date_columns:
            result['warnings'].append("No temporal data found")
            return result
        
        # Check year data
        if 'year' in df.columns:
            year_validation = self._validate_year_column(df['year'])
            result['temporal_coverage'] = year_validation['valid_percentage']
            if not year_validation['is_valid']:
                result['warnings'].extend(year_validation['warnings'])
        
        return result
    
    def _validate_numeric_column(self, df: pd.DataFrame, column: str) -> Dict[str, Any]:
        """Validate a numeric column."""
        result = {
            'is_valid': True,
            'warnings': [],
            'valid_count': 0,
            'invalid_count': 0,
            'outliers': 0
        }
        
        if column not in df.columns:
            result['warnings'].append(f"Column {column} not found")
            return result
        
        # Convert to numeric
        numeric_data = pd.to_numeric(df[column], errors='coerce')
        valid_data = numeric_data.dropna()
        
        result['valid_count'] = len(valid_data)
        result['invalid_count'] = len(df) - len(valid_data)
        
        if len(valid_data) == 0:
            result['warnings'].append(f"No valid numeric data in {column}")
            result['is_valid'] = False
            return result
        
        # Check for outliers using IQR method
        Q1 = valid_data.quantile(0.25)
        Q3 = valid_data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = valid_data[(valid_data < lower_bound) | (valid_data > upper_bound)]
        result['outliers'] = len(outliers)
        
        if len(outliers) > len(valid_data) * 0.1:  # More than 10% outliers
            result['warnings'].append(f"High number of outliers in {column}: {len(outliers)}")
        
        return result
    
    def _validate_year_column(self, year_series: pd.Series) -> Dict[str, Any]:
        """Validate year data."""
        result = {
            'is_valid': True,
            'warnings': [],
            'valid_percentage': 0
        }
        
        current_year = pd.Timestamp.now().year
        valid_years = year_series[
            (year_series >= 1800) & 
            (year_series <= current_year) & 
            year_series.notna()
        ]
        
        result['valid_percentage'] = len(valid_years) / len(year_series) * 100
        
        if result['valid_percentage'] < 50:
            result['warnings'].append(f"Low temporal coverage: {result['valid_percentage']:.1f}%")
        
        return result
    
    def _suggest_species_name(self, species_name: str) -> Optional[str]:
        """Suggest corrections for species names."""
        # Simple suggestions based on common patterns
        suggestions = {
            'tuber melanosporum': 'Tuber melanosporum',
            'tuber magnatum': 'Tuber magnatum',
            'tuber aestivum': 'Tuber aestivum',
            'tuber borchii': 'Tuber borchii',
            'tuber brumale': 'Tuber brumale'
        }
        
        return suggestions.get(species_name.lower())
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize data."""
        cleaned_df = df.copy()
        
        # Clean species names
        if 'species' in cleaned_df.columns:
            cleaned_df['species'] = cleaned_df['species'].str.strip().str.title()
        
        # Clean coordinates
        if 'latitude' in cleaned_df.columns and 'longitude' in cleaned_df.columns:
            cleaned_df = cleaned_df.dropna(subset=['latitude', 'longitude'])
            cleaned_df = cleaned_df[
                (cleaned_df['latitude'] >= -90) & (cleaned_df['latitude'] <= 90) &
                (cleaned_df['longitude'] >= -180) & (cleaned_df['longitude'] <= 180)
            ]
        
        # Clean numeric columns
        numeric_columns = cleaned_df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='coerce')
        
        return cleaned_df
    
    def validate_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Assess overall data quality."""
        quality_report = {
            'completeness_score': 0,
            'accuracy_score': 0,
            'consistency_score': 0,
            'overall_score': 0,
            'recommendations': []
        }
        
        # Calculate completeness score
        total_cells = df.shape[0] * df.shape[1]
        missing_cells = df.isnull().sum().sum()
        quality_report['completeness_score'] = (1 - missing_cells / total_cells) * 100
        
        # Calculate accuracy score (based on coordinate validity)
        if 'latitude' in df.columns and 'longitude' in df.columns:
            valid_coords = 0
            for _, row in df.iterrows():
                try:
                    validate_coordinates(row['latitude'], row['longitude'])
                    valid_coords += 1
                except ValidationError:
                    pass
            quality_report['accuracy_score'] = (valid_coords / len(df)) * 100
        else:
            quality_report['accuracy_score'] = 50  # Neutral score if no coordinates
        
        # Calculate consistency score (based on data type consistency)
        type_consistency = 0
        for col in df.columns:
            if df[col].dtype == 'object':
                # Check if all values are strings
                if df[col].apply(lambda x: isinstance(x, str) or pd.isna(x)).all():
                    type_consistency += 1
            else:
                # Check if all values are numeric
                if df[col].apply(lambda x: isinstance(x, (int, float)) or pd.isna(x)).all():
                    type_consistency += 1
        
        quality_report['consistency_score'] = (type_consistency / len(df.columns)) * 100
        
        # Calculate overall score
        quality_report['overall_score'] = (
            quality_report['completeness_score'] * 0.4 +
            quality_report['accuracy_score'] * 0.4 +
            quality_report['consistency_score'] * 0.2
        )
        
        # Generate recommendations
        if quality_report['completeness_score'] < 80:
            quality_report['recommendations'].append("Improve data completeness by filling missing values")
        
        if quality_report['accuracy_score'] < 90:
            quality_report['recommendations'].append("Review and correct coordinate data")
        
        if quality_report['consistency_score'] < 90:
            quality_report['recommendations'].append("Standardize data types across columns")
        
        return quality_report


def validate_habitat_dataset(df: pd.DataFrame, strict_mode: bool = True) -> Dict[str, Any]:
    """
    Convenience function to validate habitat dataset.
    
    Args:
        df: DataFrame to validate
        strict_mode: Whether to use strict validation
        
    Returns:
        Validation results dictionary
    """
    validator = DataValidator(strict_mode=strict_mode)
    return validator.validate_habitat_data(df)


def assess_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Convenience function to assess data quality.
    
    Args:
        df: DataFrame to assess
        
    Returns:
        Quality assessment dictionary
    """
    validator = DataValidator()
    return validator.validate_data_quality(df)