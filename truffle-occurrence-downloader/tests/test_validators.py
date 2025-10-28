"""
Tests for the Data Validator
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from truffle_downloader import DataValidator


class TestDataValidator:
    """Test cases for DataValidator"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.validator = DataValidator()
    
    def test_validate_data_empty(self):
        """Test validation with empty DataFrame"""
        empty_df = pd.DataFrame()
        results = self.validator.validate_data(empty_df)
        
        assert results['valid'] is False
        assert 'Empty dataset' in results['errors']
    
    def test_validate_data_valid(self):
        """Test validation with valid data"""
        valid_data = pd.DataFrame({
            'species': ['Tuber melanosporum', 'Tuber magnatum'],
            'latitude': [44.0, 45.0],
            'longitude': [5.0, 8.0],
            'event_date': ['2023-01-01', '2023-02-01'],
            'country': ['FR', 'IT'],
            'gbif_id': [1, 2]
        })
        
        results = self.validator.validate_data(valid_data)
        
        assert results['valid'] is True
        assert len(results['errors']) == 0
        assert 'overall_score' in results['quality_metrics']
    
    def test_validate_required_columns_missing(self):
        """Test validation with missing required columns"""
        invalid_data = pd.DataFrame({
            'species': ['Tuber melanosporum'],
            # Missing latitude and longitude
        })
        
        results = self.validator.validate_data(invalid_data)
        
        assert results['valid'] is False
        assert any('Missing required columns' in error for error in results['errors'])
    
    def test_validate_coordinates_invalid_latitude(self):
        """Test validation with invalid latitude values"""
        invalid_data = pd.DataFrame({
            'species': ['Tuber melanosporum', 'Tuber melanosporum'],
            'latitude': [44.0, 200.0],  # Invalid latitude
            'longitude': [5.0, 8.0],
            'gbif_id': [1, 2]
        })
        
        results = self.validator.validate_data(invalid_data)
        
        assert results['valid'] is False
        assert any('invalid latitude values' in error for error in results['errors'])
    
    def test_validate_coordinates_invalid_longitude(self):
        """Test validation with invalid longitude values"""
        invalid_data = pd.DataFrame({
            'species': ['Tuber melanosporum', 'Tuber melanosporum'],
            'latitude': [44.0, 45.0],
            'longitude': [5.0, 200.0],  # Invalid longitude
            'gbif_id': [1, 2]
        })
        
        results = self.validator.validate_data(invalid_data)
        
        assert results['valid'] is False
        assert any('invalid longitude values' in error for error in results['errors'])
    
    def test_validate_coordinates_zero_coords(self):
        """Test validation with zero coordinates (potential missing data)"""
        invalid_data = pd.DataFrame({
            'species': ['Tuber melanosporum', 'Tuber melanosporum'],
            'latitude': [44.0, 0.0],  # Zero coordinates
            'longitude': [5.0, 0.0],
            'gbif_id': [1, 2]
        })
        
        results = self.validator.validate_data(invalid_data)
        
        assert results['valid'] is True  # Should be valid but with warning
        assert any('coordinates at (0, 0)' in warning for warning in results['warnings'])
    
    def test_validate_dates_future(self):
        """Test validation with future dates"""
        future_data = pd.DataFrame({
            'species': ['Tuber melanosporum'],
            'latitude': [44.0],
            'longitude': [5.0],
            'event_date': [datetime.now() + pd.Timedelta(days=365)],  # Future date
            'gbif_id': [1]
        })
        
        results = self.validator.validate_data(future_data)
        
        assert results['valid'] is True
        assert any('future dates' in warning for warning in results['warnings'])
    
    def test_validate_dates_old(self):
        """Test validation with very old dates"""
        old_data = pd.DataFrame({
            'species': ['Tuber melanosporum'],
            'latitude': [44.0],
            'longitude': [5.0],
            'event_date': ['1800-01-01'],  # Very old date
            'gbif_id': [1]
        })
        
        results = self.validator.validate_data(old_data)
        
        assert results['valid'] is True
        assert any('very old dates' in warning for warning in results['warnings'])
    
    def test_validate_species_names_empty(self):
        """Test validation with empty species names"""
        invalid_data = pd.DataFrame({
            'species': ['Tuber melanosporum', '', None],  # Empty and None values
            'latitude': [44.0, 45.0, 46.0],
            'longitude': [5.0, 8.0, 9.0],
            'gbif_id': [1, 2, 3]
        })
        
        results = self.validator.validate_data(invalid_data)
        
        assert results['valid'] is False
        assert any('empty species names' in error for error in results['errors'])
    
    def test_validate_species_names_format(self):
        """Test validation with invalid species name format"""
        invalid_data = pd.DataFrame({
            'species': ['Tuber melanosporum', 'tuber magnatum', 'TUBER AESTIVUM'],  # Invalid formats
            'latitude': [44.0, 45.0, 46.0],
            'longitude': [5.0, 8.0, 9.0],
            'gbif_id': [1, 2, 3]
        })
        
        results = self.validator.validate_data(invalid_data)
        
        assert results['valid'] is True  # Should be valid but with warnings
        assert any('invalid species name format' in warning for warning in results['warnings'])
    
    def test_validate_geographic_data_high_uncertainty(self):
        """Test validation with high coordinate uncertainty"""
        invalid_data = pd.DataFrame({
            'species': ['Tuber melanosporum', 'Tuber melanosporum'],
            'latitude': [44.0, 45.0],
            'longitude': [5.0, 8.0],
            'coordinate_uncertainty': [1000, 15000],  # High uncertainty
            'gbif_id': [1, 2]
        })
        
        results = self.validator.validate_data(invalid_data)
        
        assert results['valid'] is True
        assert any('high coordinate uncertainty' in warning for warning in results['warnings'])
    
    def test_validate_data_completeness(self):
        """Test data completeness validation"""
        incomplete_data = pd.DataFrame({
            'species': ['Tuber melanosporum', 'Tuber melanosporum'],
            'latitude': [44.0, 45.0],
            'longitude': [5.0, 8.0],
            'event_date': ['2023-01-01', None],  # Missing date
            'country': ['FR', None],  # Missing country
            'gbif_id': [1, 2]
        })
        
        results = self.validator.validate_data(incomplete_data)
        
        assert results['valid'] is True
        assert 'completeness' in results['quality_metrics']
        assert results['quality_metrics']['completeness']['event_date'] == 0.5
        assert results['quality_metrics']['completeness']['country'] == 0.5
    
    def test_validate_duplicates_exact(self):
        """Test validation with exact duplicates"""
        duplicate_data = pd.DataFrame({
            'species': ['Tuber melanosporum', 'Tuber melanosporum'],
            'latitude': [44.0, 44.0],
            'longitude': [5.0, 5.0],
            'gbif_id': [1, 1]  # Duplicate GBIF ID
        })
        
        results = self.validator.validate_data(duplicate_data)
        
        assert results['valid'] is False
        assert any('duplicate GBIF IDs' in error for error in results['errors'])
    
    def test_validate_duplicates_spatial(self):
        """Test validation with spatial duplicates"""
        duplicate_data = pd.DataFrame({
            'species': ['Tuber melanosporum', 'Tuber magnatum'],
            'latitude': [44.0, 44.0],  # Same coordinates
            'longitude': [5.0, 5.0],
            'gbif_id': [1, 2]
        })
        
        results = self.validator.validate_data(duplicate_data)
        
        assert results['valid'] is True
        assert any('identical coordinates' in warning for warning in results['warnings'])
    
    def test_calculate_quality_score(self):
        """Test quality score calculation"""
        # Test with perfect data
        perfect_data = pd.DataFrame({
            'species': ['Tuber melanosporum', 'Tuber magnatum'],
            'latitude': [44.0, 45.0],
            'longitude': [5.0, 8.0],
            'event_date': ['2023-01-01', '2023-02-01'],
            'country': ['FR', 'IT'],
            'gbif_id': [1, 2]
        })
        
        results = self.validator.validate_data(perfect_data)
        
        assert results['quality_metrics']['overall_score'] >= 90
        assert results['quality_metrics']['rating'] == "Excellent"
    
    def test_get_validation_summary(self):
        """Test validation summary generation"""
        test_data = pd.DataFrame({
            'species': ['Tuber melanosporum'],
            'latitude': [44.0],
            'longitude': [5.0],
            'gbif_id': [1]
        })
        
        self.validator.validate_data(test_data)
        summary = self.validator.get_validation_summary()
        
        assert isinstance(summary, str)
        assert "Data Quality:" in summary
        assert "Overall Score:" in summary
    
    def test_filter_high_quality_data(self):
        """Test high-quality data filtering"""
        mixed_data = pd.DataFrame({
            'species': ['Tuber melanosporum', 'Tuber melanosporum', 'Tuber melanosporum'],
            'latitude': [44.0, 200.0, 45.0],  # One invalid latitude
            'longitude': [5.0, 8.0, 9.0],
            'coordinate_uncertainty': [1000, 5000, 15000],  # One high uncertainty
            'gbif_id': [1, 2, 3]
        })
        
        filtered_data = self.validator.filter_high_quality_data(mixed_data)
        
        # Should remove invalid coordinates and high uncertainty
        assert len(filtered_data) == 1
        assert filtered_data['latitude'].iloc[0] == 45.0
        assert filtered_data['coordinate_uncertainty'].iloc[0] == 5000
    
    def test_filter_high_quality_data_empty(self):
        """Test high-quality data filtering with empty DataFrame"""
        empty_df = pd.DataFrame()
        filtered_data = self.validator.filter_high_quality_data(empty_df)
        
        assert filtered_data.empty