"""
Tests for the GBIF Truffle Downloader
"""

import pytest
import pandas as pd
from unittest.mock import Mock, patch
from truffle_downloader import GBIFTruffleDownloader


class TestGBIFTruffleDownloader:
    """Test cases for GBIFTruffleDownloader"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.downloader = GBIFTruffleDownloader()
    
    def test_init(self):
        """Test downloader initialization"""
        assert self.downloader.base_url == "https://api.gbif.org/v1"
        assert self.downloader.stats['total_requests'] == 0
        assert self.downloader.stats['records_downloaded'] == 0
    
    def test_init_with_config(self):
        """Test downloader initialization with config file"""
        # This would test loading from a config file
        # For now, we'll test the default config
        downloader = GBIFTruffleDownloader()
        assert downloader.config['gbif']['base_url'] == "https://api.gbif.org/v1"
    
    @patch('truffle_downloader.downloader.requests.Session.get')
    def test_get_species_key_success(self, mock_get):
        """Test successful species key retrieval"""
        # Mock response
        mock_response = Mock()
        mock_response.json.return_value = {
            'results': [{'key': 12345, 'scientificName': 'Tuber melanosporum'}]
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        species_key = self.downloader._get_species_key("Tuber melanosporum")
        
        assert species_key == 12345
        mock_get.assert_called_once()
    
    @patch('truffle_downloader.downloader.requests.Session.get')
    def test_get_species_key_not_found(self, mock_get):
        """Test species key retrieval when species not found"""
        # Mock response
        mock_response = Mock()
        mock_response.json.return_value = {'results': []}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        species_key = self.downloader._get_species_key("Unknown species")
        
        assert species_key is None
    
    @patch('truffle_downloader.downloader.requests.Session.get')
    def test_get_species_key_error(self, mock_get):
        """Test species key retrieval with error"""
        # Mock response to raise an exception
        mock_get.side_effect = Exception("API Error")
        
        species_key = self.downloader._get_species_key("Tuber melanosporum")
        
        assert species_key is None
    
    def test_build_search_params(self):
        """Test search parameter building"""
        params = self.downloader._build_search_params(
            species_key=12345,
            countries=["FR", "IT"],
            year_from=2020,
            year_to=2023,
            max_records=1000
        )
        
        assert params['taxonKey'] == 12345
        assert params['country'] == "FR|IT"
        assert params['year'] == "2020,2023"
        assert params['hasCoordinate'] is True
        assert params['hasGeospatialIssue'] is False
    
    def test_build_search_params_coordinate_bounds(self):
        """Test search parameter building with coordinate bounds"""
        coordinate_bounds = {
            'min_lat': 40.0,
            'max_lat': 50.0,
            'min_lon': -5.0,
            'max_lon': 10.0
        }
        
        params = self.downloader._build_search_params(
            species_key=12345,
            coordinate_bounds=coordinate_bounds
        )
        
        assert params['decimalLatitude'] == "40.0,50.0"
        assert params['decimalLongitude'] == "-5.0,10.0"
    
    def test_clean_and_validate_data(self):
        """Test data cleaning and validation"""
        # Create test data
        test_data = pd.DataFrame({
            'species': ['Tuber melanosporum', 'Tuber magnatum'],
            'decimalLatitude': [44.0, 45.0],
            'decimalLongitude': [5.0, 8.0],
            'eventDate': ['2023-01-01', '2023-02-01'],
            'country': ['FR', 'IT'],
            'gbifID': [1, 2]
        })
        
        cleaned_data = self.downloader._clean_and_validate_data(test_data)
        
        # Check column renaming
        assert 'latitude' in cleaned_data.columns
        assert 'longitude' in cleaned_data.columns
        assert 'event_date' in cleaned_data.columns
        
        # Check data types
        assert pd.api.types.is_numeric_dtype(cleaned_data['latitude'])
        assert pd.api.types.is_numeric_dtype(cleaned_data['longitude'])
        assert pd.api.types.is_datetime64_any_dtype(cleaned_data['event_date'])
        
        # Check source information
        assert 'source' in cleaned_data.columns
        assert all(cleaned_data['source'] == 'GBIF')
    
    def test_clean_and_validate_data_empty(self):
        """Test data cleaning with empty DataFrame"""
        empty_df = pd.DataFrame()
        cleaned_data = self.downloader._clean_and_validate_data(empty_df)
        
        assert cleaned_data.empty
    
    def test_clean_and_validate_data_invalid_coordinates(self):
        """Test data cleaning with invalid coordinates"""
        test_data = pd.DataFrame({
            'species': ['Tuber melanosporum', 'Tuber melanosporum'],
            'decimalLatitude': [44.0, 200.0],  # Invalid latitude
            'decimalLongitude': [5.0, 8.0],
            'gbifID': [1, 2]
        })
        
        cleaned_data = self.downloader._clean_and_validate_data(test_data)
        
        # Should remove invalid coordinates
        assert len(cleaned_data) == 1
        assert cleaned_data['latitude'].iloc[0] == 44.0
    
    def test_get_download_stats(self):
        """Test download statistics retrieval"""
        # Set some test stats
        self.downloader.stats['total_requests'] = 10
        self.downloader.stats['successful_requests'] = 8
        self.downloader.stats['failed_requests'] = 2
        self.downloader.stats['records_downloaded'] = 100
        
        stats = self.downloader.get_download_stats()
        
        assert stats['total_requests'] == 10
        assert stats['successful_requests'] == 8
        assert stats['failed_requests'] == 2
        assert stats['records_downloaded'] == 100
    
    @patch('truffle_downloader.downloader.requests.Session.get')
    def test_search_species(self, mock_get):
        """Test species search functionality"""
        # Mock response
        mock_response = Mock()
        mock_response.json.return_value = {
            'results': [
                {'key': 1, 'scientificName': 'Tuber melanosporum'},
                {'key': 2, 'scientificName': 'Tuber magnatum'}
            ]
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        results = self.downloader.search_species("Tuber", limit=2)
        
        assert len(results) == 2
        assert results[0]['scientificName'] == 'Tuber melanosporum'
        assert results[1]['scientificName'] == 'Tuber magnatum'
    
    @patch('truffle_downloader.downloader.requests.Session.get')
    def test_get_species_info(self, mock_get):
        """Test species information retrieval"""
        # Mock response
        mock_response = Mock()
        mock_response.json.return_value = {
            'key': 12345,
            'scientificName': 'Tuber melanosporum',
            'canonicalName': 'Tuber melanosporum',
            'rank': 'SPECIES'
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        info = self.downloader.get_species_info(12345)
        
        assert info['key'] == 12345
        assert info['scientificName'] == 'Tuber melanosporum'
        assert info['rank'] == 'SPECIES'
    
    def test_duration_calculation(self):
        """Test duration calculation"""
        from datetime import datetime, timedelta
        
        # Set start and end times
        self.downloader.stats['start_time'] = datetime.now()
        self.downloader.stats['end_time'] = datetime.now() + timedelta(seconds=30)
        
        duration = self.downloader._get_duration()
        
        # Should return a formatted duration string
        assert isinstance(duration, str)
        assert "0:00:30" in duration or "0:00:29" in duration or "0:00:31" in duration