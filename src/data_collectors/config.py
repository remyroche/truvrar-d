"""
Configuration system for unified data collector.

This module provides default configurations for all supported data sources
and utilities for managing collector configurations.
"""

from typing import Dict, Any, List, Optional
from pathlib import Path
import json


class DataCollectorConfig:
    """Configuration manager for data collectors."""
    
    # Default configurations for all data sources
    DEFAULT_CONFIGS = {
        'gbif': {
            'base_url': 'https://api.gbif.org/v1',
            'timeout': 30,
            'max_retries': 3,
            'rate_limit_delay': 0.1
        },
        'inaturalist': {
            'base_url': 'https://api.inaturalist.org/v1',
            'timeout': 30,
            'max_retries': 3,
            'rate_limit_delay': 0.1
        },
        'soilgrids': {
            'base_url': 'https://rest.isric.org/soilgrids/v2.0',
            'timeout': 30,
            'max_retries': 3,
            'rate_limit_delay': 0.1
        },
        'worldclim': {
            'base_url': 'https://biogeo.ucdavis.edu/data/worldclim/v2.1',
            'timeout': 60,
            'max_retries': 3,
            'rate_limit_delay': 0.5
        },
        'glim': {
            'base_url': 'https://www.geo.uni-hamburg.de/en/geologie/forschung/geodynamik/glim.html',
            'timeout': 30,
            'max_retries': 3,
            'rate_limit_delay': 0.1
        },
        'ebi_metagenomics': {
            'base_url': 'https://www.ebi.ac.uk/metagenomics/api/latest',
            'timeout': 30,
            'max_retries': 3,
            'rate_limit_delay': 0.5
        }
    }
    
    # Default parameters for each data source
    DEFAULT_PARAMS = {
        'gbif': {
            'limit': 10000,
            'has_coordinate': True,
            'has_geospatial_issue': False
        },
        'inaturalist': {
            'limit': 10000,
            'has_geo': True,
            'quality_grade': 'research,needs_id'
        },
        'soilgrids': {
            'variables': [
                'phh2o', 'cac03', 'soc', 'nitrogen', 'phosporus',
                'sand', 'silt', 'clay', 'bdod', 'cec', 'cfvo'
            ]
        },
        'worldclim': {
            'variables': [
                'bio1', 'bio2', 'bio3', 'bio4', 'bio5', 'bio6', 'bio7',
                'bio8', 'bio9', 'bio10', 'bio11', 'bio12', 'bio13', 'bio14',
                'bio15', 'bio16', 'bio17', 'bio18', 'bio19'
            ],
            'resolution': '30s'
        },
        'glim': {
            'buffer_distance': 0.01
        },
        'ebi_metagenomics': {
            'search_term': 'Tuber',
            'limit': 1000,
            'include_samples': True,
            'include_abundance': False
        }
    }
    
    def __init__(self, config_file: Optional[Path] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_file: Optional path to custom configuration file
        """
        self.config = self.DEFAULT_CONFIGS.copy()
        self.params = self.DEFAULT_PARAMS.copy()
        
        if config_file and config_file.exists():
            self.load_config(config_file)
    
    def load_config(self, config_file: Path) -> None:
        """Load configuration from file."""
        try:
            with open(config_file, 'r') as f:
                custom_config = json.load(f)
            
            # Update configurations
            if 'sources' in custom_config:
                for source, config in custom_config['sources'].items():
                    if source in self.config:
                        self.config[source].update(config)
            
            # Update parameters
            if 'parameters' in custom_config:
                for source, params in custom_config['parameters'].items():
                    if source in self.params:
                        self.params[source].update(params)
                        
        except Exception as e:
            print(f"Warning: Could not load config file {config_file}: {e}")
    
    def get_source_config(self, source: str) -> Dict[str, Any]:
        """Get configuration for a specific data source."""
        return self.config.get(source, {})
    
    def get_source_params(self, source: str) -> Dict[str, Any]:
        """Get default parameters for a specific data source."""
        return self.params.get(source, {})
    
    def get_all_configs(self) -> Dict[str, Dict[str, Any]]:
        """Get all source configurations."""
        return self.config
    
    def get_all_params(self) -> Dict[str, Dict[str, Any]]:
        """Get all default parameters."""
        return self.params
    
    def update_source_config(self, source: str, config: Dict[str, Any]) -> None:
        """Update configuration for a specific source."""
        if source in self.config:
            self.config[source].update(config)
        else:
            self.config[source] = config
    
    def update_source_params(self, source: str, params: Dict[str, Any]) -> None:
        """Update parameters for a specific source."""
        if source in self.params:
            self.params[source].update(params)
        else:
            self.params[source] = params
    
    def save_config(self, config_file: Path) -> None:
        """Save current configuration to file."""
        config_data = {
            'sources': self.config,
            'parameters': self.params
        }
        
        with open(config_file, 'w') as f:
            json.dump(config_data, f, indent=2)
    
    def get_supported_sources(self) -> List[str]:
        """Get list of supported data sources."""
        return list(self.config.keys())
    
    def validate_source(self, source: str) -> bool:
        """Validate if a source is supported."""
        return source in self.config


def create_default_config_file(config_file: Path) -> None:
    """Create a default configuration file."""
    config_manager = DataCollectorConfig()
    config_manager.save_config(config_file)


def load_collector_config(config_file: Optional[Path] = None) -> DataCollectorConfig:
    """Load collector configuration from file or create default."""
    if config_file is None:
        config_file = Path(__file__).parent / "default_config.json"
    
    if not config_file.exists():
        create_default_config_file(config_file)
    
    return DataCollectorConfig(config_file)