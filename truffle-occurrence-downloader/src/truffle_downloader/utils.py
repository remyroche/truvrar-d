"""
Utility functions for Truffle Occurrence Data Downloader
"""

import logging
import pandas as pd
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import json
import yaml
from datetime import datetime

logger = logging.getLogger(__name__)


def load_species_list(file_path: Union[str, Path]) -> List[str]:
    """
    Load species list from a text file.
    
    Args:
        file_path: Path to the species list file
        
    Returns:
        List of species names
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Species list file not found: {file_path}")
    
    species_list = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):  # Skip empty lines and comments
                species_list.append(line)
    
    logger.info(f"Loaded {len(species_list)} species from {file_path}")
    return species_list


def save_species_list(species_list: List[str], file_path: Union[str, Path]) -> None:
    """
    Save species list to a text file.
    
    Args:
        species_list: List of species names
        file_path: Path to save the species list
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        for species in species_list:
            f.write(f"{species}\n")
    
    logger.info(f"Saved {len(species_list)} species to {file_path}")


def load_config(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load configuration from YAML or JSON file.
    
    Args:
        file_path: Path to the configuration file
        
    Returns:
        Configuration dictionary
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        if file_path.suffix.lower() in ['.yaml', '.yml']:
            config = yaml.safe_load(f)
        else:
            config = json.load(f)
    
    logger.info(f"Loaded configuration from {file_path}")
    return config


def save_config(config: Dict[str, Any], file_path: Union[str, Path]) -> None:
    """
    Save configuration to YAML or JSON file.
    
    Args:
        config: Configuration dictionary
        file_path: Path to save the configuration
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        if file_path.suffix.lower() in ['.yaml', '.yml']:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        else:
            json.dump(config, f, indent=2)
    
    logger.info(f"Saved configuration to {file_path}")


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted size string
    """
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.1f} {size_names[i]}"


def format_duration(seconds: float) -> str:
    """
    Format duration in human-readable format.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted duration string
    """
    if seconds < 60:
        return f"{seconds:.1f} seconds"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f} minutes"
    else:
        hours = seconds / 3600
        return f"{hours:.1f} hours"


def get_data_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Get summary statistics for a DataFrame.
    
    Args:
        df: DataFrame to summarize
        
    Returns:
        Summary dictionary
    """
    if df.empty:
        return {"total_records": 0}
    
    summary = {
        "total_records": len(df),
        "columns": list(df.columns),
        "memory_usage": df.memory_usage(deep=True).sum(),
        "dtypes": df.dtypes.to_dict()
    }
    
    # Add species information
    if 'species' in df.columns:
        summary["species_count"] = df['species'].nunique()
        summary["species_list"] = df['species'].unique().tolist()
    
    # Add geographic information
    if 'latitude' in df.columns and 'longitude' in df.columns:
        summary["geographic_bounds"] = {
            "min_latitude": float(df['latitude'].min()),
            "max_latitude": float(df['latitude'].max()),
            "min_longitude": float(df['longitude'].min()),
            "max_longitude": float(df['longitude'].max())
        }
    
    # Add temporal information
    if 'event_date' in df.columns:
        date_col = pd.to_datetime(df['event_date'], errors='coerce')
        summary["temporal_range"] = {
            "earliest": date_col.min().isoformat() if not date_col.isna().all() else None,
            "latest": date_col.max().isoformat() if not date_col.isna().all() else None
        }
    
    # Add country information
    if 'country' in df.columns:
        summary["country_count"] = df['country'].nunique()
        summary["country_list"] = df['country'].unique().tolist()
    
    return summary


def validate_species_name(species_name: str) -> bool:
    """
    Validate species name format.
    
    Args:
        species_name: Species name to validate
        
    Returns:
        True if valid, False otherwise
    """
    import re
    
    # Basic validation: Genus species format
    pattern = r'^[A-Z][a-z]+\s+[a-z]+'
    return bool(re.match(pattern, species_name))


def clean_species_name(species_name: str) -> str:
    """
    Clean and standardize species name.
    
    Args:
        species_name: Species name to clean
        
    Returns:
        Cleaned species name
    """
    # Remove extra whitespace
    cleaned = species_name.strip()
    
    # Ensure proper capitalization
    parts = cleaned.split()
    if len(parts) >= 2:
        parts[0] = parts[0].capitalize()  # Genus
        parts[1] = parts[1].lower()       # species
        cleaned = ' '.join(parts)
    
    return cleaned


def create_output_directory(base_dir: Union[str, Path], 
                          subdir: Optional[str] = None) -> Path:
    """
    Create output directory with timestamp.
    
    Args:
        base_dir: Base directory path
        subdir: Optional subdirectory name
        
    Returns:
        Path to created directory
    """
    base_dir = Path(base_dir)
    
    if subdir:
        output_dir = base_dir / subdir
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = base_dir / f"truffle_data_{timestamp}"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Created output directory: {output_dir}")
    
    return output_dir


def setup_logging(log_level: str = "INFO", 
                 log_file: Optional[Union[str, Path]] = None) -> None:
    """
    Set up logging configuration.
    
    Args:
        log_level: Logging level
        log_file: Optional log file path
    """
    level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Set up root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Add console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # Add file handler if specified
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    logger.info(f"Logging configured: level={log_level}, file={log_file}")


def check_dependencies() -> Dict[str, bool]:
    """
    Check if required dependencies are available.
    
    Returns:
        Dictionary of dependency availability
    """
    dependencies = {
        'pandas': True,
        'numpy': True,
        'requests': True,
        'yaml': True,
        'click': True,
        'tqdm': True
    }
    
    # Check optional dependencies
    optional_deps = {
        'geopandas': False,
        'shapely': False,
        'fiona': False,
        'openpyxl': False,
        'xlsxwriter': False
    }
    
    for dep in optional_deps:
        try:
            __import__(dep)
            optional_deps[dep] = True
        except ImportError:
            pass
    
    dependencies.update(optional_deps)
    
    return dependencies


def print_dependency_status() -> None:
    """Print dependency status information."""
    deps = check_dependencies()
    
    print("Dependency Status:")
    print("=" * 50)
    
    for dep, available in deps.items():
        status = "✓" if available else "✗"
        print(f"{status} {dep}")
    
    print("\nNote: Optional dependencies (✗) are not required for basic functionality")
    print("Install with: pip install -e .[geospatial] for geospatial features")


def create_sample_config(output_path: Union[str, Path]) -> None:
    """
    Create a sample configuration file.
    
    Args:
        output_path: Path to save the sample configuration
    """
    sample_config = {
        'gbif': {
            'base_url': 'https://api.gbif.org/v1',
            'timeout': 30,
            'max_retries': 3,
            'rate_limit': 1000
        },
        'download': {
            'batch_size': 300,
            'max_records': 10000,
            'coordinate_uncertainty_max': 10000,
            'include_duplicates': False,
            'has_coordinate': True,
            'has_geospatial_issue': False
        },
        'output': {
            'default_format': 'csv',
            'include_metadata': True,
            'compress_output': False
        },
        'logging': {
            'level': 'INFO',
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        }
    }
    
    output_path = Path(output_path)
    save_config(sample_config, output_path)
    logger.info(f"Sample configuration created: {output_path}")


def create_sample_species_list(output_path: Union[str, Path]) -> None:
    """
    Create a sample species list file.
    
    Args:
        output_path: Path to save the sample species list
    """
    sample_species = [
        "# Common truffle species",
        "Tuber melanosporum",  # Black truffle
        "Tuber magnatum",      # White truffle
        "Tuber aestivum",      # Summer truffle
        "Tuber borchii",       # Bianchetto truffle
        "Tuber brumale",       # Winter truffle
        "Tuber mesentericum",  # Bagnoli truffle
        "Tuber macrosporum",   # Smooth black truffle
        "Tuber indicum",       # Chinese truffle
        "Tuber himalayense",   # Himalayan truffle
        "Tuber oregonense",    # Oregon white truffle
        "Tuber gibbosum",      # Oregon black truffle
        "Tuber canaliculatum"  # Pecan truffle
    ]
    
    output_path = Path(output_path)
    with open(output_path, 'w', encoding='utf-8') as f:
        for species in sample_species:
            f.write(f"{species}\n")
    
    logger.info(f"Sample species list created: {output_path}")