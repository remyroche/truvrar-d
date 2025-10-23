"""
Configuration settings for the Global Truffle Habitat Atlas (GTHA)
"""
import os
from pathlib import Path
from typing import Dict, List, Optional

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
LOGS_DIR = PROJECT_ROOT / "logs"

# Create directories if they don't exist
for dir_path in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, OUTPUTS_DIR, LOGS_DIR]:
    dir_path.mkdir(exist_ok=True)

# API endpoints and configurations
API_CONFIG = {
    "gbif": {
        "base_url": "https://api.gbif.org/v1",
        "timeout": 30,
        "max_retries": 3
    },
    "inaturalist": {
        "base_url": "https://api.inaturalist.org/v1",
        "timeout": 30,
        "max_retries": 3
    },
    "soilgrids": {
        "base_url": "https://rest.soilgrids.org",
        "timeout": 30,
        "max_retries": 3
    },
    "worldclim": {
        "base_url": "https://biogeo.ucdavis.edu/data/worldclim/v2.1/base",
        "timeout": 60,
        "max_retries": 3
    }
}

# Truffle species of interest
TRUFFLE_SPECIES = [
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
    "Tuber canaliculatum", # Pecan truffle
]

# Environmental variables to extract
CLIMATE_VARIABLES = [
    "bio1",   # Annual Mean Temperature
    "bio2",   # Mean Diurnal Range
    "bio3",   # Isothermality
    "bio4",   # Temperature Seasonality
    "bio5",   # Max Temperature of Warmest Month
    "bio6",   # Min Temperature of Coldest Month
    "bio7",   # Temperature Annual Range
    "bio8",   # Mean Temperature of Wettest Quarter
    "bio9",   # Mean Temperature of Driest Quarter
    "bio10",  # Mean Temperature of Warmest Quarter
    "bio11",  # Mean Temperature of Coldest Quarter
    "bio12",  # Annual Precipitation
    "bio13",  # Precipitation of Wettest Month
    "bio14",  # Precipitation of Driest Month
    "bio15",  # Precipitation Seasonality
    "bio16",  # Precipitation of Wettest Quarter
    "bio17",  # Precipitation of Driest Quarter
    "bio18",  # Precipitation of Warmest Quarter
    "bio19",  # Precipitation of Coldest Quarter
]

SOIL_VARIABLES = [
    "phh2o",      # pH in H2O
    "cac03",      # Calcium carbonate
    "soc",        # Soil organic carbon
    "nitrogen",   # Total nitrogen
    "phosporus",  # Phosphorus
    "sand",       # Sand content
    "silt",       # Silt content
    "clay",       # Clay content
    "bdod",       # Bulk density
    "cec",        # Cation exchange capacity
    "cfvo",       # Coarse fragments
    "clay",       # Clay content
]

# Database configuration
DATABASE_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": os.getenv("DB_PORT", "5432"),
    "database": os.getenv("DB_NAME", "truffle_habitat"),
    "username": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", "password")
}

# Model configuration
MODEL_CONFIG = {
    "test_size": 0.2,
    "random_state": 42,
    "n_estimators": 100,
    "max_depth": 10,
    "min_samples_split": 5,
    "min_samples_leaf": 2
}

# Export formats
EXPORT_FORMATS = ["csv", "geojson", "shp", "parquet"]

# Logging configuration
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        },
    },
    "handlers": {
        "default": {
            "level": "INFO",
            "formatter": "standard",
            "class": "logging.StreamHandler",
        },
        "file": {
            "level": "DEBUG",
            "formatter": "standard",
            "class": "logging.FileHandler",
            "filename": LOGS_DIR / "gtha.log",
            "mode": "a",
        },
    },
    "loggers": {
        "": {
            "handlers": ["default", "file"],
            "level": "DEBUG",
            "propagate": False
        }
    }
}