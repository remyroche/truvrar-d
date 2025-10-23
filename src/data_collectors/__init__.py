"""
Data collection modules for various environmental and biodiversity data sources.

This module provides both individual collectors and a unified collector that can
handle multiple data sources through a single interface.
"""

from .base_collector import BaseCollector
from .unified_collector import UnifiedDataCollector
from .config import DataCollectorConfig, load_collector_config

# Individual collectors (for backward compatibility)
from .gbif_collector import GBIFCollector
from .inaturalist_collector import iNaturalistCollector
from .soilgrids_collector import SoilGridsCollector
from .worldclim_collector import WorldClimCollector
from .glim_collector import GLiMCollector
from .ebi_metagenomics_collector import EBIMetagenomicsCollector

__all__ = [
    # Core classes
    "BaseCollector",
    "UnifiedDataCollector",
    "DataCollectorConfig",
    "load_collector_config",
    
    # Individual collectors (for backward compatibility)
    "GBIFCollector", 
    "iNaturalistCollector",
    "SoilGridsCollector",
    "WorldClimCollector",
    "GLiMCollector",
    "EBIMetagenomicsCollector"
]