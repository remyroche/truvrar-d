#!/usr/bin/env python3
"""
Example usage of the unified data collector.

This script demonstrates how to use the UnifiedDataCollector to collect data
from multiple sources with a simplified interface.
"""

import logging
from pathlib import Path
from src.data_collectors import UnifiedDataCollector, load_collector_config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Demonstrate unified data collector usage."""
    
    # Set up paths
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # Load configuration
    config_manager = load_collector_config()
    
    # Create unified collector
    collector = UnifiedDataCollector(
        config=config_manager.get_all_configs(),
        data_dir=data_dir
    )
    
    # Example 1: Collect GBIF data
    logger.info("=== Collecting GBIF Data ===")
    try:
        gbif_data = collector.collect(
            source='gbif',
            species=['Tuber melanosporum', 'Tuber magnatum'],
            limit=100
        )
        logger.info(f"Collected {len(gbif_data)} GBIF records")
        if not gbif_data.empty:
            logger.info(f"Columns: {list(gbif_data.columns)}")
    except Exception as e:
        logger.error(f"Error collecting GBIF data: {e}")
    
    # Example 2: Collect iNaturalist data
    logger.info("=== Collecting iNaturalist Data ===")
    try:
        inat_data = collector.collect(
            source='inaturalist',
            species=['Tuber melanosporum'],
            limit=50
        )
        logger.info(f"Collected {len(inat_data)} iNaturalist records")
    except Exception as e:
        logger.error(f"Error collecting iNaturalist data: {e}")
    
    # Example 3: Collect soil data for specific coordinates
    logger.info("=== Collecting Soil Data ===")
    try:
        # Example coordinates (some locations in France and Italy)
        coordinates = [
            (44.0, 4.0),  # France
            (45.0, 7.0),  # Italy
            (43.0, 2.0)   # France
        ]
        
        soil_data = collector.collect(
            source='soilgrids',
            coordinates=coordinates,
            variables=['phh2o', 'soc', 'sand', 'silt', 'clay']
        )
        logger.info(f"Collected soil data for {len(soil_data)} locations")
    except Exception as e:
        logger.error(f"Error collecting soil data: {e}")
    
    # Example 4: Collect climate data
    logger.info("=== Collecting Climate Data ===")
    try:
        climate_data = collector.collect(
            source='worldclim',
            coordinates=coordinates,
            variables=['bio1', 'bio12', 'bio4']  # Temperature, precipitation, seasonality
        )
        logger.info(f"Collected climate data for {len(climate_data)} locations")
    except Exception as e:
        logger.error(f"Error collecting climate data: {e}")
    
    # Example 5: Collect from multiple sources at once
    logger.info("=== Collecting from Multiple Sources ===")
    try:
        multi_data = collector.collect_multiple(
            sources=['gbif', 'inaturalist'],
            species=['Tuber melanosporum'],
            limit=25
        )
        
        for source, data in multi_data.items():
            logger.info(f"{source}: {len(data)} records")
    except Exception as e:
        logger.error(f"Error collecting from multiple sources: {e}")
    
    # Example 6: Collect EBI Metagenomics data
    logger.info("=== Collecting EBI Metagenomics Data ===")
    try:
        ebi_data = collector.collect(
            source='ebi_metagenomics',
            search_term='Tuber',
            limit=50,
            include_samples=True
        )
        logger.info(f"Collected {len(ebi_data)} EBI Metagenomics records")
    except Exception as e:
        logger.error(f"Error collecting EBI Metagenomics data: {e}")
    
    logger.info("=== Data Collection Complete ===")
    
    # Show available data files
    data_files = list(data_dir.glob("*.csv"))
    if data_files:
        logger.info(f"Data files created: {[f.name for f in data_files]}")
    else:
        logger.info("No data files were created (possibly due to API errors)")

if __name__ == "__main__":
    main()