#!/usr/bin/env python3
"""
Enhanced example usage of the unified data collector with harmonization and caching.

This script demonstrates the new features including:
- Data harmonization and quality scoring
- Intelligent caching
- Licensing and attribution tracking
- Comprehensive metadata and statistics
"""

import logging
from pathlib import Path
from src.data_collectors import UnifiedDataCollector, load_collector_config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Demonstrate enhanced unified data collector usage."""
    
    # Set up paths
    data_dir = Path("enhanced_data")
    data_dir.mkdir(exist_ok=True)
    
    # Load configuration
    config_manager = load_collector_config()
    
    # Create enhanced unified collector with harmonization and caching
    collector = UnifiedDataCollector(
        config=config_manager.get_all_configs(),
        data_dir=data_dir,
        enable_caching=True,      # Enable intelligent caching
        enable_harmonization=True # Enable data harmonization
    )
    
    logger.info("=== Enhanced Data Collection with Harmonization ===")
    
    # Example 1: Collect and harmonize data from multiple sources
    logger.info("Collecting data from multiple sources...")
    
    try:
        # Collect from multiple sources
        harmonized_results = collector.collect_multiple(
            sources=['gbif', 'inaturalist'],
            species=['Tuber melanosporum'],
            limit=100
        )
        
        # Display harmonized results
        if 'records_df' in harmonized_results:
            records_df = harmonized_results['records_df']
            metadata_df = harmonized_results['metadata_df']
            summary_stats = harmonized_results['summary_stats']
            quality_scores = harmonized_results['quality_scores']
            licensing_info = harmonized_results['licensing_info']
            
            logger.info(f"=== Harmonized Data Summary ===")
            logger.info(f"Total records: {summary_stats.get('total_records', 0)}")
            logger.info(f"Unique sources: {summary_stats.get('unique_sources', 0)}")
            logger.info(f"Coordinate coverage: {summary_stats.get('coordinate_coverage', 0):.1f}%")
            logger.info(f"Temporal coverage: {summary_stats.get('temporal_coverage', 0):.1f}%")
            logger.info(f"Missing data: {summary_stats.get('missing_data_percentage', 0):.1f}%")
            
            # Display quality scores
            logger.info(f"\n=== Quality Scores ===")
            for source, scores in quality_scores.items():
                logger.info(f"{source}: {scores.get('overall_score', 0):.3f}")
                logger.info(f"  - Coordinate quality: {scores.get('coordinate_quality', 0):.3f}")
                logger.info(f"  - Temporal quality: {scores.get('temporal_quality', 0):.3f}")
                logger.info(f"  - Completeness: {scores.get('completeness', 0):.3f}")
                logger.info(f"  - Source reliability: {scores.get('source_reliability', 0):.3f}")
            
            # Display licensing information
            logger.info(f"\n=== Licensing Information ===")
            for source, license_info in licensing_info.items():
                logger.info(f"{source}:")
                logger.info(f"  - License: {license_info.get('license', 'Unknown')}")
                logger.info(f"  - Attribution: {license_info.get('attribution', 'Unknown')}")
                logger.info(f"  - Records: {license_info.get('record_count', 0)}")
            
            # Display harmonized data columns
            logger.info(f"\n=== Harmonized Data Columns ===")
            logger.info(f"Columns: {list(records_df.columns)}")
            
            # Show sample of harmonized data
            if not records_df.empty:
                logger.info(f"\n=== Sample Harmonized Data ===")
                sample_cols = ['data_source', 'species_standardized', 'latitude', 'longitude', 
                             'coord_quality', 'has_good_coordinates', 'env_richness_index']
                available_cols = [col for col in sample_cols if col in records_df.columns]
                logger.info(f"\n{records_df[available_cols].head().to_string()}")
        
    except Exception as e:
        logger.error(f"Error in harmonized collection: {e}")
    
    # Example 2: Demonstrate caching
    logger.info("\n=== Caching Demonstration ===")
    
    try:
        # First call - will hit API
        logger.info("First call (will hit API)...")
        start_time = time.time()
        gbif_data1 = collector.collect(
            source='gbif',
            species=['Tuber magnatum'],
            limit=50
        )
        first_call_time = time.time() - start_time
        logger.info(f"First call took {first_call_time:.2f} seconds")
        
        # Second call - should use cache
        logger.info("Second call (should use cache)...")
        start_time = time.time()
        gbif_data2 = collector.collect(
            source='gbif',
            species=['Tuber magnatum'],
            limit=50
        )
        second_call_time = time.time() - start_time
        logger.info(f"Second call took {second_call_time:.2f} seconds")
        
        # Show cache statistics
        cache_stats = collector.get_cache_stats()
        if cache_stats:
            logger.info(f"\n=== Cache Statistics ===")
            logger.info(f"Total entries: {cache_stats['total_entries']}")
            logger.info(f"Total size: {cache_stats['total_size_mb']:.2f} MB")
            logger.info(f"Total records: {cache_stats['total_records']}")
            logger.info(f"Average access count: {cache_stats['avg_access_count']:.1f}")
            
            logger.info(f"\nSource breakdown:")
            for source, info in cache_stats['source_breakdown'].items():
                logger.info(f"  {source}: {info['count']} entries, {info['size_mb']:.2f} MB")
        
    except Exception as e:
        logger.error(f"Error in caching demonstration: {e}")
    
    # Example 3: Environmental data collection with harmonization
    logger.info("\n=== Environmental Data Collection ===")
    
    try:
        # Collect environmental data
        coordinates = [(44.0, 4.0), (45.0, 7.0), (43.0, 2.0)]
        
        # Soil data
        soil_result = collector.collect(
            source='soilgrids',
            coordinates=coordinates,
            variables=['phh2o', 'soc', 'sand', 'silt', 'clay']
        )
        
        if isinstance(soil_result, dict) and 'records_df' in soil_result:
            soil_df = soil_result['records_df']
            logger.info(f"Collected soil data: {len(soil_df)} records")
            logger.info(f"Soil data quality score: {soil_result.get('quality_scores', {}).get('soilgrids', {}).get('overall_score', 0):.3f}")
        
        # Climate data
        climate_result = collector.collect(
            source='worldclim',
            coordinates=coordinates,
            variables=['bio1', 'bio12', 'bio4']
        )
        
        if isinstance(climate_result, dict) and 'records_df' in climate_result:
            climate_df = climate_result['records_df']
            logger.info(f"Collected climate data: {len(climate_df)} records")
            logger.info(f"Climate data quality score: {climate_result.get('quality_scores', {}).get('worldclim', {}).get('overall_score', 0):.3f}")
        
    except Exception as e:
        logger.error(f"Error in environmental data collection: {e}")
    
    # Example 4: Data quality analysis
    logger.info("\n=== Data Quality Analysis ===")
    
    try:
        # Get a sample dataset for quality analysis
        sample_data = collector.collect(
            source='gbif',
            species=['Tuber melanosporum'],
            limit=200
        )
        
        if isinstance(sample_data, dict) and 'records_df' in sample_data:
            df = sample_data['records_df']
            
            # Analyze coordinate quality
            if 'coord_quality' in df.columns:
                coord_quality_counts = df['coord_quality'].value_counts()
                logger.info(f"Coordinate quality distribution:")
                for quality, count in coord_quality_counts.items():
                    logger.info(f"  {quality}: {count} records")
            
            # Analyze data completeness
            if 'has_good_coordinates' in df.columns:
                good_coords = df['has_good_coordinates'].sum()
                logger.info(f"Records with good coordinates: {good_coords}/{len(df)} ({good_coords/len(df)*100:.1f}%)")
            
            # Analyze environmental richness
            if 'env_richness_index' in df.columns:
                richness_stats = df['env_richness_index'].describe()
                logger.info(f"Environmental richness index statistics:")
                logger.info(f"  Mean: {richness_stats['mean']:.3f}")
                logger.info(f"  Std: {richness_stats['std']:.3f}")
                logger.info(f"  Min: {richness_stats['min']:.3f}")
                logger.info(f"  Max: {richness_stats['max']:.3f}")
        
    except Exception as e:
        logger.error(f"Error in data quality analysis: {e}")
    
    logger.info("\n=== Enhanced Data Collection Complete ===")
    
    # Show final cache statistics
    final_cache_stats = collector.get_cache_stats()
    if final_cache_stats:
        logger.info(f"\n=== Final Cache Statistics ===")
        logger.info(f"Total cached entries: {final_cache_stats['total_entries']}")
        logger.info(f"Total cache size: {final_cache_stats['total_size_mb']:.2f} MB")
        logger.info(f"Total cached records: {final_cache_stats['total_records']}")

if __name__ == "__main__":
    import time
    main()