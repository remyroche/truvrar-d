#!/usr/bin/env python3
"""
Comprehensive example workflow for microbiome-environmental atlas.

This script demonstrates the complete workflow for collecting, processing,
and analyzing microbiome data integrated with environmental layers.
"""

import logging
import pandas as pd
from pathlib import Path
import json
from typing import Dict, List, Any

# Import our custom modules
from src.data_collectors.ebi_metagenomics_collector import EBIMetagenomicsCollector
from src.data_collectors.soilgrids_collector import SoilGridsCollector
from src.data_collectors.worldclim_collector import WorldClimCollector
from src.data_collectors.glim_collector import GLiMCollector
from src.data_processing.abundance_processor import AbundanceProcessor
from src.data_processing.data_merger import DataMerger

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main workflow for microbiome-environmental atlas."""
    logger.info("Starting microbiome-environmental atlas workflow")
    
    # Configuration
    config = {
        'data_dir': Path('data'),
        'output_dir': Path('output'),
        'search_terms': ['Tuber', 'truffle', 'mycorrhiza'],
        'max_studies': 100,
        'include_abundance': True
    }
    
    # Create directories
    config['data_dir'].mkdir(exist_ok=True)
    config['output_dir'].mkdir(exist_ok=True)
    
    # Step 1: Collect microbiome data from EBI Metagenomics
    logger.info("Step 1: Collecting microbiome data from EBI Metagenomics")
    microbiome_data = collect_microbiome_data(config)
    
    if microbiome_data.empty:
        logger.error("No microbiome data collected. Exiting.")
        return
    
    # Step 2: Collect environmental data
    logger.info("Step 2: Collecting environmental data")
    environmental_data = collect_environmental_data(config, microbiome_data)
    
    # Step 3: Process abundance tables
    logger.info("Step 3: Processing abundance tables")
    abundance_data = process_abundance_data(config, microbiome_data)
    
    # Step 4: Merge all data
    logger.info("Step 4: Merging microbiome and environmental data")
    merged_data = merge_all_data(microbiome_data, environmental_data, abundance_data)
    
    # Step 5: Analyze and visualize results
    logger.info("Step 5: Analyzing results")
    analyze_results(merged_data, config)
    
    logger.info("Workflow completed successfully!")


def collect_microbiome_data(config: Dict[str, Any]) -> pd.DataFrame:
    """Collect microbiome data from EBI Metagenomics API."""
    logger.info("Collecting microbiome data from EBI Metagenomics")
    
    # Initialize collector
    collector = EBIMetagenomicsCollector(config, config['data_dir'])
    
    all_microbiome_data = []
    
    # Collect data for each search term
    for search_term in config['search_terms']:
        logger.info(f"Searching for: {search_term}")
        
        try:
            data = collector.collect(
                search_term=search_term,
                limit=config['max_studies'],
                include_samples=True,
                include_abundance=config['include_abundance']
            )
            
            if not data.empty:
                all_microbiome_data.append(data)
                logger.info(f"Collected {len(data)} records for '{search_term}'")
            else:
                logger.warning(f"No data found for '{search_term}'")
                
        except Exception as e:
            logger.error(f"Error collecting data for '{search_term}': {e}")
            continue
    
    if not all_microbiome_data:
        logger.warning("No microbiome data collected")
        return pd.DataFrame()
    
    # Combine all data
    combined_data = pd.concat(all_microbiome_data, ignore_index=True)
    logger.info(f"Total microbiome records collected: {len(combined_data)}")
    
    return combined_data


def collect_environmental_data(config: Dict[str, Any], microbiome_data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Collect environmental data (soil, climate, geology)."""
    logger.info("Collecting environmental data")
    
    environmental_data = {}
    
    # Get coordinates from microbiome data
    coords_data = microbiome_data[
        microbiome_data['latitude'].notna() & 
        microbiome_data['longitude'].notna()
    ]
    
    if coords_data.empty:
        logger.warning("No coordinates available for environmental data collection")
        return environmental_data
    
    coordinates = list(zip(coords_data['latitude'], coords_data['longitude']))
    logger.info(f"Collecting environmental data for {len(coordinates)} locations")
    
    # Collect soil data
    try:
        logger.info("Collecting soil data from SoilGrids")
        soil_collector = SoilGridsCollector(config, config['data_dir'])
        soil_data = soil_collector.collect(coordinates)
        environmental_data['soil'] = soil_data
        logger.info(f"Collected soil data: {len(soil_data)} records")
    except Exception as e:
        logger.error(f"Error collecting soil data: {e}")
        environmental_data['soil'] = pd.DataFrame()
    
    # Collect climate data
    try:
        logger.info("Collecting climate data from WorldClim")
        climate_collector = WorldClimCollector(config, config['data_dir'])
        climate_data = climate_collector.collect(coordinates)
        environmental_data['climate'] = climate_data
        logger.info(f"Collected climate data: {len(climate_data)} records")
    except Exception as e:
        logger.error(f"Error collecting climate data: {e}")
        environmental_data['climate'] = pd.DataFrame()
    
    # Collect geological data
    try:
        logger.info("Collecting geological data from GLiM")
        glim_collector = GLiMCollector(config, config['data_dir'])
        glim_data = glim_collector.collect(coordinates)
        environmental_data['geology'] = glim_data
        logger.info(f"Collected geological data: {len(glim_data)} records")
    except Exception as e:
        logger.error(f"Error collecting geological data: {e}")
        environmental_data['geology'] = pd.DataFrame()
    
    return environmental_data


def process_abundance_data(config: Dict[str, Any], microbiome_data: pd.DataFrame) -> pd.DataFrame:
    """Process abundance tables from various sources."""
    logger.info("Processing abundance data")
    
    # Initialize processor
    processor = AbundanceProcessor(config['data_dir'])
    
    # Create abundance sources from microbiome data
    abundance_sources = []
    
    for _, sample in microbiome_data.iterrows():
        if pd.notna(sample.get('sample_id')):
            source = {
                'type': 'mgnify',
                'name': f"Sample_{sample['sample_id']}",
                'sample_id': sample['sample_id'],
                'study_id': sample.get('study_id')
            }
            abundance_sources.append(source)
    
    if not abundance_sources:
        logger.warning("No abundance sources found")
        return pd.DataFrame()
    
    # Process abundance data
    try:
        abundance_data = processor.process_abundance_tables(
            abundance_sources=abundance_sources,
            target_taxonomic_levels=['family', 'genus', 'species'],
            min_abundance_threshold=0.001
        )
        
        logger.info(f"Processed abundance data: {len(abundance_data)} records")
        return abundance_data
        
    except Exception as e:
        logger.error(f"Error processing abundance data: {e}")
        return pd.DataFrame()


def merge_all_data(microbiome_data: pd.DataFrame, 
                  environmental_data: Dict[str, pd.DataFrame],
                  abundance_data: pd.DataFrame) -> pd.DataFrame:
    """Merge all data sources."""
    logger.info("Merging all data sources")
    
    # Initialize merger
    merger = DataMerger(config, config['data_dir'])
    
    # Merge microbiome and environmental data
    merged_data = merger.merge_microbiome_environmental_data(
        microbiome_data=microbiome_data,
        soil_data=environmental_data.get('soil', pd.DataFrame()),
        climate_data=environmental_data.get('climate', pd.DataFrame()),
        glim_data=environmental_data.get('geology', pd.DataFrame()),
        abundance_data=abundance_data
    )
    
    logger.info(f"Merged data: {len(merged_data)} records")
    
    # Save merged data
    output_file = config['output_dir'] / 'merged_microbiome_environmental_data.csv'
    merged_data.to_csv(output_file, index=False)
    logger.info(f"Saved merged data to {output_file}")
    
    return merged_data


def analyze_results(merged_data: pd.DataFrame, config: Dict[str, Any]):
    """Analyze and visualize results."""
    logger.info("Analyzing results")
    
    # Basic statistics
    logger.info(f"Total samples: {len(merged_data)}")
    logger.info(f"Samples with coordinates: {merged_data['has_coordinates'].sum()}")
    logger.info(f"Samples with fruiting evidence: {merged_data['fruiting_evidence'].sum()}")
    
    # Environmental analysis
    if 'ph_combined' in merged_data.columns:
        ph_stats = merged_data['ph_combined'].describe()
        logger.info(f"pH statistics: {ph_stats}")
    
    if 'temp_combined' in merged_data.columns:
        temp_stats = merged_data['temp_combined'].describe()
        logger.info(f"Temperature statistics: {temp_stats}")
    
    # Microbiome analysis
    if 'microbiome_diversity' in merged_data.columns:
        diversity_counts = merged_data['microbiome_diversity'].value_counts()
        logger.info(f"Microbiome diversity distribution: {diversity_counts}")
    
    if 'env_microbiome_compatibility' in merged_data.columns:
        compatibility_stats = merged_data['env_microbiome_compatibility'].describe()
        logger.info(f"Environmental-microbiome compatibility: {compatibility_stats}")
    
    # Create summary report
    create_summary_report(merged_data, config)


def create_summary_report(merged_data: pd.DataFrame, config: Dict[str, Any]):
    """Create a summary report of the analysis."""
    logger.info("Creating summary report")
    
    report = {
        'analysis_summary': {
            'total_samples': len(merged_data),
            'samples_with_coordinates': int(merged_data['has_coordinates'].sum()),
            'samples_with_fruiting_evidence': int(merged_data['fruiting_evidence'].sum()),
            'unique_studies': merged_data['study_id'].nunique() if 'study_id' in merged_data.columns else 0,
            'unique_species': merged_data['host'].nunique() if 'host' in merged_data.columns else 0
        },
        'environmental_conditions': {},
        'microbiome_characteristics': {},
        'key_findings': []
    }
    
    # Environmental conditions
    if 'ph_combined' in merged_data.columns:
        report['environmental_conditions']['ph'] = {
            'mean': float(merged_data['ph_combined'].mean()),
            'std': float(merged_data['ph_combined'].std()),
            'min': float(merged_data['ph_combined'].min()),
            'max': float(merged_data['ph_combined'].max())
        }
    
    if 'temp_combined' in merged_data.columns:
        report['environmental_conditions']['temperature'] = {
            'mean': float(merged_data['temp_combined'].mean()),
            'std': float(merged_data['temp_combined'].std()),
            'min': float(merged_data['temp_combined'].min()),
            'max': float(merged_data['temp_combined'].max())
        }
    
    # Microbiome characteristics
    if 'microbiome_diversity' in merged_data.columns:
        diversity_dist = merged_data['microbiome_diversity'].value_counts().to_dict()
        report['microbiome_characteristics']['diversity_distribution'] = diversity_dist
    
    if 'env_microbiome_compatibility' in merged_data.columns:
        compatibility_stats = merged_data['env_microbiome_compatibility'].describe()
        report['microbiome_characteristics']['compatibility'] = {
            'mean': float(compatibility_stats['mean']),
            'std': float(compatibility_stats['std']),
            'min': float(compatibility_stats['min']),
            'max': float(compatibility_stats['max'])
        }
    
    # Key findings
    if 'fruiting_evidence' in merged_data.columns:
        fruiting_rate = merged_data['fruiting_evidence'].mean()
        report['key_findings'].append(f"Fruiting evidence rate: {fruiting_rate:.2%}")
    
    if 'env_microbiome_compatibility' in merged_data.columns:
        high_compatibility = (merged_data['env_microbiome_compatibility'] >= 6).sum()
        report['key_findings'].append(f"Samples with high environmental-microbiome compatibility: {high_compatibility}")
    
    # Save report
    report_file = config['output_dir'] / 'analysis_report.json'
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"Summary report saved to {report_file}")
    
    # Print key findings
    logger.info("Key findings:")
    for finding in report['key_findings']:
        logger.info(f"  - {finding}")


if __name__ == "__main__":
    main()
