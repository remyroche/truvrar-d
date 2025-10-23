#!/usr/bin/env python3
"""
Global Truffle Habitat Atlas (GTHA) - Main Application

This script provides the main interface for collecting, processing, and analyzing
truffle habitat data from various online sources.
"""
import logging
import argparse
from pathlib import Path
from typing import List, Optional, Dict, Any
import json
import pandas as pd

from config import *
from src.data_processing.habitat_processor import HabitatProcessor
from src.models.habitat_model import HabitatModel
from src.visualization.mapping_tools import MappingTools
from src.data_collectors.academic_collector import AcademicDataCollector
from src.data_collectors.biodiversity_collector import BiodiversityDataCollector

# Set up logging
logging.basicConfig(**LOGGING_CONFIG)
logger = logging.getLogger(__name__)


def main():
    """Main application entry point."""
    parser = argparse.ArgumentParser(description='Global Truffle Habitat Atlas')
    parser.add_argument('--action', choices=['collect', 'analyze', 'visualize', 'export'], 
                       required=True, help='Action to perform')
    parser.add_argument('--species', nargs='+', default=TRUFFLE_SPECIES,
                       help='Truffle species to analyze')
    parser.add_argument('--countries', nargs='+', 
                       help='Country codes to filter by (e.g., FR IT ES)')
    parser.add_argument('--year-from', type=int, 
                       help='Start year for data collection')
    parser.add_argument('--year-to', type=int, 
                       help='End year for data collection')
    parser.add_argument('--output-dir', type=Path, default=OUTPUTS_DIR,
                       help='Output directory for results')
    parser.add_argument('--config-file', type=Path,
                       help='Custom configuration file')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config_file) if args.config_file else {
        'gbif': API_CONFIG['gbif'],
        'inaturalist': API_CONFIG['inaturalist'],
        'soilgrids': API_CONFIG['soilgrids'],
        'worldclim': API_CONFIG['worldclim'],
        **MODEL_CONFIG
    }
    
    # Initialize processors
    processor = HabitatProcessor(config, DATA_DIR)
    model = HabitatModel(config, MODELS_DIR)
    mapper = MappingTools()
    
    # Initialize specialized collectors
    academic_collector = AcademicDataCollector(config, DATA_DIR)
    biodiversity_collector = BiodiversityDataCollector(config, DATA_DIR)
    
    try:
        if args.action == 'collect':
            collect_data(processor, academic_collector, biodiversity_collector, args)
        elif args.action == 'analyze':
            analyze_data(processor, model, args)
        elif args.action == 'visualize':
            visualize_data(mapper, args)
        elif args.action == 'export':
            export_data(processor, args)
            
    except Exception as e:
        logger.error(f"Error during {args.action}: {e}")
        raise


def collect_data(processor: HabitatProcessor, academic_collector: AcademicDataCollector, 
                biodiversity_collector: BiodiversityDataCollector, args):
    """Collect truffle habitat data from all sources."""
    logger.info("Starting data collection")
    
    # Collect biodiversity data
    logger.info("Collecting biodiversity data...")
    biodiversity_data = biodiversity_collector.collect_biodiversity_data(
        source='gbif',
        species=args.species,
        limit=10000,
        countries=args.countries,
        year_from=args.year_from,
        year_to=args.year_to
    )
    
    # Collect academic data
    logger.info("Collecting academic data...")
    academic_data = academic_collector.collect_academic_data(
        source='pubmed',
        search_terms=args.species,
        limit=1000
    )
    
    # Process and merge data
    data = processor.collect_all_data(
        species=args.species,
        countries=args.countries,
        year_from=args.year_from,
        year_to=args.year_to
    )
    
    if data.empty:
        logger.warning("No data collected")
        return
        
    # Save raw data
    raw_data_path = args.output_dir / "raw_habitat_data.csv"
    data.to_csv(raw_data_path, index=False)
    logger.info(f"Raw data saved to {raw_data_path}")
    
    # Generate collection report
    report = processor.analyze_habitat_characteristics(data)
    report_path = args.output_dir / "collection_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    logger.info(f"Collection report saved to {report_path}")


def analyze_data(processor: HabitatProcessor, model: HabitatModel, args):
    """Analyze collected truffle habitat data."""
    logger.info("Starting data analysis")
    
    # Load data
    data_path = args.output_dir / "raw_habitat_data.csv"
    if not data_path.exists():
        logger.error(f"Data file not found: {data_path}")
        return
        
    data = pd.read_csv(data_path)
    
    if data.empty:
        logger.warning("No data to analyze")
        return
        
    # Train models
    training_results = model.train_models(data)
    
    # Save training results
    results_path = args.output_dir / "training_results.json"
    with open(results_path, 'w') as f:
        json.dump(training_results, f, indent=2)
    logger.info(f"Training results saved to {results_path}")
    
    # Generate habitat report
    habitat_report = model.generate_habitat_report(data)
    report_path = args.output_dir / "habitat_analysis_report.json"
    with open(report_path, 'w') as f:
        json.dump(habitat_report, f, indent=2)
    logger.info(f"Habitat analysis report saved to {report_path}")
    
    # Plot feature importance
    importance_path = args.output_dir / "feature_importance.png"
    model.plot_feature_importance(save_path=importance_path)
    
    # Export habitat parameters
    exported_files = processor.export_habitat_parameters(data, args.output_dir)
    logger.info(f"Exported files: {list(exported_files.keys())}")


def visualize_data(mapper: MappingTools, args):
    """Create visualizations of truffle habitat data."""
    logger.info("Starting data visualization")
    
    # Load data
    data_path = args.output_dir / "raw_habitat_data.csv"
    if not data_path.exists():
        logger.error(f"Data file not found: {data_path}")
        return
        
    data = pd.read_csv(data_path)
    
    if data.empty:
        logger.warning("No data to visualize")
        return
        
    # Create species distribution map
    species_map_path = args.output_dir / "species_distribution_map.html"
    mapper.create_species_distribution_map(data, save_path=species_map_path)
    
    # Create environmental variable maps
    env_vars = ['soil_pH', 'soil_CaCO3_pct', 'mean_annual_temp_C', 'annual_precip_mm']
    for var in env_vars:
        if var in data.columns:
            var_map_path = args.output_dir / f"{var}_map.html"
            mapper.create_environmental_map(data, var, save_path=var_map_path)
    
    # Create correlation heatmap
    heatmap_path = args.output_dir / "correlation_heatmap.png"
    mapper.create_correlation_heatmap(data, save_path=heatmap_path)
    
    logger.info("Visualization complete")


def export_data(processor: HabitatProcessor, args):
    """Export processed data in various formats."""
    logger.info("Starting data export")
    
    # Load data
    data_path = args.output_dir / "raw_habitat_data.csv"
    if not data_path.exists():
        logger.error(f"Data file not found: {data_path}")
        return
        
    data = pd.read_csv(data_path)
    
    if data.empty:
        logger.warning("No data to export")
        return
        
    # Export in different formats
    export_formats = ['csv', 'geojson', 'parquet']
    exported_files = {}
    
    for fmt in export_formats:
        if fmt == 'csv':
            file_path = args.output_dir / f"truffle_habitat_data.{fmt}"
            data.to_csv(file_path, index=False)
        elif fmt == 'geojson':
            file_path = args.output_dir / f"truffle_habitat_data.{fmt}"
            # Convert to GeoJSON
            import geopandas as gpd
            from shapely.geometry import Point
            gdf = gpd.GeoDataFrame(
                data,
                geometry=[Point(xy) for xy in zip(data['longitude'], data['latitude'])],
                crs='EPSG:4326'
            )
            gdf.to_file(file_path, driver='GeoJSON')
        elif fmt == 'parquet':
            file_path = args.output_dir / f"truffle_habitat_data.{fmt}"
            data.to_parquet(file_path, index=False)
            
        exported_files[fmt] = str(file_path)
        
    # Export hydroponic parameters
    hydroponic_params = generate_hydroponic_parameters(data)
    params_path = args.output_dir / "hydroponic_parameters.json"
    with open(params_path, 'w') as f:
        json.dump(hydroponic_params, f, indent=2)
    exported_files['hydroponic_parameters'] = str(params_path)
    
    logger.info(f"Export complete. Files: {list(exported_files.keys())}")


def generate_hydroponic_parameters(data: pd.DataFrame) -> Dict[str, Any]:
    """Generate hydroponic parameters from natural habitat data."""
    params = {}
    
    # Calculate parameter ranges by species
    for species in data['species'].unique():
        species_data = data[data['species'] == species]
        
        species_params = {
            'pH_range': {
                'min': float(species_data['soil_pH'].min()) if 'soil_pH' in species_data.columns else None,
                'max': float(species_data['soil_pH'].max()) if 'soil_pH' in species_data.columns else None,
                'recommended': float(species_data['soil_pH'].mean()) if 'soil_pH' in species_data.columns else None
            },
            'temperature_range': {
                'min': float(species_data['mean_annual_temp_C'].min()) if 'mean_annual_temp_C' in species_data.columns else None,
                'max': float(species_data['mean_annual_temp_C'].max()) if 'mean_annual_temp_C' in species_data.columns else None,
                'recommended': float(species_data['mean_annual_temp_C'].mean()) if 'mean_annual_temp_C' in species_data.columns else None
            },
            'precipitation_range': {
                'min': float(species_data['annual_precip_mm'].min()) if 'annual_precip_mm' in species_data.columns else None,
                'max': float(species_data['annual_precip_mm'].max()) if 'annual_precip_mm' in species_data.columns else None,
                'recommended': float(species_data['annual_precip_mm'].mean()) if 'annual_precip_mm' in species_data.columns else None
            },
            'calcium_carbonate_range': {
                'min': float(species_data['soil_CaCO3_pct'].min()) if 'soil_CaCO3_pct' in species_data.columns else None,
                'max': float(species_data['soil_CaCO3_pct'].max()) if 'soil_CaCO3_pct' in species_data.columns else None,
                'recommended': float(species_data['soil_CaCO3_pct'].mean()) if 'soil_CaCO3_pct' in species_data.columns else None
            }
        }
        
        params[species] = species_params
        
    return params


def load_config(config_file: Path) -> Dict[str, Any]:
    """Load configuration from file."""
    with open(config_file, 'r') as f:
        return json.load(f)


if __name__ == "__main__":
    main()