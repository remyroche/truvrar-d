#!/usr/bin/env python3
"""
Complete workflow script for the Global Truffle Habitat Atlas (GTHA)

This script runs the entire GTHA pipeline from data collection to visualization.
"""
import logging
import argparse
from pathlib import Path
import time
import json

from config import *
from src.data_processing.habitat_processor import HabitatProcessor
from src.models.habitat_model import HabitatModel
from src.visualization.mapping_tools import MappingTools
from src.visualization.plotting_tools import PlottingTools

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.FileHandler(LOGS_DIR / 'gtha_workflow.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def run_complete_workflow(species=None, countries=None, year_from=None, year_to=None, output_dir=None):
    """Run the complete GTHA workflow."""
    
    if species is None:
        species = TRUFFLE_SPECIES[:3]  # Use first 3 species for demo
    
    if output_dir is None:
        output_dir = OUTPUTS_DIR
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    logger.info("🍄 Starting Global Truffle Habitat Atlas Workflow")
    logger.info(f"Species: {species}")
    logger.info(f"Countries: {countries}")
    logger.info(f"Years: {year_from}-{year_to}")
    logger.info(f"Output directory: {output_dir}")
    
    # Initialize configuration
    config = {
        'gbif': API_CONFIG['gbif'],
        'inaturalist': API_CONFIG['inaturalist'],
        'soilgrids': API_CONFIG['soilgrids'],
        'worldclim': API_CONFIG['worldclim'],
        **MODEL_CONFIG
    }
    
    # Initialize components
    processor = HabitatProcessor(config, DATA_DIR)
    model = HabitatModel(config, MODELS_DIR)
    mapper = MappingTools()
    plotter = PlottingTools()
    
    workflow_results = {
        'start_time': time.time(),
        'steps_completed': [],
        'errors': [],
        'output_files': {}
    }
    
    try:
        # Step 1: Data Collection
        logger.info("📥 Step 1: Collecting data from all sources...")
        start_time = time.time()
        
        data = processor.collect_all_data(
            species=species,
            countries=countries,
            year_from=year_from,
            year_to=year_to
        )
        
        if data.empty:
            logger.error("No data collected. Exiting workflow.")
            return workflow_results
        
        collection_time = time.time() - start_time
        logger.info(f"✅ Data collection completed in {collection_time:.1f}s: {len(data)} records")
        
        # Save raw data
        raw_data_path = output_dir / "workflow_raw_data.csv"
        data.to_csv(raw_data_path, index=False)
        workflow_results['output_files']['raw_data'] = str(raw_data_path)
        workflow_results['steps_completed'].append('data_collection')
        
        # Step 2: Data Analysis
        logger.info("🔍 Step 2: Analyzing habitat characteristics...")
        start_time = time.time()
        
        analysis = processor.analyze_habitat_characteristics(data)
        analysis_time = time.time() - start_time
        logger.info(f"✅ Habitat analysis completed in {analysis_time:.1f}s")
        
        # Save analysis results
        analysis_path = output_dir / "habitat_analysis.json"
        with open(analysis_path, 'w') as f:
            json.dump(analysis, f, indent=2)
        workflow_results['output_files']['habitat_analysis'] = str(analysis_path)
        workflow_results['steps_completed'].append('habitat_analysis')
        
        # Step 3: Machine Learning
        logger.info("🤖 Step 3: Training machine learning models...")
        start_time = time.time()
        
        training_results = model.train_models(data)
        training_time = time.time() - start_time
        logger.info(f"✅ Model training completed in {training_time:.1f}s")
        
        # Save training results
        training_path = output_dir / "training_results.json"
        with open(training_path, 'w') as f:
            json.dump(training_results, f, indent=2)
        workflow_results['output_files']['training_results'] = str(training_path)
        workflow_results['steps_completed'].append('model_training')
        
        # Step 4: Visualization
        logger.info("📊 Step 4: Creating visualizations...")
        start_time = time.time()
        
        # Species distribution map
        species_map_path = output_dir / "species_distribution_map.html"
        mapper.create_species_distribution_map(data, save_path=species_map_path)
        workflow_results['output_files']['species_map'] = str(species_map_path)
        
        # Environmental correlation heatmap
        heatmap_path = output_dir / "correlation_heatmap.png"
        mapper.create_correlation_heatmap(data, save_path=heatmap_path)
        workflow_results['output_files']['correlation_heatmap'] = str(heatmap_path)
        
        # Feature importance plot
        if model.feature_importance_ is not None:
            importance_path = output_dir / "feature_importance.png"
            model.plot_feature_importance(save_path=importance_path)
            workflow_results['output_files']['feature_importance'] = str(importance_path)
        
        # Additional plots
        env_vars = ['soil_pH', 'soil_CaCO3_pct', 'mean_annual_temp_C', 'annual_precip_mm']
        available_vars = [var for var in env_vars if var in data.columns]
        
        if available_vars:
            # Distribution plots
            dist_plots = plotter.create_environmental_distribution_plots(data, available_vars, output_dir)
            logger.info(f"Created {len(dist_plots)} distribution plots")
            
            # Species comparison plots
            comp_plots = plotter.create_species_comparison_plots(data, available_vars, output_dir)
            logger.info(f"Created {len(comp_plots)} species comparison plots")
        
        visualization_time = time.time() - start_time
        logger.info(f"✅ Visualization completed in {visualization_time:.1f}s")
        workflow_results['steps_completed'].append('visualization')
        
        # Step 5: Export Results
        logger.info("📤 Step 5: Exporting results...")
        start_time = time.time()
        
        # Export habitat parameters
        exported_files = processor.export_habitat_parameters(data, output_dir)
        workflow_results['output_files'].update(exported_files)
        
        # Generate hydroponic parameters
        hydroponic_params = generate_hydroponic_parameters(data)
        params_path = output_dir / "hydroponic_parameters.json"
        with open(params_path, 'w') as f:
            json.dump(hydroponic_params, f, indent=2)
        workflow_results['output_files']['hydroponic_parameters'] = str(params_path)
        
        # Generate comprehensive report
        comprehensive_report = model.generate_habitat_report(data)
        report_path = output_dir / "comprehensive_report.json"
        with open(report_path, 'w') as f:
            json.dump(comprehensive_report, f, indent=2)
        workflow_results['output_files']['comprehensive_report'] = str(report_path)
        
        export_time = time.time() - start_time
        logger.info(f"✅ Export completed in {export_time:.1f}s")
        workflow_results['steps_completed'].append('export')
        
        # Workflow Summary
        total_time = time.time() - workflow_results['start_time']
        workflow_results['total_time'] = total_time
        workflow_results['status'] = 'completed'
        
        logger.info("🎉 Workflow completed successfully!")
        logger.info(f"⏱️  Total time: {total_time:.1f}s")
        logger.info(f"📁 Output directory: {output_dir}")
        logger.info(f"📊 Records processed: {len(data)}")
        logger.info(f"🔬 Species analyzed: {data['species'].nunique()}")
        
        # Print output files
        logger.info("📋 Generated files:")
        for file_type, file_path in workflow_results['output_files'].items():
            logger.info(f"  {file_type}: {file_path}")
        
    except Exception as e:
        logger.error(f"❌ Workflow failed: {e}")
        workflow_results['errors'].append(str(e))
        workflow_results['status'] = 'failed'
    
    # Save workflow results
    results_path = output_dir / "workflow_results.json"
    with open(results_path, 'w') as f:
        json.dump(workflow_results, f, indent=2)
    
    return workflow_results


def generate_hydroponic_parameters(data):
    """Generate hydroponic parameters from natural habitat data."""
    params = {}
    
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


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description='Run complete GTHA workflow')
    parser.add_argument('--species', nargs='+', default=TRUFFLE_SPECIES[:3],
                       help='Truffle species to analyze')
    parser.add_argument('--countries', nargs='+',
                       help='Country codes to filter by')
    parser.add_argument('--year-from', type=int,
                       help='Start year for data collection')
    parser.add_argument('--year-to', type=int,
                       help='End year for data collection')
    parser.add_argument('--output-dir', type=Path, default=OUTPUTS_DIR,
                       help='Output directory for results')
    parser.add_argument('--quick', action='store_true',
                       help='Run quick demo with limited data')
    
    args = parser.parse_args()
    
    # Quick demo mode
    if args.quick:
        args.species = ['Tuber melanosporum']
        args.countries = ['FR']
        args.year_from = 2022
        args.year_to = 2023
        logger.info("🚀 Running in quick demo mode")
    
    # Run workflow
    results = run_complete_workflow(
        species=args.species,
        countries=args.countries,
        year_from=args.year_from,
        year_to=args.year_to,
        output_dir=args.output_dir
    )
    
    # Print final status
    if results['status'] == 'completed':
        print(f"\n🎉 Workflow completed successfully!")
        print(f"📁 Check results in: {args.output_dir}")
    else:
        print(f"\n❌ Workflow failed. Check logs for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()