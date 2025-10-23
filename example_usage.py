#!/usr/bin/env python3
"""
Example usage script for the Global Truffle Habitat Atlas (GTHA)

This script demonstrates how to use the GTHA system for collecting,
analyzing, and visualizing truffle habitat data.
"""
import logging
from pathlib import Path
import pandas as pd
import json

# Import GTHA modules
from config import *
from src.data_processing.habitat_processor import HabitatProcessor
from src.models.habitat_model import HabitatModel
from src.visualization.mapping_tools import MappingTools
from src.visualization.plotting_tools import PlottingTools

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_basic_workflow():
    """Example of basic GTHA workflow."""
    print("🌍 Global Truffle Habitat Atlas - Basic Workflow Example")
    print("=" * 60)
    
    # Initialize components
    config = {
        'gbif': API_CONFIG['gbif'],
        'inaturalist': API_CONFIG['inaturalist'],
        'soilgrids': API_CONFIG['soilgrids'],
        'worldclim': API_CONFIG['worldclim'],
        **MODEL_CONFIG
    }
    
    processor = HabitatProcessor(config, DATA_DIR)
    model = HabitatModel(config, MODELS_DIR)
    mapper = MappingTools()
    plotter = PlottingTools()
    
    # Example 1: Collect data for specific species
    print("\n1. Collecting data for Tuber melanosporum and Tuber magnatum...")
    species = ['Tuber melanosporum', 'Tuber magnatum']
    
    try:
        data = processor.collect_all_data(
            species=species,
            countries=['FR', 'IT', 'ES'],  # France, Italy, Spain
            year_from=2020,
            year_to=2023
        )
        
        if not data.empty:
            print(f"✅ Collected {len(data)} records")
            print(f"   Species: {data['species'].value_counts().to_dict()}")
        else:
            print("❌ No data collected")
            return
            
    except Exception as e:
        print(f"❌ Error collecting data: {e}")
        return
    
    # Example 2: Analyze habitat characteristics
    print("\n2. Analyzing habitat characteristics...")
    try:
        analysis = processor.analyze_habitat_characteristics(data)
        
        print("📊 Habitat Analysis Results:")
        print(f"   Total records: {len(data)}")
        print(f"   Species count: {data['species'].nunique()}")
        print(f"   Geographic bounds:")
        print(f"     Latitude: {analysis['geographic_bounds']['min_lat']:.2f} to {analysis['geographic_bounds']['max_lat']:.2f}")
        print(f"     Longitude: {analysis['geographic_bounds']['min_lon']:.2f} to {analysis['geographic_bounds']['max_lon']:.2f}")
        
    except Exception as e:
        print(f"❌ Error analyzing data: {e}")
    
    # Example 3: Train machine learning models
    print("\n3. Training machine learning models...")
    try:
        training_results = model.train_models(data)
        
        if training_results['species_classification']:
            accuracy = training_results['species_classification']['accuracy']
            print(f"✅ Species classification accuracy: {accuracy:.3f}")
        
        if training_results['habitat_suitability']:
            r2 = training_results['habitat_suitability']['r2_score']
            print(f"✅ Habitat suitability R²: {r2:.3f}")
            
    except Exception as e:
        print(f"❌ Error training models: {e}")
    
    # Example 4: Create visualizations
    print("\n4. Creating visualizations...")
    try:
        # Species distribution map
        map_path = OUTPUTS_DIR / "example_species_map.html"
        mapper.create_species_distribution_map(data, save_path=map_path)
        print(f"✅ Species distribution map: {map_path}")
        
        # Environmental correlation heatmap
        heatmap_path = OUTPUTS_DIR / "example_correlation_heatmap.png"
        mapper.create_correlation_heatmap(data, save_path=heatmap_path)
        print(f"✅ Correlation heatmap: {heatmap_path}")
        
    except Exception as e:
        print(f"❌ Error creating visualizations: {e}")
    
    # Example 5: Export results
    print("\n5. Exporting results...")
    try:
        exported_files = processor.export_habitat_parameters(data, OUTPUTS_DIR)
        print("✅ Exported files:")
        for file_type, file_path in exported_files.items():
            print(f"   {file_type}: {file_path}")
            
    except Exception as e:
        print(f"❌ Error exporting data: {e}")
    
    print("\n🎉 Basic workflow completed!")


def example_advanced_analysis():
    """Example of advanced analysis features."""
    print("\n🔬 Advanced Analysis Example")
    print("=" * 40)
    
    # Load existing data
    data_path = OUTPUTS_DIR / "raw_habitat_data.csv"
    if not data_path.exists():
        print("❌ No data file found. Run basic workflow first.")
        return
        
    data = pd.read_csv(data_path)
    
    if data.empty:
        print("❌ No data to analyze")
        return
    
    # Initialize components
    config = {**MODEL_CONFIG}
    model = HabitatModel(config, MODELS_DIR)
    plotter = PlottingTools()
    
    # Load trained models
    try:
        model.load_models()
        print("✅ Models loaded successfully")
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        return
    
    # Example 1: Feature importance analysis
    print("\n1. Analyzing feature importance...")
    try:
        if model.feature_importance_ is not None:
            top_features = model.feature_importance_.head(10)
            print("🔍 Top 10 most important features:")
            for _, row in top_features.iterrows():
                print(f"   {row['feature']}: {row['total_importance']:.3f}")
        else:
            print("❌ No feature importance data available")
    except Exception as e:
        print(f"❌ Error analyzing feature importance: {e}")
    
    # Example 2: Create detailed plots
    print("\n2. Creating detailed analysis plots...")
    try:
        # Environmental distribution plots
        env_vars = ['soil_pH', 'soil_CaCO3_pct', 'mean_annual_temp_C', 'annual_precip_mm']
        available_vars = [var for var in env_vars if var in data.columns]
        
        if available_vars:
            plots = plotter.create_environmental_distribution_plots(
                data, available_vars, OUTPUTS_DIR
            )
            print(f"✅ Created {len(plots)} distribution plots")
        
        # Species comparison plots
        plots = plotter.create_species_comparison_plots(
            data, available_vars, OUTPUTS_DIR
        )
        print(f"✅ Created {len(plots)} species comparison plots")
        
    except Exception as e:
        print(f"❌ Error creating plots: {e}")
    
    # Example 3: Generate habitat report
    print("\n3. Generating comprehensive habitat report...")
    try:
        report = model.generate_habitat_report(data)
        
        report_path = OUTPUTS_DIR / "comprehensive_habitat_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"✅ Comprehensive report saved: {report_path}")
        
        # Print summary
        print("\n📋 Report Summary:")
        print(f"   Total records: {report['dataset_summary']['total_records']}")
        print(f"   Species count: {report['dataset_summary']['species_count']}")
        print(f"   Environmental variables: {len(report['environmental_variables'])}")
        
    except Exception as e:
        print(f"❌ Error generating report: {e}")
    
    print("\n🎉 Advanced analysis completed!")


def example_hydroponic_parameters():
    """Example of generating hydroponic parameters."""
    print("\n🌱 Hydroponic Parameters Example")
    print("=" * 40)
    
    # Load data
    data_path = OUTPUTS_DIR / "raw_habitat_data.csv"
    if not data_path.exists():
        print("❌ No data file found. Run basic workflow first.")
        return
        
    data = pd.read_csv(data_path)
    
    if data.empty:
        print("❌ No data to analyze")
        return
    
    # Generate hydroponic parameters
    print("1. Generating hydroponic parameters from natural habitats...")
    
    try:
        # Calculate parameter ranges by species
        hydroponic_params = {}
        
        for species in data['species'].unique():
            species_data = data[data['species'] == species]
            
            params = {
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
            
            hydroponic_params[species] = params
        
        # Save parameters
        params_path = OUTPUTS_DIR / "hydroponic_parameters.json"
        with open(params_path, 'w') as f:
            json.dump(hydroponic_params, f, indent=2)
        
        print(f"✅ Hydroponic parameters saved: {params_path}")
        
        # Display parameters for first species
        first_species = list(hydroponic_params.keys())[0]
        print(f"\n📊 Example parameters for {first_species}:")
        params = hydroponic_params[first_species]
        
        for param_type, values in params.items():
            if values['min'] is not None:
                print(f"   {param_type}:")
                print(f"     Range: {values['min']:.2f} - {values['max']:.2f}")
                print(f"     Recommended: {values['recommended']:.2f}")
        
    except Exception as e:
        print(f"❌ Error generating hydroponic parameters: {e}")
    
    print("\n🎉 Hydroponic parameters generated!")


def main():
    """Main example function."""
    print("🍄 Global Truffle Habitat Atlas - Example Usage")
    print("=" * 60)
    
    # Create output directory
    OUTPUTS_DIR.mkdir(exist_ok=True)
    
    # Run examples
    try:
        example_basic_workflow()
        example_advanced_analysis()
        example_hydroponic_parameters()
        
        print("\n" + "=" * 60)
        print("🎉 All examples completed successfully!")
        print(f"📁 Check the '{OUTPUTS_DIR}' directory for generated files.")
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Examples interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        logger.exception("Unexpected error in examples")


if __name__ == "__main__":
    main()