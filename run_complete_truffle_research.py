#!/usr/bin/env python3
"""
Complete Truffle Research Workflow - Example Implementation

This script demonstrates the complete 6-phase research workflow for truffle ecology
analysis, from data collection through model-agnostic inference, with full
reproducibility and automation capabilities.
"""

import logging
import sys
from pathlib import Path
from typing import List, Tuple
import argparse

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from src.research.truffle_research_workflow import (
    TruffleResearchWorkflow, ResearchConfig, run_truffle_research
)
from src.research.automated_data_pipeline import (
    AutomatedDataPipeline, PipelineConfig, start_automated_pipeline
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('truffle_research.log')
    ]
)
logger = logging.getLogger(__name__)


def run_complete_research_workflow(
    truffle_species: List[str] = None,
    tree_genera: List[str] = None,
    study_regions: List[Tuple[float, float, float, float]] = None,
    output_dir: str = "complete_truffle_research",
    enable_automation: bool = False
):
    """
    Run the complete truffle research workflow.
    
    Args:
        truffle_species: List of truffle species to study
        tree_genera: List of tree genera to include
        study_regions: List of study regions as (min_lat, max_lat, min_lon, max_lon)
        output_dir: Output directory for results
        enable_automation: Whether to enable automated data collection
    """
    
    logger.info("=== Starting Complete Truffle Research Workflow ===")
    
    # Create research configuration
    research_config = ResearchConfig(
        truffle_species=truffle_species or [
            'Tuber melanosporum', 'Tuber magnatum', 'Tuber aestivum',
            'Tuber borchii', 'Tuber brumale'
        ],
        tree_genera=tree_genera or [
            'Quercus', 'Corylus', 'Tilia', 'Populus', 'Salix',
            'Pinus', 'Carpinus', 'Fagus', 'Castanea'
        ],
        study_regions=study_regions or [
            (35.0, 70.0, -15.0, 40.0),  # Europe
            (25.0, 50.0, -125.0, -65.0)  # North America
        ],
        output_dir=Path(output_dir),
        occurrence_limit=5000,  # Reduced for example
        enable_caching=True,
        enable_harmonization=True
    )
    
    logger.info(f"Research configuration:")
    logger.info(f"  - Truffle species: {research_config.truffle_species}")
    logger.info(f"  - Tree genera: {research_config.tree_genera}")
    logger.info(f"  - Study regions: {research_config.study_regions}")
    logger.info(f"  - Output directory: {research_config.output_dir}")
    
    if enable_automation:
        # Run automated pipeline
        logger.info("Starting automated data collection pipeline...")
        
        pipeline_config = PipelineConfig(
            occurrence_schedule="monthly",
            environmental_schedule="on_demand",
            metagenomics_schedule="quarterly",
            output_dir=Path(output_dir) / "automated_data",
            enable_notifications=False
        )
        
        # Start pipeline in background (simplified for example)
        logger.info("Automated pipeline configuration created")
        logger.info("Note: Full automation requires running the pipeline separately")
        
    else:
        # Run complete research workflow
        logger.info("Running complete research workflow...")
        
        try:
            # Initialize workflow
            workflow = TruffleResearchWorkflow(research_config)
            
            # Run all 6 phases
            results = workflow.run_complete_workflow()
            
            # Display results summary
            display_results_summary(results)
            
            logger.info("=== Research Workflow Completed Successfully ===")
            logger.info(f"Results saved to: {research_config.output_dir}")
            
            return results
            
        except Exception as e:
            logger.error(f"Research workflow failed: {e}")
            raise


def display_results_summary(results: dict):
    """Display a summary of research results."""
    logger.info("\n=== RESEARCH RESULTS SUMMARY ===")
    
    # Data collection summary
    if 'harmonized_data' in results and results['harmonized_data']:
        harmonized = results['harmonized_data']
        if 'summary_stats' in harmonized:
            stats = harmonized['summary_stats']
            logger.info(f"Total harmonized records: {stats.get('total_records', 0)}")
            logger.info(f"Unique sources: {stats.get('unique_sources', 0)}")
            logger.info(f"Coordinate coverage: {stats.get('coordinate_coverage', 0):.1f}%")
            logger.info(f"Temporal coverage: {stats.get('temporal_coverage', 0):.1f}%")
            logger.info(f"Missing data: {stats.get('missing_data_percentage', 0):.1f}%")
    
    # Quality scores
    if 'harmonized_data' in results and 'quality_scores' in results['harmonized_data']:
        quality_scores = results['harmonized_data']['quality_scores']
        logger.info(f"\nData Quality Scores:")
        for source, scores in quality_scores.items():
            overall_score = scores.get('overall_score', 0)
            logger.info(f"  {source}: {overall_score:.3f}")
    
    # Analysis results
    if 'analysis_results' in results:
        analysis = results['analysis_results']
        
        # Geographic analysis
        if 'geographic_analysis' in analysis:
            geo = analysis['geographic_analysis']
            if 'lat_range' in geo and 'lon_range' in geo:
                logger.info(f"\nGeographic Coverage:")
                logger.info(f"  Latitude range: {geo['lat_range'][0]:.2f} to {geo['lat_range'][1]:.2f}")
                logger.info(f"  Longitude range: {geo['lon_range'][0]:.2f} to {geo['lon_range'][1]:.2f}")
                logger.info(f"  Unique locations: {geo.get('unique_locations', 0)}")
        
        # Species analysis
        if 'species_analysis' in analysis:
            species = analysis['species_analysis']
            if 'unique_species' in species:
                logger.info(f"\nSpecies Diversity:")
                logger.info(f"  Unique species: {species['unique_species']}")
                if 'most_common_species' in species and species['most_common_species']:
                    logger.info(f"  Most common species: {species['most_common_species']}")
        
        # Model performance
        if 'model_performance' in results:
            models = results['model_performance']
            logger.info(f"\nModel Performance:")
            for target_name, model_info in models.items():
                if model_info['model_type'] == 'classification':
                    report = model_info.get('classification_report', {})
                    if 'accuracy' in report:
                        logger.info(f"  {target_name}: {report['accuracy']:.3f} accuracy")
                elif model_info['model_type'] == 'regression':
                    r2 = model_info.get('r2', 0)
                    mse = model_info.get('mse', 0)
                    logger.info(f"  {target_name}: R² = {r2:.3f}, MSE = {mse:.3f}")


def run_quick_example():
    """Run a quick example with minimal data collection."""
    logger.info("Running quick example...")
    
    # Quick example with just two species
    results = run_complete_research_workflow(
        truffle_species=['Tuber melanosporum', 'Tuber magnatum'],
        tree_genera=['Quercus', 'Corylus'],
        study_regions=[(40.0, 50.0, 0.0, 15.0)],  # Focus on France/Italy
        output_dir="quick_truffle_example",
        enable_automation=False
    )
    
    return results


def run_comprehensive_study():
    """Run a comprehensive study with all species and regions."""
    logger.info("Running comprehensive study...")
    
    results = run_complete_research_workflow(
        truffle_species=[
            'Tuber melanosporum', 'Tuber magnatum', 'Tuber aestivum',
            'Tuber borchii', 'Tuber brumale', 'Tuber indicum'
        ],
        tree_genera=[
            'Quercus', 'Corylus', 'Tilia', 'Populus', 'Salix',
            'Pinus', 'Carpinus', 'Fagus', 'Castanea', 'Betula'
        ],
        study_regions=[
            (35.0, 70.0, -15.0, 40.0),  # Europe
            (25.0, 50.0, -125.0, -65.0),  # North America
            (-40.0, -10.0, 110.0, 160.0)  # Australia
        ],
        output_dir="comprehensive_truffle_study",
        enable_automation=False
    )
    
    return results


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description="Run truffle research workflow")
    parser.add_argument(
        "--mode", 
        choices=["quick", "comprehensive", "custom"],
        default="quick",
        help="Research mode to run"
    )
    parser.add_argument(
        "--output-dir",
        default="truffle_research_output",
        help="Output directory for results"
    )
    parser.add_argument(
        "--truffle-species",
        nargs="+",
        help="Truffle species to study"
    )
    parser.add_argument(
        "--tree-genera",
        nargs="+",
        help="Tree genera to include"
    )
    parser.add_argument(
        "--enable-automation",
        action="store_true",
        help="Enable automated data collection pipeline"
    )
    
    args = parser.parse_args()
    
    try:
        if args.mode == "quick":
            logger.info("Running quick example...")
            results = run_quick_example()
            
        elif args.mode == "comprehensive":
            logger.info("Running comprehensive study...")
            results = run_comprehensive_study()
            
        elif args.mode == "custom":
            logger.info("Running custom research workflow...")
            results = run_complete_research_workflow(
                truffle_species=args.truffle_species,
                tree_genera=args.tree_genera,
                output_dir=args.output_dir,
                enable_automation=args.enable_automation
            )
        
        logger.info("Research workflow completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("Research workflow interrupted by user")
    except Exception as e:
        logger.error(f"Research workflow failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()