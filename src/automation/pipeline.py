"""
Automated pipeline for end-to-end data processing.
"""
import logging
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from pathlib import Path
import json
import pandas as pd
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..data_processing.habitat_processor import HabitatProcessor
from ..models.habitat_model import HabitatModel
from ..visualization.mapping_tools import MappingTools

logger = logging.getLogger(__name__)


class AutomatedPipeline:
    """Automated pipeline for data collection, processing, and analysis."""
    
    def __init__(self, config: Dict[str, Any], data_dir: Path, output_dir: Path):
        self.config = config
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.processor = HabitatProcessor(config, data_dir)
        self.model = HabitatModel(config, data_dir / "models")
        self.mapper = MappingTools()
        
        # Pipeline state
        self.pipeline_state = {
            'last_run': None,
            'total_runs': 0,
            'successful_runs': 0,
            'failed_runs': 0,
            'total_records_processed': 0
        }
        
    def run_full_pipeline(self, species: List[str], 
                         countries: Optional[List[str]] = None,
                         parallel: bool = True) -> Dict[str, Any]:
        """Run the complete pipeline from data collection to visualization."""
        logger.info("Starting full automated pipeline")
        start_time = datetime.now()
        
        pipeline_results = {
            'start_time': start_time.isoformat(),
            'species': species,
            'countries': countries,
            'steps': {},
            'errors': [],
            'status': 'running'
        }
        
        try:
            # Step 1: Data Collection
            logger.info("Step 1: Collecting data from all sources")
            step_start = datetime.now()
            
            if parallel:
                data = self._collect_data_parallel(species, countries)
            else:
                data = self.processor.collect_all_data(species, countries)
                
            if data.empty:
                raise ValueError("No data collected")
                
            pipeline_results['steps']['data_collection'] = {
                'status': 'completed',
                'duration': (datetime.now() - step_start).total_seconds(),
                'records': len(data)
            }
            
            # Step 2: Data Processing
            logger.info("Step 2: Processing and cleaning data")
            step_start = datetime.now()
            
            processed_data = self._process_data(data)
            
            pipeline_results['steps']['data_processing'] = {
                'status': 'completed',
                'duration': (datetime.now() - step_start).total_seconds(),
                'records': len(processed_data)
            }
            
            # Step 3: Model Training/Update
            logger.info("Step 3: Training/updating models")
            step_start = datetime.now()
            
            model_results = self._train_models(processed_data)
            
            pipeline_results['steps']['model_training'] = {
                'status': 'completed',
                'duration': (datetime.now() - step_start).total_seconds(),
                'results': model_results
            }
            
            # Step 4: Analysis
            logger.info("Step 4: Running habitat analysis")
            step_start = datetime.now()
            
            analysis_results = self.processor.analyze_habitat_characteristics(processed_data)
            
            pipeline_results['steps']['analysis'] = {
                'status': 'completed',
                'duration': (datetime.now() - step_start).total_seconds(),
                'results': analysis_results
            }
            
            # Step 5: Visualization
            logger.info("Step 5: Generating visualizations")
            step_start = datetime.now()
            
            viz_results = self._generate_visualizations(processed_data)
            
            pipeline_results['steps']['visualization'] = {
                'status': 'completed',
                'duration': (datetime.now() - step_start).total_seconds(),
                'files': viz_results
            }
            
            # Step 6: Export
            logger.info("Step 6: Exporting results")
            step_start = datetime.now()
            
            export_results = self.processor.export_habitat_parameters(processed_data, self.output_dir)
            
            pipeline_results['steps']['export'] = {
                'status': 'completed',
                'duration': (datetime.now() - step_start).total_seconds(),
                'files': export_results
            }
            
            # Update pipeline state
            self.pipeline_state['last_run'] = start_time
            self.pipeline_state['total_runs'] += 1
            self.pipeline_state['successful_runs'] += 1
            self.pipeline_state['total_records_processed'] += len(processed_data)
            
            pipeline_results['status'] = 'completed'
            pipeline_results['total_duration'] = (datetime.now() - start_time).total_seconds()
            
            logger.info(f"Pipeline completed successfully in {pipeline_results['total_duration']:.1f}s")
            
        except Exception as e:
            error_msg = f"Pipeline failed: {e}"
            logger.error(error_msg)
            pipeline_results['errors'].append(error_msg)
            pipeline_results['status'] = 'failed'
            
            self.pipeline_state['total_runs'] += 1
            self.pipeline_state['failed_runs'] += 1
            
        finally:
            # Save pipeline results
            self._save_pipeline_results(pipeline_results)
            
        return pipeline_results
        
    def _collect_data_parallel(self, species: List[str], 
                             countries: Optional[List[str]] = None) -> pd.DataFrame:
        """Collect data from multiple sources in parallel."""
        logger.info("Collecting data in parallel mode")
        
        # Split species into chunks for parallel processing
        chunk_size = max(1, len(species) // 4)  # Use 4 parallel workers
        species_chunks = [species[i:i + chunk_size] for i in range(0, len(species), chunk_size)]
        
        all_data = []
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            # Submit collection tasks
            future_to_chunk = {
                executor.submit(
                    self.processor.collect_all_data, 
                    chunk, 
                    countries
                ): chunk for chunk in species_chunks
            }
            
            # Collect results
            for future in as_completed(future_to_chunk):
                chunk = future_to_chunk[future]
                try:
                    data = future.result()
                    if not data.empty:
                        all_data.append(data)
                        logger.info(f"Collected {len(data)} records for chunk {chunk}")
                except Exception as e:
                    logger.error(f"Error collecting data for chunk {chunk}: {e}")
                    
        # Combine all data
        if all_data:
            combined_data = pd.concat(all_data, ignore_index=True)
            # Remove duplicates
            combined_data = combined_data.drop_duplicates(
                subset=['species', 'latitude', 'longitude']
            )
            logger.info(f"Total records collected: {len(combined_data)}")
            return combined_data
        else:
            return pd.DataFrame()
            
    def _process_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Process and clean the collected data."""
        logger.info(f"Processing {len(data)} records")
        
        # Basic cleaning
        processed_data = data.copy()
        
        # Remove records with invalid coordinates
        processed_data = processed_data.dropna(subset=['latitude', 'longitude'])
        
        # Remove duplicates
        processed_data = processed_data.drop_duplicates(
            subset=['species', 'latitude', 'longitude']
        )
        
        # Add processing metadata
        processed_data['processed_at'] = datetime.now().isoformat()
        processed_data['pipeline_version'] = '1.0.0'
        
        logger.info(f"Processed data: {len(processed_data)} records")
        return processed_data
        
    def _train_models(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Train or update machine learning models."""
        logger.info("Training/updating models")
        
        try:
            # Check if models exist
            model_path = self.data_dir / "models"
            if not model_path.exists():
                model_path.mkdir(parents=True)
                
            # Train models
            training_results = self.model.train_models(data)
            
            logger.info("Models trained successfully")
            return training_results
            
        except Exception as e:
            logger.error(f"Error training models: {e}")
            return {'error': str(e)}
            
    def _generate_visualizations(self, data: pd.DataFrame) -> Dict[str, str]:
        """Generate visualizations."""
        logger.info("Generating visualizations")
        
        viz_files = {}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        try:
            # Species distribution map
            map_path = self.output_dir / f"species_map_{timestamp}.html"
            self.mapper.create_species_distribution_map(data, save_path=map_path)
            viz_files['species_map'] = str(map_path)
            
            # Correlation heatmap
            heatmap_path = self.output_dir / f"correlation_heatmap_{timestamp}.png"
            self.mapper.create_correlation_heatmap(data, save_path=heatmap_path)
            viz_files['correlation_heatmap'] = str(heatmap_path)
            
            # Feature importance plot
            if self.model.feature_importance_ is not None:
                importance_path = self.output_dir / f"feature_importance_{timestamp}.png"
                self.model.plot_feature_importance(save_path=importance_path)
                viz_files['feature_importance'] = str(importance_path)
                
            logger.info(f"Generated {len(viz_files)} visualizations")
            
        except Exception as e:
            logger.error(f"Error generating visualizations: {e}")
            
        return viz_files
        
    def _save_pipeline_results(self, results: Dict[str, Any]):
        """Save pipeline results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = self.output_dir / f"pipeline_results_{timestamp}.json"
        
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
            
        # Also save to latest results
        latest_path = self.output_dir / "latest_pipeline_results.json"
        with open(latest_path, 'w') as f:
            json.dump(results, f, indent=2)
            
        logger.info(f"Pipeline results saved to {results_path}")
        
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current pipeline status."""
        return {
            'state': self.pipeline_state,
            'last_run': self.pipeline_state['last_run'].isoformat() if self.pipeline_state['last_run'] else None,
            'success_rate': (
                self.pipeline_state['successful_runs'] / self.pipeline_state['total_runs'] 
                if self.pipeline_state['total_runs'] > 0 else 0
            )
        }
        
    def run_incremental_update(self, species: List[str], 
                             countries: Optional[List[str]] = None) -> Dict[str, Any]:
        """Run incremental update with only new data."""
        logger.info("Running incremental update")
        
        # Load existing data
        existing_data_path = self.output_dir / "latest_data.csv"
        if existing_data_path.exists():
            existing_data = pd.read_csv(existing_data_path)
            logger.info(f"Loaded {len(existing_data)} existing records")
        else:
            existing_data = pd.DataFrame()
            
        # Collect new data (last 30 days)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        new_data = self.processor.collect_all_data(
            species=species,
            countries=countries,
            year_from=start_date.year,
            year_to=end_date.year
        )
        
        if new_data.empty:
            logger.info("No new data found")
            return {'status': 'no_new_data', 'records': 0}
            
        # Merge with existing data
        if not existing_data.empty:
            combined_data = pd.concat([existing_data, new_data], ignore_index=True)
            combined_data = combined_data.drop_duplicates(
                subset=['species', 'latitude', 'longitude']
            )
        else:
            combined_data = new_data
            
        # Run pipeline on combined data
        results = self.run_full_pipeline(species, countries)
        results['incremental'] = True
        results['new_records'] = len(new_data)
        results['total_records'] = len(combined_data)
        
        return results