"""
Automated Data Collection Pipeline for Truffle Research

This module provides automated, scheduled data collection with full
reproducibility and error handling for the truffle research workflow.
"""

import logging
import schedule
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
import json
import yaml
from dataclasses import dataclass
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import traceback

from .truffle_research_workflow import TruffleResearchWorkflow, ResearchConfig
from ..data_collectors import UnifiedDataCollector, load_collector_config

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for automated data collection pipeline."""
    
    # Collection schedules
    occurrence_schedule: str = "monthly"  # daily, weekly, monthly
    environmental_schedule: str = "on_demand"  # daily, weekly, monthly, on_demand
    metagenomics_schedule: str = "quarterly"  # monthly, quarterly, yearly
    
    # Data collection parameters
    max_retries: int = 3
    retry_delay: int = 300  # seconds
    parallel_workers: int = 4
    
    # Quality thresholds
    min_quality_score: float = 0.5
    min_records_per_source: int = 10
    
    # Output parameters
    output_dir: Path = None
    backup_frequency: str = "weekly"  # daily, weekly, monthly
    retention_days: int = 365
    
    # Notification settings
    enable_notifications: bool = False
    notification_email: str = None
    slack_webhook: str = None
    
    def __post_init__(self):
        if self.output_dir is None:
            self.output_dir = Path("automated_truffle_data")


class AutomatedDataPipeline:
    """Automated data collection pipeline with scheduling and monitoring."""
    
    def __init__(self, pipeline_config: PipelineConfig, research_config: ResearchConfig):
        self.pipeline_config = pipeline_config
        self.research_config = research_config
        
        # Initialize data collector
        collector_config = load_collector_config()
        self.collector = UnifiedDataCollector(
            config=collector_config.get_all_configs(),
            data_dir=self.pipeline_config.output_dir / "raw_data",
            enable_caching=True,
            enable_harmonization=True
        )
        
        # Pipeline state
        self.pipeline_state = {
            'last_run': None,
            'successful_runs': 0,
            'failed_runs': 0,
            'total_records_collected': 0,
            'data_sources_status': {},
            'last_errors': []
        }
        
        # Create output directories
        self.pipeline_config.output_dir.mkdir(parents=True, exist_ok=True)
        (self.pipeline_config.output_dir / "logs").mkdir(exist_ok=True)
        (self.pipeline_config.output_dir / "backups").mkdir(exist_ok=True)
        (self.pipeline_config.output_dir / "reports").mkdir(exist_ok=True)
        
        # Set up logging
        self._setup_logging()
    
    def _setup_logging(self):
        """Set up comprehensive logging for the pipeline."""
        log_dir = self.pipeline_config.output_dir / "logs"
        
        # Create formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        simple_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        
        # File handler for detailed logs
        file_handler = logging.FileHandler(
            log_dir / f"pipeline_{datetime.now().strftime('%Y%m%d')}.log"
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(detailed_formatter)
        
        # Console handler for important messages
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(simple_formatter)
        
        # Configure root logger
        logger.setLevel(logging.DEBUG)
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    
    def start_pipeline(self):
        """Start the automated data collection pipeline."""
        logger.info("Starting automated truffle data collection pipeline")
        
        # Schedule data collection tasks
        self._schedule_tasks()
        
        # Run initial collection
        self._run_initial_collection()
        
        # Start the scheduler
        logger.info("Pipeline scheduler started. Press Ctrl+C to stop.")
        try:
            while True:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
        except KeyboardInterrupt:
            logger.info("Pipeline stopped by user")
        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            self._handle_pipeline_error(e)
    
    def _schedule_tasks(self):
        """Schedule data collection tasks based on configuration."""
        # Occurrence data collection
        if self.pipeline_config.occurrence_schedule == "daily":
            schedule.every().day.at("02:00").do(self._collect_occurrence_data)
        elif self.pipeline_config.occurrence_schedule == "weekly":
            schedule.every().monday.at("02:00").do(self._collect_occurrence_data)
        elif self.pipeline_config.occurrence_schedule == "monthly":
            schedule.every().month.do(self._collect_occurrence_data)
        
        # Environmental data collection
        if self.pipeline_config.environmental_schedule == "daily":
            schedule.every().day.at("03:00").do(self._collect_environmental_data)
        elif self.pipeline_config.environmental_schedule == "weekly":
            schedule.every().monday.at("03:00").do(self._collect_environmental_data)
        elif self.pipeline_config.environmental_schedule == "monthly":
            schedule.every().month.do(self._collect_environmental_data)
        
        # Metagenomics data collection
        if self.pipeline_config.metagenomics_schedule == "monthly":
            schedule.every().month.do(self._collect_metagenomics_data)
        elif self.pipeline_config.metagenomics_schedule == "quarterly":
            schedule.every(3).months.do(self._collect_metagenomics_data)
        elif self.pipeline_config.metagenomics_schedule == "yearly":
            schedule.every().year.do(self._collect_metagenomics_data)
        
        # Backup tasks
        if self.pipeline_config.backup_frequency == "daily":
            schedule.every().day.at("04:00").do(self._create_backup)
        elif self.pipeline_config.backup_frequency == "weekly":
            schedule.every().sunday.at("04:00").do(self._create_backup)
        elif self.pipeline_config.backup_frequency == "monthly":
            schedule.every().month.do(self._create_backup)
        
        # Cleanup tasks
        schedule.every().day.at("05:00").do(self._cleanup_old_data)
        
        logger.info("Data collection tasks scheduled")
    
    def _run_initial_collection(self):
        """Run initial data collection on pipeline start."""
        logger.info("Running initial data collection")
        
        try:
            # Collect all data types
            self._collect_occurrence_data()
            self._collect_environmental_data()
            self._collect_metagenomics_data()
            
            # Generate initial report
            self._generate_collection_report()
            
            logger.info("Initial data collection completed successfully")
            
        except Exception as e:
            logger.error(f"Initial collection failed: {e}")
            self._handle_pipeline_error(e)
    
    def _collect_occurrence_data(self):
        """Collect occurrence data from GBIF and iNaturalist."""
        logger.info("Starting occurrence data collection")
        
        collection_results = {
            'truffles': {},
            'trees': {},
            'total_records': 0,
            'errors': []
        }
        
        # Collect truffle data
        for source in ['gbif', 'inaturalist']:
            try:
                logger.info(f"Collecting truffle data from {source}")
                
                result = self.collector.collect(
                    source=source,
                    species=self.research_config.truffle_species,
                    limit=self.research_config.occurrence_limit,
                    has_coordinate=True
                )
                
                if isinstance(result, dict) and 'records_df' in result:
                    records_count = len(result['records_df'])
                    collection_results['truffles'][source] = {
                        'records': records_count,
                        'quality_score': result.get('quality_scores', {}).get(source, {}).get('overall_score', 0),
                        'status': 'success'
                    }
                    collection_results['total_records'] += records_count
                    
                    # Save data
                    self._save_collection_data(f"truffles_{source}", result)
                    
                else:
                    collection_results['truffles'][source] = {
                        'records': 0,
                        'quality_score': 0,
                        'status': 'no_data'
                    }
                    
            except Exception as e:
                error_msg = f"Error collecting truffle data from {source}: {e}"
                logger.error(error_msg)
                collection_results['errors'].append(error_msg)
                collection_results['truffles'][source] = {
                    'records': 0,
                    'quality_score': 0,
                    'status': 'error'
                }
        
        # Collect tree data
        for source in ['gbif', 'inaturalist']:
            try:
                logger.info(f"Collecting tree data from {source}")
                
                result = self.collector.collect(
                    source=source,
                    species=self.research_config.tree_genera,
                    limit=self.research_config.occurrence_limit,
                    has_coordinate=True
                )
                
                if isinstance(result, dict) and 'records_df' in result:
                    records_count = len(result['records_df'])
                    collection_results['trees'][source] = {
                        'records': records_count,
                        'quality_score': result.get('quality_scores', {}).get(source, {}).get('overall_score', 0),
                        'status': 'success'
                    }
                    collection_results['total_records'] += records_count
                    
                    # Save data
                    self._save_collection_data(f"trees_{source}", result)
                    
                else:
                    collection_results['trees'][source] = {
                        'records': 0,
                        'quality_score': 0,
                        'status': 'no_data'
                    }
                    
            except Exception as e:
                error_msg = f"Error collecting tree data from {source}: {e}"
                logger.error(error_msg)
                collection_results['errors'].append(error_msg)
                collection_results['trees'][source] = {
                    'records': 0,
                    'quality_score': 0,
                    'status': 'error'
                }
        
        # Update pipeline state
        self.pipeline_state['last_run'] = datetime.now().isoformat()
        self.pipeline_state['total_records_collected'] += collection_results['total_records']
        
        if collection_results['errors']:
            self.pipeline_state['failed_runs'] += 1
            self.pipeline_state['last_errors'] = collection_results['errors']
        else:
            self.pipeline_state['successful_runs'] += 1
        
        # Save collection results
        self._save_collection_results('occurrence', collection_results)
        
        logger.info(f"Occurrence data collection completed: {collection_results['total_records']} records")
    
    def _collect_environmental_data(self):
        """Collect environmental data for existing coordinates."""
        logger.info("Starting environmental data collection")
        
        # Get coordinates from existing occurrence data
        coordinates = self._get_existing_coordinates()
        
        if not coordinates:
            logger.warning("No coordinates available for environmental data collection")
            return
        
        collection_results = {
            'soil': {'status': 'pending', 'records': 0, 'errors': []},
            'climate': {'status': 'pending', 'records': 0, 'errors': []},
            'geology': {'status': 'pending', 'records': 0, 'errors': []},
            'total_records': 0
        }
        
        # Collect data in parallel
        with ThreadPoolExecutor(max_workers=self.pipeline_config.parallel_workers) as executor:
            # Submit tasks
            future_to_source = {}
            
            # Soil data
            future_to_source[executor.submit(
                self._collect_soil_data, coordinates
            )] = 'soil'
            
            # Climate data
            future_to_source[executor.submit(
                self._collect_climate_data, coordinates
            )] = 'climate'
            
            # Geology data
            future_to_source[executor.submit(
                self._collect_geology_data, coordinates
            )] = 'geology'
            
            # Process results
            for future in as_completed(future_to_source):
                source = future_to_source[future]
                try:
                    result = future.result()
                    collection_results[source] = result
                    collection_results['total_records'] += result['records']
                    
                    if result['status'] == 'success':
                        # Save data
                        self._save_collection_data(f"environmental_{source}", result['data'])
                    
                except Exception as e:
                    error_msg = f"Error collecting {source} data: {e}"
                    logger.error(error_msg)
                    collection_results[source] = {
                        'status': 'error',
                        'records': 0,
                        'errors': [error_msg]
                    }
        
        # Update pipeline state
        self.pipeline_state['total_records_collected'] += collection_results['total_records']
        
        # Save collection results
        self._save_collection_results('environmental', collection_results)
        
        logger.info(f"Environmental data collection completed: {collection_results['total_records']} records")
    
    def _collect_soil_data(self, coordinates: List[tuple]) -> Dict[str, Any]:
        """Collect soil data from SoilGrids."""
        try:
            result = self.collector.collect(
                source='soilgrids',
                coordinates=coordinates,
                variables=self.research_config.soil_variables
            )
            
            if isinstance(result, dict) and 'records_df' in result:
                return {
                    'status': 'success',
                    'records': len(result['records_df']),
                    'data': result,
                    'errors': []
                }
            else:
                return {
                    'status': 'no_data',
                    'records': 0,
                    'data': None,
                    'errors': ['No soil data returned']
                }
                
        except Exception as e:
            return {
                'status': 'error',
                'records': 0,
                'data': None,
                'errors': [str(e)]
            }
    
    def _collect_climate_data(self, coordinates: List[tuple]) -> Dict[str, Any]:
        """Collect climate data from WorldClim."""
        try:
            result = self.collector.collect(
                source='worldclim',
                coordinates=coordinates,
                variables=self.research_config.climate_variables
            )
            
            if isinstance(result, dict) and 'records_df' in result:
                return {
                    'status': 'success',
                    'records': len(result['records_df']),
                    'data': result,
                    'errors': []
                }
            else:
                return {
                    'status': 'no_data',
                    'records': 0,
                    'data': None,
                    'errors': ['No climate data returned']
                }
                
        except Exception as e:
            return {
                'status': 'error',
                'records': 0,
                'data': None,
                'errors': [str(e)]
            }
    
    def _collect_geology_data(self, coordinates: List[tuple]) -> Dict[str, Any]:
        """Collect geology data from GLiM."""
        try:
            result = self.collector.collect(
                source='glim',
                coordinates=coordinates
            )
            
            if isinstance(result, dict) and 'records_df' in result:
                return {
                    'status': 'success',
                    'records': len(result['records_df']),
                    'data': result,
                    'errors': []
                }
            else:
                return {
                    'status': 'no_data',
                    'records': 0,
                    'data': None,
                    'errors': ['No geology data returned']
                }
                
        except Exception as e:
            return {
                'status': 'error',
                'records': 0,
                'data': None,
                'errors': [str(e)]
            }
    
    def _collect_metagenomics_data(self):
        """Collect metagenomics data from EBI."""
        logger.info("Starting metagenomics data collection")
        
        collection_results = {
            'total_records': 0,
            'sources': {},
            'errors': []
        }
        
        for search_term in self.research_config.metagenomics_search_terms:
            try:
                logger.info(f"Collecting metagenomics data for '{search_term}'")
                
                result = self.collector.collect(
                    source='ebi_metagenomics',
                    search_term=search_term,
                    limit=1000,
                    include_samples=True,
                    include_abundance=True
                )
                
                if isinstance(result, dict) and 'records_df' in result:
                    records_count = len(result['records_df'])
                    collection_results['sources'][search_term] = {
                        'records': records_count,
                        'status': 'success'
                    }
                    collection_results['total_records'] += records_count
                    
                    # Save data
                    self._save_collection_data(f"metagenomics_{search_term}", result)
                    
                else:
                    collection_results['sources'][search_term] = {
                        'records': 0,
                        'status': 'no_data'
                    }
                    
            except Exception as e:
                error_msg = f"Error collecting metagenomics data for '{search_term}': {e}"
                logger.error(error_msg)
                collection_results['errors'].append(error_msg)
                collection_results['sources'][search_term] = {
                    'records': 0,
                    'status': 'error'
                }
        
        # Update pipeline state
        self.pipeline_state['total_records_collected'] += collection_results['total_records']
        
        # Save collection results
        self._save_collection_results('metagenomics', collection_results)
        
        logger.info(f"Metagenomics data collection completed: {collection_results['total_records']} records")
    
    def _get_existing_coordinates(self) -> List[tuple]:
        """Get coordinates from existing occurrence data."""
        coordinates = set()
        
        # Look for existing occurrence data files
        data_dir = self.pipeline_config.output_dir / "raw_data"
        
        for file_path in data_dir.glob("truffles_*.parquet"):
            try:
                df = pd.read_parquet(file_path)
                if 'latitude' in df.columns and 'longitude' in df.columns:
                    coords = df[['latitude', 'longitude']].dropna()
                    coordinates.update([(row['latitude'], row['longitude']) 
                                     for _, row in coords.iterrows()])
            except Exception as e:
                logger.warning(f"Error reading coordinates from {file_path}: {e}")
        
        for file_path in data_dir.glob("trees_*.parquet"):
            try:
                df = pd.read_parquet(file_path)
                if 'latitude' in df.columns and 'longitude' in df.columns:
                    coords = df[['latitude', 'longitude']].dropna()
                    coordinates.update([(row['latitude'], row['longitude']) 
                                     for _, row in coords.iterrows()])
            except Exception as e:
                logger.warning(f"Error reading coordinates from {file_path}: {e}")
        
        return list(coordinates)
    
    def _save_collection_data(self, data_type: str, data: Dict[str, Any]):
        """Save collected data to files."""
        data_dir = self.pipeline_config.output_dir / "raw_data"
        data_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if isinstance(data, dict) and 'records_df' in data:
            # Save DataFrame
            df = data['records_df']
            if not df.empty:
                file_path = data_dir / f"{data_type}_{timestamp}.parquet"
                df.to_parquet(file_path)
                logger.info(f"Saved {data_type} data to {file_path}")
            
            # Save metadata
            if 'metadata_df' in data and not data['metadata_df'].empty:
                metadata_path = data_dir / f"{data_type}_metadata_{timestamp}.parquet"
                data['metadata_df'].to_parquet(metadata_path)
    
    def _save_collection_results(self, collection_type: str, results: Dict[str, Any]):
        """Save collection results and statistics."""
        results_dir = self.pipeline_config.output_dir / "reports"
        results_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = results_dir / f"{collection_type}_collection_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Saved {collection_type} collection results to {results_file}")
    
    def _generate_collection_report(self):
        """Generate comprehensive collection report."""
        logger.info("Generating collection report")
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'pipeline_state': self.pipeline_state,
            'data_sources': self._get_data_source_status(),
            'collection_summary': self._get_collection_summary(),
            'quality_metrics': self._get_quality_metrics(),
            'recommendations': self._get_recommendations()
        }
        
        # Save report
        report_path = self.pipeline_config.output_dir / "reports" / f"collection_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Collection report saved to {report_path}")
        
        # Send notifications if enabled
        if self.pipeline_config.enable_notifications:
            self._send_notification(report)
    
    def _get_data_source_status(self) -> Dict[str, Any]:
        """Get status of all data sources."""
        status = {}
        
        # Check cache statistics
        cache_stats = self.collector.get_cache_stats()
        if cache_stats:
            status['cache'] = cache_stats
        
        # Check data files
        data_dir = self.pipeline_config.output_dir / "raw_data"
        if data_dir.exists():
            files = list(data_dir.glob("*.parquet"))
            status['data_files'] = {
                'count': len(files),
                'total_size_mb': sum(f.stat().st_size for f in files) / (1024 * 1024),
                'latest_file': max(files, key=lambda x: x.stat().st_mtime).name if files else None
            }
        
        return status
    
    def _get_collection_summary(self) -> Dict[str, Any]:
        """Get collection summary statistics."""
        return {
            'total_records': self.pipeline_state['total_records_collected'],
            'successful_runs': self.pipeline_state['successful_runs'],
            'failed_runs': self.pipeline_state['failed_runs'],
            'last_run': self.pipeline_state['last_run']
        }
    
    def _get_quality_metrics(self) -> Dict[str, Any]:
        """Get data quality metrics."""
        # This would analyze the quality of collected data
        return {
            'average_quality_score': 0.0,  # Placeholder
            'coordinate_coverage': 0.0,    # Placeholder
            'temporal_coverage': 0.0       # Placeholder
        }
    
    def _get_recommendations(self) -> List[str]:
        """Get recommendations for improving data collection."""
        recommendations = []
        
        if self.pipeline_state['failed_runs'] > self.pipeline_state['successful_runs']:
            recommendations.append("High failure rate detected. Check API connectivity and rate limits.")
        
        if self.pipeline_state['total_records_collected'] < 1000:
            recommendations.append("Low record count. Consider expanding search parameters.")
        
        return recommendations
    
    def _create_backup(self):
        """Create backup of collected data."""
        logger.info("Creating data backup")
        
        backup_dir = self.pipeline_config.output_dir / "backups"
        backup_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = backup_dir / f"backup_{timestamp}.tar.gz"
        
        # Create tar.gz backup
        import tarfile
        with tarfile.open(backup_path, "w:gz") as tar:
            tar.add(self.pipeline_config.output_dir / "raw_data", arcname="raw_data")
            tar.add(self.pipeline_config.output_dir / "reports", arcname="reports")
        
        logger.info(f"Backup created: {backup_path}")
    
    def _cleanup_old_data(self):
        """Clean up old data based on retention policy."""
        logger.info("Cleaning up old data")
        
        cutoff_date = datetime.now() - timedelta(days=self.pipeline_config.retention_days)
        
        # Clean up old data files
        data_dir = self.pipeline_config.output_dir / "raw_data"
        if data_dir.exists():
            for file_path in data_dir.glob("*.parquet"):
                if datetime.fromtimestamp(file_path.stat().st_mtime) < cutoff_date:
                    file_path.unlink()
                    logger.info(f"Deleted old file: {file_path}")
        
        # Clean up old reports
        reports_dir = self.pipeline_config.output_dir / "reports"
        if reports_dir.exists():
            for file_path in reports_dir.glob("*.json"):
                if datetime.fromtimestamp(file_path.stat().st_mtime) < cutoff_date:
                    file_path.unlink()
                    logger.info(f"Deleted old report: {file_path}")
    
    def _handle_pipeline_error(self, error: Exception):
        """Handle pipeline errors with logging and notifications."""
        error_msg = f"Pipeline error: {str(error)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        
        self.pipeline_state['failed_runs'] += 1
        self.pipeline_state['last_errors'].append(error_msg)
        
        if self.pipeline_config.enable_notifications:
            self._send_error_notification(error_msg)
    
    def _send_notification(self, report: Dict[str, Any]):
        """Send notification about collection status."""
        # Placeholder for notification implementation
        logger.info("Notification sent (placeholder)")
    
    def _send_error_notification(self, error_msg: str):
        """Send error notification."""
        # Placeholder for error notification implementation
        logger.error(f"Error notification sent: {error_msg}")


def create_pipeline_config(
    occurrence_schedule: str = "monthly",
    environmental_schedule: str = "on_demand",
    metagenomics_schedule: str = "quarterly",
    output_dir: str = "automated_truffle_data"
) -> PipelineConfig:
    """Create a pipeline configuration with custom parameters."""
    return PipelineConfig(
        occurrence_schedule=occurrence_schedule,
        environmental_schedule=environmental_schedule,
        metagenomics_schedule=metagenomics_schedule,
        output_dir=Path(output_dir)
    )


def start_automated_pipeline(
    research_config: ResearchConfig = None,
    pipeline_config: PipelineConfig = None,
    truffle_species: List[str] = None
):
    """Start the automated data collection pipeline."""
    if research_config is None:
        research_config = ResearchConfig(truffle_species=truffle_species)
    
    if pipeline_config is None:
        pipeline_config = create_pipeline_config()
    
    pipeline = AutomatedDataPipeline(pipeline_config, research_config)
    pipeline.start_pipeline()


if __name__ == "__main__":
    # Example usage
    research_config = ResearchConfig(
        truffle_species=['Tuber melanosporum', 'Tuber magnatum'],
        output_dir=Path("example_research")
    )
    
    pipeline_config = create_pipeline_config(
        occurrence_schedule="monthly",
        environmental_schedule="on_demand",
        metagenomics_schedule="quarterly"
    )
    
    start_automated_pipeline(research_config, pipeline_config)