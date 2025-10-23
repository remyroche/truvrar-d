"""
Automated data collection scheduler for regular updates.
"""
import logging
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
import schedule
import time
from pathlib import Path
import json
import pandas as pd

from ..data_processing.habitat_processor import HabitatProcessor
from ..models.habitat_model import HabitatModel
from ..visualization.mapping_tools import MappingTools

logger = logging.getLogger(__name__)


class DataCollectionScheduler:
    """Scheduler for automated data collection and processing."""
    
    def __init__(self, config: Dict[str, Any], data_dir: Path, output_dir: Path):
        self.config = config
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.processor = HabitatProcessor(config, data_dir)
        self.model = HabitatModel(config, data_dir / "models")
        self.mapper = MappingTools()
        
        # Scheduler state
        self.is_running = False
        self.last_run = None
        self.run_count = 0
        self.errors = []
        
    def schedule_daily_collection(self, species: List[str], 
                                countries: Optional[List[str]] = None,
                                hour: int = 2, minute: int = 0):
        """Schedule daily data collection at specified time."""
        schedule.every().day.at(f"{hour:02d}:{minute:02d}").do(
            self._run_collection_job, species, countries
        )
        logger.info(f"Scheduled daily collection at {hour:02d}:{minute:02d}")
        
    def schedule_weekly_collection(self, species: List[str], 
                                 countries: Optional[List[str]] = None,
                                 day: str = 'monday', hour: int = 3, minute: int = 0):
        """Schedule weekly data collection."""
        getattr(schedule.every(), day).at(f"{hour:02d}:{minute:02d}").do(
            self._run_collection_job, species, countries
        )
        logger.info(f"Scheduled weekly collection on {day} at {hour:02d}:{minute:02d}")
        
    def schedule_monthly_collection(self, species: List[str], 
                                  countries: Optional[List[str]] = None,
                                  day: int = 1, hour: int = 4, minute: int = 0):
        """Schedule monthly data collection."""
        schedule.every().month.do(
            self._run_collection_job, species, countries
        )
        logger.info(f"Scheduled monthly collection on day {day} at {hour:02d}:{minute:02d}")
        
    def schedule_custom_interval(self, species: List[str], 
                               countries: Optional[List[str]] = None,
                               interval_hours: int = 6):
        """Schedule collection at custom intervals."""
        schedule.every(interval_hours).hours.do(
            self._run_collection_job, species, countries
        )
        logger.info(f"Scheduled collection every {interval_hours} hours")
        
    def _run_collection_job(self, species: List[str], 
                          countries: Optional[List[str]] = None):
        """Run the data collection job."""
        try:
            logger.info(f"Starting scheduled data collection job #{self.run_count + 1}")
            start_time = datetime.now()
            
            # Collect data
            data = self.processor.collect_all_data(
                species=species,
                countries=countries,
                year_from=datetime.now().year - 1,  # Last year
                year_to=datetime.now().year
            )
            
            if data.empty:
                logger.warning("No new data collected in scheduled run")
                return
                
            # Save data with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            data_path = self.output_dir / f"scheduled_data_{timestamp}.csv"
            data.to_csv(data_path, index=False)
            
            # Update model if enough new data
            if len(data) >= 10:  # Minimum threshold for model update
                self._update_models(data)
                
            # Generate updated visualizations
            self._update_visualizations(data, timestamp)
            
            # Update state
            self.last_run = start_time
            self.run_count += 1
            
            # Save run log
            self._save_run_log(timestamp, len(data), "success")
            
            logger.info(f"Scheduled collection completed: {len(data)} records")
            
        except Exception as e:
            error_msg = f"Error in scheduled collection: {e}"
            logger.error(error_msg)
            self.errors.append(error_msg)
            self._save_run_log(timestamp, 0, "error", str(e))
            
    def _update_models(self, data: pd.DataFrame):
        """Update machine learning models with new data."""
        try:
            logger.info("Updating models with new data")
            
            # Load existing data if available
            existing_data_path = self.output_dir / "latest_data.csv"
            if existing_data_path.exists():
                existing_data = pd.read_csv(existing_data_path)
                combined_data = pd.concat([existing_data, data], ignore_index=True)
            else:
                combined_data = data
                
            # Train models
            training_results = self.model.train_models(combined_data)
            
            # Save updated data
            combined_data.to_csv(existing_data_path, index=False)
            
            # Save model update log
            update_log = {
                'timestamp': datetime.now().isoformat(),
                'new_records': len(data),
                'total_records': len(combined_data),
                'training_results': training_results
            }
            
            log_path = self.output_dir / "model_updates.json"
            if log_path.exists():
                with open(log_path, 'r') as f:
                    logs = json.load(f)
            else:
                logs = []
                
            logs.append(update_log)
            
            with open(log_path, 'w') as f:
                json.dump(logs, f, indent=2)
                
            logger.info("Models updated successfully")
            
        except Exception as e:
            logger.error(f"Error updating models: {e}")
            
    def _update_visualizations(self, data: pd.DataFrame, timestamp: str):
        """Update visualizations with new data."""
        try:
            logger.info("Updating visualizations")
            
            # Create timestamped output directory
            viz_dir = self.output_dir / f"visualizations_{timestamp}"
            viz_dir.mkdir(exist_ok=True)
            
            # Species distribution map
            map_path = viz_dir / "species_distribution_map.html"
            self.mapper.create_species_distribution_map(data, save_path=map_path)
            
            # Environmental correlation heatmap
            heatmap_path = viz_dir / "correlation_heatmap.png"
            self.mapper.create_correlation_heatmap(data, save_path=heatmap_path)
            
            logger.info(f"Visualizations updated in {viz_dir}")
            
        except Exception as e:
            logger.error(f"Error updating visualizations: {e}")
            
    def _save_run_log(self, timestamp: str, record_count: int, 
                     status: str, error: Optional[str] = None):
        """Save run log for monitoring."""
        log_entry = {
            'timestamp': timestamp,
            'record_count': record_count,
            'status': status,
            'error': error
        }
        
        log_path = self.output_dir / "run_logs.json"
        if log_path.exists():
            with open(log_path, 'r') as f:
                logs = json.load(f)
        else:
            logs = []
            
        logs.append(log_entry)
        
        # Keep only last 100 runs
        if len(logs) > 100:
            logs = logs[-100:]
            
        with open(log_path, 'w') as f:
            json.dump(logs, f, indent=2)
            
    def start_scheduler(self):
        """Start the scheduler."""
        self.is_running = True
        logger.info("Starting data collection scheduler")
        
        while self.is_running:
            try:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
            except KeyboardInterrupt:
                logger.info("Scheduler stopped by user")
                break
            except Exception as e:
                logger.error(f"Scheduler error: {e}")
                time.sleep(300)  # Wait 5 minutes before retrying
                
    def stop_scheduler(self):
        """Stop the scheduler."""
        self.is_running = False
        logger.info("Data collection scheduler stopped")
        
    def get_status(self) -> Dict[str, Any]:
        """Get scheduler status."""
        return {
            'is_running': self.is_running,
            'last_run': self.last_run.isoformat() if self.last_run else None,
            'run_count': self.run_count,
            'error_count': len(self.errors),
            'recent_errors': self.errors[-5:] if self.errors else []
        }
        
    def clear_schedule(self):
        """Clear all scheduled jobs."""
        schedule.clear()
        logger.info("All scheduled jobs cleared")
        
    def list_scheduled_jobs(self) -> List[str]:
        """List all scheduled jobs."""
        jobs = []
        for job in schedule.jobs:
            jobs.append(str(job))
        return jobs