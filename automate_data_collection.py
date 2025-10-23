#!/usr/bin/env python3
"""
Automated Data Collection Script for Global Truffle Habitat Atlas

This script provides various automation options for downloading data from
internet-based sources on a scheduled basis.
"""
import logging
import argparse
import time
import signal
import sys
from pathlib import Path
from datetime import datetime, timedelta
import json

from config import *
from src.automation.scheduler import DataCollectionScheduler
from src.automation.pipeline import AutomatedPipeline
from src.automation.monitoring import DataQualityMonitor

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.FileHandler(LOGS_DIR / 'automation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class DataCollectionAutomation:
    """Main automation controller."""
    
    def __init__(self, config_file: Path = None):
        # Load configuration
        if config_file and config_file.exists():
            with open(config_file, 'r') as f:
                self.config = json.load(f)
        else:
            self.config = {
                'gbif': API_CONFIG['gbif'],
                'inaturalist': API_CONFIG['inaturalist'],
                'soilgrids': API_CONFIG['soilgrids'],
                'worldclim': API_CONFIG['worldclim'],
                **MODEL_CONFIG
            }
            
        # Initialize components
        self.scheduler = DataCollectionScheduler(self.config, DATA_DIR, OUTPUTS_DIR)
        self.pipeline = AutomatedPipeline(self.config, DATA_DIR, OUTPUTS_DIR)
        self.monitor = DataQualityMonitor(self.config, OUTPUTS_DIR)
        
        # Automation state
        self.running = False
        
    def run_scheduled_collection(self, species: list, countries: list = None, 
                               schedule_type: str = 'daily', **kwargs):
        """Run scheduled data collection."""
        logger.info(f"Setting up {schedule_type} data collection")
        
        if schedule_type == 'daily':
            hour = kwargs.get('hour', 2)
            minute = kwargs.get('minute', 0)
            self.scheduler.schedule_daily_collection(species, countries, hour, minute)
            
        elif schedule_type == 'weekly':
            day = kwargs.get('day', 'monday')
            hour = kwargs.get('hour', 3)
            minute = kwargs.get('minute', 0)
            self.scheduler.schedule_weekly_collection(species, countries, day, hour, minute)
            
        elif schedule_type == 'monthly':
            day = kwargs.get('day', 1)
            hour = kwargs.get('hour', 4)
            minute = kwargs.get('minute', 0)
            self.scheduler.schedule_monthly_collection(species, countries, day, hour, minute)
            
        elif schedule_type == 'custom':
            interval_hours = kwargs.get('interval_hours', 6)
            self.scheduler.schedule_custom_interval(species, countries, interval_hours)
            
        else:
            raise ValueError(f"Unknown schedule type: {schedule_type}")
            
        # Start scheduler
        self.running = True
        logger.info("Starting scheduler...")
        
        try:
            self.scheduler.start_scheduler()
        except KeyboardInterrupt:
            logger.info("Scheduler stopped by user")
        finally:
            self.running = False
            
    def run_one_time_collection(self, species: list, countries: list = None, 
                              parallel: bool = True, monitor_quality: bool = True):
        """Run one-time data collection and processing."""
        logger.info("Starting one-time data collection")
        
        try:
            # Run full pipeline
            results = self.pipeline.run_full_pipeline(
                species=species,
                countries=countries,
                parallel=parallel
            )
            
            if results['status'] == 'completed':
                logger.info("One-time collection completed successfully")
                
                # Monitor data quality if requested
                if monitor_quality:
                    self._monitor_quality(results)
                    
                return results
            else:
                logger.error(f"One-time collection failed: {results.get('errors', [])}")
                return results
                
        except Exception as e:
            logger.error(f"Error in one-time collection: {e}")
            return {'status': 'failed', 'error': str(e)}
            
    def run_incremental_update(self, species: list, countries: list = None):
        """Run incremental update with only new data."""
        logger.info("Starting incremental update")
        
        try:
            results = self.pipeline.run_incremental_update(species, countries)
            
            if results['status'] == 'completed':
                logger.info(f"Incremental update completed: {results.get('new_records', 0)} new records")
            else:
                logger.info(f"Incremental update: {results.get('status', 'unknown')}")
                
            return results
            
        except Exception as e:
            logger.error(f"Error in incremental update: {e}")
            return {'status': 'failed', 'error': str(e)}
            
    def _monitor_quality(self, results: dict):
        """Monitor data quality for collected data."""
        try:
            # Load the collected data
            data_path = OUTPUTS_DIR / "latest_data.csv"
            if not data_path.exists():
                logger.warning("No data file found for quality monitoring")
                return
                
            import pandas as pd
            data = pd.read_csv(data_path)
            
            # Monitor quality
            alerts = self.monitor.monitor_data_quality(data, "automated_collection")
            
            # Log quality score
            quality_score = self.monitor.get_quality_score(data)
            logger.info(f"Data quality score: {quality_score:.2f}")
            
            # Log alerts
            if alerts:
                alert_summary = self.monitor.get_alert_summary()
                logger.info(f"Quality alerts: {alert_summary['total_alerts']} total")
                
                for level, count in alert_summary['by_level'].items():
                    if count > 0:
                        logger.warning(f"  {level.upper()}: {count} alerts")
                        
        except Exception as e:
            logger.error(f"Error in quality monitoring: {e}")
            
    def get_status(self) -> dict:
        """Get automation status."""
        return {
            'scheduler': self.scheduler.get_status(),
            'pipeline': self.pipeline.get_pipeline_status(),
            'running': self.running
        }
        
    def stop(self):
        """Stop automation."""
        if self.running:
            self.scheduler.stop_scheduler()
            self.running = False
            logger.info("Automation stopped")


def signal_handler(signum, frame):
    """Handle shutdown signals."""
    logger.info("Received shutdown signal")
    if 'automation' in globals():
        automation.stop()
    sys.exit(0)


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description='Automated Data Collection for GTHA')
    
    # Mode selection
    parser.add_argument('--mode', choices=['scheduled', 'one-time', 'incremental'], 
                       required=True, help='Automation mode')
    
    # Species and location
    parser.add_argument('--species', nargs='+', default=TRUFFLE_SPECIES[:3],
                       help='Truffle species to collect data for')
    parser.add_argument('--countries', nargs='+',
                       help='Country codes to filter by')
    
    # Scheduling options
    parser.add_argument('--schedule-type', choices=['daily', 'weekly', 'monthly', 'custom'],
                       default='daily', help='Schedule type for scheduled mode')
    parser.add_argument('--hour', type=int, default=2,
                       help='Hour for scheduled collection (0-23)')
    parser.add_argument('--minute', type=int, default=0,
                       help='Minute for scheduled collection (0-59)')
    parser.add_argument('--day', default='monday',
                       help='Day for weekly/monthly collection')
    parser.add_argument('--interval-hours', type=int, default=6,
                       help='Interval in hours for custom scheduling')
    
    # Processing options
    parser.add_argument('--parallel', action='store_true', default=True,
                       help='Use parallel processing')
    parser.add_argument('--no-quality-monitoring', action='store_true',
                       help='Disable data quality monitoring')
    
    # Configuration
    parser.add_argument('--config-file', type=Path,
                       help='Custom configuration file')
    parser.add_argument('--output-dir', type=Path, default=OUTPUTS_DIR,
                       help='Output directory')
    
    args = parser.parse_args()
    
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Initialize automation
    global automation
    automation = DataCollectionAutomation(args.config_file)
    
    try:
        if args.mode == 'scheduled':
            # Run scheduled collection
            automation.run_scheduled_collection(
                species=args.species,
                countries=args.countries,
                schedule_type=args.schedule_type,
                hour=args.hour,
                minute=args.minute,
                day=args.day,
                interval_hours=args.interval_hours
            )
            
        elif args.mode == 'one-time':
            # Run one-time collection
            results = automation.run_one_time_collection(
                species=args.species,
                countries=args.countries,
                parallel=args.parallel,
                monitor_quality=not args.no_quality_monitoring
            )
            
            # Print results
            print(f"\n📊 Collection Results:")
            print(f"Status: {results['status']}")
            if 'total_duration' in results:
                print(f"Duration: {results['total_duration']:.1f}s")
            if 'errors' in results and results['errors']:
                print(f"Errors: {len(results['errors'])}")
                
        elif args.mode == 'incremental':
            # Run incremental update
            results = automation.run_incremental_update(
                species=args.species,
                countries=args.countries
            )
            
            print(f"\n📊 Incremental Update Results:")
            print(f"Status: {results['status']}")
            if 'new_records' in results:
                print(f"New records: {results['new_records']}")
            if 'total_records' in results:
                print(f"Total records: {results['total_records']}")
                
    except KeyboardInterrupt:
        logger.info("Automation interrupted by user")
    except Exception as e:
        logger.error(f"Automation failed: {e}")
        sys.exit(1)
    finally:
        if 'automation' in globals():
            automation.stop()


if __name__ == "__main__":
    main()