#!/usr/bin/env python3
"""
Setup script for automated data fetching
========================================

This script sets up the automated data fetching system for truffle cultivation research.
It creates necessary directories, initializes the database, and sets up monitoring.

Usage:
    python scripts/setup_data_automation.py
    python scripts/setup_data_automation.py --create-config
    python scripts/setup_data_automation.py --test-connections
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path
import yaml
import sqlite3
import aiohttp

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from etl.automated_data_fetcher import AutomatedDataFetcher, FetchConfig, create_default_config

logger = logging.getLogger(__name__)

def setup_directories():
    """Create necessary directories for data storage."""
    directories = [
        "data/raw/papers",
        "data/raw/patents", 
        "data/processed",
        "data/exports",
        "logs",
        "configs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {directory}")

def setup_database():
    """Initialize the SQLite database for tracking fetch history."""
    db_path = "data/fetch_history.db"
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    
    with sqlite3.connect(db_path) as conn:
        # Create tables
        conn.execute("""
            CREATE TABLE IF NOT EXISTS fetch_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT NOT NULL,
                query TEXT NOT NULL,
                fetch_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                papers_count INTEGER DEFAULT 0,
                patents_count INTEGER DEFAULT 0,
                status TEXT DEFAULT 'success',
                error_message TEXT,
                UNIQUE(source, query, fetch_time)
            )
        """)
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS fetched_papers (
                doi TEXT PRIMARY KEY,
                title TEXT,
                source TEXT,
                fetch_time TIMESTAMP,
                processed BOOLEAN DEFAULT FALSE,
                quality_score REAL,
                entities_extracted BOOLEAN DEFAULT FALSE,
                kg_ingested BOOLEAN DEFAULT FALSE
            )
        """)
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS fetched_patents (
                pub_number TEXT PRIMARY KEY,
                title TEXT,
                source TEXT,
                fetch_time TIMESTAMP,
                processed BOOLEAN DEFAULT FALSE,
                quality_score REAL,
                entities_extracted BOOLEAN DEFAULT FALSE,
                kg_ingested BOOLEAN DEFAULT FALSE
            )
        """)
        
        # Create indexes for better performance
        conn.execute("CREATE INDEX IF NOT EXISTS idx_fetch_time ON fetch_history(fetch_time)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_papers_source ON fetched_papers(source)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_patents_source ON fetched_patents(source)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_papers_processed ON fetched_papers(processed)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_patents_processed ON fetched_patents(processed)")
    
    logger.info(f"Database initialized at: {db_path}")

async def test_api_connections():
    """Test connections to various APIs."""
    logger.info("Testing API connections...")
    
    apis = {
        "OpenAlex": "https://api.openalex.org/works?search=truffle&per-page=1",
        "Crossref": "https://api.crossref.org/works?query=truffle&rows=1",
        "PubMed": "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term=truffle&retmax=1",
        "EPO OPS": "https://ops.epo.org/3.2/rest-services/published-data/search?q=truffle&range=1-1",
        "WIPO": "https://patentscope.wipo.int/search/api/search?q=truffle&start=0&rows=1"
    }
    
    async with aiohttp.ClientSession() as session:
        for name, url in apis.items():
            try:
                async with session.get(url, timeout=10) as response:
                    if response.status == 200:
                        logger.info(f"✅ {name}: Connection successful")
                    else:
                        logger.warning(f"⚠️  {name}: HTTP {response.status}")
            except Exception as e:
                logger.error(f"❌ {name}: Connection failed - {e}")

def create_systemd_service():
    """Create a systemd service file for running the data fetcher as a service."""
    service_content = """[Unit]
Description=Truffle Cultivation Data Fetcher
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/workspace/truffle_cultivation_research
ExecStart=/usr/bin/python3 -m etl.automated_data_fetcher --mode monitor --config configs/data_fetching.yaml
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
"""
    
    service_path = Path("truffle-data-fetcher.service")
    with open(service_path, 'w') as f:
        f.write(service_content)
    
    logger.info(f"Systemd service file created: {service_path}")
    logger.info("To install: sudo cp truffle-data-fetcher.service /etc/systemd/system/")
    logger.info("To enable: sudo systemctl enable truffle-data-fetcher")
    logger.info("To start: sudo systemctl start truffle-data-fetcher")

def create_cron_job():
    """Create a cron job for scheduled data fetching."""
    cron_content = """# Truffle Cultivation Data Fetcher
# Run every 6 hours
0 */6 * * * cd /workspace/truffle_cultivation_research && python3 -m etl.automated_data_fetcher --mode batch --config configs/data_fetching.yaml >> logs/cron.log 2>&1

# Run entity extraction every 12 hours
0 */12 * * * cd /workspace/truffle_cultivation_research && python3 scripts/process_fetched_data.py >> logs/entity_extraction.log 2>&1
"""
    
    cron_path = Path("truffle-data-fetcher.cron")
    with open(cron_path, 'w') as f:
        f.write(cron_content)
    
    logger.info(f"Cron job file created: {cron_path}")
    logger.info("To install: crontab truffle-data-fetcher.cron")

def create_docker_compose_override():
    """Create a Docker Compose override for data fetching services."""
    override_content = """version: '3.8'

services:
  data-fetcher:
    build: .
    command: python -m etl.automated_data_fetcher --mode monitor --config configs/data_fetching.yaml
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
      - ./configs:/app/configs
    depends_on:
      - neo4j
      - graphdb
    environment:
      - PYTHONPATH=/app
    restart: unless-stopped
    
  data-processor:
    build: .
    command: python scripts/process_fetched_data.py --mode continuous
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
    depends_on:
      - neo4j
      - graphdb
    environment:
      - PYTHONPATH=/app
    restart: unless-stopped
"""
    
    override_path = Path("docker-compose.override.yml")
    with open(override_path, 'w') as f:
        f.write(override_content)
    
    logger.info(f"Docker Compose override created: {override_path}")

def create_monitoring_script():
    """Create a monitoring script for the data fetcher."""
    monitoring_script = """#!/bin/bash
# Monitoring script for truffle data fetcher

LOG_FILE="logs/monitoring.log"
DB_PATH="data/fetch_history.db"

echo "$(date): Starting monitoring check" >> $LOG_FILE

# Check if database exists
if [ ! -f "$DB_PATH" ]; then
    echo "$(date): ERROR - Database not found at $DB_PATH" >> $LOG_FILE
    exit 1
fi

# Check recent fetch activity
RECENT_FETCHES=$(sqlite3 $DB_PATH "SELECT COUNT(*) FROM fetch_history WHERE fetch_time > datetime('now', '-24 hours')")

if [ "$RECENT_FETCHES" -eq 0 ]; then
    echo "$(date): WARNING - No fetches in last 24 hours" >> $LOG_FILE
    # Could send alert here
fi

# Check for errors
ERROR_COUNT=$(sqlite3 $DB_PATH "SELECT COUNT(*) FROM fetch_history WHERE status = 'error' AND fetch_time > datetime('now', '-24 hours')")

if [ "$ERROR_COUNT" -gt 0 ]; then
    echo "$(date): WARNING - $ERROR_COUNT errors in last 24 hours" >> $LOG_FILE
    # Could send alert here
fi

echo "$(date): Monitoring check completed" >> $LOG_FILE
"""
    
    monitoring_path = Path("scripts/monitor_data_fetcher.sh")
    monitoring_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(monitoring_path, 'w') as f:
        f.write(monitoring_script)
    
    monitoring_path.chmod(0o755)  # Make executable
    logger.info(f"Monitoring script created: {monitoring_path}")

def main():
    """Main setup function."""
    parser = argparse.ArgumentParser(description='Setup automated data fetching')
    parser.add_argument('--create-config', action='store_true',
                       help='Create default configuration file')
    parser.add_argument('--test-connections', action='store_true',
                       help='Test API connections')
    parser.add_argument('--create-services', action='store_true',
                       help='Create systemd service and cron job files')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info("Setting up automated data fetching system...")
    
    # Create directories
    setup_directories()
    
    # Setup database
    setup_database()
    
    # Create configuration if requested
    if args.create_config:
        create_default_config("configs/data_fetching.yaml")
        logger.info("Default configuration created")
    
    # Test connections if requested
    if args.test_connections:
        asyncio.run(test_api_connections())
    
    # Create service files if requested
    if args.create_services:
        create_systemd_service()
        create_cron_job()
        create_docker_compose_override()
        create_monitoring_script()
        logger.info("Service files created")
    
    logger.info("Setup completed successfully!")
    logger.info("\nNext steps:")
    logger.info("1. Review and edit configs/data_fetching.yaml")
    logger.info("2. Test with: python -m etl.automated_data_fetcher --mode batch --query 'truffle'")
    logger.info("3. Start monitoring: python -m etl.automated_data_fetcher --mode monitor")
    logger.info("4. Or use Docker: docker-compose up -d data-fetcher")

if __name__ == '__main__':
    main()