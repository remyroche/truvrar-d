"""
Automated Data Fetcher for Truffle Cultivation Research
======================================================

This module provides automated data downloading from various internet sources
for truffle cultivation research. It includes:

1. Scheduled data fetching from multiple APIs
2. Real-time monitoring of new publications
3. Automated entity extraction and enrichment
4. Data quality assessment and validation
5. Integration with the knowledge graph

Usage:
    python -m etl.automated_data_fetcher --mode schedule
    python -m etl.automated_data_fetcher --mode monitor
    python -m etl.automated_data_fetcher --mode batch --query "truffle cultivation"
"""

import asyncio
import aiohttp
import logging
import schedule
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import json
import yaml
from pathlib import Path
import argparse
import signal
import sys
from concurrent.futures import ThreadPoolExecutor
import hashlib
import sqlite3
from contextlib import asynccontextmanager

# Import our data ingestion pipeline
from .data_ingestion import DataIngestionPipeline, PaperMetadata, PatentMetadata

logger = logging.getLogger(__name__)

@dataclass
class FetchConfig:
    """Configuration for automated data fetching."""
    # Data sources
    sources: List[str]  # ['openalex', 'crossref', 'pubmed', 'epo', 'wipo']
    
    # Queries to monitor
    queries: List[str]
    
    # Scheduling
    fetch_interval_hours: int = 24
    batch_size: int = 1000
    max_retries: int = 3
    
    # Quality filters
    min_confidence: float = 0.7
    min_relevance_score: float = 0.5
    
    # Storage
    output_dir: str = "data/raw"
    db_path: str = "data/fetch_history.db"
    
    # Rate limiting
    requests_per_minute: int = 60
    delay_between_requests: float = 1.0

class AutomatedDataFetcher:
    """Automated data fetcher with scheduling and monitoring capabilities."""
    
    def __init__(self, config: FetchConfig):
        self.config = config
        self.pipeline = DataIngestionPipeline()
        self.db_path = config.db_path
        self.running = False
        self.session = None
        
        # Initialize database
        self._init_database()
        
        # Rate limiting
        self.request_times = []
        self.semaphore = asyncio.Semaphore(config.requests_per_minute)
    
    def _init_database(self):
        """Initialize SQLite database for tracking fetch history."""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
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
                    entities_extracted BOOLEAN DEFAULT FALSE
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
                    entities_extracted BOOLEAN DEFAULT FALSE
                )
            """)
    
    async def start_monitoring(self):
        """Start real-time monitoring for new data."""
        logger.info("Starting real-time data monitoring...")
        self.running = True
        
        # Set up signal handlers for graceful shutdown
        def signal_handler(signum, frame):
            logger.info("Received shutdown signal, stopping monitoring...")
            self.running = False
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        try:
            while self.running:
                await self._monitor_cycle()
                await asyncio.sleep(300)  # Check every 5 minutes
        except KeyboardInterrupt:
            logger.info("Monitoring stopped by user")
        finally:
            await self._cleanup()
    
    async def _monitor_cycle(self):
        """Single monitoring cycle."""
        logger.info("Starting monitoring cycle...")
        
        for query in self.config.queries:
            if not self.running:
                break
                
            try:
                await self._fetch_and_process_query(query)
            except Exception as e:
                logger.error(f"Error processing query '{query}': {e}")
        
        logger.info("Monitoring cycle completed")
    
    async def _fetch_and_process_query(self, query: str):
        """Fetch and process data for a specific query."""
        logger.info(f"Fetching data for query: {query}")
        
        # Check if we've already fetched recently
        if self._recently_fetched(query):
            logger.info(f"Query '{query}' fetched recently, skipping")
            return
        
        # Fetch papers
        papers = await self._fetch_papers_with_retry(query)
        if papers:
            await self._store_papers(papers, query)
            logger.info(f"Stored {len(papers)} papers for query: {query}")
        
        # Fetch patents
        patents = await self._fetch_patents_with_retry(query)
        if patents:
            await self._store_patents(patents, query)
            logger.info(f"Stored {len(patents)} patents for query: {query}")
        
        # Update fetch history
        self._update_fetch_history(query, len(papers), len(patents))
    
    async def _fetch_papers_with_retry(self, query: str) -> List[PaperMetadata]:
        """Fetch papers with retry logic."""
        for attempt in range(self.config.max_retries):
            try:
                async with self.semaphore:
                    papers = await self.pipeline.fetch_papers(
                        query=query,
                        max_results=self.config.batch_size
                    )
                    return papers
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed for papers: {e}")
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                else:
                    raise
    
    async def _fetch_patents_with_retry(self, query: str) -> List[PatentMetadata]:
        """Fetch patents with retry logic."""
        for attempt in range(self.config.max_retries):
            try:
                async with self.semaphore:
                    patents = await self.pipeline.fetch_patents(
                        query=query,
                        max_results=self.config.batch_size
                    )
                    return patents
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed for patents: {e}")
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                else:
                    raise
    
    async def _store_papers(self, papers: List[PaperMetadata], query: str):
        """Store papers in database and filesystem."""
        with sqlite3.connect(self.db_path) as conn:
            for paper in papers:
                # Store in database
                conn.execute("""
                    INSERT OR REPLACE INTO fetched_papers 
                    (doi, title, source, fetch_time, quality_score)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    paper.doi,
                    paper.title,
                    paper.source,
                    paper.fetched_at,
                    self._calculate_quality_score(paper)
                ))
                
                # Store raw data
                output_path = Path(self.config.output_dir) / "papers" / f"{paper.doi.replace('/', '_')}.json"
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                with open(output_path, 'w') as f:
                    json.dump(paper.__dict__, f, default=str, indent=2)
    
    async def _store_patents(self, patents: List[PatentMetadata], query: str):
        """Store patents in database and filesystem."""
        with sqlite3.connect(self.db_path) as conn:
            for patent in patents:
                # Store in database
                conn.execute("""
                    INSERT OR REPLACE INTO fetched_patents 
                    (pub_number, title, source, fetch_time, quality_score)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    patent.pub_number,
                    patent.title,
                    patent.source,
                    patent.fetched_at,
                    self._calculate_quality_score(patent)
                ))
                
                # Store raw data
                output_path = Path(self.config.output_dir) / "patents" / f"{patent.pub_number}.json"
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                with open(output_path, 'w') as f:
                    json.dump(patent.__dict__, f, default=str, indent=2)
    
    def _calculate_quality_score(self, item) -> float:
        """Calculate quality score for a paper or patent."""
        score = 0.0
        
        # Title quality
        if hasattr(item, 'title') and item.title:
            score += 0.3 if len(item.title) > 20 else 0.1
        
        # Abstract quality
        if hasattr(item, 'abstract') and item.abstract:
            score += 0.4 if len(item.abstract) > 100 else 0.2
        
        # Author/Inventor quality
        if hasattr(item, 'authors') and item.authors:
            score += 0.2 if len(item.authors) > 0 else 0.0
        elif hasattr(item, 'inventors') and item.inventors:
            score += 0.2 if len(item.inventors) > 0 else 0.0
        
        # Year quality (recent is better)
        if hasattr(item, 'year') and item.year:
            current_year = datetime.now().year
            if item.year >= current_year - 5:
                score += 0.1
        
        return min(score, 1.0)
    
    def _recently_fetched(self, query: str) -> bool:
        """Check if query was fetched recently."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT COUNT(*) FROM fetch_history 
                WHERE query = ? AND fetch_time > datetime('now', '-1 hour')
            """, (query,))
            return cursor.fetchone()[0] > 0
    
    def _update_fetch_history(self, query: str, papers_count: int, patents_count: int):
        """Update fetch history in database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO fetch_history (source, query, papers_count, patents_count)
                VALUES (?, ?, ?, ?)
            """, ('automated', query, papers_count, patents_count))
    
    async def _cleanup(self):
        """Cleanup resources."""
        if self.session:
            await self.session.close()
        logger.info("Cleanup completed")
    
    def start_scheduled_fetching(self):
        """Start scheduled data fetching."""
        logger.info("Starting scheduled data fetching...")
        
        # Schedule jobs
        for query in self.config.queries:
            schedule.every(self.config.fetch_interval_hours).hours.do(
                self._scheduled_fetch, query
            )
        
        # Run scheduler
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
    
    def _scheduled_fetch(self, query: str):
        """Scheduled fetch function."""
        logger.info(f"Running scheduled fetch for query: {query}")
        
        # Run async fetch in new event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            loop.run_until_complete(self._fetch_and_process_query(query))
        finally:
            loop.close()
    
    async def batch_fetch(self, queries: List[str]):
        """Batch fetch data for multiple queries."""
        logger.info(f"Starting batch fetch for {len(queries)} queries...")
        
        tasks = []
        for query in queries:
            task = asyncio.create_task(self._fetch_and_process_query(query))
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        success_count = sum(1 for r in results if not isinstance(r, Exception))
        logger.info(f"Batch fetch completed: {success_count}/{len(queries)} successful")
        
        return results

def load_config(config_path: str) -> FetchConfig:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config_data = yaml.safe_load(f)
    
    return FetchConfig(**config_data)

def create_default_config(output_path: str):
    """Create default configuration file."""
    config = {
        'sources': ['openalex', 'crossref', 'pubmed', 'epo', 'wipo'],
        'queries': [
            '"Tuber melanosporum" AND mycorrhiz* AND hydroponic*',
            '"Tuber aestivum" AND cultivation AND controlled environment',
            'truffle AND mycorrhiza AND nutrient*',
            'ectomycorrhiza AND host tree AND inoculation',
            'truffle AND fruiting AND environmental control'
        ],
        'fetch_interval_hours': 24,
        'batch_size': 1000,
        'max_retries': 3,
        'min_confidence': 0.7,
        'min_relevance_score': 0.5,
        'output_dir': 'data/raw',
        'db_path': 'data/fetch_history.db',
        'requests_per_minute': 60,
        'delay_between_requests': 1.0
    }
    
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    
    logger.info(f"Default configuration created at: {output_path}")

async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Automated Data Fetcher for Truffle Research')
    parser.add_argument('--mode', choices=['monitor', 'schedule', 'batch'], 
                       default='monitor', help='Operation mode')
    parser.add_argument('--config', default='configs/data_fetching.yaml',
                       help='Configuration file path')
    parser.add_argument('--query', help='Single query for batch mode')
    parser.add_argument('--queries', nargs='+', help='Multiple queries for batch mode')
    parser.add_argument('--create-config', action='store_true',
                       help='Create default configuration file')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    if args.create_config:
        create_default_config(args.config)
        return
    
    # Load configuration
    if not Path(args.config).exists():
        logger.error(f"Configuration file not found: {args.config}")
        logger.info("Use --create-config to create a default configuration")
        return
    
    config = load_config(args.config)
    fetcher = AutomatedDataFetcher(config)
    
    if args.mode == 'monitor':
        await fetcher.start_monitoring()
    elif args.mode == 'schedule':
        fetcher.start_scheduled_fetching()
    elif args.mode == 'batch':
        if args.query:
            queries = [args.query]
        elif args.queries:
            queries = args.queries
        else:
            queries = config.queries
        
        await fetcher.batch_fetch(queries)

if __name__ == '__main__':
    asyncio.run(main())