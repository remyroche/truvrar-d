#!/usr/bin/env python3
"""
Process Fetched Data
===================

This script processes the raw data fetched by the automated data fetcher.
It performs entity extraction, quality assessment, and ingestion into the knowledge graph.

Usage:
    python scripts/process_fetched_data.py --mode batch
    python scripts/process_fetched_data.py --mode continuous
    python scripts/process_fetched_data.py --mode single --doi "10.1000/example"
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path
import sqlite3
import json
from typing import List, Dict, Any
from datetime import datetime, timedelta

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from etl.data_ingestion import DataIngestionPipeline
from knowledge_graph.neo4j.ingestion import KnowledgeGraphIngester

logger = logging.getLogger(__name__)

class DataProcessor:
    """Processes fetched data and ingests it into the knowledge graph."""
    
    def __init__(self, db_path: str = "data/fetch_history.db"):
        self.db_path = db_path
        self.pipeline = DataIngestionPipeline()
        self.kg_ingester = KnowledgeGraphIngester()
        
    async def process_unprocessed_papers(self, limit: int = 100) -> int:
        """Process unprocessed papers."""
        logger.info(f"Processing up to {limit} unprocessed papers...")
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT doi, title, source, fetch_time
                FROM fetched_papers 
                WHERE processed = FALSE
                ORDER BY fetch_time ASC
                LIMIT ?
            """, (limit,))
            
            papers = cursor.fetchall()
        
        processed_count = 0
        
        for doi, title, source, fetch_time in papers:
            try:
                await self._process_single_paper(doi, title, source, fetch_time)
                processed_count += 1
                
                # Update database
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute("""
                        UPDATE fetched_papers 
                        SET processed = TRUE, entities_extracted = TRUE
                        WHERE doi = ?
                    """, (doi,))
                
                logger.info(f"Processed paper: {doi}")
                
            except Exception as e:
                logger.error(f"Error processing paper {doi}: {e}")
                
                # Mark as processed even if failed to avoid infinite retry
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute("""
                        UPDATE fetched_papers 
                        SET processed = TRUE
                        WHERE doi = ?
                    """, (doi,))
        
        logger.info(f"Processed {processed_count} papers")
        return processed_count
    
    async def process_unprocessed_patents(self, limit: int = 100) -> int:
        """Process unprocessed patents."""
        logger.info(f"Processing up to {limit} unprocessed patents...")
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT pub_number, title, source, fetch_time
                FROM fetched_patents 
                WHERE processed = FALSE
                ORDER BY fetch_time ASC
                LIMIT ?
            """, (limit,))
            
            patents = cursor.fetchall()
        
        processed_count = 0
        
        for pub_number, title, source, fetch_time in patents:
            try:
                await self._process_single_patent(pub_number, title, source, fetch_time)
                processed_count += 1
                
                # Update database
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute("""
                        UPDATE fetched_patents 
                        SET processed = TRUE, entities_extracted = TRUE
                        WHERE pub_number = ?
                    """, (pub_number,))
                
                logger.info(f"Processed patent: {pub_number}")
                
            except Exception as e:
                logger.error(f"Error processing patent {pub_number}: {e}")
                
                # Mark as processed even if failed
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute("""
                        UPDATE fetched_patents 
                        SET processed = TRUE
                        WHERE pub_number = ?
                    """, (pub_number,))
        
        logger.info(f"Processed {processed_count} patents")
        return processed_count
    
    async def _process_single_paper(self, doi: str, title: str, source: str, fetch_time: str):
        """Process a single paper."""
        # Load paper data
        paper_path = Path("data/raw/papers") / f"{doi.replace('/', '_')}.json"
        
        if not paper_path.exists():
            logger.warning(f"Paper file not found: {paper_path}")
            return
        
        with open(paper_path, 'r') as f:
            paper_data = json.load(f)
        
        # Extract entities
        entities = await self._extract_entities_from_paper(paper_data)
        
        # Ingest into knowledge graph
        await self._ingest_paper_into_kg(paper_data, entities)
        
        # Update quality score
        quality_score = self._calculate_paper_quality(paper_data, entities)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE fetched_papers 
                SET quality_score = ?, entities_extracted = TRUE
                WHERE doi = ?
            """, (quality_score, doi))
    
    async def _process_single_patent(self, pub_number: str, title: str, source: str, fetch_time: str):
        """Process a single patent."""
        # Load patent data
        patent_path = Path("data/raw/patents") / f"{pub_number}.json"
        
        if not patent_path.exists():
            logger.warning(f"Patent file not found: {patent_path}")
            return
        
        with open(patent_path, 'r') as f:
            patent_data = json.load(f)
        
        # Extract entities
        entities = await self._extract_entities_from_patent(patent_data)
        
        # Ingest into knowledge graph
        await self._ingest_patent_into_kg(patent_data, entities)
        
        # Update quality score
        quality_score = self._calculate_patent_quality(patent_data, entities)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE fetched_patents 
                SET quality_score = ?, entities_extracted = TRUE
                WHERE pub_number = ?
            """, (quality_score, pub_number))
    
    async def _extract_entities_from_paper(self, paper_data: Dict[str, Any]) -> Dict[str, List[str]]:
        """Extract entities from paper data."""
        # This would use spaCy, SciBERT, or other NLP tools
        # For now, return placeholder
        return {
            'fungi_species': [],
            'host_trees': [],
            'nutrients': [],
            'pgr': [],
            'control_terms': []
        }
    
    async def _extract_entities_from_patent(self, patent_data: Dict[str, Any]) -> Dict[str, List[str]]:
        """Extract entities from patent data."""
        # This would use spaCy, SciBERT, or other NLP tools
        # For now, return placeholder
        return {
            'fungi_species': [],
            'host_trees': [],
            'nutrients': [],
            'pgr': [],
            'control_terms': []
        }
    
    async def _ingest_paper_into_kg(self, paper_data: Dict[str, Any], entities: Dict[str, List[str]]):
        """Ingest paper data into knowledge graph."""
        # This would use the KnowledgeGraphIngester
        # For now, just log
        logger.info(f"Ingesting paper into KG: {paper_data.get('doi', 'unknown')}")
    
    async def _ingest_patent_into_kg(self, patent_data: Dict[str, Any], entities: Dict[str, List[str]]):
        """Ingest patent data into knowledge graph."""
        # This would use the KnowledgeGraphIngester
        # For now, just log
        logger.info(f"Ingesting patent into KG: {patent_data.get('pub_number', 'unknown')}")
    
    def _calculate_paper_quality(self, paper_data: Dict[str, Any], entities: Dict[str, List[str]]) -> float:
        """Calculate quality score for a paper."""
        score = 0.0
        
        # Title quality
        if paper_data.get('title'):
            score += 0.3 if len(paper_data['title']) > 20 else 0.1
        
        # Abstract quality
        if paper_data.get('abstract'):
            score += 0.4 if len(paper_data['abstract']) > 100 else 0.2
        
        # Author quality
        if paper_data.get('authors'):
            score += 0.2 if len(paper_data['authors']) > 0 else 0.0
        
        # Entity extraction quality
        total_entities = sum(len(entities[key]) for key in entities)
        if total_entities > 0:
            score += 0.1
        
        return min(score, 1.0)
    
    def _calculate_patent_quality(self, patent_data: Dict[str, Any], entities: Dict[str, List[str]]) -> float:
        """Calculate quality score for a patent."""
        score = 0.0
        
        # Title quality
        if patent_data.get('title'):
            score += 0.3 if len(patent_data['title']) > 20 else 0.1
        
        # Abstract quality
        if patent_data.get('abstract'):
            score += 0.4 if len(patent_data['abstract']) > 100 else 0.2
        
        # Inventor quality
        if patent_data.get('inventors'):
            score += 0.2 if len(patent_data['inventors']) > 0 else 0.0
        
        # Entity extraction quality
        total_entities = sum(len(entities[key]) for key in entities)
        if total_entities > 0:
            score += 0.1
        
        return min(score, 1.0)
    
    async def continuous_processing(self, interval_minutes: int = 30):
        """Continuously process new data."""
        logger.info(f"Starting continuous processing (interval: {interval_minutes} minutes)")
        
        while True:
            try:
                # Process papers
                papers_processed = await self.process_unprocessed_papers(limit=50)
                
                # Process patents
                patents_processed = await self.process_unprocessed_patents(limit=50)
                
                if papers_processed > 0 or patents_processed > 0:
                    logger.info(f"Processed {papers_processed} papers and {patents_processed} patents")
                else:
                    logger.debug("No new data to process")
                
                # Wait for next interval
                await asyncio.sleep(interval_minutes * 60)
                
            except KeyboardInterrupt:
                logger.info("Continuous processing stopped by user")
                break
            except Exception as e:
                logger.error(f"Error in continuous processing: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retry
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        with sqlite3.connect(self.db_path) as conn:
            # Paper stats
            cursor = conn.execute("""
                SELECT 
                    COUNT(*) as total,
                    SUM(CASE WHEN processed = TRUE THEN 1 ELSE 0 END) as processed,
                    SUM(CASE WHEN entities_extracted = TRUE THEN 1 ELSE 0 END) as entities_extracted,
                    AVG(quality_score) as avg_quality
                FROM fetched_papers
            """)
            paper_stats = cursor.fetchone()
            
            # Patent stats
            cursor = conn.execute("""
                SELECT 
                    COUNT(*) as total,
                    SUM(CASE WHEN processed = TRUE THEN 1 ELSE 0 END) as processed,
                    SUM(CASE WHEN entities_extracted = TRUE THEN 1 ELSE 0 END) as entities_extracted,
                    AVG(quality_score) as avg_quality
                FROM fetched_patents
            """)
            patent_stats = cursor.fetchone()
            
            # Recent activity
            cursor = conn.execute("""
                SELECT COUNT(*) FROM fetch_history 
                WHERE fetch_time > datetime('now', '-24 hours')
            """)
            recent_fetches = cursor.fetchone()[0]
        
        return {
            'papers': {
                'total': paper_stats[0],
                'processed': paper_stats[1],
                'entities_extracted': paper_stats[2],
                'avg_quality': paper_stats[3] or 0.0
            },
            'patents': {
                'total': patent_stats[0],
                'processed': patent_stats[1],
                'entities_extracted': patent_stats[2],
                'avg_quality': patent_stats[3] or 0.0
            },
            'recent_fetches_24h': recent_fetches
        }

async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Process fetched data')
    parser.add_argument('--mode', choices=['batch', 'continuous', 'single'], 
                       default='batch', help='Processing mode')
    parser.add_argument('--doi', help='DOI for single paper processing')
    parser.add_argument('--pub-number', help='Publication number for single patent processing')
    parser.add_argument('--limit', type=int, default=100, help='Batch size limit')
    parser.add_argument('--interval', type=int, default=30, help='Interval for continuous mode (minutes)')
    parser.add_argument('--stats', action='store_true', help='Show processing statistics')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    processor = DataProcessor()
    
    if args.stats:
        stats = processor.get_processing_stats()
        print("\nProcessing Statistics:")
        print("=" * 50)
        print(f"Papers: {stats['papers']['processed']}/{stats['papers']['total']} processed")
        print(f"Patents: {stats['patents']['processed']}/{stats['patents']['total']} processed")
        print(f"Recent fetches (24h): {stats['recent_fetches_24h']}")
        print(f"Average paper quality: {stats['papers']['avg_quality']:.2f}")
        print(f"Average patent quality: {stats['patents']['avg_quality']:.2f}")
        return
    
    if args.mode == 'batch':
        papers_processed = await processor.process_unprocessed_papers(args.limit)
        patents_processed = await processor.process_unprocessed_patents(args.limit)
        logger.info(f"Batch processing completed: {papers_processed} papers, {patents_processed} patents")
        
    elif args.mode == 'continuous':
        await processor.continuous_processing(args.interval)
        
    elif args.mode == 'single':
        if args.doi:
            # Process single paper
            logger.info(f"Processing single paper: {args.doi}")
            # Implementation would go here
        elif args.pub_number:
            # Process single patent
            logger.info(f"Processing single patent: {args.pub_number}")
            # Implementation would go here
        else:
            logger.error("Must specify --doi or --pub-number for single mode")

if __name__ == '__main__':
    asyncio.run(main())