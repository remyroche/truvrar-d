"""
Main data collection pipeline for academic research extraction.
"""

import asyncio
import yaml
from pathlib import Path
from typing import List, Dict, Any, Optional, Set
from datetime import datetime
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from loguru import logger
import hashlib
import uuid

from .data_sources import (
    OpenAlexConnector,
    CrossrefConnector,
    EuropePMCConnector,
    SemanticScholarConnector,
    PubMedConnector,
    ArxivConnector,
    DOAJConnector
)
from .data_schema import PaperMetadata, Manifest


class DataCollectionPipeline:
    """Main pipeline for collecting academic papers from multiple sources."""
    
    def __init__(self, config_path: str):
        """Initialize pipeline with configuration."""
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.connectors = self._initialize_connectors()
        self.collected_papers: Dict[str, PaperMetadata] = {}
        self.duplicates: Set[str] = set()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _initialize_connectors(self) -> Dict[str, Any]:
        """Initialize data source connectors."""
        connectors = {}
        
        # Primary sources
        for source_name, source_config in self.config.get("primary_sources", {}).items():
            if source_config.get("enabled", True):
                connector_class = self._get_connector_class(source_name)
                if connector_class:
                    connectors[source_name] = connector_class(source_config)
                    logger.info(f"Initialized {source_name} connector")
        
        # Optional sources
        for source_name, source_config in self.config.get("optional_sources", {}).items():
            if source_config.get("enabled", False):
                connector_class = self._get_connector_class(source_name)
                if connector_class:
                    connectors[source_name] = connector_class(source_config)
                    logger.info(f"Initialized {source_name} connector")
        
        return connectors
    
    def _get_connector_class(self, source_name: str):
        """Get connector class by name."""
        connector_map = {
            "openalex": OpenAlexConnector,
            "crossref": CrossrefConnector,
            "europe_pmc": EuropePMCConnector,
            "semantic_scholar": SemanticScholarConnector,
            "pubmed": PubMedConnector,
            "arxiv": ArxivConnector,
            "doaj": DOAJConnector
        }
        return connector_map.get(source_name)
    
    async def collect_papers(self, output_dir: str = "outputs") -> Manifest:
        """Main method to collect papers from all sources."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        run_id = str(uuid.uuid4())[:8]
        start_time = datetime.now()
        
        logger.info(f"Starting data collection pipeline (run_id: {run_id})")
        
        # Generate search queries
        queries = self._generate_search_queries()
        
        # Collect papers from each source
        total_papers = 0
        for source_name, connector in self.connectors.items():
            logger.info(f"Collecting from {source_name}")
            
            async with connector:
                for query in queries:
                    try:
                        async for paper in connector.search_papers(
                            query=query,
                            year_range=tuple(self.config["search_config"]["year_range"]),
                            max_results=self.config["search_config"]["max_results_per_source"]
                        ):
                            if self._add_paper(paper):
                                total_papers += 1
                                
                    except Exception as e:
                        logger.error(f"Error collecting from {source_name} with query '{query}': {e}")
                        continue
        
        # Deduplicate papers
        logger.info("Deduplicating papers...")
        unique_papers = self._deduplicate_papers()
        
        # Save results
        self._save_papers(unique_papers, output_path)
        
        end_time = datetime.now()
        
        # Create manifest
        manifest = Manifest(
            run_id=run_id,
            start_time=start_time,
            end_time=end_time,
            sources_used=list(self.connectors.keys()),
            total_papers=total_papers,
            unique_papers=len(unique_papers),
            duplicates_removed=len(self.duplicates),
            fulltext_downloaded=0,  # TODO: Implement fulltext download
            processing_errors=0,
            api_versions={},  # TODO: Track API versions
            checksums={},  # TODO: Calculate checksums
            configuration=self.config
        )
        
        # Save manifest
        manifest_path = output_path / "manifest.yaml"
        with open(manifest_path, 'w') as f:
            yaml.dump(manifest.dict(), f, default_flow_style=False)
        
        logger.info(f"Data collection completed. Found {len(unique_papers)} unique papers.")
        return manifest
    
    def _generate_search_queries(self) -> List[str]:
        """Generate search queries from configuration."""
        queries = []
        
        taxa_terms = self.config["search_config"]["taxa_terms"]
        context_terms = self.config["search_config"]["context_terms"]
        
        # Create combinations of taxa and context terms
        for taxa in taxa_terms:
            # Single taxa term
            queries.append(taxa)
            
            # Taxa + context combinations
            for context in context_terms[:5]:  # Limit to first 5 context terms
                queries.append(f"{taxa} AND {context}")
        
        # Add some broader queries
        queries.extend([
            "truffle cultivation",
            "ectomycorrhizal fungi",
            "Tuber species",
            "truffle ecology"
        ])
        
        return queries[:20]  # Limit to 20 queries to avoid rate limits
    
    def _add_paper(self, paper: PaperMetadata) -> bool:
        """Add paper to collection, checking for duplicates."""
        paper_id = paper.paper_id
        
        if paper_id in self.collected_papers:
            # Check if this version is better (more complete, better OA status)
            existing = self.collected_papers[paper_id]
            if self._is_better_paper(paper, existing):
                self.collected_papers[paper_id] = paper
                # Add source to source_list
                if paper.source not in paper.source_list:
                    paper.source_list.append(paper.source)
            return False
        else:
            self.collected_papers[paper_id] = paper
            return True
    
    def _is_better_paper(self, new_paper: PaperMetadata, existing_paper: PaperMetadata) -> bool:
        """Determine if new paper is better than existing one."""
        # Prefer papers with abstracts
        if new_paper.abstract and not existing_paper.abstract:
            return True
        if not new_paper.abstract and existing_paper.abstract:
            return False
        
        # Prefer open access papers
        if new_paper.oa_status.value == "open" and existing_paper.oa_status.value != "open":
            return True
        
        # Prefer papers with more citations
        if new_paper.citations > existing_paper.citations:
            return True
        
        return False
    
    def _deduplicate_papers(self) -> List[PaperMetadata]:
        """Remove duplicate papers using fuzzy matching."""
        papers = list(self.collected_papers.values())
        unique_papers = []
        processed = set()
        
        for i, paper in enumerate(papers):
            if paper.paper_id in processed:
                continue
            
            # Find similar papers
            similar_papers = [paper]
            for j, other_paper in enumerate(papers[i+1:], i+1):
                if other_paper.paper_id in processed:
                    continue
                
                if self._are_similar_papers(paper, other_paper):
                    similar_papers.append(other_paper)
                    processed.add(other_paper.paper_id)
            
            # Keep the best paper from similar group
            best_paper = max(similar_papers, key=lambda p: (
                bool(p.abstract),
                p.oa_status.value == "open",
                p.citations
            ))
            
            unique_papers.append(best_paper)
            processed.add(paper.paper_id)
        
        return unique_papers
    
    def _are_similar_papers(self, paper1: PaperMetadata, paper2: PaperMetadata) -> bool:
        """Check if two papers are similar (potential duplicates)."""
        # Check title similarity
        title_sim = self._calculate_title_similarity(paper1.title, paper2.title)
        if title_sim < self.config["processing"]["deduplication"]["title_similarity_threshold"]:
            return False
        
        # Check author overlap
        authors1 = {a.name.lower() for a in paper1.authors}
        authors2 = {a.name.lower() for a in paper2.authors}
        
        if authors1 and authors2:
            overlap = len(authors1.intersection(authors2)) / max(len(authors1), len(authors2))
            if overlap < self.config["processing"]["deduplication"]["author_overlap_threshold"]:
                return False
        
        # Check year similarity
        if paper1.year and paper2.year and abs(paper1.year - paper2.year) > 1:
            return False
        
        return True
    
    def _calculate_title_similarity(self, title1: str, title2: str) -> float:
        """Calculate similarity between two titles."""
        try:
            from fuzzywuzzy import fuzz
            return fuzz.ratio(title1.lower(), title2.lower()) / 100.0
        except ImportError:
            # Fallback to simple similarity
            words1 = set(title1.lower().split())
            words2 = set(title2.lower().split())
            if not words1 or not words2:
                return 0.0
            return len(words1.intersection(words2)) / max(len(words1), len(words2))
    
    def _save_papers(self, papers: List[PaperMetadata], output_path: Path):
        """Save papers to Parquet file."""
        # Convert to DataFrame
        data = []
        for paper in papers:
            paper_dict = paper.dict()
            # Convert datetime to string for JSON serialization
            paper_dict["collection_date"] = paper_dict["collection_date"].isoformat()
            data.append(paper_dict)
        
        df = pd.DataFrame(data)
        
        # Save as Parquet
        parquet_path = output_path / "papers_index.parquet"
        df.to_parquet(parquet_path, index=False)
        
        logger.info(f"Saved {len(papers)} papers to {parquet_path}")
        
        # Also save as JSON for inspection
        json_path = output_path / "papers_index.json"
        df.to_json(json_path, orient="records", indent=2)
        
        logger.info(f"Saved {len(papers)} papers to {json_path}")


async def main():
    """Main entry point for data collection."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Collect academic papers about truffle research")
    parser.add_argument("--config", default="configs/sources.yaml", help="Configuration file path")
    parser.add_argument("--output", default="outputs", help="Output directory")
    
    args = parser.parse_args()
    
    # Setup logging
    logger.add("logs/data_collection.log", rotation="1 day", retention="7 days")
    
    # Run pipeline
    pipeline = DataCollectionPipeline(args.config)
    manifest = await pipeline.collect_papers(args.output)
    
    print(f"Collection completed. Found {manifest.unique_papers} unique papers.")
    print(f"Manifest saved to {args.output}/manifest.yaml")


if __name__ == "__main__":
    asyncio.run(main())
