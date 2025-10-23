"""
Main entry point for academic research extraction system.
"""

import asyncio
import argparse
import yaml
from pathlib import Path
from loguru import logger
import pandas as pd

from src.data_collection_pipeline import DataCollectionPipeline
from src.processing.semantic_analysis import analyze_papers_batch
from src.processing.entity_extraction import extract_entities_and_relations
from src.analysis.trend_analysis import analyze_trends_batch
from src.data_schema import PaperMetadata, ClassificationLabels


async def run_full_pipeline(config_path: str, output_dir: str = "outputs"):
    """Run the complete academic research extraction pipeline."""
    
    # Setup logging
    log_dir = Path(output_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger.add(log_dir / "pipeline.log", rotation="1 day", retention="7 days")
    
    logger.info("Starting academic research extraction pipeline...")
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Step 1: Collect papers
    logger.info("Step 1: Collecting papers from academic sources...")
    pipeline = DataCollectionPipeline(config_path)
    manifest = await pipeline.collect_papers(output_dir)
    
    # Load collected papers
    papers_df = pd.read_parquet(Path(output_dir) / "papers_index.parquet")
    papers = []
    
    for _, row in papers_df.iterrows():
        try:
            # Convert DataFrame row to PaperMetadata
            paper_dict = row.to_dict()
            
            # Handle datetime conversion
            if 'collection_date' in paper_dict and isinstance(paper_dict['collection_date'], str):
                from datetime import datetime
                paper_dict['collection_date'] = datetime.fromisoformat(paper_dict['collection_date'])
            
            # Handle authors list
            if 'authors' in paper_dict and isinstance(paper_dict['authors'], str):
                import json
                paper_dict['authors'] = json.loads(paper_dict['authors'])
            
            paper = PaperMetadata(**paper_dict)
            papers.append(paper)
        except Exception as e:
            logger.error(f"Error loading paper: {e}")
            continue
    
    logger.info(f"Loaded {len(papers)} papers for processing")
    
    # Step 2: Semantic analysis and classification
    logger.info("Step 2: Performing semantic analysis and classification...")
    classifications, topic_summary = analyze_papers_batch(papers, config, output_dir)
    
    # Step 3: Entity and relation extraction
    logger.info("Step 3: Extracting entities and relations...")
    entities, relations, knowledge_graph = extract_entities_and_relations(papers, config, output_dir)
    
    # Step 4: Trend analysis
    logger.info("Step 4: Analyzing trends...")
    trends = analyze_trends_batch(papers, classifications, config, output_dir)
    
    # Step 5: Generate final report
    logger.info("Step 5: Generating final report...")
    generate_final_report(manifest, papers, classifications, entities, relations, trends, output_dir)
    
    logger.info("Pipeline completed successfully!")
    return manifest


def generate_final_report(
    manifest, 
    papers, 
    classifications, 
    entities, 
    relations, 
    trends, 
    output_dir: str
):
    """Generate final summary report."""
    
    report_path = Path(output_dir) / "FINAL_REPORT.md"
    
    with open(report_path, 'w') as f:
        f.write("# Academic Research Extraction - Final Report\n\n")
        f.write(f"**Run ID:** {manifest.run_id}\n")
        f.write(f"**Start Time:** {manifest.start_time}\n")
        f.write(f"**End Time:** {manifest.end_time}\n\n")
        
        f.write("## Summary Statistics\n\n")
        f.write(f"- **Total Papers Collected:** {manifest.total_papers}\n")
        f.write(f"- **Unique Papers:** {manifest.unique_papers}\n")
        f.write(f"- **Duplicates Removed:** {manifest.duplicates_removed}\n")
        f.write(f"- **Sources Used:** {', '.join(manifest.sources_used)}\n\n")
        
        f.write("## Processing Results\n\n")
        f.write(f"- **Papers Classified:** {len(classifications)}\n")
        f.write(f"- **Entities Extracted:** {len(entities)}\n")
        f.write(f"- **Relations Extracted:** {len(relations)}\n")
        f.write(f"- **Trend Analyses:** {len(trends)}\n\n")
        
        f.write("## Output Files\n\n")
        f.write("- `papers_index.parquet` - Main paper metadata\n")
        f.write("- `labels_multilabel.parquet` - Classification results\n")
        f.write("- `topics_summary.json` - Topic analysis summary\n")
        f.write("- `entities.parquet` - Extracted entities\n")
        f.write("- `relations.parquet` - Extracted relations\n")
        f.write("- `knowledge_graph.gml` - Knowledge graph\n")
        f.write("- `trend_analysis.csv` - Trend analysis results\n")
        f.write("- `manifest.yaml` - Collection manifest\n\n")
        
        f.write("## Next Steps\n\n")
        f.write("1. Review the collected papers in `papers_index.parquet`\n")
        f.write("2. Explore classifications in `labels_multilabel.parquet`\n")
        f.write("3. Analyze trends using `trend_analysis.csv`\n")
        f.write("4. Visualize the knowledge graph using `knowledge_graph.gml`\n")
        f.write("5. Use the Streamlit dashboard for interactive exploration\n")
    
    logger.info(f"Final report saved to {report_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Academic Research Extraction System")
    parser.add_argument("--config", default="configs/sources.yaml", help="Configuration file path")
    parser.add_argument("--output", default="outputs", help="Output directory")
    parser.add_argument("--dashboard", action="store_true", help="Launch Streamlit dashboard")
    
    args = parser.parse_args()
    
    if args.dashboard:
        # Launch Streamlit dashboard
        import subprocess
        import sys
        
        dashboard_path = Path(__file__).parent / "dashboard.py"
        if dashboard_path.exists():
            subprocess.run([sys.executable, "-m", "streamlit", "run", str(dashboard_path)])
        else:
            logger.error("Dashboard not found. Please run the full pipeline first.")
    else:
        # Run full pipeline
        asyncio.run(run_full_pipeline(args.config, args.output))


if __name__ == "__main__":
    main()
