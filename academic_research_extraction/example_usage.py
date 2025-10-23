"""
Example usage of the academic research extraction system.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from data_collection_pipeline import DataCollectionPipeline
from processing.semantic_analysis import analyze_papers_batch
from processing.entity_extraction import extract_entities_and_relations
from analysis.trend_analysis import analyze_trends_batch
import pandas as pd


async def run_example():
    """Run a complete example of the system."""
    print("🚀 Academic Research Extraction - Example Usage")
    print("=" * 60)
    
    # Configuration
    config_path = "configs/sources.yaml"
    output_dir = "example_outputs"
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    print(f"\n📁 Output directory: {output_dir}")
    print(f"⚙️  Configuration: {config_path}")
    
    # Step 1: Collect papers (limited for example)
    print("\n1️⃣ Collecting papers...")
    try:
        pipeline = DataCollectionPipeline(config_path)
        
        # Modify config for example (limit results)
        pipeline.config["search_config"]["max_results_per_source"] = 50
        
        manifest = await pipeline.collect_papers(output_dir)
        print(f"✅ Collected {manifest.unique_papers} unique papers")
        
    except Exception as e:
        print(f"❌ Collection failed: {e}")
        return
    
    # Step 2: Load papers for processing
    print("\n2️⃣ Loading papers for processing...")
    try:
        papers_df = pd.read_parquet(Path(output_dir) / "papers_index.parquet")
        print(f"✅ Loaded {len(papers_df)} papers from Parquet file")
        
        # Convert to PaperMetadata objects (simplified)
        papers = []
        for _, row in papers_df.head(10).iterrows():  # Process only first 10 for example
            try:
                from data_schema import PaperMetadata, Author, OpenAccessStatus, Language
                
                # Handle authors
                authors = []
                if pd.notna(row.get('authors')):
                    if isinstance(row['authors'], str):
                        import json
                        authors_data = json.loads(row['authors'])
                    else:
                        authors_data = row['authors']
                    
                    for author_data in authors_data:
                        if isinstance(author_data, dict):
                            authors.append(Author(
                                name=author_data.get('name', ''),
                                orcid=author_data.get('orcid'),
                                affiliation=author_data.get('affiliation')
                            ))
                
                paper = PaperMetadata(
                    paper_id=row['paper_id'],
                    title=row['title'],
                    abstract=row.get('abstract', ''),
                    authors=authors,
                    year=row.get('year'),
                    journal=row.get('journal'),
                    source=row['source'],
                    oa_status=OpenAccessStatus(row.get('oa_status', 'unknown')),
                    language=Language(row.get('language', 'en')) if pd.notna(row.get('language')) else None
                )
                papers.append(paper)
                
            except Exception as e:
                print(f"⚠️  Skipping paper due to error: {e}")
                continue
        
        print(f"✅ Converted {len(papers)} papers to metadata objects")
        
    except Exception as e:
        print(f"❌ Loading failed: {e}")
        return
    
    # Step 3: Semantic analysis
    print("\n3️⃣ Performing semantic analysis...")
    try:
        config = pipeline.config
        classifications, topic_summary = analyze_papers_batch(papers, config, output_dir)
        print(f"✅ Classified {len(classifications)} papers")
        print(f"📊 Topic summary: {topic_summary.get('total_papers', 0)} papers analyzed")
        
    except Exception as e:
        print(f"❌ Semantic analysis failed: {e}")
        return
    
    # Step 4: Entity extraction
    print("\n4️⃣ Extracting entities and relations...")
    try:
        entities, relations, knowledge_graph = extract_entities_and_relations(
            papers, config, output_dir
        )
        print(f"✅ Extracted {len(entities)} entities and {len(relations)} relations")
        print(f"🌐 Knowledge graph: {len(knowledge_graph.nodes)} nodes, {len(knowledge_graph.edges)} edges")
        
    except Exception as e:
        print(f"❌ Entity extraction failed: {e}")
        return
    
    # Step 5: Trend analysis
    print("\n5️⃣ Analyzing trends...")
    try:
        trends = analyze_trends_batch(papers, classifications, config, output_dir)
        print(f"✅ Generated {len(trends)} trend analyses")
        
    except Exception as e:
        print(f"❌ Trend analysis failed: {e}")
        return
    
    # Step 6: Summary
    print("\n📊 EXAMPLE COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(f"📁 Output files saved to: {output_dir}/")
    print(f"📄 Papers: {len(papers)}")
    print(f"🏷️  Classifications: {len(classifications)}")
    print(f"🔍 Entities: {len(entities)}")
    print(f"🔗 Relations: {len(relations)}")
    print(f"📈 Trends: {len(trends)}")
    
    print("\n🎯 Next steps:")
    print("1. Explore the output files in the output directory")
    print("2. Run the full pipeline: python main.py")
    print("3. Launch the dashboard: python main.py --dashboard")
    
    return True


def main():
    """Main function."""
    try:
        success = asyncio.run(run_example())
        if success:
            print("\n✅ Example completed successfully!")
            return 0
        else:
            print("\n❌ Example failed!")
            return 1
    except KeyboardInterrupt:
        print("\n⏹️  Example interrupted by user")
        return 1
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
