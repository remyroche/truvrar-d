"""
Test script for academic research extraction system.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from data_collection_pipeline import DataCollectionPipeline
from data_schema import PaperMetadata, Author, OpenAccessStatus, Language


async def test_basic_functionality():
    """Test basic system functionality."""
    print("🧪 Testing Academic Research Extraction System")
    print("=" * 50)
    
    # Test 1: Data schema
    print("\n1. Testing data schema...")
    try:
        author = Author(name="Test Author", orcid="0000-0000-0000-0000")
        paper = PaperMetadata(
            paper_id="test_001",
            title="Test Paper on Truffle Biology",
            abstract="This is a test abstract about truffle cultivation.",
            authors=[author],
            year=2023,
            journal="Test Journal",
            source="test",
            oa_status=OpenAccessStatus.OPEN,
            language=Language.ENGLISH
        )
        print("✅ Data schema test passed")
    except Exception as e:
        print(f"❌ Data schema test failed: {e}")
        return False
    
    # Test 2: Configuration loading
    print("\n2. Testing configuration loading...")
    try:
        config_path = Path("configs/sources.yaml")
        if not config_path.exists():
            print("❌ Configuration file not found")
            return False
        
        pipeline = DataCollectionPipeline(str(config_path))
        print("✅ Configuration loading test passed")
    except Exception as e:
        print(f"❌ Configuration loading test failed: {e}")
        return False
    
    # Test 3: Connector initialization
    print("\n3. Testing connector initialization...")
    try:
        connectors = pipeline.connectors
        print(f"✅ Initialized {len(connectors)} connectors: {list(connectors.keys())}")
    except Exception as e:
        print(f"❌ Connector initialization test failed: {e}")
        return False
    
    # Test 4: Search query generation
    print("\n4. Testing search query generation...")
    try:
        queries = pipeline._generate_search_queries()
        print(f"✅ Generated {len(queries)} search queries")
        print(f"   Sample queries: {queries[:3]}")
    except Exception as e:
        print(f"❌ Search query generation test failed: {e}")
        return False
    
    print("\n🎉 All basic tests passed!")
    return True


def test_data_processing():
    """Test data processing modules."""
    print("\n5. Testing data processing modules...")
    
    try:
        from processing.semantic_analysis import SemanticAnalyzer
        from processing.entity_extraction import EntityExtractor
        from processing.deduplication import DeduplicationEngine
        
        # Test with minimal config
        config = {
            "classification": {
                "taxonomy_labels": ["truffle_biology", "cultivation"],
                "min_confidence_threshold": 0.3
            },
            "entity_extraction": {
                "species_patterns": [r"Tuber\s+\w+"],
                "host_patterns": [r"Quercus\s+\w+"],
                "soil_patterns": [r"pH\s*[=<>]\s*[\d.]+"]
            },
            "deduplication": {
                "title_similarity_threshold": 0.97,
                "author_overlap_threshold": 0.6,
                "simhash_threshold": 0.85
            }
        }
        
        # Test analyzers
        semantic_analyzer = SemanticAnalyzer(config)
        entity_extractor = EntityExtractor(config)
        dedup_engine = DeduplicationEngine(config)
        
        print("✅ Data processing modules initialized successfully")
        return True
        
    except Exception as e:
        print(f"❌ Data processing test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("Starting system tests...")
    
    # Test basic functionality
    basic_test_passed = asyncio.run(test_basic_functionality())
    
    # Test data processing
    processing_test_passed = test_data_processing()
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    print(f"Basic functionality: {'✅ PASSED' if basic_test_passed else '❌ FAILED'}")
    print(f"Data processing: {'✅ PASSED' if processing_test_passed else '❌ FAILED'}")
    
    if basic_test_passed and processing_test_passed:
        print("\n🎉 All tests passed! System is ready to use.")
        print("\nNext steps:")
        print("1. Run: python main.py --config configs/sources.yaml --output outputs")
        print("2. Launch dashboard: python main.py --dashboard")
    else:
        print("\n❌ Some tests failed. Please check the errors above.")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
