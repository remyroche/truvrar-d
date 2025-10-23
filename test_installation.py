#!/usr/bin/env python3
"""
Test script to verify GTHA installation and basic functionality.
"""
import sys
import importlib
from pathlib import Path

def test_imports():
    """Test that all required modules can be imported."""
    print("🔍 Testing imports...")
    
    required_modules = [
        'pandas',
        'numpy', 
        'requests',
        'rasterio',
        'geopandas',
        'sklearn',
        'folium',
        'matplotlib',
        'seaborn',
        'plotly'
    ]
    
    failed_imports = []
    
    for module in required_modules:
        try:
            importlib.import_module(module)
            print(f"  ✅ {module}")
        except ImportError as e:
            print(f"  ❌ {module}: {e}")
            failed_imports.append(module)
    
    if failed_imports:
        print(f"\n❌ Failed to import: {', '.join(failed_imports)}")
        print("Please install missing dependencies with: pip install -r requirements.txt")
        return False
    else:
        print("\n✅ All required modules imported successfully!")
        return True

def test_gtha_modules():
    """Test that GTHA modules can be imported."""
    print("\n🔍 Testing GTHA modules...")
    
    try:
        from config import TRUFFLE_SPECIES, API_CONFIG
        print("  ✅ config")
    except ImportError as e:
        print(f"  ❌ config: {e}")
        return False
    
    try:
        from src.data_collectors import GBIFCollector, SoilGridsCollector
        print("  ✅ data_collectors")
    except ImportError as e:
        print(f"  ❌ data_collectors: {e}")
        return False
    
    try:
        from src.data_processing import HabitatProcessor
        print("  ✅ data_processing")
    except ImportError as e:
        print(f"  ❌ data_processing: {e}")
        return False
    
    try:
        from src.models import HabitatModel
        print("  ✅ models")
    except ImportError as e:
        print(f"  ❌ models: {e}")
        return False
    
    try:
        from src.visualization import MappingTools
        print("  ✅ visualization")
    except ImportError as e:
        print(f"  ❌ visualization: {e}")
        return False
    
    print("\n✅ All GTHA modules imported successfully!")
    return True

def test_configuration():
    """Test configuration loading."""
    print("\n🔍 Testing configuration...")
    
    try:
        from config import TRUFFLE_SPECIES, API_CONFIG, MODEL_CONFIG
        
        print(f"  ✅ Truffle species: {len(TRUFFLE_SPECIES)} configured")
        print(f"  ✅ API configurations: {len(API_CONFIG)} sources")
        print(f"  ✅ Model configuration: {len(MODEL_CONFIG)} parameters")
        
        return True
    except Exception as e:
        print(f"  ❌ Configuration error: {e}")
        return False

def test_directory_structure():
    """Test that required directories exist."""
    print("\n🔍 Testing directory structure...")
    
    required_dirs = [
        'src',
        'src/data_collectors',
        'src/data_processing', 
        'src/models',
        'src/visualization',
        'data',
        'models',
        'outputs',
        'logs'
    ]
    
    missing_dirs = []
    
    for dir_path in required_dirs:
        if Path(dir_path).exists():
            print(f"  ✅ {dir_path}")
        else:
            print(f"  ❌ {dir_path} (missing)")
            missing_dirs.append(dir_path)
    
    if missing_dirs:
        print(f"\n❌ Missing directories: {', '.join(missing_dirs)}")
        print("Creating missing directories...")
        
        for dir_path in missing_dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
            print(f"  ✅ Created {dir_path}")
    
    return True

def test_basic_functionality():
    """Test basic GTHA functionality."""
    print("\n🔍 Testing basic functionality...")
    
    try:
        from config import TRUFFLE_SPECIES
        from src.data_collectors import GBIFCollector
        from src.data_processing import HabitatProcessor
        
        # Test configuration
        config = {
            'gbif': {'base_url': 'https://api.gbif.org/v1', 'timeout': 30, 'max_retries': 3},
            'inaturalist': {'base_url': 'https://api.inaturalist.org/v1', 'timeout': 30, 'max_retries': 3},
            'soilgrids': {'base_url': 'https://rest.soilgrids.org', 'timeout': 30, 'max_retries': 3},
            'worldclim': {'base_url': 'https://biogeo.ucdavis.edu/data/worldclim/v2.1/base', 'timeout': 60, 'max_retries': 3}
        }
        
        # Test processor initialization
        processor = HabitatProcessor(config, Path('data'))
        print("  ✅ HabitatProcessor initialized")
        
        # Test collector initialization
        gbif_collector = GBIFCollector(config, Path('data'))
        print("  ✅ GBIFCollector initialized")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Basic functionality test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🍄 Global Truffle Habitat Atlas - Installation Test")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_gtha_modules,
        test_configuration,
        test_directory_structure,
        test_basic_functionality
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
    
    print("\n" + "=" * 60)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! GTHA is ready to use.")
        print("\nNext steps:")
        print("1. Run: python example_usage.py")
        print("2. Or run: python main.py --action collect --help")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()