#!/usr/bin/env python3
"""
Test script to verify Truffle Occurrence Data Downloader installation
"""

import sys
import importlib
from pathlib import Path

def test_imports():
    """Test that all required modules can be imported"""
    print("Testing imports...")
    
    required_modules = [
        'pandas',
        'numpy', 
        'requests',
        'yaml',
        'click',
        'tqdm'
    ]
    
    optional_modules = [
        'geopandas',
        'shapely',
        'fiona'
    ]
    
    # Test required modules
    for module in required_modules:
        try:
            importlib.import_module(module)
            print(f"✓ {module}")
        except ImportError as e:
            print(f"✗ {module}: {e}")
            return False
    
    # Test optional modules
    print("\nTesting optional modules...")
    for module in optional_modules:
        try:
            importlib.import_module(module)
            print(f"✓ {module}")
        except ImportError:
            print(f"✗ {module} (optional)")
    
    return True

def test_package_import():
    """Test that the package can be imported"""
    print("\nTesting package import...")
    
    try:
        from truffle_downloader import GBIFTruffleDownloader, DataValidator, DataExporter
        print("✓ truffle_downloader package")
        return True
    except ImportError as e:
        print(f"✗ truffle_downloader package: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality"""
    print("\nTesting basic functionality...")
    
    try:
        from truffle_downloader import GBIFTruffleDownloader, DataValidator, DataExporter
        
        # Test downloader initialization
        downloader = GBIFTruffleDownloader()
        print("✓ GBIFTruffleDownloader initialization")
        
        # Test validator initialization
        validator = DataValidator()
        print("✓ DataValidator initialization")
        
        # Test exporter initialization
        exporter = DataExporter("./test_output")
        print("✓ DataExporter initialization")
        
        return True
    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
        return False

def test_cli():
    """Test CLI availability"""
    print("\nTesting CLI...")
    
    try:
        import subprocess
        result = subprocess.run([sys.executable, "-m", "truffle_downloader.cli", "--help"], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✓ CLI available")
            return True
        else:
            print(f"✗ CLI test failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"✗ CLI test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("Truffle Occurrence Data Downloader - Installation Test")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_package_import,
        test_basic_functionality,
        test_cli
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 60)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✓ All tests passed! Installation is working correctly.")
        return 0
    else:
        print("✗ Some tests failed. Please check the installation.")
        return 1

if __name__ == "__main__":
    sys.exit(main())