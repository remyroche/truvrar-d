#!/usr/bin/env python3
"""
Basic usage example for Truffle Occurrence Data Downloader

This example demonstrates how to download truffle occurrence data
from GBIF using the Python API.
"""

from truffle_downloader import GBIFTruffleDownloader, DataValidator, DataExporter
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Basic usage example"""
    
    # Initialize the downloader
    downloader = GBIFTruffleDownloader()
    
    # Define species to download
    species = [
        "Tuber melanosporum",  # Black truffle
        "Tuber magnatum",      # White truffle
        "Tuber aestivum"       # Summer truffle
    ]
    
    # Download data
    print("Downloading truffle occurrence data...")
    data = downloader.download_species(
        species=species,
        countries=["FR", "IT", "ES"],  # France, Italy, Spain
        year_from=2010,
        year_to=2023,
        max_records=1000
    )
    
    if data.empty:
        print("No data found for the specified criteria")
        return
    
    print(f"Downloaded {len(data)} records")
    print(f"Species found: {data['species'].unique()}")
    print(f"Countries: {data['country'].unique()}")
    
    # Validate data quality
    print("\nValidating data quality...")
    validator = DataValidator()
    validation_results = validator.validate_data(data)
    
    print(f"Data quality: {validation_results['quality_metrics']['rating']}")
    print(f"Overall score: {validation_results['quality_metrics']['overall_score']:.1f}/100")
    
    # Export data
    print("\nExporting data...")
    exporter = DataExporter("./output")
    
    # Export as CSV
    csv_file = exporter.export_data(data, "truffle_occurrences", "csv")
    print(f"CSV file saved: {csv_file}")
    
    # Export as GeoJSON (requires geopandas)
    try:
        geojson_file = exporter.export_data(data, "truffle_occurrences", "geojson")
        print(f"GeoJSON file saved: {geojson_file}")
    except ImportError:
        print("GeoJSON export requires geopandas. Install with: pip install geopandas")
    
    # Create summary report
    download_stats = downloader.get_download_stats()
    report_file = exporter.export_summary_report(data, validation_results, download_stats)
    print(f"Summary report saved: {report_file}")
    
    print("\nDownload complete!")


if __name__ == "__main__":
    main()