#!/usr/bin/env python3
"""
Advanced filtering example for Truffle Occurrence Data Downloader

This example demonstrates advanced filtering options including
geographic bounds, temporal filtering, and data quality filtering.
"""

from truffle_downloader import GBIFTruffleDownloader, DataValidator, DataExporter
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Advanced filtering example"""
    
    # Initialize the downloader with custom config
    downloader = GBIFTruffleDownloader("config.yaml")
    
    # Example 1: Download data for specific geographic region
    print("=== Example 1: Geographic Filtering ===")
    
    # Define coordinate bounds for France
    france_bounds = {
        'min_lat': 41.0,
        'max_lat': 51.0,
        'min_lon': -5.0,
        'max_lon': 10.0
    }
    
    data_france = downloader.download_species(
        species=["Tuber melanosporum"],
        coordinate_bounds=france_bounds,
        year_from=2020,
        max_records=500
    )
    
    print(f"Downloaded {len(data_france)} records from France")
    
    # Example 2: Download data with specific quality criteria
    print("\n=== Example 2: Quality Filtering ===")
    
    data_quality = downloader.download_species(
        species=["Tuber magnatum"],
        countries=["IT"],
        year_from=2015,
        year_to=2023,
        max_records=1000
    )
    
    # Filter for high-quality data
    validator = DataValidator()
    high_quality_data = validator.filter_high_quality_data(data_quality, min_quality_score=80.0)
    
    print(f"Original records: {len(data_quality)}")
    print(f"High-quality records: {len(high_quality_data)}")
    
    # Example 3: Seasonal analysis
    print("\n=== Example 3: Seasonal Analysis ===")
    
    # Download data for different seasons
    seasons = {
        "winter": (12, 2),  # Dec, Jan, Feb
        "spring": (3, 5),   # Mar, Apr, May
        "summer": (6, 8),   # Jun, Jul, Aug
        "autumn": (9, 11)   # Sep, Oct, Nov
    }
    
    seasonal_data = {}
    
    for season_name, (start_month, end_month) in seasons.items():
        print(f"Downloading {season_name} data...")
        
        # Note: GBIF doesn't support month filtering directly in the API
        # We'll download all data and filter by month later
        data = downloader.download_species(
            species=["Tuber aestivum"],
            countries=["FR", "IT", "ES"],
            year_from=2020,
            year_to=2023,
            max_records=2000
        )
        
        if not data.empty and 'month' in data.columns:
            # Filter by season
            if start_month > end_month:  # Winter spans year boundary
                season_mask = (data['month'] >= start_month) | (data['month'] <= end_month)
            else:
                season_mask = (data['month'] >= start_month) & (data['month'] <= end_month)
            
            seasonal_data[season_name] = data[season_mask]
            print(f"  {season_name}: {len(seasonal_data[season_name])} records")
    
    # Example 4: Export with different formats
    print("\n=== Example 4: Multi-format Export ===")
    
    exporter = DataExporter("./output")
    
    # Export all seasonal data
    for season, data in seasonal_data.items():
        if not data.empty:
            # Export as CSV
            csv_file = exporter.export_data(
                data, f"truffle_aestivum_{season}", "csv"
            )
            print(f"  {season} CSV: {csv_file}")
            
            # Export as GeoJSON
            try:
                geojson_file = exporter.export_data(
                    data, f"truffle_aestivum_{season}", "geojson"
                )
                print(f"  {season} GeoJSON: {geojson_file}")
            except ImportError:
                print(f"  {season} GeoJSON: Requires geopandas")
    
    # Example 5: Create comprehensive export package
    print("\n=== Example 5: Export Package ===")
    
    # Combine all data
    all_data = data_quality.copy()
    if not data_france.empty:
        all_data = all_data.append(data_france, ignore_index=True)
    
    # Remove duplicates
    all_data = all_data.drop_duplicates(subset=['gbif_id'], keep='first')
    
    # Validate combined data
    validation_results = validator.validate_data(all_data)
    
    # Create export package
    download_stats = downloader.get_download_stats()
    package_file = exporter.create_export_package(
        all_data, validation_results, download_stats
    )
    
    print(f"Export package created: {package_file}")
    
    # Example 6: Data analysis
    print("\n=== Example 6: Data Analysis ===")
    
    if not all_data.empty:
        print(f"Total records: {len(all_data)}")
        print(f"Species: {all_data['species'].nunique()}")
        print(f"Countries: {all_data['country'].nunique()}")
        
        if 'event_date' in all_data.columns:
            date_range = all_data['event_date'].agg(['min', 'max'])
            print(f"Date range: {date_range['min']} to {date_range['max']}")
        
        # Species breakdown
        species_counts = all_data['species'].value_counts()
        print("\nSpecies breakdown:")
        for species, count in species_counts.items():
            print(f"  {species}: {count} records")
        
        # Country breakdown
        if 'country' in all_data.columns:
            country_counts = all_data['country'].value_counts()
            print("\nCountry breakdown:")
            for country, count in country_counts.head(10).items():
                print(f"  {country}: {count} records")
    
    print("\nAdvanced filtering example complete!")


if __name__ == "__main__":
    main()