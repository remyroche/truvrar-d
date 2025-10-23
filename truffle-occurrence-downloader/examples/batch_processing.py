#!/usr/bin/env python3
"""
Batch processing example for Truffle Occurrence Data Downloader

This example demonstrates how to process multiple species and countries
in batches, with progress tracking and error handling.
"""

from truffle_downloader import GBIFTruffleDownloader, DataValidator, DataExporter
import logging
import pandas as pd
from pathlib import Path
import time
from typing import List, Dict, Any

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def process_species_batch(species_list: List[str], 
                         countries: List[str],
                         downloader: GBIFTruffleDownloader,
                         batch_size: int = 5) -> pd.DataFrame:
    """
    Process species in batches to avoid overwhelming the API.
    
    Args:
        species_list: List of species to process
        countries: List of country codes
        downloader: GBIF downloader instance
        batch_size: Number of species to process in each batch
        
    Returns:
        Combined DataFrame with all data
    """
    all_data = []
    
    for i in range(0, len(species_list), batch_size):
        batch = species_list[i:i + batch_size]
        print(f"Processing batch {i//batch_size + 1}: {batch}")
        
        try:
            batch_data = downloader.download_species(
                species=batch,
                countries=countries,
                year_from=2010,
                year_to=2023,
                max_records=2000
            )
            
            if not batch_data.empty:
                all_data.append(batch_data)
                print(f"  Downloaded {len(batch_data)} records")
            else:
                print(f"  No data found for batch")
            
            # Rate limiting - wait between batches
            time.sleep(2)
            
        except Exception as e:
            print(f"  Error processing batch: {e}")
            continue
    
    if all_data:
        return pd.concat(all_data, ignore_index=True)
    else:
        return pd.DataFrame()


def process_country_batch(species: str,
                         country_list: List[str],
                         downloader: GBIFTruffleDownloader,
                         batch_size: int = 3) -> pd.DataFrame:
    """
    Process countries in batches for a single species.
    
    Args:
        species: Species name to download
        country_list: List of country codes
        downloader: GBIF downloader instance
        batch_size: Number of countries to process in each batch
        
    Returns:
        Combined DataFrame with all data
    """
    all_data = []
    
    for i in range(0, len(country_list), batch_size):
        batch_countries = country_list[i:i + batch_size]
        print(f"Processing {species} for countries: {batch_countries}")
        
        try:
            batch_data = downloader.download_species(
                species=species,
                countries=batch_countries,
                year_from=2010,
                year_to=2023,
                max_records=1000
            )
            
            if not batch_data.empty:
                all_data.append(batch_data)
                print(f"  Downloaded {len(batch_data)} records")
            else:
                print(f"  No data found for {species} in {batch_countries}")
            
            # Rate limiting
            time.sleep(1)
            
        except Exception as e:
            print(f"  Error processing {species} for {batch_countries}: {e}")
            continue
    
    if all_data:
        return pd.concat(all_data, ignore_index=True)
    else:
        return pd.DataFrame()


def main():
    """Batch processing example"""
    
    # Initialize components
    downloader = GBIFTruffleDownloader("config.yaml")
    validator = DataValidator()
    exporter = DataExporter("./output")
    
    # Define species and countries to process
    species_list = [
        "Tuber melanosporum",
        "Tuber magnatum", 
        "Tuber aestivum",
        "Tuber borchii",
        "Tuber brumale",
        "Tuber mesentericum",
        "Tuber macrosporum"
    ]
    
    country_list = [
        "FR", "IT", "ES", "PT", "HR", "SI", "AT", "CH", "DE", "GB"
    ]
    
    print("=== Batch Processing Example ===")
    print(f"Species: {len(species_list)}")
    print(f"Countries: {len(country_list)}")
    print()
    
    # Example 1: Process species in batches
    print("=== Example 1: Species Batch Processing ===")
    start_time = time.time()
    
    all_species_data = process_species_batch(
        species_list, country_list, downloader, batch_size=3
    )
    
    species_time = time.time() - start_time
    print(f"Species batch processing completed in {species_time:.1f} seconds")
    print(f"Total records: {len(all_species_data)}")
    
    # Example 2: Process countries in batches for specific species
    print("\n=== Example 2: Country Batch Processing ===")
    start_time = time.time()
    
    # Focus on the most important species
    priority_species = ["Tuber melanosporum", "Tuber magnatum"]
    
    for species in priority_species:
        print(f"\nProcessing {species} by country batches...")
        country_data = process_country_batch(
            species, country_list, downloader, batch_size=2
        )
        
        if not country_data.empty:
            # Save individual species data
            species_file = exporter.export_data(
                country_data, f"{species.replace(' ', '_')}_by_country", "csv"
            )
            print(f"  Saved {species} data: {species_file}")
    
    country_time = time.time() - start_time
    print(f"Country batch processing completed in {country_time:.1f} seconds")
    
    # Example 3: Validate and clean all data
    print("\n=== Example 3: Data Validation and Cleaning ===")
    
    if not all_species_data.empty:
        print("Validating combined data...")
        validation_results = validator.validate_data(all_species_data)
        
        print(f"Data quality: {validation_results['quality_metrics']['rating']}")
        print(f"Overall score: {validation_results['quality_metrics']['overall_score']:.1f}/100")
        
        # Filter for high-quality data
        high_quality_data = validator.filter_high_quality_data(
            all_species_data, min_quality_score=70.0
        )
        
        print(f"High-quality records: {len(high_quality_data)}")
        
        # Example 4: Export with progress tracking
        print("\n=== Example 4: Export with Progress Tracking ===")
        
        # Export in multiple formats
        formats = ['csv', 'parquet']
        
        for fmt in formats:
            try:
                print(f"Exporting as {fmt.upper()}...")
                file_path = exporter.export_data(
                    high_quality_data, f"batch_processed_truffles", fmt
                )
                print(f"  {fmt.upper()}: {file_path}")
            except Exception as e:
                print(f"  Error exporting {fmt}: {e}")
        
        # Try GeoJSON export (requires geopandas)
        try:
            print("Exporting as GeoJSON...")
            geojson_file = exporter.export_data(
                high_quality_data, "batch_processed_truffles", "geojson"
            )
            print(f"  GeoJSON: {geojson_file}")
        except ImportError:
            print("  GeoJSON: Requires geopandas (pip install geopandas)")
        except Exception as e:
            print(f"  Error exporting GeoJSON: {e}")
        
        # Example 5: Create comprehensive report
        print("\n=== Example 5: Comprehensive Report ===")
        
        # Get download statistics
        download_stats = downloader.get_download_stats()
        
        # Create summary report
        report_file = exporter.export_summary_report(
            high_quality_data, validation_results, download_stats
        )
        print(f"Summary report: {report_file}")
        
        # Create species list
        species_file = exporter.export_species_list(high_quality_data)
        print(f"Species list: {species_file}")
        
        # Create export package
        package_file = exporter.create_export_package(
            high_quality_data, validation_results, download_stats
        )
        print(f"Export package: {package_file}")
        
        # Example 6: Data analysis summary
        print("\n=== Example 6: Data Analysis Summary ===")
        
        print(f"Total processing time: {species_time + country_time:.1f} seconds")
        print(f"Total records downloaded: {len(all_species_data)}")
        print(f"High-quality records: {len(high_quality_data)}")
        print(f"Quality improvement: {len(high_quality_data)/len(all_species_data)*100:.1f}%")
        
        # Species breakdown
        species_counts = high_quality_data['species'].value_counts()
        print(f"\nSpecies breakdown:")
        for species, count in species_counts.items():
            print(f"  {species}: {count} records")
        
        # Country breakdown
        if 'country' in high_quality_data.columns:
            country_counts = high_quality_data['country'].value_counts()
            print(f"\nTop countries:")
            for country, count in country_counts.head(5).items():
                print(f"  {country}: {count} records")
        
        # Temporal analysis
        if 'event_date' in high_quality_data.columns:
            high_quality_data['year'] = pd.to_datetime(high_quality_data['event_date']).dt.year
            year_counts = high_quality_data['year'].value_counts().sort_index()
            print(f"\nRecords by year:")
            for year, count in year_counts.tail(5).items():
                print(f"  {year}: {count} records")
    
    print("\nBatch processing example complete!")


if __name__ == "__main__":
    main()