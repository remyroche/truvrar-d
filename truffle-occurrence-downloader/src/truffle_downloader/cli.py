"""
Command-line interface for Truffle Occurrence Data Downloader
"""

import click
import logging
import json
from pathlib import Path
from typing import List, Optional
import sys

from .downloader import GBIFTruffleDownloader
from .validators import DataValidator
from .exporters import DataExporter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@click.group()
@click.option('--config', '-c', type=click.Path(exists=True), help='Configuration file path')
@click.option('--output-dir', '-o', type=click.Path(), default='./output', help='Output directory')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.pass_context
def cli(ctx, config, output_dir, verbose):
    """Truffle Occurrence Data Downloader - Download truffle data from GBIF"""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    ctx.ensure_object(dict)
    ctx.obj['config'] = config
    ctx.obj['output_dir'] = Path(output_dir)
    ctx.obj['output_dir'].mkdir(parents=True, exist_ok=True)


@cli.command()
@click.option('--species', '-s', multiple=True, help='Species names to download (can be used multiple times)')
@click.option('--species-file', type=click.Path(exists=True), help='File containing species names (one per line)')
@click.option('--countries', multiple=True, help='Country codes to filter by (e.g., FR IT ES)')
@click.option('--year-from', type=int, help='Start year for temporal filtering')
@click.option('--year-to', type=int, help='End year for temporal filtering')
@click.option('--max-records', type=int, help='Maximum number of records to download')
@click.option('--format', 'output_format', type=click.Choice(['csv', 'geojson', 'parquet', 'shp']), 
              default='csv', help='Output format')
@click.option('--filename', help='Output filename (without extension)')
@click.option('--compress', is_flag=True, help='Compress output files')
@click.option('--validate', is_flag=True, help='Validate data quality')
@click.option('--package', is_flag=True, help='Create complete export package')
@click.pass_context
def download(ctx, species, species_file, countries, year_from, year_to, max_records, 
            output_format, filename, compress, validate, package):
    """Download truffle occurrence data from GBIF"""
    
    # Get species list
    species_list = list(species)
    if species_file:
        with open(species_file, 'r') as f:
            species_list.extend([line.strip() for line in f if line.strip()])
    
    if not species_list:
        click.echo("Error: No species specified. Use --species or --species-file", err=True)
        sys.exit(1)
    
    # Initialize downloader
    downloader = GBIFTruffleDownloader(ctx.obj['config'])
    
    # Download data
    click.echo(f"Downloading data for {len(species_list)} species...")
    try:
        data = downloader.download_species(
            species=species_list,
            countries=list(countries) if countries else None,
            year_from=year_from,
            year_to=year_to,
            max_records=max_records
        )
        
        if data.empty:
            click.echo("No data found for the specified criteria", err=True)
            sys.exit(1)
        
        click.echo(f"Downloaded {len(data)} records")
        
    except Exception as e:
        click.echo(f"Error downloading data: {e}", err=True)
        sys.exit(1)
    
    # Validate data if requested
    validation_results = None
    if validate:
        click.echo("Validating data quality...")
        validator = DataValidator()
        validation_results = validator.validate_data(data)
        
        click.echo(f"Data quality: {validation_results['quality_metrics'].get('rating', 'Unknown')}")
        click.echo(f"Overall score: {validation_results['quality_metrics'].get('overall_score', 0):.1f}/100")
        
        if validation_results['errors']:
            click.echo(f"Errors: {len(validation_results['errors'])}")
            for error in validation_results['errors'][:3]:
                click.echo(f"  - {error}")
        
        if validation_results['warnings']:
            click.echo(f"Warnings: {len(validation_results['warnings'])}")
            for warning in validation_results['warnings'][:3]:
                click.echo(f"  - {warning}")
    
    # Export data
    exporter = DataExporter(ctx.obj['output_dir'])
    
    if package:
        # Create complete export package
        click.echo("Creating export package...")
        download_stats = downloader.get_download_stats()
        package_path = exporter.create_export_package(
            data, validation_results, download_stats
        )
        click.echo(f"Export package created: {package_path}")
    else:
        # Export in single format
        if not filename:
            filename = f"truffle_occurrences_{len(species_list)}_species"
        
        try:
            file_path = exporter.export_data(
                data, filename, output_format, compress=compress
            )
            click.echo(f"Data exported to: {file_path}")
        except Exception as e:
            click.echo(f"Error exporting data: {e}", err=True)
            sys.exit(1)


@cli.command()
@click.option('--species', '-s', help='Species name to search for')
@click.option('--limit', type=int, default=10, help='Maximum number of results')
@click.pass_context
def search(ctx, species, limit):
    """Search for species in GBIF"""
    
    if not species:
        click.echo("Error: Species name required", err=True)
        sys.exit(1)
    
    downloader = GBIFTruffleDownloader(ctx.obj['config'])
    
    try:
        results = downloader.search_species(species, limit)
        
        if not results:
            click.echo("No species found")
            return
        
        click.echo(f"Found {len(results)} species matching '{species}':")
        click.echo()
        
        for i, result in enumerate(results, 1):
            click.echo(f"{i}. {result.get('scientificName', 'Unknown')}")
            click.echo(f"   Key: {result.get('key', 'Unknown')}")
            click.echo(f"   Rank: {result.get('rank', 'Unknown')}")
            if 'canonicalName' in result:
                click.echo(f"   Canonical: {result['canonicalName']}")
            click.echo()
            
    except Exception as e:
        click.echo(f"Error searching for species: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--species-key', type=int, help='GBIF species key')
@click.option('--species-name', help='Species name (will be resolved to key)')
@click.pass_context
def info(ctx, species_key, species_name):
    """Get detailed information about a species"""
    
    if not species_key and not species_name:
        click.echo("Error: Either --species-key or --species-name required", err=True)
        sys.exit(1)
    
    downloader = GBIFTruffleDownloader(ctx.obj['config'])
    
    try:
        if species_name and not species_key:
            # Resolve species name to key
            results = downloader.search_species(species_name, 1)
            if not results:
                click.echo(f"Species '{species_name}' not found")
                return
            species_key = results[0]['key']
        
        info = downloader.get_species_info(species_key)
        
        if not info:
            click.echo("Species information not found")
            return
        
        click.echo(f"Species Information:")
        click.echo(f"Scientific Name: {info.get('scientificName', 'Unknown')}")
        click.echo(f"Canonical Name: {info.get('canonicalName', 'Unknown')}")
        click.echo(f"Key: {info.get('key', 'Unknown')}")
        click.echo(f"Rank: {info.get('rank', 'Unknown')}")
        click.echo(f"Kingdom: {info.get('kingdom', 'Unknown')}")
        click.echo(f"Phylum: {info.get('phylum', 'Unknown')}")
        click.echo(f"Class: {info.get('class', 'Unknown')}")
        click.echo(f"Order: {info.get('order', 'Unknown')}")
        click.echo(f"Family: {info.get('family', 'Unknown')}")
        click.echo(f"Genus: {info.get('genus', 'Unknown')}")
        
        if 'vernacularNames' in info and info['vernacularNames']:
            click.echo(f"Common Names:")
            for name in info['vernacularNames'][:5]:  # Show first 5
                click.echo(f"  - {name.get('vernacularName', 'Unknown')} ({name.get('language', 'Unknown')})")
        
    except Exception as e:
        click.echo(f"Error getting species info: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--input-file', '-i', type=click.Path(exists=True), required=True, 
              help='Input CSV file with occurrence data')
@click.option('--output-file', '-o', help='Output file for validation report')
@click.pass_context
def validate(ctx, input_file, output_file):
    """Validate existing occurrence data"""
    
    try:
        import pandas as pd
        data = pd.read_csv(input_file)
    except Exception as e:
        click.echo(f"Error reading input file: {e}", err=True)
        sys.exit(1)
    
    click.echo("Validating data...")
    validator = DataValidator()
    validation_results = validator.validate_data(data)
    
    # Display results
    click.echo(validator.get_validation_summary())
    
    # Save detailed report if requested
    if output_file:
        exporter = DataExporter(ctx.obj['output_dir'])
        report_path = exporter.export_quality_report(validation_results)
        click.echo(f"Detailed validation report saved to: {report_path}")


@cli.command()
@click.option('--input-file', '-i', type=click.Path(exists=True), required=True,
              help='Input CSV file with occurrence data')
@click.option('--format', 'output_format', type=click.Choice(['csv', 'geojson', 'parquet', 'shp']),
              default='csv', help='Output format')
@click.option('--filename', help='Output filename (without extension)')
@click.option('--compress', is_flag=True, help='Compress output files')
@click.pass_context
def convert(ctx, input_file, output_format, filename, compress):
    """Convert existing occurrence data to different formats"""
    
    try:
        import pandas as pd
        data = pd.read_csv(input_file)
    except Exception as e:
        click.echo(f"Error reading input file: {e}", err=True)
        sys.exit(1)
    
    if not filename:
        filename = f"converted_data_{output_format}"
    
    exporter = DataExporter(ctx.obj['output_dir'])
    
    try:
        file_path = exporter.export_data(data, filename, output_format, compress=compress)
        click.echo(f"Data converted and saved to: {file_path}")
    except Exception as e:
        click.echo(f"Error converting data: {e}", err=True)
        sys.exit(1)


def main():
    """Main entry point for the CLI"""
    cli()


if __name__ == '__main__':
    main()