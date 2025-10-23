"""
Data export module for truffle occurrence data.

This module provides various export formats and utilities for saving
truffle occurrence data in different formats.
"""

import logging
import pandas as pd
import json
from typing import Dict, List, Optional, Union, Any
from pathlib import Path
import gzip
import zipfile
from datetime import datetime

logger = logging.getLogger(__name__)


class DataExporter:
    """
    Data exporter for truffle occurrence data.
    
    Supports multiple export formats including CSV, GeoJSON, Parquet, and Shapefile.
    """
    
    def __init__(self, output_dir: Optional[Union[str, Path]] = None):
        """
        Initialize the data exporter.
        
        Args:
            output_dir: Default output directory for exported files
        """
        self.output_dir = Path(output_dir) if output_dir else Path.cwd()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def export_data(self, 
                   df: pd.DataFrame, 
                   filename: str,
                   format: str = 'csv',
                   include_metadata: bool = True,
                   compress: bool = False) -> Path:
        """
        Export data in the specified format.
        
        Args:
            df: DataFrame with occurrence data
            filename: Output filename (without extension)
            format: Export format ('csv', 'geojson', 'parquet', 'shp')
            include_metadata: Whether to include metadata in the export
            compress: Whether to compress the output file
            
        Returns:
            Path to the exported file
        """
        if df.empty:
            raise ValueError("Cannot export empty DataFrame")
        
        # Add timestamp to filename if not present
        if not any(char.isdigit() for char in filename):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{filename}_{timestamp}"
        
        # Prepare data for export
        export_df = self._prepare_data_for_export(df, include_metadata)
        
        # Export based on format
        if format.lower() == 'csv':
            file_path = self._export_csv(export_df, filename, compress)
        elif format.lower() == 'geojson':
            file_path = self._export_geojson(export_df, filename, compress)
        elif format.lower() == 'parquet':
            file_path = self._export_parquet(export_df, filename, compress)
        elif format.lower() == 'shp':
            file_path = self._export_shapefile(export_df, filename, compress)
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        logger.info(f"Data exported to {file_path}")
        return file_path
    
    def _prepare_data_for_export(self, df: pd.DataFrame, include_metadata: bool) -> pd.DataFrame:
        """Prepare data for export by cleaning and adding metadata."""
        export_df = df.copy()
        
        if include_metadata:
            # Add export metadata
            export_df['export_date'] = datetime.now().isoformat()
            export_df['export_tool'] = 'TruffleOccurrenceDownloader'
            export_df['export_version'] = '1.0.0'
        
        # Ensure required columns for geospatial formats
        if 'latitude' in export_df.columns and 'longitude' in export_df.columns:
            # Round coordinates to reasonable precision
            export_df['latitude'] = export_df['latitude'].round(6)
            export_df['longitude'] = export_df['longitude'].round(6)
        
        return export_df
    
    def _export_csv(self, df: pd.DataFrame, filename: str, compress: bool) -> Path:
        """Export data as CSV."""
        file_path = self.output_dir / f"{filename}.csv"
        
        if compress:
            file_path = file_path.with_suffix('.csv.gz')
            with gzip.open(file_path, 'wt', encoding='utf-8') as f:
                df.to_csv(f, index=False)
        else:
            df.to_csv(file_path, index=False)
        
        return file_path
    
    def _export_geojson(self, df: pd.DataFrame, filename: str, compress: bool) -> Path:
        """Export data as GeoJSON."""
        try:
            import geopandas as gpd
            from shapely.geometry import Point
        except ImportError:
            raise ImportError("geopandas and shapely are required for GeoJSON export")
        
        if 'latitude' not in df.columns or 'longitude' not in df.columns:
            raise ValueError("Latitude and longitude columns are required for GeoJSON export")
        
        # Create GeoDataFrame
        geometry = [Point(xy) for xy in zip(df['longitude'], df['latitude'])]
        gdf = gpd.GeoDataFrame(df, geometry=geometry, crs='EPSG:4326')
        
        file_path = self.output_dir / f"{filename}.geojson"
        
        if compress:
            file_path = file_path.with_suffix('.geojson.gz')
            with gzip.open(file_path, 'wt', encoding='utf-8') as f:
                f.write(gdf.to_json())
        else:
            gdf.to_file(file_path, driver='GeoJSON')
        
        return file_path
    
    def _export_parquet(self, df: pd.DataFrame, filename: str, compress: bool) -> Path:
        """Export data as Parquet."""
        file_path = self.output_dir / f"{filename}.parquet"
        
        # Parquet has built-in compression
        compression = 'gzip' if compress else None
        df.to_parquet(file_path, index=False, compression=compression)
        
        return file_path
    
    def _export_shapefile(self, df: pd.DataFrame, filename: str, compress: bool) -> Path:
        """Export data as Shapefile."""
        try:
            import geopandas as gpd
            from shapely.geometry import Point
        except ImportError:
            raise ImportError("geopandas and shapely are required for Shapefile export")
        
        if 'latitude' not in df.columns or 'longitude' not in df.columns:
            raise ValueError("Latitude and longitude columns are required for Shapefile export")
        
        # Create GeoDataFrame
        geometry = [Point(xy) for xy in zip(df['longitude'], df['latitude'])]
        gdf = gpd.GeoDataFrame(df, geometry=geometry, crs='EPSG:4326')
        
        # Shapefile doesn't support compression directly, so we'll zip it
        file_path = self.output_dir / f"{filename}.shp"
        
        if compress:
            # Export to temporary directory first
            temp_dir = self.output_dir / "temp_shapefile"
            temp_dir.mkdir(exist_ok=True)
            temp_path = temp_dir / filename
            gdf.to_file(temp_path, driver='ESRI Shapefile')
            
            # Create zip file
            zip_path = self.output_dir / f"{filename}.zip"
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for file in temp_dir.glob(f"{filename}.*"):
                    zipf.write(file, file.name)
            
            # Clean up temp directory
            import shutil
            shutil.rmtree(temp_dir)
            
            return zip_path
        else:
            gdf.to_file(file_path, driver='ESRI Shapefile')
            return file_path
    
    def export_summary_report(self, 
                             df: pd.DataFrame, 
                             validation_results: Optional[Dict[str, Any]] = None,
                             download_stats: Optional[Dict[str, Any]] = None) -> Path:
        """
        Export a summary report of the downloaded data.
        
        Args:
            df: DataFrame with occurrence data
            validation_results: Results from data validation
            download_stats: Download statistics
            
        Returns:
            Path to the summary report
        """
        report = {
            'export_info': {
                'export_date': datetime.now().isoformat(),
                'total_records': len(df),
                'export_tool': 'TruffleOccurrenceDownloader',
                'export_version': '1.0.0'
            }
        }
        
        # Add data summary
        if not df.empty:
            report['data_summary'] = {
                'species_count': df['species'].nunique() if 'species' in df.columns else 0,
                'countries_count': df['country'].nunique() if 'country' in df.columns else 0,
                'date_range': {
                    'earliest': df['event_date'].min().isoformat() if 'event_date' in df.columns and not df['event_date'].isna().all() else None,
                    'latest': df['event_date'].max().isoformat() if 'event_date' in df.columns and not df['event_date'].isna().all() else None
                },
                'coordinate_bounds': {
                    'min_latitude': float(df['latitude'].min()) if 'latitude' in df.columns else None,
                    'max_latitude': float(df['latitude'].max()) if 'latitude' in df.columns else None,
                    'min_longitude': float(df['longitude'].min()) if 'longitude' in df.columns else None,
                    'max_longitude': float(df['longitude'].max()) if 'longitude' in df.columns else None
                }
            }
            
            # Add species breakdown
            if 'species' in df.columns:
                species_counts = df['species'].value_counts().to_dict()
                report['species_breakdown'] = species_counts
            
            # Add country breakdown
            if 'country' in df.columns:
                country_counts = df['country'].value_counts().to_dict()
                report['country_breakdown'] = country_counts
        
        # Add validation results
        if validation_results:
            report['validation_results'] = validation_results
        
        # Add download statistics
        if download_stats:
            report['download_statistics'] = download_stats
        
        # Save report
        report_path = self.output_dir / f"summary_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Summary report exported to {report_path}")
        return report_path
    
    def export_species_list(self, df: pd.DataFrame) -> Path:
        """
        Export a list of unique species found in the data.
        
        Args:
            df: DataFrame with occurrence data
            
        Returns:
            Path to the species list file
        """
        if 'species' not in df.columns:
            raise ValueError("Species column not found in data")
        
        species_list = df['species'].unique().tolist()
        species_list.sort()
        
        species_data = {
            'export_date': datetime.now().isoformat(),
            'total_species': len(species_list),
            'species': species_list
        }
        
        file_path = self.output_dir / f"species_list_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(file_path, 'w') as f:
            json.dump(species_data, f, indent=2)
        
        logger.info(f"Species list exported to {file_path}")
        return file_path
    
    def export_quality_report(self, validation_results: Dict[str, Any]) -> Path:
        """
        Export a detailed quality report.
        
        Args:
            validation_results: Results from data validation
            
        Returns:
            Path to the quality report
        """
        file_path = self.output_dir / f"quality_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(file_path, 'w') as f:
            json.dump(validation_results, f, indent=2, default=str)
        
        logger.info(f"Quality report exported to {file_path}")
        return file_path
    
    def create_export_package(self, 
                             df: pd.DataFrame,
                             validation_results: Optional[Dict[str, Any]] = None,
                             download_stats: Optional[Dict[str, Any]] = None,
                             formats: List[str] = ['csv', 'geojson', 'parquet']) -> Path:
        """
        Create a complete export package with data in multiple formats.
        
        Args:
            df: DataFrame with occurrence data
            validation_results: Results from data validation
            download_stats: Download statistics
            formats: List of export formats to include
            
        Returns:
            Path to the export package (ZIP file)
        """
        package_name = f"truffle_occurrence_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        package_dir = self.output_dir / package_name
        package_dir.mkdir(exist_ok=True)
        
        # Export data in multiple formats
        for format in formats:
            try:
                self.export_data(df, "occurrence_data", format, include_metadata=True)
                # Move to package directory
                source_file = self.output_dir / f"occurrence_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{format}"
                if source_file.exists():
                    source_file.rename(package_dir / f"occurrence_data.{format}")
            except Exception as e:
                logger.warning(f"Failed to export {format} format: {e}")
        
        # Export summary report
        self.export_summary_report(df, validation_results, download_stats)
        summary_file = self.output_dir / f"summary_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        if summary_file.exists():
            summary_file.rename(package_dir / "summary_report.json")
        
        # Export species list
        self.export_species_list(df)
        species_file = self.output_dir / f"species_list_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        if species_file.exists():
            species_file.rename(package_dir / "species_list.json")
        
        # Export quality report if available
        if validation_results:
            self.export_quality_report(validation_results)
            quality_file = self.output_dir / f"quality_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            if quality_file.exists():
                quality_file.rename(package_dir / "quality_report.json")
        
        # Create ZIP package
        zip_path = self.output_dir / f"{package_name}.zip"
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file in package_dir.rglob('*'):
                if file.is_file():
                    zipf.write(file, file.relative_to(package_dir))
        
        # Clean up package directory
        import shutil
        shutil.rmtree(package_dir)
        
        logger.info(f"Export package created: {zip_path}")
        return zip_path