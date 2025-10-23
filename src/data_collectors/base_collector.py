"""
Base class for data collectors with common functionality.
"""
import logging
import time
import json
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Tuple
import requests
import pandas as pd
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)


class BaseCollector(ABC):
    """Base class for all data collectors."""
    
    def __init__(self, config: Dict[str, Any], data_dir: Path):
        self.config = config
        self.data_dir = data_dir
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'GTHA/0.1.0 (https://github.com/gtha/truffle-atlas)'
        })
        
        # Ensure data directory exists
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
    def _make_request(self, url: str, params: Optional[Dict] = None, 
                     timeout: int = 30, max_retries: int = 3, 
                     rate_limit_delay: float = 0.1) -> requests.Response:
        """Make HTTP request with retry logic and rate limiting."""
        for attempt in range(max_retries):
            try:
                response = self.session.get(url, params=params, timeout=timeout)
                response.raise_for_status()
                
                # Apply rate limiting
                if rate_limit_delay > 0:
                    time.sleep(rate_limit_delay)
                    
                return response
            except requests.exceptions.RequestException as e:
                logger.warning(f"Request attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    raise
                time.sleep(2 ** attempt)  # Exponential backoff
                
    def _save_data(self, data: Union[pd.DataFrame, Dict], filename: str) -> Path:
        """Save data to file with automatic format detection."""
        filepath = self.data_dir / filename
        
        if isinstance(data, pd.DataFrame):
            if filename.endswith('.csv'):
                data.to_csv(filepath, index=False)
            elif filename.endswith('.parquet'):
                data.to_parquet(filepath, index=False)
            elif filename.endswith('.json'):
                data.to_json(filepath, orient='records', indent=2)
            else:
                # Default to CSV if format not recognized
                data.to_csv(filepath, index=False)
        else:
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
                
        logger.info(f"Data saved to {filepath}")
        return filepath
    
    def _validate_coordinates(self, df: pd.DataFrame, 
                             lat_col: str = 'latitude', 
                             lon_col: str = 'longitude') -> pd.DataFrame:
        """Validate and clean coordinate data."""
        if lat_col not in df.columns or lon_col not in df.columns:
            logger.warning(f"Coordinate columns {lat_col} or {lon_col} not found")
            return df
        
        # Convert to numeric
        df[lat_col] = pd.to_numeric(df[lat_col], errors='coerce')
        df[lon_col] = pd.to_numeric(df[lon_col], errors='coerce')
        
        # Remove invalid coordinates
        initial_count = len(df)
        df = df.dropna(subset=[lat_col, lon_col])
        
        # Check coordinate ranges
        valid_lat = (df[lat_col] >= -90) & (df[lat_col] <= 90)
        valid_lon = (df[lon_col] >= -180) & (df[lon_col] <= 180)
        df = df[valid_lat & valid_lon]
        
        removed_count = initial_count - len(df)
        if removed_count > 0:
            logger.info(f"Removed {removed_count} records with invalid coordinates")
        
        return df
    
    def _standardize_species_names(self, df: pd.DataFrame, 
                                  species_col: str = 'species') -> pd.DataFrame:
        """Standardize species names in the dataset."""
        if species_col not in df.columns:
            return df
        
        # Create standardized species name
        df[f'{species_col}_standardized'] = (
            df[species_col]
            .str.strip()
            .str.lower()
            .str.replace(r'\s+', ' ', regex=True)  # Normalize whitespace
        )
        
        return df
    
    def _add_derived_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add common derived fields to the dataset."""
        if df.empty:
            return df
        
        # Add coordinate presence indicator
        if 'latitude' in df.columns and 'longitude' in df.columns:
            df['has_coordinates'] = df['latitude'].notna() & df['longitude'].notna()
        
        # Add data quality indicators
        if 'coordinate_uncertainty' in df.columns:
            df['coordinate_quality'] = df['coordinate_uncertainty'].apply(
                lambda x: 'high' if pd.isna(x) or x <= 1000 else 
                         'medium' if x <= 10000 else 'low'
            )
        
        # Add temporal indicators
        date_cols = [col for col in df.columns if 'date' in col.lower()]
        for col in date_cols:
            if col in df.columns:
                df[f'{col}_year'] = pd.to_datetime(df[col], errors='coerce').dt.year
                df[f'{col}_month'] = pd.to_datetime(df[col], errors='coerce').dt.month
        
        return df
    
    def _clean_numeric_columns(self, df: pd.DataFrame, 
                              columns: List[str]) -> pd.DataFrame:
        """Clean and convert numeric columns."""
        for col in columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        return df
    
    def _clean_date_columns(self, df: pd.DataFrame, 
                           columns: List[str]) -> pd.DataFrame:
        """Clean and convert date columns."""
        for col in columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        return df
    
    def _get_data_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get summary statistics for the dataset."""
        if df.empty:
            return {'record_count': 0, 'columns': 0}
        
        summary = {
            'record_count': len(df),
            'columns': len(df.columns),
            'coordinate_coverage': 0,
            'temporal_coverage': 0,
            'missing_data_percentage': 0
        }
        
        # Calculate coordinate coverage
        if 'latitude' in df.columns and 'longitude' in df.columns:
            coord_count = df[['latitude', 'longitude']].dropna().shape[0]
            summary['coordinate_coverage'] = coord_count / len(df) * 100
        
        # Calculate temporal coverage
        date_cols = [col for col in df.columns if 'date' in col.lower()]
        if date_cols:
            date_count = df[date_cols].dropna(how='all').shape[0]
            summary['temporal_coverage'] = date_count / len(df) * 100
        
        # Calculate missing data percentage
        total_cells = df.shape[0] * df.shape[1]
        missing_cells = df.isnull().sum().sum()
        summary['missing_data_percentage'] = missing_cells / total_cells * 100
        
        return summary
    
    def _log_data_summary(self, df: pd.DataFrame, source: str) -> None:
        """Log summary statistics for collected data."""
        summary = self._get_data_summary(df)
        logger.info(f"{source} data summary: {summary['record_count']} records, "
                   f"{summary['columns']} columns, "
                   f"{summary['coordinate_coverage']:.1f}% coordinate coverage")
        
    def _merge_dataframes(self, dataframes: List[pd.DataFrame], 
                         on: str = 'latitude', 
                         how: str = 'outer') -> pd.DataFrame:
        """Merge multiple dataframes on common columns."""
        if not dataframes:
            return pd.DataFrame()
        
        if len(dataframes) == 1:
            return dataframes[0]
        
        result = dataframes[0]
        for df in dataframes[1:]:
            if not df.empty:
                result = pd.merge(result, df, on=on, how=how, suffixes=('', '_dup'))
        
        return result
    
    def _filter_by_bounds(self, df: pd.DataFrame, 
                         min_lat: float, max_lat: float,
                         min_lon: float, max_lon: float,
                         lat_col: str = 'latitude', 
                         lon_col: str = 'longitude') -> pd.DataFrame:
        """Filter dataframe by geographic bounds."""
        if df.empty:
            return df
        
        if lat_col not in df.columns or lon_col not in df.columns:
            logger.warning("Coordinate columns not found for filtering")
            return df
        
        mask = (
            (df[lat_col] >= min_lat) & (df[lat_col] <= max_lat) &
            (df[lon_col] >= min_lon) & (df[lon_col] <= max_lon)
        )
        
        filtered_df = df[mask].copy()
        logger.info(f"Filtered to {len(filtered_df)} records within bounds")
        
        return filtered_df
        
    @abstractmethod
    def collect(self, **kwargs) -> pd.DataFrame:
        """Collect data from the source."""
        pass
        
    @abstractmethod
    def validate_data(self, data: pd.DataFrame) -> bool:
        """Validate collected data."""
        pass