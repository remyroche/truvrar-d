"""
Comprehensive error handling utilities for the GTHA project.
"""
import logging
import traceback
from typing import Any, Dict, Optional, Callable, Type
from functools import wraps
import requests
import pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)


class GTHAError(Exception):
    """Base exception for GTHA-specific errors."""
    pass


class DataCollectionError(GTHAError):
    """Error during data collection."""
    pass


class DataProcessingError(GTHAError):
    """Error during data processing."""
    pass


class ValidationError(GTHAError):
    """Error during data validation."""
    pass


class ConfigurationError(GTHAError):
    """Error in configuration."""
    pass


class APIError(GTHAError):
    """Error from external API calls."""
    pass


def handle_api_errors(func: Callable) -> Callable:
    """Decorator to handle API-related errors."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except requests.exceptions.ConnectionError as e:
            logger.error(f"Connection error in {func.__name__}: {e}")
            raise APIError(f"Failed to connect to API: {e}")
        except requests.exceptions.Timeout as e:
            logger.error(f"Timeout error in {func.__name__}: {e}")
            raise APIError(f"API request timed out: {e}")
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error in {func.__name__}: {e}")
            raise APIError(f"HTTP error from API: {e}")
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error in {func.__name__}: {e}")
            raise APIError(f"Request failed: {e}")
        except Exception as e:
            logger.error(f"Unexpected error in {func.__name__}: {e}")
            raise DataCollectionError(f"Unexpected error during data collection: {e}")
    return wrapper


def handle_data_processing_errors(func: Callable) -> Callable:
    """Decorator to handle data processing errors."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except pd.errors.EmptyDataError as e:
            logger.error(f"Empty data error in {func.__name__}: {e}")
            raise DataProcessingError(f"No data to process: {e}")
        except pd.errors.ParserError as e:
            logger.error(f"Data parsing error in {func.__name__}: {e}")
            raise DataProcessingError(f"Failed to parse data: {e}")
        except ValueError as e:
            logger.error(f"Value error in {func.__name__}: {e}")
            raise DataProcessingError(f"Invalid data value: {e}")
        except Exception as e:
            logger.error(f"Unexpected error in {func.__name__}: {e}")
            raise DataProcessingError(f"Unexpected error during data processing: {e}")
    return wrapper


def handle_file_operations(func: Callable) -> Callable:
    """Decorator to handle file operation errors."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except FileNotFoundError as e:
            logger.error(f"File not found in {func.__name__}: {e}")
            raise DataProcessingError(f"Required file not found: {e}")
        except PermissionError as e:
            logger.error(f"Permission error in {func.__name__}: {e}")
            raise DataProcessingError(f"Permission denied for file operation: {e}")
        except OSError as e:
            logger.error(f"OS error in {func.__name__}: {e}")
            raise DataProcessingError(f"File system error: {e}")
        except Exception as e:
            logger.error(f"Unexpected error in {func.__name__}: {e}")
            raise DataProcessingError(f"Unexpected error during file operation: {e}")
    return wrapper


class ErrorHandler:
    """Centralized error handling class."""
    
    def __init__(self, log_errors: bool = True, raise_errors: bool = True):
        self.log_errors = log_errors
        self.raise_errors = raise_errors
        
    def handle_error(self, error: Exception, context: str = "", 
                    custom_message: str = "") -> Optional[Any]:
        """
        Handle an error with logging and optional re-raising.
        
        Args:
            error: The exception to handle
            context: Additional context about where the error occurred
            custom_message: Custom error message
            
        Returns:
            None or default value if not re-raising
        """
        error_msg = custom_message or str(error)
        full_context = f"{context}: {error_msg}" if context else error_msg
        
        if self.log_errors:
            logger.error(f"Error in {context}: {error}")
            logger.debug(f"Traceback: {traceback.format_exc()}")
        
        if self.raise_errors:
            raise error
        else:
            return None
    
    def safe_execute(self, func: Callable, *args, default_return: Any = None, 
                    context: str = "", **kwargs) -> Any:
        """
        Safely execute a function with error handling.
        
        Args:
            func: Function to execute
            *args: Function arguments
            default_return: Value to return if function fails
            context: Context for error logging
            **kwargs: Function keyword arguments
            
        Returns:
            Function result or default_return if error occurs
        """
        try:
            return func(*args, **kwargs)
        except Exception as e:
            self.handle_error(e, context, f"Failed to execute {func.__name__}")
            return default_return


def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate configuration dictionary.
    
    Args:
        config: Configuration dictionary to validate
        
    Returns:
        True if valid, raises ConfigurationError if invalid
    """
    required_sections = ['gbif', 'inaturalist', 'soilgrids', 'worldclim']
    
    for section in required_sections:
        if section not in config:
            raise ConfigurationError(f"Missing required configuration section: {section}")
        
        section_config = config[section]
        if not isinstance(section_config, dict):
            raise ConfigurationError(f"Configuration section {section} must be a dictionary")
        
        required_keys = ['base_url', 'timeout', 'max_retries']
        for key in required_keys:
            if key not in section_config:
                raise ConfigurationError(f"Missing required key {key} in {section} configuration")
    
    return True


def validate_dataframe(df: pd.DataFrame, required_columns: list = None, 
                      min_rows: int = 0) -> bool:
    """
    Validate a pandas DataFrame.
    
    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        min_rows: Minimum number of rows required
        
    Returns:
        True if valid, raises ValidationError if invalid
    """
    if df is None:
        raise ValidationError("DataFrame is None")
    
    if not isinstance(df, pd.DataFrame):
        raise ValidationError("Object is not a pandas DataFrame")
    
    if len(df) < min_rows:
        raise ValidationError(f"DataFrame has {len(df)} rows, minimum required: {min_rows}")
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValidationError(f"Missing required columns: {missing_columns}")
    
    return True


def validate_coordinates(latitude: float, longitude: float) -> bool:
    """
    Validate coordinate values.
    
    Args:
        latitude: Latitude value
        longitude: Longitude value
        
    Returns:
        True if valid, raises ValidationError if invalid
    """
    if latitude is None or longitude is None:
        raise ValidationError("Coordinates cannot be None")
    
    if not isinstance(latitude, (int, float)) or not isinstance(longitude, (int, float)):
        raise ValidationError("Coordinates must be numeric")
    
    if not (-90 <= latitude <= 90):
        raise ValidationError(f"Latitude {latitude} is out of valid range [-90, 90]")
    
    if not (-180 <= longitude <= 180):
        raise ValidationError(f"Longitude {longitude} is out of valid range [-180, 180]")
    
    return True


def retry_on_failure(max_retries: int = 3, delay: float = 1.0, 
                    backoff_factor: float = 2.0, 
                    exceptions: tuple = (Exception,)) -> Callable:
    """
    Decorator to retry function on failure.
    
    Args:
        max_retries: Maximum number of retry attempts
        delay: Initial delay between retries in seconds
        backoff_factor: Multiplier for delay after each retry
        exceptions: Tuple of exceptions to catch and retry on
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            current_delay = delay
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        logger.warning(f"Attempt {attempt + 1} failed for {func.__name__}: {e}")
                        logger.info(f"Retrying in {current_delay} seconds...")
                        import time
                        time.sleep(current_delay)
                        current_delay *= backoff_factor
                    else:
                        logger.error(f"All {max_retries + 1} attempts failed for {func.__name__}")
                        raise last_exception
            
            raise last_exception
        return wrapper
    return decorator


def create_error_report(error: Exception, context: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Create a detailed error report.
    
    Args:
        error: The exception that occurred
        context: Additional context information
        
    Returns:
        Dictionary with error details
    """
    report = {
        'error_type': type(error).__name__,
        'error_message': str(error),
        'traceback': traceback.format_exc(),
        'timestamp': pd.Timestamp.now().isoformat(),
        'context': context or {}
    }
    
    return report


def log_error_report(report: Dict[str, Any], log_file: Optional[Path] = None):
    """
    Log error report to file.
    
    Args:
        report: Error report dictionary
        log_file: Optional log file path
    """
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(log_file, 'a') as f:
            f.write(f"\n{'='*50}\n")
            f.write(f"Error Report - {report['timestamp']}\n")
            f.write(f"{'='*50}\n")
            f.write(f"Type: {report['error_type']}\n")
            f.write(f"Message: {report['error_message']}\n")
            f.write(f"Context: {report['context']}\n")
            f.write(f"Traceback:\n{report['traceback']}\n")
    else:
        logger.error(f"Error Report: {report}")