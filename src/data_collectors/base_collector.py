"""
Base class for data collectors with common functionality.
"""
import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
import requests
import pandas as pd
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
        
    def _make_request(self, url: str, params: Optional[Dict] = None, 
                     timeout: int = 30, max_retries: int = 3) -> requests.Response:
        """Make HTTP request with retry logic."""
        for attempt in range(max_retries):
            try:
                response = self.session.get(url, params=params, timeout=timeout)
                response.raise_for_status()
                return response
            except requests.exceptions.RequestException as e:
                logger.warning(f"Request attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    raise
                time.sleep(2 ** attempt)  # Exponential backoff
                
    def _save_data(self, data: Union[pd.DataFrame, Dict], filename: str) -> Path:
        """Save data to file."""
        filepath = self.data_dir / filename
        
        if isinstance(data, pd.DataFrame):
            if filename.endswith('.csv'):
                data.to_csv(filepath, index=False)
            elif filename.endswith('.parquet'):
                data.to_parquet(filepath, index=False)
            elif filename.endswith('.json'):
                data.to_json(filepath, orient='records', indent=2)
        else:
            import json
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
                
        logger.info(f"Data saved to {filepath}")
        return filepath
        
    @abstractmethod
    def collect(self, **kwargs) -> pd.DataFrame:
        """Collect data from the source."""
        pass
        
    @abstractmethod
    def validate_data(self, data: pd.DataFrame) -> bool:
        """Validate collected data."""
        pass