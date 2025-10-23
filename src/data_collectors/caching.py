"""
Caching system for data collectors to avoid repeated API calls.

This module provides intelligent caching with TTL, compression, and
metadata tracking for efficient data collection workflows.
"""

import logging
import sqlite3
import json
import pickle
import hashlib
import gzip
from pathlib import Path
from typing import Any, Dict, Optional, Union, List
from datetime import datetime, timedelta
import pandas as pd

logger = logging.getLogger(__name__)


class DataCache:
    """Intelligent caching system for data collection."""
    
    def __init__(self, cache_dir: Path, ttl_hours: int = 24, max_size_mb: int = 1000):
        """
        Initialize the caching system.
        
        Args:
            cache_dir: Directory to store cache files
            ttl_hours: Time-to-live for cached data in hours
            max_size_mb: Maximum cache size in MB
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl_hours = ttl_hours
        self.max_size_mb = max_size_mb
        
        # Initialize SQLite database for metadata
        self.db_path = self.cache_dir / "cache_metadata.db"
        self._init_database()
        
    def _init_database(self):
        """Initialize the cache metadata database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache_entries (
                    cache_key TEXT PRIMARY KEY,
                    source TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    created_at TIMESTAMP NOT NULL,
                    expires_at TIMESTAMP NOT NULL,
                    file_size INTEGER NOT NULL,
                    record_count INTEGER,
                    metadata TEXT,
                    access_count INTEGER DEFAULT 0,
                    last_accessed TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_expires_at ON cache_entries(expires_at)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_source ON cache_entries(source)
            """)
    
    def _generate_cache_key(self, source: str, params: Dict[str, Any]) -> str:
        """Generate a unique cache key for the request."""
        # Sort parameters for consistent hashing
        sorted_params = json.dumps(params, sort_keys=True)
        
        # Create hash of source + parameters
        key_string = f"{source}:{sorted_params}"
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def get(self, source: str, params: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """
        Retrieve data from cache.
        
        Args:
            source: Data source name
            params: Request parameters
            
        Returns:
            Cached DataFrame or None if not found/expired
        """
        cache_key = self._generate_cache_key(source, params)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT file_path, expires_at, record_count, metadata
                FROM cache_entries 
                WHERE cache_key = ? AND expires_at > ?
            """, (cache_key, datetime.now()))
            
            row = cursor.fetchone()
            
            if row:
                file_path, expires_at, record_count, metadata = row
                
                # Update access statistics
                conn.execute("""
                    UPDATE cache_entries 
                    SET access_count = access_count + 1, last_accessed = ?
                    WHERE cache_key = ?
                """, (datetime.now(), cache_key))
                
                # Load cached data
                try:
                    df = self._load_cached_data(Path(file_path))
                    logger.info(f"Cache hit for {source}: {record_count} records")
                    return df
                except Exception as e:
                    logger.warning(f"Error loading cached data: {e}")
                    # Remove corrupted cache entry
                    self._remove_cache_entry(cache_key)
                    return None
            else:
                logger.debug(f"Cache miss for {source}")
                return None
    
    def put(self, source: str, params: Dict[str, Any], data: pd.DataFrame, 
            metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Store data in cache.
        
        Args:
            source: Data source name
            params: Request parameters
            data: DataFrame to cache
            metadata: Optional metadata
            
        Returns:
            True if successfully cached, False otherwise
        """
        if data.empty:
            logger.warning(f"Not caching empty DataFrame for {source}")
            return False
        
        cache_key = self._generate_cache_key(source, params)
        expires_at = datetime.now() + timedelta(hours=self.ttl_hours)
        
        # Generate file path
        file_path = self.cache_dir / f"{cache_key}.parquet.gz"
        
        try:
            # Save data with compression
            self._save_cached_data(data, file_path)
            
            # Calculate file size
            file_size = file_path.stat().st_size
            
            # Store metadata in database
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO cache_entries 
                    (cache_key, source, file_path, created_at, expires_at, 
                     file_size, record_count, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    cache_key, source, str(file_path), datetime.now(), expires_at,
                    file_size, len(data), json.dumps(metadata) if metadata else None
                ))
            
            logger.info(f"Cached {len(data)} records for {source}")
            
            # Clean up old entries if cache is too large
            self._cleanup_cache()
            
            return True
            
        except Exception as e:
            logger.error(f"Error caching data for {source}: {e}")
            return False
    
    def _save_cached_data(self, data: pd.DataFrame, file_path: Path):
        """Save DataFrame to compressed file."""
        # Use parquet for efficient storage
        temp_path = file_path.with_suffix('.tmp')
        data.to_parquet(temp_path, compression='gzip')
        temp_path.rename(file_path)
    
    def _load_cached_data(self, file_path: Path) -> pd.DataFrame:
        """Load DataFrame from compressed file."""
        return pd.read_parquet(file_path)
    
    def _remove_cache_entry(self, cache_key: str):
        """Remove a cache entry and its file."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT file_path FROM cache_entries WHERE cache_key = ?
            """, (cache_key,))
            
            row = cursor.fetchone()
            if row:
                file_path = Path(row[0])
                if file_path.exists():
                    file_path.unlink()
                
                conn.execute("""
                    DELETE FROM cache_entries WHERE cache_key = ?
                """, (cache_key,))
    
    def _cleanup_cache(self):
        """Clean up old and oversized cache entries."""
        # Remove expired entries
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT cache_key, file_path FROM cache_entries 
                WHERE expires_at <= ?
            """, (datetime.now(),))
            
            for cache_key, file_path in cursor.fetchall():
                self._remove_cache_entry(cache_key)
        
        # Check total cache size
        total_size = self._get_cache_size()
        if total_size > self.max_size_mb * 1024 * 1024:  # Convert to bytes
            self._evict_oldest_entries()
    
    def _get_cache_size(self) -> int:
        """Get total cache size in bytes."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT SUM(file_size) FROM cache_entries")
            result = cursor.fetchone()
            return result[0] if result[0] else 0
    
    def _evict_oldest_entries(self):
        """Evict oldest cache entries to make space."""
        target_size = self.max_size_mb * 1024 * 1024 * 0.8  # 80% of max size
        
        with sqlite3.connect(self.db_path) as conn:
            # Get entries sorted by last accessed (oldest first)
            cursor = conn.execute("""
                SELECT cache_key, file_size FROM cache_entries 
                ORDER BY last_accessed ASC, created_at ASC
            """)
            
            current_size = self._get_cache_size()
            for cache_key, file_size in cursor.fetchall():
                if current_size <= target_size:
                    break
                
                self._remove_cache_entry(cache_key)
                current_size -= file_size
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT 
                    COUNT(*) as total_entries,
                    SUM(file_size) as total_size,
                    SUM(record_count) as total_records,
                    AVG(access_count) as avg_access_count
                FROM cache_entries
            """)
            
            stats = cursor.fetchone()
            
            # Get source breakdown
            cursor = conn.execute("""
                SELECT source, COUNT(*) as count, SUM(file_size) as size
                FROM cache_entries 
                GROUP BY source
            """)
            
            source_breakdown = {
                row[0]: {'count': row[1], 'size_mb': row[2] / (1024 * 1024)}
                for row in cursor.fetchall()
            }
            
            return {
                'total_entries': stats[0] or 0,
                'total_size_mb': (stats[1] or 0) / (1024 * 1024),
                'total_records': stats[2] or 0,
                'avg_access_count': stats[3] or 0,
                'source_breakdown': source_breakdown
            }
    
    def clear_cache(self, source: Optional[str] = None):
        """Clear cache entries, optionally for a specific source."""
        with sqlite3.connect(self.db_path) as conn:
            if source:
                cursor = conn.execute("""
                    SELECT cache_key, file_path FROM cache_entries WHERE source = ?
                """, (source,))
            else:
                cursor = conn.execute("""
                    SELECT cache_key, file_path FROM cache_entries
                """)
            
            for cache_key, file_path in cursor.fetchall():
                self._remove_cache_entry(cache_key)
            
            logger.info(f"Cleared cache for {source or 'all sources'}")
    
    def get_cache_info(self, source: str, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Get information about a cached entry."""
        cache_key = self._generate_cache_key(source, params)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT created_at, expires_at, file_size, record_count, 
                       metadata, access_count, last_accessed
                FROM cache_entries WHERE cache_key = ?
            """, (cache_key,))
            
            row = cursor.fetchone()
            if row:
                return {
                    'created_at': row[0],
                    'expires_at': row[1],
                    'file_size_mb': row[2] / (1024 * 1024),
                    'record_count': row[3],
                    'metadata': json.loads(row[4]) if row[4] else None,
                    'access_count': row[5],
                    'last_accessed': row[6],
                    'is_expired': datetime.fromisoformat(row[1]) < datetime.now()
                }
        
        return None