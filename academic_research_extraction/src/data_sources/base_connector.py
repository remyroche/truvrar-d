"""
Base connector class for academic data sources.
"""

import asyncio
import httpx
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, AsyncGenerator
from datetime import datetime
import yaml
from pathlib import Path
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential
import orjson

from ..data_schema import PaperMetadata, Author, License, OpenAccessStatus


class BaseConnector(ABC):
    """Base class for all data source connectors."""
    
    def __init__(self, config: Dict[str, Any], rate_limit: float = 1.0):
        self.config = config
        self.rate_limit = rate_limit
        self.base_url = config.get("base_url", "")
        self.batch_size = config.get("batch_size", 100)
        self.enabled = config.get("enabled", True)
        self.session: Optional[httpx.AsyncClient] = None
        
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = httpx.AsyncClient(
            timeout=30.0,
            limits=httpx.Limits(max_keepalive_connections=20, max_connections=100)
        )
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.aclose()
    
    @abstractmethod
    async def search_papers(
        self, 
        query: str, 
        year_range: tuple = (1950, 2024),
        max_results: int = 1000
    ) -> AsyncGenerator[PaperMetadata, None]:
        """Search for papers matching the query."""
        pass
    
    @abstractmethod
    def normalize_paper(self, raw_data: Dict[str, Any]) -> PaperMetadata:
        """Normalize raw data to canonical schema."""
        pass
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def _make_request(
        self, 
        url: str, 
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """Make HTTP request with retry logic."""
        if not self.session:
            raise RuntimeError("Session not initialized. Use async context manager.")
        
        # Rate limiting
        await asyncio.sleep(1.0 / self.rate_limit)
        
        try:
            response = await self.session.get(url, params=params, headers=headers)
            response.raise_for_status()
            
            # Try to parse as JSON
            try:
                return response.json()
            except Exception:
                # Fallback to orjson for better performance
                return orjson.loads(response.content)
                
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error {e.response.status_code} for {url}: {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"Request failed for {url}: {str(e)}")
            raise
    
    def _extract_authors(self, authors_data: List[Dict[str, Any]]) -> List[Author]:
        """Extract and normalize author information."""
        authors = []
        for author_data in authors_data:
            if isinstance(author_data, dict):
                author = Author(
                    name=author_data.get("name", "").strip(),
                    orcid=author_data.get("orcid"),
                    affiliation=author_data.get("affiliation"),
                    email=author_data.get("email")
                )
                if author.name:  # Only add if name is not empty
                    authors.append(author)
        return authors
    
    def _extract_license(self, license_data: Optional[Dict[str, Any]]) -> Optional[License]:
        """Extract license information."""
        if not license_data:
            return None
            
        return License(
            name=license_data.get("name"),
            url=license_data.get("url"),
            oa_status=OpenAccessStatus(license_data.get("oa_status", "unknown")),
            best_oa_url=license_data.get("best_oa_url"),
            publisher_policy=license_data.get("publisher_policy")
        )
    
    def _normalize_doi(self, doi: str) -> str:
        """Normalize DOI format."""
        if not doi:
            return ""
        doi = doi.strip().lower()
        if not doi.startswith("10."):
            return ""
        return doi
    
    def _normalize_title(self, title: str) -> str:
        """Normalize title for deduplication."""
        if not title:
            return ""
        return title.strip()
    
    def _extract_year(self, date_str: str) -> Optional[int]:
        """Extract year from date string."""
        if not date_str:
            return None
        try:
            # Try to extract year from various date formats
            if isinstance(date_str, str):
                # Look for 4-digit year
                import re
                year_match = re.search(r'\b(19|20)\d{2}\b', date_str)
                if year_match:
                    return int(year_match.group())
            elif isinstance(date_str, (int, float)):
                return int(date_str)
        except (ValueError, TypeError):
            pass
        return None
    
    def _detect_language(self, text: str) -> Optional[str]:
        """Detect language of text."""
        if not text:
            return None
        try:
            from langdetect import detect
            return detect(text)
        except Exception:
            return None
    
    async def get_paper_details(self, paper_id: str) -> Optional[PaperMetadata]:
        """Get detailed information for a specific paper."""
        # Override in subclasses if needed
        return None
    
    def _generate_paper_id(self, raw_data: Dict[str, Any]) -> str:
        """Generate unique paper ID from raw data."""
        # Try DOI first
        doi = raw_data.get("doi", "")
        if doi:
            return f"doi_{self._normalize_doi(doi)}"
        
        # Fallback to title + first author + year
        title = raw_data.get("title", "")
        authors = raw_data.get("authors", [])
        year = raw_data.get("year", "")
        
        first_author = ""
        if authors and len(authors) > 0:
            first_author = authors[0].get("name", "") if isinstance(authors[0], dict) else str(authors[0])
        
        # Create hash from normalized components
        import hashlib
        content = f"{title}_{first_author}_{year}".lower().strip()
        return f"hash_{hashlib.md5(content.encode()).hexdigest()[:12]}"
