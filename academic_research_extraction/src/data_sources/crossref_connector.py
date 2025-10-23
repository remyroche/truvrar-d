"""
Crossref API connector for academic research extraction.
"""

from typing import List, Dict, Any, Optional, AsyncGenerator
from loguru import logger

from .base_connector import BaseConnector
from ..data_schema import PaperMetadata, Author, OpenAccessStatus, Language


class CrossrefConnector(BaseConnector):
    """Crossref API connector."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, rate_limit=config.get("rate_limit", 50))
        self.mailto = config.get("mailto", "research@example.com")
    
    async def search_papers(
        self, 
        query: str, 
        year_range: tuple = (1950, 2024),
        max_results: int = 1000
    ) -> AsyncGenerator[PaperMetadata, None]:
        """Search for papers using Crossref API."""
        if not self.enabled:
            logger.info("Crossref connector is disabled")
            return
        
        logger.info(f"Searching Crossref for: {query}")
        
        # Build search query
        search_params = {
            "query": query,
            "filter": f"from-pub-date:{year_range[0]},until-pub-date:{year_range[1]}",
            "rows": min(self.batch_size, 1000),  # Crossref max is 1000
            "mailto": self.mailto
        }
        
        offset = 0
        total_found = 0
        
        while total_found < max_results:
            search_params["offset"] = offset
            url = f"{self.base_url}/works"
            
            try:
                data = await self._make_request(url, params=search_params)
                
                if not data or "message" not in data:
                    logger.warning("No results from Crossref")
                    break
                
                results = data["message"].get("items", [])
                if not results:
                    logger.info("No more results from Crossref")
                    break
                
                for result in results:
                    if total_found >= max_results:
                        break
                    
                    try:
                        paper = self.normalize_paper(result)
                        if paper:
                            yield paper
                            total_found += 1
                    except Exception as e:
                        logger.error(f"Error normalizing Crossref paper: {e}")
                        continue
                
                # Check if there are more results
                total_results = data["message"].get("total-results", 0)
                if offset + len(results) >= total_results:
                    break
                
                offset += len(results)
                    
            except Exception as e:
                logger.error(f"Error searching Crossref: {e}")
                break
        
        logger.info(f"Crossref search completed. Found {total_found} papers.")
    
    def normalize_paper(self, raw_data: Dict[str, Any]) -> Optional[PaperMetadata]:
        """Normalize Crossref data to canonical schema."""
        try:
            # Extract basic information
            title = raw_data.get("title", [""])[0] if raw_data.get("title") else ""
            if not title:
                return None
            
            abstract = raw_data.get("abstract", "")
            if abstract and isinstance(abstract, str):
                abstract_text = abstract
            else:
                abstract_text = ""
            
            # Extract authors
            authors = []
            for author_data in raw_data.get("author", []):
                given = author_data.get("given", "")
                family = author_data.get("family", "")
                name = f"{given} {family}".strip()
                
                if name:
                    author = Author(
                        name=name,
                        orcid=author_data.get("ORCID"),
                        affiliation=None
                    )
                    authors.append(author)
            
            # Extract publication info
            published_date = raw_data.get("published-print", {})
            if not published_date:
                published_date = raw_data.get("published-online", {})
            
            year = None
            if published_date.get("date-parts"):
                year = published_date["date-parts"][0][0]
            
            # Extract journal
            journal = None
            if raw_data.get("container-title"):
                journal = raw_data["container-title"][0]
            
            # Extract DOI
            doi = raw_data.get("DOI", "")
            
            # Extract subjects
            subjects = raw_data.get("subject", [])
            
            # Extract citations (not available in Crossref)
            citations = 0
            
            # Generate paper ID
            paper_id = self._generate_paper_id({
                "doi": doi,
                "title": title,
                "authors": [{"name": a.name} for a in authors],
                "year": year
            })
            
            # Detect language
            language = self._detect_language(abstract_text or title)
            if language and language not in ["en", "fr", "it", "es"]:
                language = "en"
            
            return PaperMetadata(
                paper_id=paper_id,
                doi=self._normalize_doi(doi) if doi else None,
                title=title,
                abstract=abstract_text,
                authors=authors,
                year=year,
                journal=journal,
                source="crossref",
                url=raw_data.get("URL"),
                oa_status=OpenAccessStatus.UNKNOWN,
                subjects=subjects,
                citations=citations,
                language=Language(language) if language else None,
                source_list=["crossref"]
            )
            
        except Exception as e:
            logger.error(f"Error normalizing Crossref paper: {e}")
            return None
