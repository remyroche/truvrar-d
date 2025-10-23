"""
OpenAlex API connector for academic research extraction.
"""

import asyncio
from typing import List, Dict, Any, Optional, AsyncGenerator
from urllib.parse import urlencode
from loguru import logger

from .base_connector import BaseConnector
from ..data_schema import PaperMetadata, Author, License, OpenAccessStatus, Language


class OpenAlexConnector(BaseConnector):
    """OpenAlex API connector."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, rate_limit=config.get("rate_limit", 10))
        self.api_key = config.get("api_key")  # Optional for OpenAlex
    
    async def search_papers(
        self, 
        query: str, 
        year_range: tuple = (1950, 2024),
        max_results: int = 1000
    ) -> AsyncGenerator[PaperMetadata, None]:
        """Search for papers using OpenAlex API."""
        if not self.enabled:
            logger.info("OpenAlex connector is disabled")
            return
        
        logger.info(f"Searching OpenAlex for: {query}")
        
        # Build search query
        search_params = {
            "search": query,
            "filter": f"publication_year:{year_range[0]}-{year_range[1]}",
            "per_page": min(self.batch_size, 200),  # OpenAlex max is 200
            "cursor": "*"
        }
        
        total_found = 0
        cursor = "*"
        
        while total_found < max_results:
            search_params["cursor"] = cursor
            url = f"{self.base_url}/works"
            
            try:
                data = await self._make_request(url, params=search_params)
                
                if not data or "results" not in data:
                    logger.warning("No results from OpenAlex")
                    break
                
                results = data["results"]
                if not results:
                    logger.info("No more results from OpenAlex")
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
                        logger.error(f"Error normalizing OpenAlex paper: {e}")
                        continue
                
                # Check if there are more results
                if "meta" in data and "next_cursor" in data["meta"]:
                    cursor = data["meta"]["next_cursor"]
                else:
                    break
                    
            except Exception as e:
                logger.error(f"Error searching OpenAlex: {e}")
                break
        
        logger.info(f"OpenAlex search completed. Found {total_found} papers.")
    
    def normalize_paper(self, raw_data: Dict[str, Any]) -> Optional[PaperMetadata]:
        """Normalize OpenAlex data to canonical schema."""
        try:
            # Extract basic information
            title = raw_data.get("title", "").strip()
            if not title:
                return None
            
            abstract = raw_data.get("abstract_inverted_index")
            abstract_text = ""
            if abstract:
                # Reconstruct abstract from inverted index
                abstract_text = self._reconstruct_abstract(abstract)
            
            # Extract authors
            authors = []
            for author_data in raw_data.get("authorships", []):
                author_info = author_data.get("author", {})
                if author_info:
                    author = Author(
                        name=author_info.get("display_name", "").strip(),
                        orcid=author_info.get("orcid"),
                        affiliation=None  # Will be extracted from institutions
                    )
                    if author.name:
                        authors.append(author)
            
            # Extract affiliations
            affiliations = []
            for authorship in raw_data.get("authorships", []):
                for institution in authorship.get("institutions", []):
                    if institution.get("display_name"):
                        affiliations.append(institution["display_name"])
            
            # Extract publication info
            publication_date = raw_data.get("publication_date", "")
            year = self._extract_year(publication_date)
            
            # Extract journal
            primary_location = raw_data.get("primary_location", {})
            journal = None
            if primary_location and primary_location.get("source"):
                journal = primary_location["source"].get("display_name")
            
            # Extract DOI
            doi = None
            for identifier in raw_data.get("ids", {}):
                if identifier == "doi":
                    doi = raw_data["ids"][identifier]
                    break
            
            # Extract open access info
            oa_info = raw_data.get("open_access", {})
            oa_status = OpenAccessStatus.UNKNOWN
            if oa_info.get("is_oa"):
                oa_status = OpenAccessStatus.OPEN
            
            # Extract best PDF/HTML URLs
            best_pdf_url = None
            best_html_url = None
            for location in raw_data.get("locations", []):
                if location.get("is_oa"):
                    pdf_url = location.get("pdf_url")
                    html_url = location.get("landing_page_url")
                    if pdf_url and not best_pdf_url:
                        best_pdf_url = pdf_url
                    if html_url and not best_html_url:
                        best_html_url = html_url
            
            # Extract subjects
            subjects = []
            for concept in raw_data.get("concepts", []):
                if concept.get("score", 0) > 0.3:  # Only high-confidence concepts
                    subjects.append(concept.get("display_name", ""))
            
            # Extract citations
            citations = raw_data.get("cited_by_count", 0)
            
            # Extract references
            references = []
            for reference in raw_data.get("referenced_works", []):
                if reference:
                    references.append(reference)
            
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
                language = "en"  # Default to English for unsupported languages
            
            return PaperMetadata(
                paper_id=paper_id,
                doi=self._normalize_doi(doi) if doi else None,
                title=title,
                abstract=abstract_text,
                authors=authors,
                year=year,
                journal=journal,
                source="openalex",
                url=raw_data.get("id"),  # OpenAlex URL
                best_pdf_url=best_pdf_url,
                best_html_url=best_html_url,
                oa_status=oa_status,
                subjects=subjects,
                citations=citations,
                references=references,
                affiliations=affiliations,
                language=Language(language) if language else None,
                source_list=["openalex"]
            )
            
        except Exception as e:
            logger.error(f"Error normalizing OpenAlex paper: {e}")
            return None
    
    def _reconstruct_abstract(self, inverted_index: Dict[str, List[int]]) -> str:
        """Reconstruct abstract text from OpenAlex inverted index."""
        if not inverted_index:
            return ""
        
        # Create a list of (position, word) tuples
        word_positions = []
        for word, positions in inverted_index.items():
            for pos in positions:
                word_positions.append((pos, word))
        
        # Sort by position
        word_positions.sort(key=lambda x: x[0])
        
        # Join words
        return " ".join([word for _, word in word_positions])
    
    async def get_paper_details(self, paper_id: str) -> Optional[PaperMetadata]:
        """Get detailed information for a specific paper by OpenAlex ID."""
        if not paper_id.startswith("W"):
            # Convert DOI to OpenAlex ID if needed
            if paper_id.startswith("doi_"):
                doi = paper_id[4:]  # Remove "doi_" prefix
                paper_id = f"https://openalex.org/W{doi.replace('/', '')}"
            else:
                return None
        
        url = f"{self.base_url}/works/{paper_id}"
        
        try:
            data = await self._make_request(url)
            return self.normalize_paper(data)
        except Exception as e:
            logger.error(f"Error getting OpenAlex paper details: {e}")
            return None
