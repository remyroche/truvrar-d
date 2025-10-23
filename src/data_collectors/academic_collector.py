"""
Academic data collector for research papers and scholarly sources.

This collector specializes in gathering data from academic sources including:
- PubMed/NCBI
- Google Scholar
- ResearchGate
- Academic databases
- University repositories
"""
import logging
import pandas as pd
import requests
import time
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import json
from datetime import datetime, timezone
import xml.etree.ElementTree as ET

from .base_collector import BaseCollector
from ..utils.error_handling import handle_api_errors, retry_on_failure, APIError, DataCollectionError

logger = logging.getLogger(__name__)


class AcademicDataCollector(BaseCollector):
    """Specialized collector for academic and research data sources."""
    
    def __init__(self, config: Dict[str, Any], data_dir: Path):
        super().__init__(config, data_dir)
        self.academic_sources = {
            'pubmed': self._collect_pubmed_data,
            'google_scholar': self._collect_google_scholar_data,
            'researchgate': self._collect_researchgate_data,
            'crossref': self._collect_crossref_data
        }
        
    def collect_academic_data(self, source: str, search_terms: List[str], 
                            limit: int = 1000, **kwargs) -> Union[pd.DataFrame, Dict[str, Any]]:
        """
        Collect data from academic sources.
        
        Args:
            source: Academic source name
            search_terms: List of search terms
            limit: Maximum number of records to collect
            **kwargs: Additional source-specific parameters
            
        Returns:
            DataFrame with academic data or harmonized results
        """
        if source not in self.academic_sources:
            raise ValueError(f"Unknown academic source: {source}")
        
        logger.info(f"Collecting academic data from {source} for terms: {search_terms}")
        
        try:
            data = self.academic_sources[source](search_terms, limit, **kwargs)
            
            if data.empty:
                logger.warning(f"No academic data found for {source}")
                return pd.DataFrame()
            
            # Add academic-specific metadata
            data = self._add_academic_metadata(data, source)
            
            # Validate academic data
            if not self.validate_data(data):
                logger.warning(f"Academic data validation failed for {source}")
                return pd.DataFrame()
            
            logger.info(f"Collected {len(data)} academic records from {source}")
            return data
            
        except Exception as e:
            logger.error(f"Error collecting academic data from {source}: {e}")
            raise DataCollectionError(f"Failed to collect academic data: {e}")
    
    @handle_api_errors
    @retry_on_failure(max_retries=3, delay=2.0)
    def _collect_pubmed_data(self, search_terms: List[str], limit: int, 
                           **kwargs) -> pd.DataFrame:
        """Collect data from PubMed/NCBI."""
        base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
        all_records = []
        
        for term in search_terms:
            logger.info(f"Searching PubMed for: {term}")
            
            # Search for papers
            search_url = f"{base_url}esearch.fcgi"
            search_params = {
                'db': 'pubmed',
                'term': f"{term}[Title/Abstract] AND (truffle OR Tuber OR mycorrhiza)",
                'retmax': min(limit, 10000),
                'retmode': 'json',
                'sort': 'relevance'
            }
            
            try:
                response = self._make_request(search_url, search_params)
                search_data = response.json()
                
                if 'esearchresult' not in search_data:
                    continue
                
                pmids = search_data['esearchresult'].get('idlist', [])
                if not pmids:
                    logger.info(f"No PubMed results for: {term}")
                    continue
                
                # Fetch detailed records
                records = self._fetch_pubmed_details(pmids[:limit], base_url)
                all_records.extend(records)
                
            except Exception as e:
                logger.error(f"Error searching PubMed for {term}: {e}")
                continue
        
        return pd.DataFrame(all_records)
    
    def _fetch_pubmed_details(self, pmids: List[str], base_url: str) -> List[Dict]:
        """Fetch detailed PubMed records."""
        if not pmids:
            return []
        
        fetch_url = f"{base_url}efetch.fcgi"
        fetch_params = {
            'db': 'pubmed',
            'id': ','.join(pmids),
            'retmode': 'xml',
            'rettype': 'abstract'
        }
        
        try:
            response = self._make_request(fetch_url, fetch_params)
            root = ET.fromstring(response.content)
            
            records = []
            for article in root.findall('.//PubmedArticle'):
                record = self._parse_pubmed_article(article)
                if record:
                    records.append(record)
            
            return records
            
        except Exception as e:
            logger.error(f"Error fetching PubMed details: {e}")
            return []
    
    def _parse_pubmed_article(self, article) -> Optional[Dict]:
        """Parse a PubMed article XML element."""
        try:
            # Extract basic information
            medline_citation = article.find('.//MedlineCitation')
            if medline_citation is None:
                return None
            
            pmid = medline_citation.find('.//PMID')
            pmid_text = pmid.text if pmid is not None else None
            
            # Extract title
            title_elem = article.find('.//ArticleTitle')
            title = title_elem.text if title_elem is not None else ""
            
            # Extract abstract
            abstract_elem = article.find('.//AbstractText')
            abstract = abstract_elem.text if abstract_elem is not None else ""
            
            # Extract authors
            authors = []
            for author in article.findall('.//Author'):
                last_name = author.find('.//LastName')
                first_name = author.find('.//ForeName')
                if last_name is not None:
                    author_name = last_name.text
                    if first_name is not None:
                        author_name += f", {first_name.text}"
                    authors.append(author_name)
            
            # Extract journal
            journal_elem = article.find('.//Journal/Title')
            journal = journal_elem.text if journal_elem is not None else ""
            
            # Extract publication date
            pub_date = self._extract_publication_date(article)
            
            # Extract keywords
            keywords = []
            for keyword in article.findall('.//Keyword'):
                if keyword.text:
                    keywords.append(keyword.text)
            
            # Extract DOI
            doi = None
            for article_id in article.findall('.//ArticleId'):
                if article_id.get('IdType') == 'doi':
                    doi = article_id.text
                    break
            
            return {
                'pmid': pmid_text,
                'title': title,
                'abstract': abstract,
                'authors': '; '.join(authors),
                'journal': journal,
                'publication_date': pub_date,
                'keywords': '; '.join(keywords),
                'doi': doi,
                'source': 'PubMed',
                'search_terms': '; '.join(self._extract_search_terms(title, abstract)),
                'relevance_score': self._calculate_relevance_score(title, abstract),
                'data_type': 'academic_paper'
            }
            
        except Exception as e:
            logger.error(f"Error parsing PubMed article: {e}")
            return None
    
    def _extract_publication_date(self, article) -> Optional[str]:
        """Extract publication date from PubMed article."""
        try:
            pub_date = article.find('.//PubDate')
            if pub_date is None:
                return None
            
            year = pub_date.find('Year')
            month = pub_date.find('Month')
            day = pub_date.find('Day')
            
            if year is not None:
                date_parts = [year.text]
                if month is not None:
                    date_parts.append(month.text.zfill(2))
                if day is not None:
                    date_parts.append(day.text.zfill(2))
                
                return '-'.join(date_parts)
            
        except Exception as e:
            logger.error(f"Error extracting publication date: {e}")
        
        return None
    
    def _extract_search_terms(self, title: str, abstract: str) -> List[str]:
        """Extract relevant search terms from title and abstract."""
        text = f"{title} {abstract}".lower()
        
        truffle_terms = [
            'truffle', 'tuber', 'mycorrhiza', 'ectomycorrhiza',
            'melanosporum', 'magnatum', 'aestivum', 'borchii',
            'brumale', 'mesentericum', 'macrosporum'
        ]
        
        found_terms = [term for term in truffle_terms if term in text]
        return found_terms
    
    def _calculate_relevance_score(self, title: str, abstract: str) -> float:
        """Calculate relevance score for academic paper."""
        text = f"{title} {abstract}".lower()
        
        # Weight different terms
        term_weights = {
            'truffle': 2.0,
            'tuber': 1.5,
            'mycorrhiza': 1.0,
            'ectomycorrhiza': 1.0,
            'melanosporum': 2.0,
            'magnatum': 2.0,
            'aestivum': 1.5,
            'borchii': 1.5,
            'brumale': 1.5,
            'habitat': 1.0,
            'ecology': 1.0,
            'distribution': 1.0,
            'environment': 1.0
        }
        
        score = 0.0
        for term, weight in term_weights.items():
            if term in text:
                score += weight
        
        # Normalize score
        return min(score / 10.0, 1.0)
    
    @handle_api_errors
    def _collect_google_scholar_data(self, search_terms: List[str], limit: int, 
                                   **kwargs) -> pd.DataFrame:
        """Collect data from Google Scholar (limited due to no official API)."""
        logger.warning("Google Scholar collection not implemented - requires web scraping")
        return pd.DataFrame()
    
    @handle_api_errors
    def _collect_researchgate_data(self, search_terms: List[str], limit: int, 
                                 **kwargs) -> pd.DataFrame:
        """Collect data from ResearchGate (limited due to no official API)."""
        logger.warning("ResearchGate collection not implemented - requires web scraping")
        return pd.DataFrame()
    
    @handle_api_errors
    @retry_on_failure(max_retries=3, delay=1.0)
    def _collect_crossref_data(self, search_terms: List[str], limit: int, 
                             **kwargs) -> pd.DataFrame:
        """Collect data from Crossref API."""
        base_url = "https://api.crossref.org/works"
        all_records = []
        
        for term in search_terms:
            logger.info(f"Searching Crossref for: {term}")
            
            params = {
                'query': f"{term} truffle",
                'rows': min(limit, 1000),
                'sort': 'relevance',
                'order': 'desc'
            }
            
            try:
                response = self._make_request(base_url, params)
                data = response.json()
                
                if 'message' not in data or 'items' not in data['message']:
                    continue
                
                for item in data['message']['items']:
                    record = self._parse_crossref_item(item, term)
                    if record:
                        all_records.append(record)
                
            except Exception as e:
                logger.error(f"Error searching Crossref for {term}: {e}")
                continue
        
        return pd.DataFrame(all_records)
    
    def _parse_crossref_item(self, item: Dict, search_term: str) -> Optional[Dict]:
        """Parse a Crossref item."""
        try:
            # Extract basic information
            title = item.get('title', [''])[0] if item.get('title') else ""
            
            # Extract authors
            authors = []
            if 'author' in item:
                for author in item['author']:
                    given = author.get('given', '')
                    family = author.get('family', '')
                    if family:
                        author_name = family
                        if given:
                            author_name += f", {given}"
                        authors.append(author_name)
            
            # Extract publication date
            pub_date = None
            if 'published-print' in item:
                date_parts = item['published-print']['date-parts'][0]
                pub_date = '-'.join(str(part) for part in date_parts)
            elif 'published-online' in item:
                date_parts = item['published-online']['date-parts'][0]
                pub_date = '-'.join(str(part) for part in date_parts)
            
            # Extract journal
            journal = ""
            if 'container-title' in item:
                journal = item['container-title'][0] if item['container-title'] else ""
            
            # Extract DOI
            doi = item.get('DOI', '')
            
            # Extract abstract (if available)
            abstract = ""
            if 'abstract' in item:
                abstract = item['abstract']
            
            return {
                'crossref_id': item.get('id', ''),
                'title': title,
                'abstract': abstract,
                'authors': '; '.join(authors),
                'journal': journal,
                'publication_date': pub_date,
                'doi': doi,
                'source': 'Crossref',
                'search_terms': search_term,
                'relevance_score': self._calculate_relevance_score(title, abstract),
                'data_type': 'academic_paper'
            }
            
        except Exception as e:
            logger.error(f"Error parsing Crossref item: {e}")
            return None
    
    def _add_academic_metadata(self, data: pd.DataFrame, source: str) -> pd.DataFrame:
        """Add academic-specific metadata."""
        data = data.copy()
        
        # Add collection metadata
        data['collection_date'] = datetime.now(timezone.utc).isoformat()
        data['collection_source'] = source
        data['data_category'] = 'academic'
        
        # Add quality indicators
        data['has_abstract'] = data['abstract'].notna() & (data['abstract'] != '')
        data['has_doi'] = data['doi'].notna() & (data['doi'] != '')
        data['has_authors'] = data['authors'].notna() & (data['authors'] != '')
        
        # Calculate academic quality score
        quality_indicators = ['has_abstract', 'has_doi', 'has_authors']
        data['academic_quality_score'] = data[quality_indicators].sum(axis=1) / len(quality_indicators)
        
        return data
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """Validate academic data."""
        if data.empty:
            return False
        
        # Check for required academic columns
        required_columns = ['title', 'source']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            logger.error(f"Missing required academic columns: {missing_columns}")
            return False
        
        # Check for valid titles
        if data['title'].isna().all():
            logger.error("All academic records have missing titles")
            return False
        
        logger.info(f"Academic data validation passed: {len(data)} records")
        return True