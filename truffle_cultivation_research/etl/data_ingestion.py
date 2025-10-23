"""
ETL Pipeline for ingesting scientific papers, patents, and experimental data
into the truffle cultivation knowledge graph.
"""

import asyncio
import aiohttp
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
import xml.etree.ElementTree as ET
from pathlib import Path
import hashlib

logger = logging.getLogger(__name__)

@dataclass
class PaperMetadata:
    """Metadata for a scientific paper."""
    doi: str
    title: str
    abstract: str
    authors: List[str]
    venue: str
    year: int
    source: str
    url_pdf: Optional[str] = None
    oa_status: str = "unknown"
    keywords: List[str] = None
    entities: Dict[str, Any] = None
    embeddings: List[float] = None
    fetched_at: datetime = None
    
    def __post_init__(self):
        if self.keywords is None:
            self.keywords = []
        if self.entities is None:
            self.entities = {}
        if self.fetched_at is None:
            self.fetched_at = datetime.now()

@dataclass
class PatentMetadata:
    """Metadata for a patent."""
    pub_number: str
    app_number: str
    title: str
    abstract: str
    assignees: List[str]
    inventors: List[str]
    cpc: List[str]
    ipc: List[str]
    family_id: str
    jurisdiction: str
    url_pdf: Optional[str] = None
    status: str = "unknown"
    claims_text: str = ""
    embeddings: List[float] = None
    fetched_at: datetime = None
    
    def __post_init__(self):
        if self.fetched_at is None:
            self.fetched_at = datetime.now()

class DataIngestionPipeline:
    """Main ETL pipeline for data ingestion."""
    
    def __init__(self, 
                 neo4j_uri: str = "bolt://localhost:7687",
                 neo4j_user: str = "neo4j",
                 neo4j_password: str = "password",
                 graphdb_uri: str = "http://localhost:7200",
                 graphdb_repo: str = "truffle-kg"):
        
        self.neo4j_uri = neo4j_uri
        self.neo4j_user = neo4j_user
        self.neo4j_password = neo4j_password
        self.graphdb_uri = graphdb_uri
        self.graphdb_repo = graphdb_repo
        
        # API endpoints
        self.openalex_base = "https://api.openalex.org"
        self.crossref_base = "https://api.crossref.org"
        self.pubmed_base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
        self.unpaywall_base = "https://api.unpaywall.org/v2"
        self.epo_base = "https://ops.epo.org/3.2"
        self.wipo_base = "https://patentscope.wipo.int"
        
        # Rate limiting
        self.rate_limits = {
            'openalex': 10,  # requests per second
            'crossref': 50,
            'pubmed': 3,
            'unpaywall': 1,
            'epo': 1,
            'wipo': 1
        }
        
        # Entity extraction
        self.entity_extractors = self._initialize_entity_extractors()
        
        # Deduplication
        self.seen_dois = set()
        self.seen_patents = set()
    
    def _initialize_entity_extractors(self) -> Dict[str, Any]:
        """Initialize entity extraction models."""
        # This would typically load spaCy models, SciBERT, etc.
        # For now, return placeholder
        return {
            'fungi_species': [],
            'host_trees': [],
            'nutrients': [],
            'pgr': [],
            'control_terms': []
        }
    
    async def fetch_papers(self, 
                          query: str,
                          from_date: Optional[datetime] = None,
                          to_date: Optional[datetime] = None,
                          max_results: int = 1000) -> List[PaperMetadata]:
        """Fetch papers from multiple sources."""
        papers = []
        
        # OpenAlex
        openalex_papers = await self._fetch_openalex_papers(query, from_date, to_date, max_results)
        papers.extend(openalex_papers)
        
        # Crossref
        crossref_papers = await self._fetch_crossref_papers(query, from_date, to_date, max_results)
        papers.extend(crossref_papers)
        
        # PubMed
        pubmed_papers = await self._fetch_pubmed_papers(query, from_date, to_date, max_results)
        papers.extend(pubmed_papers)
        
        # Deduplicate
        papers = self._deduplicate_papers(papers)
        
        # Enrich with Unpaywall
        papers = await self._enrich_with_unpaywall(papers)
        
        # Extract entities
        papers = await self._extract_entities_papers(papers)
        
        return papers
    
    async def fetch_patents(self,
                           query: str,
                           from_date: Optional[datetime] = None,
                           to_date: Optional[datetime] = None,
                           max_results: int = 1000) -> List[PatentMetadata]:
        """Fetch patents from multiple sources."""
        patents = []
        
        # EPO OPS
        epo_patents = await self._fetch_epo_patents(query, from_date, to_date, max_results)
        patents.extend(epo_patents)
        
        # WIPO PATENTSCOPE
        wipo_patents = await self._fetch_wipo_patents(query, from_date, to_date, max_results)
        patents.extend(wipo_patents)
        
        # Deduplicate
        patents = self._deduplicate_patents(patents)
        
        # Extract entities
        patents = await self._extract_entities_patents(patents)
        
        return patents
    
    async def _fetch_openalex_papers(self, query: str, from_date: Optional[datetime], 
                                   to_date: Optional[datetime], max_results: int) -> List[PaperMetadata]:
        """Fetch papers from OpenAlex API."""
        papers = []
        
        # Build query parameters
        params = {
            'search': query,
            'per_page': 200,
            'sort': 'publication_date:desc'
        }
        
        if from_date:
            params['from_publication_date'] = from_date.strftime('%Y-%m-%d')
        if to_date:
            params['to_publication_date'] = to_date.strftime('%Y-%m-%d')
        
        async with aiohttp.ClientSession() as session:
            page = 0
            while len(papers) < max_results:
                params['page'] = page
                
                try:
                    async with session.get(f"{self.openalex_base}/works", params=params) as response:
                        if response.status == 200:
                            data = await response.json()
                            
                            for work in data.get('results', []):
                                if len(papers) >= max_results:
                                    break
                                
                                paper = self._parse_openalex_work(work)
                                if paper and paper.doi not in self.seen_dois:
                                    papers.append(paper)
                                    self.seen_dois.add(paper.doi)
                            
                            if not data.get('results'):
                                break
                            
                            page += 1
                        else:
                            logger.warning(f"OpenAlex API error: {response.status}")
                            break
                
                except Exception as e:
                    logger.error(f"Error fetching from OpenAlex: {e}")
                    break
        
        return papers
    
    async def _fetch_crossref_papers(self, query: str, from_date: Optional[datetime],
                                   to_date: Optional[datetime], max_results: int) -> List[PaperMetadata]:
        """Fetch papers from Crossref API."""
        papers = []
        
        params = {
            'query': query,
            'rows': 100,
            'sort': 'published',
            'order': 'desc'
        }
        
        if from_date:
            params['filter'] = f"from-pub-date:{from_date.strftime('%Y-%m-%d')}"
        if to_date:
            if 'filter' in params:
                params['filter'] += f",until-pub-date:{to_date.strftime('%Y-%m-%d')}"
            else:
                params['filter'] = f"until-pub-date:{to_date.strftime('%Y-%m-%d')}"
        
        async with aiohttp.ClientSession() as session:
            offset = 0
            while len(papers) < max_results:
                params['offset'] = offset
                
                try:
                    async with session.get(f"{self.crossref_base}/works", params=params) as response:
                        if response.status == 200:
                            data = await response.json()
                            
                            for item in data.get('message', {}).get('items', []):
                                if len(papers) >= max_results:
                                    break
                                
                                paper = self._parse_crossref_item(item)
                                if paper and paper.doi not in self.seen_dois:
                                    papers.append(paper)
                                    self.seen_dois.add(paper.doi)
                            
                            if not data.get('message', {}).get('items'):
                                break
                            
                            offset += 100
                        else:
                            logger.warning(f"Crossref API error: {response.status}")
                            break
                
                except Exception as e:
                    logger.error(f"Error fetching from Crossref: {e}")
                    break
        
        return papers
    
    async def _fetch_pubmed_papers(self, query: str, from_date: Optional[datetime],
                                 to_date: Optional[datetime], max_results: int) -> List[PaperMetadata]:
        """Fetch papers from PubMed API."""
        papers = []
        
        # Search for PMIDs
        search_params = {
            'db': 'pubmed',
            'term': query,
            'retmax': 10000,
            'retmode': 'json'
        }
        
        if from_date:
            search_params['mindate'] = from_date.strftime('%Y/%m/%d')
        if to_date:
            search_params['maxdate'] = to_date.strftime('%Y/%m/%d')
        
        async with aiohttp.ClientSession() as session:
            try:
                # Search for PMIDs
                async with session.get(f"{self.pubmed_base}/esearch.fcgi", params=search_params) as response:
                    if response.status == 200:
                        data = await response.json()
                        pmids = data.get('esearchresult', {}).get('idlist', [])
                        
                        # Fetch details for each PMID
                        for i in range(0, min(len(pmids), max_results), 200):
                            pmid_batch = pmids[i:i+200]
                            
                            fetch_params = {
                                'db': 'pubmed',
                                'id': ','.join(pmid_batch),
                                'retmode': 'xml'
                            }
                            
                            async with session.get(f"{self.pubmed_base}/efetch.fcgi", params=fetch_params) as fetch_response:
                                if fetch_response.status == 200:
                                    xml_data = await fetch_response.text()
                                    batch_papers = self._parse_pubmed_xml(xml_data)
                                    
                                    for paper in batch_papers:
                                        if paper.doi not in self.seen_dois:
                                            papers.append(paper)
                                            self.seen_dois.add(paper.doi)
            
            except Exception as e:
                logger.error(f"Error fetching from PubMed: {e}")
        
        return papers
    
    async def _fetch_epo_patents(self, query: str, from_date: Optional[datetime],
                               to_date: Optional[datetime], max_results: int) -> List[PatentMetadata]:
        """Fetch patents from EPO OPS API."""
        patents = []
        
        # EPO OPS requires authentication
        # This is a simplified version
        params = {
            'q': query,
            'numResults': 100
        }
        
        if from_date:
            params['dateRange'] = f"{from_date.strftime('%Y%m%d')}-{to_date.strftime('%Y%m%d') if to_date else datetime.now().strftime('%Y%m%d')}"
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(f"{self.epo_base}/rest-services/published-data/search", params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        # Parse EPO response
                        patents = self._parse_epo_response(data)
            except Exception as e:
                logger.error(f"Error fetching from EPO: {e}")
        
        return patents
    
    async def _fetch_wipo_patents(self, query: str, from_date: Optional[datetime],
                                to_date: Optional[datetime], max_results: int) -> List[PatentMetadata]:
        """Fetch patents from WIPO PATENTSCOPE API."""
        patents = []
        
        # WIPO API implementation
        # This is a simplified version
        params = {
            'q': query,
            'fq': 'publication_date:[* TO *]',
            'start': 0,
            'rows': 100
        }
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(f"{self.wipo_base}/search", params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        # Parse WIPO response
                        patents = self._parse_wipo_response(data)
            except Exception as e:
                logger.error(f"Error fetching from WIPO: {e}")
        
        return patents
    
    def _parse_openalex_work(self, work: Dict) -> Optional[PaperMetadata]:
        """Parse OpenAlex work object."""
        try:
            doi = work.get('doi', '').replace('https://doi.org/', '')
            if not doi:
                return None
            
            title = work.get('title', '')
            abstract = work.get('abstract_inverted_index', {})
            
            # Reconstruct abstract from inverted index
            if abstract:
                abstract_text = self._reconstruct_abstract(abstract)
            else:
                abstract_text = ''
            
            authors = []
            for author in work.get('authorships', []):
                author_name = author.get('author', {}).get('display_name', '')
                if author_name:
                    authors.append(author_name)
            
            venue = work.get('primary_location', {}).get('source', {}).get('display_name', '')
            year = work.get('publication_year', 0)
            
            # Get PDF URL
            url_pdf = None
            for location in work.get('locations', []):
                if location.get('is_pdf', False):
                    url_pdf = location.get('landing_page_url', '')
                    break
            
            return PaperMetadata(
                doi=doi,
                title=title,
                abstract=abstract_text,
                authors=authors,
                venue=venue,
                year=year,
                source='openalex',
                url_pdf=url_pdf,
                oa_status='open' if work.get('open_access', {}).get('is_oa', False) else 'closed'
            )
        
        except Exception as e:
            logger.error(f"Error parsing OpenAlex work: {e}")
            return None
    
    def _parse_crossref_item(self, item: Dict) -> Optional[PaperMetadata]:
        """Parse Crossref item."""
        try:
            doi = item.get('DOI', '')
            if not doi:
                return None
            
            title = item.get('title', [''])[0] if item.get('title') else ''
            
            abstract = ''
            if 'abstract' in item:
                abstract = item['abstract']
            
            authors = []
            for author in item.get('author', []):
                given = author.get('given', '')
                family = author.get('family', '')
                if given and family:
                    authors.append(f"{given} {family}")
                elif family:
                    authors.append(family)
            
            venue = ''
            if 'container-title' in item:
                venue = item['container-title'][0] if item['container-title'] else ''
            
            year = item.get('published-print', {}).get('date-parts', [[0]])[0][0]
            if not year:
                year = item.get('published-online', {}).get('date-parts', [[0]])[0][0]
            
            return PaperMetadata(
                doi=doi,
                title=title,
                abstract=abstract,
                authors=authors,
                venue=venue,
                year=year,
                source='crossref'
            )
        
        except Exception as e:
            logger.error(f"Error parsing Crossref item: {e}")
            return None
    
    def _parse_pubmed_xml(self, xml_data: str) -> List[PaperMetadata]:
        """Parse PubMed XML response."""
        papers = []
        
        try:
            root = ET.fromstring(xml_data)
            
            for article in root.findall('.//PubmedArticle'):
                paper = self._parse_pubmed_article(article)
                if paper:
                    papers.append(paper)
        
        except Exception as e:
            logger.error(f"Error parsing PubMed XML: {e}")
        
        return papers
    
    def _parse_pubmed_article(self, article) -> Optional[PaperMetadata]:
        """Parse individual PubMed article."""
        try:
            # Extract DOI
            doi = ''
            for article_id in article.findall('.//ArticleId'):
                if article_id.get('IdType') == 'doi':
                    doi = article_id.text
                    break
            
            if not doi:
                return None
            
            # Extract title
            title = ''
            title_elem = article.find('.//ArticleTitle')
            if title_elem is not None:
                title = title_elem.text or ''
            
            # Extract abstract
            abstract = ''
            abstract_elem = article.find('.//AbstractText')
            if abstract_elem is not None:
                abstract = abstract_elem.text or ''
            
            # Extract authors
            authors = []
            for author in article.findall('.//Author'):
                last_name = author.find('LastName')
                first_name = author.find('ForeName')
                
                if last_name is not None and first_name is not None:
                    authors.append(f"{first_name.text} {last_name.text}")
                elif last_name is not None:
                    authors.append(last_name.text)
            
            # Extract journal
            venue = ''
            journal_elem = article.find('.//Journal/Title')
            if journal_elem is not None:
                venue = journal_elem.text or ''
            
            # Extract year
            year = 0
            pub_date = article.find('.//PubDate')
            if pub_date is not None:
                year_elem = pub_date.find('Year')
                if year_elem is not None:
                    year = int(year_elem.text)
            
            return PaperMetadata(
                doi=doi,
                title=title,
                abstract=abstract,
                authors=authors,
                venue=venue,
                year=year,
                source='pubmed'
            )
        
        except Exception as e:
            logger.error(f"Error parsing PubMed article: {e}")
            return None
    
    def _reconstruct_abstract(self, abstract_inverted: Dict) -> str:
        """Reconstruct abstract from inverted index."""
        words = {}
        for word, positions in abstract_inverted.items():
            for pos in positions:
                words[pos] = word
        
        if not words:
            return ''
        
        max_pos = max(words.keys())
        abstract_words = []
        
        for i in range(max_pos + 1):
            if i in words:
                abstract_words.append(words[i])
            else:
                abstract_words.append('')
        
        return ' '.join(abstract_words)
    
    def _deduplicate_papers(self, papers: List[PaperMetadata]) -> List[PaperMetadata]:
        """Remove duplicate papers based on DOI."""
        seen = set()
        unique_papers = []
        
        for paper in papers:
            if paper.doi not in seen:
                seen.add(paper.doi)
                unique_papers.append(paper)
        
        return unique_papers
    
    def _deduplicate_patents(self, patents: List[PatentMetadata]) -> List[PatentMetadata]:
        """Remove duplicate patents based on publication number."""
        seen = set()
        unique_patents = []
        
        for patent in patents:
            if patent.pub_number not in seen:
                seen.add(patent.pub_number)
                unique_patents.append(patent)
        
        return unique_patents
    
    async def _enrich_with_unpaywall(self, papers: List[PaperMetadata]) -> List[PaperMetadata]:
        """Enrich papers with Unpaywall data."""
        async with aiohttp.ClientSession() as session:
            for paper in papers:
                if not paper.url_pdf:
                    try:
                        async with session.get(f"{self.unpaywall_base}/{paper.doi}") as response:
                            if response.status == 200:
                                data = await response.json()
                                paper.url_pdf = data.get('best_oa_location', {}).get('url_for_pdf', '')
                                paper.oa_status = data.get('oa_status', 'unknown')
                    except Exception as e:
                        logger.error(f"Error enriching paper {paper.doi} with Unpaywall: {e}")
        
        return papers
    
    async def _extract_entities_papers(self, papers: List[PaperMetadata]) -> List[PaperMetadata]:
        """Extract entities from papers."""
        for paper in papers:
            # This would use spaCy, SciBERT, etc. for entity extraction
            # For now, just add placeholder
            paper.entities = {
                'fungi_species': [],
                'host_trees': [],
                'nutrients': [],
                'pgr': [],
                'control_terms': []
            }
        
        return papers
    
    async def _extract_entities_patents(self, patents: List[PatentMetadata]) -> List[PatentMetadata]:
        """Extract entities from patents."""
        for patent in patents:
            # This would use spaCy, SciBERT, etc. for entity extraction
            # For now, just add placeholder
            patent.entities = {
                'fungi_species': [],
                'host_trees': [],
                'nutrients': [],
                'pgr': [],
                'control_terms': []
            }
        
        return patents
    
    def _parse_epo_response(self, data: Dict) -> List[PatentMetadata]:
        """Parse EPO OPS response."""
        # Implementation would depend on EPO API response format
        return []
    
    def _parse_wipo_response(self, data: Dict) -> List[PatentMetadata]:
        """Parse WIPO PATENTSCOPE response."""
        # Implementation would depend on WIPO API response format
        return []
    
    async def store_papers(self, papers: List[PaperMetadata]):
        """Store papers in knowledge graph."""
        # This would connect to Neo4j and GraphDB
        # For now, just log
        logger.info(f"Storing {len(papers)} papers")
    
    async def store_patents(self, patents: List[PatentMetadata]):
        """Store patents in knowledge graph."""
        # This would connect to Neo4j and GraphDB
        # For now, just log
        logger.info(f"Storing {len(patents)} patents")