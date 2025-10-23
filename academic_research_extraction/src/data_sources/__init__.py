"""
Data source connectors for academic research extraction.
"""

from .base_connector import BaseConnector
from .openalex_connector import OpenAlexConnector
from .crossref_connector import CrossrefConnector
from .europe_pmc_connector import EuropePMCConnector
from .semantic_scholar_connector import SemanticScholarConnector
from .pubmed_connector import PubMedConnector
from .arxiv_connector import ArxivConnector
from .doaj_connector import DOAJConnector

__all__ = [
    "BaseConnector",
    "OpenAlexConnector", 
    "CrossrefConnector",
    "EuropePMCConnector",
    "SemanticScholarConnector",
    "PubMedConnector",
    "ArxivConnector",
    "DOAJConnector"
]
