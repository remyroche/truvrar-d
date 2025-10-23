"""
Data schema definitions for academic research extraction system.
"""

from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from pydantic import BaseModel, Field
from enum import Enum


class OpenAccessStatus(str, Enum):
    """Open access status enumeration."""
    OPEN = "open"
    CLOSED = "closed"
    HYBRID = "hybrid"
    UNKNOWN = "unknown"


class Language(str, Enum):
    """Supported languages."""
    ENGLISH = "en"
    FRENCH = "fr"
    ITALIAN = "it"
    SPANISH = "es"


class Author(BaseModel):
    """Author information."""
    name: str
    orcid: Optional[str] = None
    affiliation: Optional[str] = None
    email: Optional[str] = None


class License(BaseModel):
    """License information."""
    name: Optional[str] = None
    url: Optional[str] = None
    oa_status: OpenAccessStatus
    best_oa_url: Optional[str] = None
    publisher_policy: Optional[str] = None


class PaperMetadata(BaseModel):
    """Canonical paper metadata schema."""
    # Core identifiers
    paper_id: str = Field(..., description="Unique paper identifier")
    doi: Optional[str] = Field(None, description="Digital Object Identifier")
    pmid: Optional[str] = Field(None, description="PubMed ID")
    pmcid: Optional[str] = Field(None, description="PubMed Central ID")
    arxiv_id: Optional[str] = Field(None, description="arXiv ID")
    
    # Basic information
    title: str = Field(..., description="Paper title")
    abstract: Optional[str] = Field(None, description="Abstract text")
    authors: List[Author] = Field(default_factory=list, description="List of authors")
    year: Optional[int] = Field(None, description="Publication year")
    journal: Optional[str] = Field(None, description="Journal name")
    volume: Optional[str] = Field(None, description="Volume")
    issue: Optional[str] = Field(None, description="Issue")
    pages: Optional[str] = Field(None, description="Page numbers")
    
    # Source information
    source: str = Field(..., description="Data source (e.g., 'openalex', 'crossref')")
    url: Optional[str] = Field(None, description="Original URL")
    best_pdf_url: Optional[str] = Field(None, description="Best available PDF URL")
    best_html_url: Optional[str] = Field(None, description="Best available HTML URL")
    
    # Licensing and access
    license: Optional[License] = Field(None, description="License information")
    oa_status: OpenAccessStatus = Field(OpenAccessStatus.UNKNOWN, description="Open access status")
    
    # Content classification
    subjects: List[str] = Field(default_factory=list, description="Subject categories")
    mesh_terms: List[str] = Field(default_factory=list, description="MeSH terms")
    keywords: List[str] = Field(default_factory=list, description="Keywords")
    language: Optional[Language] = Field(None, description="Primary language")
    
    # Citations and references
    citations: int = Field(0, description="Number of citations")
    references: List[str] = Field(default_factory=list, description="Reference DOIs/IDs")
    
    # Affiliations
    affiliations: List[str] = Field(default_factory=list, description="Author affiliations")
    
    # Provenance
    source_list: List[str] = Field(default_factory=list, description="All sources that provided this paper")
    collection_date: datetime = Field(default_factory=datetime.now, description="Collection timestamp")
    dedup_method: Optional[str] = Field(None, description="Deduplication method used")
    
    # Processing flags
    has_fulltext: bool = Field(False, description="Whether full text is available")
    fulltext_path: Optional[str] = Field(None, description="Path to full text file")
    processed: bool = Field(False, description="Whether paper has been processed")
    
    class Config:
        use_enum_values = True


class ClassificationLabels(BaseModel):
    """Multi-label classification results."""
    paper_id: str
    labels: Dict[str, float] = Field(..., description="Label probabilities")
    confidence_threshold: float = Field(0.3, description="Minimum confidence threshold")
    method: str = Field(..., description="Classification method used")
    timestamp: datetime = Field(default_factory=datetime.now)


class TopicModeling(BaseModel):
    """Topic modeling results."""
    paper_id: str
    topic_id: int
    topic_probability: float
    topic_terms: List[str]
    method: str = Field(..., description="Topic modeling method (BERTopic, LDA, etc.)")
    timestamp: datetime = Field(default_factory=datetime.now)


class Entity(BaseModel):
    """Extracted entity."""
    text: str
    label: str  # e.g., "SPECIES", "HOST_TREE", "SOIL_PROPERTY"
    start_char: int
    end_char: int
    confidence: float
    paper_id: str
    sentence_id: Optional[int] = None


class Relation(BaseModel):
    """Extracted relation between entities."""
    subject: str  # Entity text
    predicate: str  # Relation type
    object: str  # Entity text
    confidence: float
    paper_id: str
    sentence_id: Optional[int] = None
    subject_entity: Optional[Entity] = None
    object_entity: Optional[Entity] = None


class KnowledgeGraph(BaseModel):
    """Knowledge graph structure."""
    nodes: List[Dict[str, Any]] = Field(default_factory=list)
    edges: List[Dict[str, Any]] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    creation_date: datetime = Field(default_factory=datetime.now)


class TrendAnalysis(BaseModel):
    """Trend analysis results."""
    year: int
    topic: str
    count: int
    oa_share: float
    top_papers: List[str]  # Paper IDs
    geographic_coverage: Dict[str, int]  # Country -> count


class Manifest(BaseModel):
    """Collection manifest for provenance tracking."""
    run_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    sources_used: List[str]
    total_papers: int
    unique_papers: int
    duplicates_removed: int
    fulltext_downloaded: int
    processing_errors: int
    api_versions: Dict[str, str]
    checksums: Dict[str, str]
    configuration: Dict[str, Any]
    
    class Config:
        use_enum_values = True
