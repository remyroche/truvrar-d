"""
Data processing modules for academic research extraction.
"""

from .semantic_analysis import SemanticAnalyzer
from .deduplication import DeduplicationEngine
from .entity_extraction import EntityExtractor
from .topic_modeling import TopicModeler

__all__ = [
    "SemanticAnalyzer",
    "DeduplicationEngine", 
    "EntityExtractor",
    "TopicModeler"
]
