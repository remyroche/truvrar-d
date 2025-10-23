"""
Entity and relation extraction for academic papers.
"""

import re
import spacy
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import networkx as nx
from loguru import logger
import pandas as pd

from ..data_schema import PaperMetadata, Entity, Relation, KnowledgeGraph


class EntityExtractor:
    """Extract entities and relations from academic papers."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize entity extractor with configuration."""
        self.config = config
        self.nlp = None
        self.species_patterns = config.get("entity_extraction", {}).get("species_patterns", [])
        self.host_patterns = config.get("entity_extraction", {}).get("host_patterns", [])
        self.soil_patterns = config.get("entity_extraction", {}).get("soil_patterns", [])
        
    def _load_spacy_model(self):
        """Load spaCy model for NLP processing."""
        if self.nlp is None:
            try:
                # Try to load scispacy model first
                import scispacy
                self.nlp = spacy.load("en_core_sci_sm")
                logger.info("Loaded scispacy model")
            except OSError:
                try:
                    # Fallback to regular spacy model
                    self.nlp = spacy.load("en_core_web_sm")
                    logger.info("Loaded regular spacy model")
                except OSError:
                    logger.error("No spaCy model available. Please install en_core_web_sm or en_core_sci_sm")
                    raise
    
    def extract_entities(self, papers: List[PaperMetadata]) -> List[Entity]:
        """Extract entities from papers."""
        logger.info(f"Extracting entities from {len(papers)} papers...")
        
        self._load_spacy_model()
        entities = []
        
        for paper in papers:
            # Combine title and abstract for analysis
            text = paper.title
            if paper.abstract:
                text += " " + paper.abstract
            
            if not text.strip():
                continue
            
            try:
                # Process with spaCy
                doc = self.nlp(text)
                
                # Extract entities using patterns
                paper_entities = self._extract_pattern_entities(text, paper.paper_id)
                
                # Extract entities using spaCy NER
                spacy_entities = self._extract_spacy_entities(doc, paper.paper_id)
                
                # Combine and deduplicate
                all_entities = paper_entities + spacy_entities
                entities.extend(all_entities)
                
            except Exception as e:
                logger.error(f"Error processing paper {paper.paper_id}: {e}")
                continue
        
        logger.info(f"Extracted {len(entities)} entities")
        return entities
    
    def _extract_pattern_entities(self, text: str, paper_id: str) -> List[Entity]:
        """Extract entities using regex patterns."""
        entities = []
        
        # Extract species
        for pattern in self.species_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                entity = Entity(
                    text=match.group(),
                    label="SPECIES",
                    start_char=match.start(),
                    end_char=match.end(),
                    confidence=0.8,  # High confidence for pattern matches
                    paper_id=paper_id
                )
                entities.append(entity)
        
        # Extract host trees
        for pattern in self.host_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                entity = Entity(
                    text=match.group(),
                    label="HOST_TREE",
                    start_char=match.start(),
                    end_char=match.end(),
                    confidence=0.8,
                    paper_id=paper_id
                )
                entities.append(entity)
        
        # Extract soil properties
        for pattern in self.soil_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                entity = Entity(
                    text=match.group(),
                    label="SOIL_PROPERTY",
                    start_char=match.start(),
                    end_char=match.end(),
                    confidence=0.7,
                    paper_id=paper_id
                )
                entities.append(entity)
        
        return entities
    
    def _extract_spacy_entities(self, doc, paper_id: str) -> List[Entity]:
        """Extract entities using spaCy NER."""
        entities = []
        
        for ent in doc.ents:
            # Map spaCy labels to our labels
            label_map = {
                "PERSON": "AUTHOR",
                "ORG": "ORGANIZATION",
                "GPE": "LOCATION",
                "LOC": "LOCATION",
                "DATE": "DATE",
                "MONEY": "MONEY",
                "PERCENT": "PERCENT"
            }
            
            label = label_map.get(ent.label_, "OTHER")
            
            entity = Entity(
                text=ent.text,
                label=label,
                start_char=ent.start_char,
                end_char=ent.end_char,
                confidence=0.5,  # Default confidence for NER
                paper_id=paper_id
            )
            entities.append(entity)
        
        return entities
    
    def extract_relations(self, papers: List[PaperMetadata], entities: List[Entity]) -> List[Relation]:
        """Extract relations between entities."""
        logger.info(f"Extracting relations from {len(papers)} papers...")
        
        self._load_spacy_model()
        relations = []
        
        for paper in papers:
            # Get entities for this paper
            paper_entities = [e for e in entities if e.paper_id == paper.paper_id]
            
            if len(paper_entities) < 2:
                continue
            
            # Combine title and abstract
            text = paper.title
            if paper.abstract:
                text += " " + paper.abstract
            
            try:
                doc = self.nlp(text)
                paper_relations = self._extract_paper_relations(doc, paper_entities, paper.paper_id)
                relations.extend(paper_relations)
                
            except Exception as e:
                logger.error(f"Error extracting relations from paper {paper.paper_id}: {e}")
                continue
        
        logger.info(f"Extracted {len(relations)} relations")
        return relations
    
    def _extract_paper_relations(self, doc, entities: List[Entity], paper_id: str) -> List[Relation]:
        """Extract relations from a single paper."""
        relations = []
        
        # Define relation patterns
        relation_patterns = [
            # Species - ASSOCIATED_WITH - Host
            (("SPECIES", "HOST_TREE"), "ASSOCIATED_WITH"),
            # Species - FAVORS - Soil Property
            (("SPECIES", "SOIL_PROPERTY"), "FAVORS"),
            # Species - INFLUENCED_BY - Climate
            (("SPECIES", "CLIMATE"), "INFLUENCED_BY"),
            # Experiment - SETTING - in_vitro/in_vivo
            (("EXPERIMENT", "SETTING"), "SETTING")
        ]
        
        # Find co-occurring entities
        for i, entity1 in enumerate(entities):
            for j, entity2 in enumerate(entities[i+1:], i+1):
                # Check if entities are close in text
                if abs(entity1.start_char - entity2.start_char) > 200:  # Within 200 characters
                    continue
                
                # Check for relation patterns
                for (label1, label2), relation_type in relation_patterns:
                    if (entity1.label == label1 and entity2.label == label2) or \
                       (entity1.label == label2 and entity2.label == label1):
                        
                        relation = Relation(
                            subject=entity1.text,
                            predicate=relation_type,
                            object=entity2.text,
                            confidence=0.6,  # Medium confidence
                            paper_id=paper_id,
                            subject_entity=entity1,
                            object_entity=entity2
                        )
                        relations.append(relation)
        
        return relations
    
    def build_knowledge_graph(self, entities: List[Entity], relations: List[Relation]) -> KnowledgeGraph:
        """Build knowledge graph from entities and relations."""
        logger.info("Building knowledge graph...")
        
        # Create NetworkX graph
        G = nx.Graph()
        
        # Add nodes (entities)
        for entity in entities:
            G.add_node(
                entity.text,
                label=entity.label,
                paper_id=entity.paper_id,
                confidence=entity.confidence
            )
        
        # Add edges (relations)
        for relation in relations:
            G.add_edge(
                relation.subject,
                relation.object,
                predicate=relation.predicate,
                paper_id=relation.paper_id,
                confidence=relation.confidence
            )
        
        # Convert to our format
        nodes = []
        for node, data in G.nodes(data=True):
            nodes.append({
                "id": node,
                "label": data.get("label", "UNKNOWN"),
                "paper_id": data.get("paper_id"),
                "confidence": data.get("confidence", 0.0)
            })
        
        edges = []
        for source, target, data in G.edges(data=True):
            edges.append({
                "source": source,
                "target": target,
                "predicate": data.get("predicate", "RELATED_TO"),
                "paper_id": data.get("paper_id"),
                "confidence": data.get("confidence", 0.0)
            })
        
        knowledge_graph = KnowledgeGraph(
            nodes=nodes,
            edges=edges,
            metadata={
                "total_entities": len(entities),
                "total_relations": len(relations),
                "total_nodes": len(nodes),
                "total_edges": len(edges)
            }
        )
        
        logger.info(f"Knowledge graph built with {len(nodes)} nodes and {len(edges)} edges")
        return knowledge_graph
    
    def save_entities(self, entities: List[Entity], output_path: Path):
        """Save entities to Parquet file."""
        data = []
        for entity in entities:
            data.append({
                "text": entity.text,
                "label": entity.label,
                "start_char": entity.start_char,
                "end_char": entity.end_char,
                "confidence": entity.confidence,
                "paper_id": entity.paper_id,
                "sentence_id": entity.sentence_id
            })
        
        df = pd.DataFrame(data)
        
        # Save as Parquet
        parquet_path = output_path / "entities.parquet"
        df.to_parquet(parquet_path, index=False)
        
        logger.info(f"Saved {len(entities)} entities to {parquet_path}")
    
    def save_relations(self, relations: List[Relation], output_path: Path):
        """Save relations to Parquet file."""
        data = []
        for relation in relations:
            data.append({
                "subject": relation.subject,
                "predicate": relation.predicate,
                "object": relation.object,
                "confidence": relation.confidence,
                "paper_id": relation.paper_id,
                "sentence_id": relation.sentence_id
            })
        
        df = pd.DataFrame(data)
        
        # Save as Parquet
        parquet_path = output_path / "relations.parquet"
        df.to_parquet(parquet_path, index=False)
        
        logger.info(f"Saved {len(relations)} relations to {parquet_path}")
    
    def save_knowledge_graph(self, knowledge_graph: KnowledgeGraph, output_path: Path):
        """Save knowledge graph to GML file."""
        import networkx as nx
        
        # Convert back to NetworkX for GML export
        G = nx.Graph()
        
        # Add nodes
        for node in knowledge_graph.nodes:
            G.add_node(
                node["id"],
                label=node["label"],
                paper_id=node["paper_id"],
                confidence=node["confidence"]
            )
        
        # Add edges
        for edge in knowledge_graph.edges:
            G.add_edge(
                edge["source"],
                edge["target"],
                predicate=edge["predicate"],
                paper_id=edge["paper_id"],
                confidence=edge["confidence"]
            )
        
        # Save as GML
        gml_path = output_path / "knowledge_graph.gml"
        nx.write_gml(G, gml_path)
        
        logger.info(f"Saved knowledge graph to {gml_path}")
        
        # Also save as JSON
        import json
        json_path = output_path / "knowledge_graph.json"
        with open(json_path, 'w') as f:
            json.dump(knowledge_graph.dict(), f, indent=2)
        
        logger.info(f"Saved knowledge graph to {json_path}")


def extract_entities_and_relations(
    papers: List[PaperMetadata],
    config: Dict[str, Any],
    output_dir: str = "outputs"
) -> Tuple[List[Entity], List[Relation], KnowledgeGraph]:
    """Convenience function to extract entities and relations."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize extractor
    extractor = EntityExtractor(config)
    
    # Extract entities
    entities = extractor.extract_entities(papers)
    
    # Extract relations
    relations = extractor.extract_relations(papers, entities)
    
    # Build knowledge graph
    knowledge_graph = extractor.build_knowledge_graph(entities, relations)
    
    # Save results
    extractor.save_entities(entities, output_path)
    extractor.save_relations(relations, output_path)
    extractor.save_knowledge_graph(knowledge_graph, output_path)
    
    return entities, relations, knowledge_graph
