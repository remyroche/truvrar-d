"""
Semantic analysis and multi-label classification for academic papers.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import pickle
from loguru import logger
from datetime import datetime

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
import torch

from ..data_schema import PaperMetadata, ClassificationLabels


class SemanticAnalyzer:
    """Semantic analysis and classification for academic papers."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize semantic analyzer with configuration."""
        self.config = config
        self.model_name = "allenai/scibert_scivocab_uncased"
        self.embedding_model = None
        self.classifier = None
        self.taxonomy_labels = config.get("classification", {}).get("taxonomy_labels", [])
        self.min_confidence = config.get("classification", {}).get("min_confidence_threshold", 0.3)
        
    def _load_models(self):
        """Load pre-trained models for analysis."""
        if self.embedding_model is None:
            logger.info("Loading embedding model...")
            self.embedding_model = SentenceTransformer(self.model_name)
        
        if self.classifier is None:
            logger.info("Loading zero-shot classifier...")
            self.classifier = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli",
                device=0 if torch.cuda.is_available() else -1
            )
    
    def analyze_papers(self, papers: List[PaperMetadata]) -> List[ClassificationLabels]:
        """Analyze papers and generate multi-label classifications."""
        logger.info(f"Analyzing {len(papers)} papers for semantic content...")
        
        self._load_models()
        
        # Prepare texts for analysis
        texts = []
        paper_ids = []
        
        for paper in papers:
            # Combine title and abstract
            text = paper.title
            if paper.abstract:
                text += " " + paper.abstract
            texts.append(text)
            paper_ids.append(paper.paper_id)
        
        # Generate embeddings
        logger.info("Generating embeddings...")
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        
        # Generate classifications
        logger.info("Generating multi-label classifications...")
        classifications = []
        
        for i, (paper_id, text, embedding) in enumerate(zip(paper_ids, texts, embeddings)):
            try:
                # Zero-shot classification
                classification_result = self.classifier(
                    text,
                    self.taxonomy_labels,
                    multi_label=True
                )
                
                # Extract label probabilities
                label_scores = {}
                for label, score in zip(classification_result["labels"], classification_result["scores"]):
                    if score >= self.min_confidence:
                        label_scores[label] = float(score)
                
                # Create classification result
                classification = ClassificationLabels(
                    paper_id=paper_id,
                    labels=label_scores,
                    confidence_threshold=self.min_confidence,
                    method="zero_shot_bart"
                )
                
                classifications.append(classification)
                
                if (i + 1) % 100 == 0:
                    logger.info(f"Processed {i + 1}/{len(papers)} papers")
                    
            except Exception as e:
                logger.error(f"Error classifying paper {paper_id}: {e}")
                # Create empty classification
                classification = ClassificationLabels(
                    paper_id=paper_id,
                    labels={},
                    confidence_threshold=self.min_confidence,
                    method="zero_shot_bart"
                )
                classifications.append(classification)
        
        logger.info(f"Completed semantic analysis for {len(papers)} papers")
        return classifications
    
    def generate_topic_summary(self, papers: List[PaperMetadata], classifications: List[ClassificationLabels]) -> Dict[str, Any]:
        """Generate topic summary from classifications."""
        logger.info("Generating topic summary...")
        
        # Aggregate label counts
        label_counts = {}
        for classification in classifications:
            for label, score in classification.labels.items():
                if label not in label_counts:
                    label_counts[label] = 0
                label_counts[label] += 1
        
        # Calculate label distributions
        total_papers = len(papers)
        label_distributions = {
            label: count / total_papers 
            for label, count in label_counts.items()
        }
        
        # Find most common label combinations
        label_combinations = {}
        for classification in classifications:
            if len(classification.labels) > 1:
                combo = tuple(sorted(classification.labels.keys()))
                if combo not in label_combinations:
                    label_combinations[combo] = 0
                label_combinations[combo] += 1
        
        # Sort by frequency
        top_combinations = sorted(
            label_combinations.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:10]
        
        # Generate summary
        summary = {
            "total_papers": total_papers,
            "label_counts": label_counts,
            "label_distributions": label_distributions,
            "top_label_combinations": [
                {"labels": list(combo), "count": count}
                for combo, count in top_combinations
            ],
            "generated_at": datetime.now().isoformat()
        }
        
        logger.info("Topic summary generated")
        return summary
    
    def save_classifications(self, classifications: List[ClassificationLabels], output_path: Path):
        """Save classifications to Parquet file."""
        # Convert to DataFrame
        data = []
        for classification in classifications:
            row = {
                "paper_id": classification.paper_id,
                "confidence_threshold": classification.confidence_threshold,
                "method": classification.method,
                "timestamp": classification.timestamp.isoformat()
            }
            
            # Add label scores
            for label, score in classification.labels.items():
                row[f"label_{label}"] = score
            
            data.append(row)
        
        df = pd.DataFrame(data)
        
        # Save as Parquet
        parquet_path = output_path / "labels_multilabel.parquet"
        df.to_parquet(parquet_path, index=False)
        
        logger.info(f"Saved {len(classifications)} classifications to {parquet_path}")
        
        # Also save as JSON
        json_path = output_path / "labels_multilabel.json"
        df.to_json(json_path, orient="records", indent=2)
        
        logger.info(f"Saved {len(classifications)} classifications to {json_path}")
    
    def save_topic_summary(self, summary: Dict[str, Any], output_path: Path):
        """Save topic summary to JSON file."""
        import json
        
        summary_path = output_path / "topics_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Saved topic summary to {summary_path}")


def analyze_papers_batch(
    papers: List[PaperMetadata], 
    config: Dict[str, Any],
    output_dir: str = "outputs"
) -> Tuple[List[ClassificationLabels], Dict[str, Any]]:
    """Convenience function to analyze a batch of papers."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize analyzer
    analyzer = SemanticAnalyzer(config)
    
    # Analyze papers
    classifications = analyzer.analyze_papers(papers)
    
    # Generate topic summary
    topic_summary = analyzer.generate_topic_summary(papers, classifications)
    
    # Save results
    analyzer.save_classifications(classifications, output_path)
    analyzer.save_topic_summary(topic_summary, output_path)
    
    return classifications, topic_summary
