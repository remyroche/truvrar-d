"""
Advanced deduplication engine for academic papers.
"""

import hashlib
import re
from typing import List, Dict, Any, Set, Tuple
from datasketch import MinHashLSH, MinHash
from fuzzywuzzy import fuzz
from loguru import logger

from ..data_schema import PaperMetadata


class DeduplicationEngine:
    """Advanced deduplication engine for academic papers."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize deduplication engine with configuration."""
        self.config = config
        self.title_threshold = config.get("deduplication", {}).get("title_similarity_threshold", 0.97)
        self.author_threshold = config.get("deduplication", {}).get("author_overlap_threshold", 0.6)
        self.simhash_threshold = config.get("deduplication", {}).get("simhash_threshold", 0.85)
        
    def deduplicate_papers(self, papers: List[PaperMetadata]) -> List[PaperMetadata]:
        """Remove duplicate papers using multiple strategies."""
        logger.info(f"Deduplicating {len(papers)} papers...")
        
        # Step 1: Deterministic deduplication
        unique_papers = self._deterministic_deduplication(papers)
        logger.info(f"After deterministic deduplication: {len(unique_papers)} papers")
        
        # Step 2: Fuzzy deduplication
        final_papers = self._fuzzy_deduplication(unique_papers)
        logger.info(f"After fuzzy deduplication: {len(final_papers)} papers")
        
        return final_papers
    
    def _deterministic_deduplication(self, papers: List[PaperMetadata]) -> List[PaperMetadata]:
        """Remove exact duplicates based on DOI and title hash."""
        seen_dois = set()
        seen_hashes = set()
        unique_papers = []
        
        for paper in papers:
            # Check DOI
            if paper.doi:
                if paper.doi in seen_dois:
                    continue
                seen_dois.add(paper.doi)
            
            # Check title hash
            title_hash = self._hash_title(paper.title)
            if title_hash in seen_hashes:
                continue
            seen_hashes.add(title_hash)
            
            unique_papers.append(paper)
        
        return unique_papers
    
    def _fuzzy_deduplication(self, papers: List[PaperMetadata]) -> List[PaperMetadata]:
        """Remove near-duplicates using fuzzy matching."""
        # Create LSH index for efficient similarity search
        lsh = MinHashLSH(threshold=self.simhash_threshold, num_perm=128)
        
        # Generate MinHashes for all papers
        paper_hashes = {}
        for i, paper in enumerate(papers):
            minhash = self._create_minhash(paper)
            lsh.insert(i, minhash)
            paper_hashes[i] = minhash
        
        # Find similar papers
        similar_groups = []
        processed = set()
        
        for i, paper in enumerate(papers):
            if i in processed:
                continue
            
            # Find similar papers using LSH
            similar_indices = lsh.query(paper_hashes[i])
            similar_indices = [idx for idx in similar_indices if idx != i and idx not in processed]
            
            if similar_indices:
                # Group similar papers
                group = [i] + similar_indices
                similar_groups.append(group)
                processed.update(group)
            else:
                processed.add(i)
        
        # Select best paper from each group
        final_papers = []
        for group in similar_groups:
            group_papers = [papers[i] for i in group]
            best_paper = self._select_best_paper(group_papers)
            final_papers.append(best_paper)
        
        # Add papers that weren't in any group
        for i, paper in enumerate(papers):
            if i not in processed:
                final_papers.append(paper)
        
        return final_papers
    
    def _hash_title(self, title: str) -> str:
        """Create hash of normalized title."""
        normalized = self._normalize_title(title)
        return hashlib.md5(normalized.encode()).hexdigest()
    
    def _normalize_title(self, title: str) -> str:
        """Normalize title for comparison."""
        if not title:
            return ""
        
        # Convert to lowercase
        normalized = title.lower()
        
        # Remove punctuation
        normalized = re.sub(r'[^\w\s]', '', normalized)
        
        # Remove extra whitespace
        normalized = ' '.join(normalized.split())
        
        return normalized
    
    def _create_minhash(self, paper: PaperMetadata) -> MinHash:
        """Create MinHash for paper content."""
        # Combine title and abstract
        content = paper.title
        if paper.abstract:
            content += " " + paper.abstract
        
        # Normalize content
        content = self._normalize_title(content)
        
        # Create MinHash
        minhash = MinHash(num_perm=128)
        
        # Add shingles
        words = content.split()
        for i in range(len(words) - 2):
            shingle = ' '.join(words[i:i+3])
            minhash.update(shingle.encode())
        
        return minhash
    
    def _select_best_paper(self, papers: List[PaperMetadata]) -> PaperMetadata:
        """Select the best paper from a group of similar papers."""
        if len(papers) == 1:
            return papers[0]
        
        # Score papers based on quality metrics
        scored_papers = []
        for paper in papers:
            score = self._calculate_paper_score(paper)
            scored_papers.append((score, paper))
        
        # Sort by score (descending)
        scored_papers.sort(key=lambda x: x[0], reverse=True)
        
        return scored_papers[0][1]
    
    def _calculate_paper_score(self, paper: PaperMetadata) -> float:
        """Calculate quality score for a paper."""
        score = 0.0
        
        # Has abstract
        if paper.abstract:
            score += 1.0
        
        # Has DOI
        if paper.doi:
            score += 0.5
        
        # Open access
        if paper.oa_status.value == "open":
            score += 0.5
        
        # Number of citations
        score += min(paper.citations / 100.0, 1.0)  # Cap at 1.0
        
        # Number of authors
        if len(paper.authors) > 0:
            score += min(len(paper.authors) / 10.0, 0.5)  # Cap at 0.5
        
        # Has journal
        if paper.journal:
            score += 0.2
        
        return score
    
    def _are_similar_papers(self, paper1: PaperMetadata, paper2: PaperMetadata) -> bool:
        """Check if two papers are similar."""
        # Check title similarity
        title_sim = fuzz.ratio(paper1.title, paper2.title) / 100.0
        if title_sim < self.title_threshold:
            return False
        
        # Check author overlap
        authors1 = {a.name.lower() for a in paper1.authors}
        authors2 = {a.name.lower() for a in paper2.authors}
        
        if authors1 and authors2:
            overlap = len(authors1.intersection(authors2)) / max(len(authors1), len(authors2))
            if overlap < self.author_threshold:
                return False
        
        # Check year similarity
        if paper1.year and paper2.year:
            if abs(paper1.year - paper2.year) > 2:  # Within 2 years
                return False
        
        return True
