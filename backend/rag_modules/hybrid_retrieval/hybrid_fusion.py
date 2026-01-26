"""
RAG Master Project - Hybrid Fusion
===================================
Combines BM25 (sparse) and Vector (dense) retrieval using
Reciprocal Rank Fusion (RRF).
"""

from typing import List, Tuple, Dict
from langchain_core.documents import Document

from config import settings


class HybridFusion:
    """
    Hybrid retrieval fusion using Reciprocal Rank Fusion (RRF).
    
    RRF combines results from multiple retrieval methods by:
    1. Converting each result list to ranks
    2. Computing RRF score: score = Σ 1/(k + rank)
    3. Re-ranking by combined score
    
    This approach:
    - Balances keyword precision (BM25) with semantic understanding (Vector)
    - Handles vocabulary mismatch
    - Improves recall without sacrificing precision
    
    Example:
        >>> fusion = HybridFusion(alpha=0.5)
        >>> combined = fusion.fuse(bm25_results, vector_results, k=5)
    """
    
    def __init__(
        self, 
        alpha: float = None,
        rrf_k: int = 60
    ):
        """
        Initialize hybrid fusion.
        
        Args:
            alpha: Weight for vector results (0=BM25 only, 1=Vector only)
            rrf_k: RRF constant (higher = more weight to lower ranks)
        """
        self.alpha = alpha if alpha is not None else settings.HYBRID_ALPHA
        self.rrf_k = rrf_k
    
    def _get_document_id(self, doc: Document) -> str:
        """Generate a unique ID for a document."""
        # Use content hash as ID
        content_preview = doc.page_content[:200]
        return str(hash(content_preview))
    
    def fuse_with_rrf(
        self,
        bm25_results: List[Document],
        vector_results: List[Document],
        k: int = 5
    ) -> List[Document]:
        """
        Fuse results using Reciprocal Rank Fusion.
        
        Args:
            bm25_results: Documents from BM25 retrieval
            vector_results: Documents from vector retrieval
            k: Number of final results to return
            
        Returns:
            Fused and re-ranked document list
        """
        # Track documents by ID
        doc_map: Dict[str, Document] = {}
        rrf_scores: Dict[str, float] = {}
        
        # Process BM25 results
        for rank, doc in enumerate(bm25_results, 1):
            doc_id = self._get_document_id(doc)
            doc_map[doc_id] = doc
            
            # RRF score for BM25 (weighted by 1-alpha)
            bm25_weight = 1 - self.alpha
            rrf_scores[doc_id] = bm25_weight * (1.0 / (self.rrf_k + rank))
        
        # Process vector results
        for rank, doc in enumerate(vector_results, 1):
            doc_id = self._get_document_id(doc)
            doc_map[doc_id] = doc
            
            # RRF score for vector (weighted by alpha)
            vector_weight = self.alpha
            current_score = rrf_scores.get(doc_id, 0)
            rrf_scores[doc_id] = current_score + vector_weight * (1.0 / (self.rrf_k + rank))
        
        # Sort by combined RRF score
        sorted_ids = sorted(
            rrf_scores.keys(),
            key=lambda x: rrf_scores[x],
            reverse=True
        )
        
        # Return top-k documents
        return [doc_map[doc_id] for doc_id in sorted_ids[:k]]
    
    def fuse_with_scores(
        self,
        bm25_results: List[Tuple[Document, float]],
        vector_results: List[Tuple[Document, float]],
        k: int = 5
    ) -> List[Tuple[Document, float]]:
        """
        Fuse results with scores using weighted combination.
        
        This method uses the actual scores from each retriever
        rather than just ranks, providing more nuanced fusion.
        
        Args:
            bm25_results: (Document, score) tuples from BM25
            vector_results: (Document, score) tuples from vector search
            k: Number of final results
            
        Returns:
            Fused (Document, combined_score) tuples
        """
        doc_map: Dict[str, Document] = {}
        combined_scores: Dict[str, float] = {}
        
        # Normalize BM25 scores (they can be any positive number)
        if bm25_results:
            max_bm25 = max(score for _, score in bm25_results)
            min_bm25 = min(score for _, score in bm25_results)
            bm25_range = max_bm25 - min_bm25 if max_bm25 != min_bm25 else 1
            
            for doc, score in bm25_results:
                doc_id = self._get_document_id(doc)
                doc_map[doc_id] = doc
                # Normalize to 0-1 range
                normalized = (score - min_bm25) / bm25_range
                combined_scores[doc_id] = (1 - self.alpha) * normalized
        
        # Vector scores are typically distances (lower = better)
        # We need to invert them for fusion
        if vector_results:
            max_vec = max(score for _, score in vector_results)
            min_vec = min(score for _, score in vector_results)
            vec_range = max_vec - min_vec if max_vec != min_vec else 1
            
            for doc, score in vector_results:
                doc_id = self._get_document_id(doc)
                doc_map[doc_id] = doc
                # Invert and normalize (lower distance = higher score)
                normalized = 1 - ((score - min_vec) / vec_range)
                current = combined_scores.get(doc_id, 0)
                combined_scores[doc_id] = current + self.alpha * normalized
        
        # Sort by combined score
        sorted_ids = sorted(
            combined_scores.keys(),
            key=lambda x: combined_scores[x],
            reverse=True
        )
        
        return [
            (doc_map[doc_id], combined_scores[doc_id]) 
            for doc_id in sorted_ids[:k]
        ]
    
    def simple_interleave(
        self,
        bm25_results: List[Document],
        vector_results: List[Document],
        k: int = 5
    ) -> List[Document]:
        """
        Simple interleaving fusion (alternating results).
        
        This is a simpler alternative to RRF that alternates
        between BM25 and vector results.
        
        Args:
            bm25_results: Documents from BM25
            vector_results: Documents from vector search
            k: Number of results
            
        Returns:
            Interleaved document list
        """
        result = []
        seen_ids = set()
        
        max_len = max(len(bm25_results), len(vector_results))
        
        for i in range(max_len):
            if len(result) >= k:
                break
            
            # Add from BM25
            if i < len(bm25_results):
                doc = bm25_results[i]
                doc_id = self._get_document_id(doc)
                if doc_id not in seen_ids:
                    result.append(doc)
                    seen_ids.add(doc_id)
            
            if len(result) >= k:
                break
            
            # Add from vector
            if i < len(vector_results):
                doc = vector_results[i]
                doc_id = self._get_document_id(doc)
                if doc_id not in seen_ids:
                    result.append(doc)
                    seen_ids.add(doc_id)
        
        return result
