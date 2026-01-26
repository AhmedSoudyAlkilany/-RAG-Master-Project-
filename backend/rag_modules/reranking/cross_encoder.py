"""
RAG Master Project - Cross-Encoder Re-ranker
=============================================
High-quality re-ranking using sentence-transformers cross-encoder models.
"""

from typing import List, Tuple, Optional
from langchain_core.documents import Document

from config import settings


class CrossEncoderReranker:
    """
    Cross-Encoder based document re-ranker.
    
    Cross-encoders jointly encode the query and document together,
    allowing for rich interaction between them. This produces more
    accurate relevance scores than bi-encoder approaches.
    
    The trade-off is speed: cross-encoders are slower because they
    can't pre-compute document embeddings.
    
    Best for:
    - High-value information needs
    - When retrieval quality is critical
    - Handling ambiguous queries
    
    Example:
        >>> reranker = CrossEncoderReranker()
        >>> ranked = reranker.rerank(query, documents, top_k=3)
    """
    
    # Default cross-encoder model
    DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    
    def __init__(self, model_name: str = None):
        """
        Initialize cross-encoder re-ranker.
        
        Args:
            model_name: HuggingFace model name for cross-encoder
        """
        self.model_name = model_name or self.DEFAULT_MODEL
        self._model = None
    
    def _load_model(self):
        """Lazy-load the cross-encoder model."""
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
                self._model = CrossEncoder(self.model_name)
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required for CrossEncoderReranker. "
                    "Install with: pip install sentence-transformers"
                )
        return self._model
    
    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: Optional[int] = None
    ) -> List[Document]:
        """
        Re-rank documents using cross-encoder.
        
        Args:
            query: Search query
            documents: Documents to re-rank
            top_k: Number of top documents to return
            
        Returns:
            Re-ranked document list
        """
        top_k = top_k or settings.TOP_K_RERANK
        
        if not documents:
            return []
        
        model = self._load_model()
        
        # Prepare query-document pairs
        pairs = [(query, doc.page_content) for doc in documents]
        
        # Get scores
        scores = model.predict(pairs)
        
        # Sort by score (descending)
        scored_docs = list(zip(documents, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        return [doc for doc, _ in scored_docs[:top_k]]
    
    def rerank_with_scores(
        self,
        query: str,
        documents: List[Document],
        top_k: Optional[int] = None
    ) -> List[Tuple[Document, float]]:
        """
        Re-rank and return documents with scores.
        
        Args:
            query: Search query
            documents: Documents to re-rank
            top_k: Number of top documents to return
            
        Returns:
            List of (document, score) tuples
        """
        top_k = top_k or settings.TOP_K_RERANK
        
        if not documents:
            return []
        
        model = self._load_model()
        
        # Prepare query-document pairs
        pairs = [(query, doc.page_content) for doc in documents]
        
        # Get scores
        scores = model.predict(pairs)
        
        # Sort by score (descending)
        scored_docs = list(zip(documents, scores.tolist()))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        return scored_docs[:top_k]
