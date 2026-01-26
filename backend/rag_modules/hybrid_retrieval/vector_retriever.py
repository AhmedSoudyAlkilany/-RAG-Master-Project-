"""
RAG Master Project - Vector Retriever
======================================
Dense retrieval using vector embeddings.
Part of the Hybrid Retrieval system.
"""

from typing import List, Optional, Tuple
from langchain_core.documents import Document

from core.vector_store import VectorStore, get_vector_store


class VectorRetriever:
    """
    Dense vector retriever using embeddings.
    
    This retriever uses semantic embeddings to find documents
    that are conceptually similar to the query, even if they
    don't share exact keywords.
    
    Key advantages:
    - Captures semantic meaning
    - Handles synonyms and paraphrases
    - Works across languages
    - Finds conceptually related content
    
    Example:
        >>> retriever = VectorRetriever()
        >>> results = retriever.retrieve("machine learning algorithms", k=5)
    """
    
    def __init__(self, vector_store: Optional[VectorStore] = None):
        """
        Initialize vector retriever.
        
        Args:
            vector_store: Optional VectorStore instance. Uses global if not provided.
        """
        self.vector_store = vector_store or get_vector_store()
    
    def retrieve(
        self, 
        query: str, 
        k: int = 5
    ) -> List[Document]:
        """
        Retrieve top-k semantically similar documents.
        
        Args:
            query: Search query
            k: Number of documents to retrieve
            
        Returns:
            List of similar documents
        """
        return self.vector_store.similarity_search(query, k=k)
    
    def retrieve_with_scores(
        self, 
        query: str, 
        k: int = 5
    ) -> List[Tuple[Document, float]]:
        """
        Retrieve top-k documents with similarity scores.
        
        Note: Lower scores indicate higher similarity for distance-based metrics.
        
        Args:
            query: Search query
            k: Number of documents to retrieve
            
        Returns:
            List of (document, score) tuples
        """
        return self.vector_store.similarity_search_with_scores(query, k=k)
    
    def retrieve_mmr(
        self,
        query: str,
        k: int = 5,
        fetch_k: int = 20,
        lambda_mult: float = 0.5
    ) -> List[Document]:
        """
        Retrieve documents using Maximum Marginal Relevance.
        
        MMR balances relevance and diversity by iteratively selecting
        documents that are both similar to the query and different
        from already selected documents.
        
        Args:
            query: Search query
            k: Number of documents to retrieve
            fetch_k: Number of candidates to consider
            lambda_mult: Balance factor (0=max diversity, 1=max relevance)
            
        Returns:
            List of diverse, relevant documents
        """
        return self.vector_store.max_marginal_relevance_search(
            query, k=k, fetch_k=fetch_k, lambda_mult=lambda_mult
        )
    
    @property
    def document_count(self) -> int:
        """Get the number of indexed documents."""
        return self.vector_store.document_count
