"""
RAG Master Project - BM25 Retriever
====================================
Sparse retrieval using BM25 (Best Matching 25) algorithm.
Part of the Hybrid Retrieval system.
"""

from typing import List, Optional, Tuple
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document


class BM25Retriever:
    """
    BM25 sparse retriever for keyword-based document search.
    
    BM25 is a bag-of-words retrieval function that ranks documents
    based on term frequency and inverse document frequency (TF-IDF).
    
    Key advantages over pure vector search:
    - Excellent for exact keyword matching
    - Handles specialized terminology well
    - Fast and efficient
    - Works without ML models
    
    Example:
        >>> retriever = BM25Retriever()
        >>> retriever.index_documents([Document(page_content="Hello world")])
        >>> results = retriever.retrieve("hello", k=3)
    """
    
    def __init__(self, tokenizer=None):
        """
        Initialize BM25 retriever.
        
        Args:
            tokenizer: Custom tokenizer function. Defaults to simple word split.
        """
        self.tokenizer = tokenizer or self._default_tokenizer
        self.documents: List[Document] = []
        self.bm25: Optional[BM25Okapi] = None
        self._corpus: List[List[str]] = []
    
    @staticmethod
    def _default_tokenizer(text: str) -> List[str]:
        """
        Simple word tokenizer.
        
        Converts text to lowercase and splits on whitespace.
        This is a basic tokenizer; for production, consider using
        a more sophisticated tokenizer like NLTK or spaCy.
        """
        # Convert to lowercase and split on whitespace
        text = text.lower()
        # Remove common punctuation
        for char in ".,!?;:\"'()[]{}":
            text = text.replace(char, " ")
        # Split and filter empty strings
        return [word for word in text.split() if word]
    
    def index_documents(self, documents: List[Document]) -> None:
        """
        Index documents for BM25 retrieval.
        
        Args:
            documents: List of documents to index
        """
        self.documents = documents
        
        # Tokenize all documents
        self._corpus = [
            self.tokenizer(doc.page_content) 
            for doc in documents
        ]
        
        # Build BM25 index
        self.bm25 = BM25Okapi(self._corpus)
    
    def add_documents(self, documents: List[Document]) -> None:
        """
        Add documents to existing index.
        
        Args:
            documents: New documents to add
        """
        if not self.documents:
            self.index_documents(documents)
            return
        
        # Append new documents
        self.documents.extend(documents)
        
        # Retokenize and rebuild index
        new_tokens = [self.tokenizer(doc.page_content) for doc in documents]
        self._corpus.extend(new_tokens)
        
        # Rebuild BM25 index
        self.bm25 = BM25Okapi(self._corpus)
    
    def retrieve(
        self, 
        query: str, 
        k: int = 5
    ) -> List[Document]:
        """
        Retrieve top-k documents for a query.
        
        Args:
            query: Search query
            k: Number of documents to retrieve
            
        Returns:
            List of relevant documents
        """
        if not self.bm25 or not self.documents:
            return []
        
        # Tokenize query
        query_tokens = self.tokenizer(query)
        
        # Get BM25 scores
        scores = self.bm25.get_scores(query_tokens)
        
        # Get top-k indices
        top_indices = sorted(
            range(len(scores)), 
            key=lambda i: scores[i], 
            reverse=True
        )[:k]
        
        # Return corresponding documents
        return [self.documents[i] for i in top_indices]
    
    def retrieve_with_scores(
        self, 
        query: str, 
        k: int = 5
    ) -> List[Tuple[Document, float]]:
        """
        Retrieve top-k documents with their BM25 scores.
        
        Args:
            query: Search query
            k: Number of documents to retrieve
            
        Returns:
            List of (document, score) tuples
        """
        if not self.bm25 or not self.documents:
            return []
        
        # Tokenize query
        query_tokens = self.tokenizer(query)
        
        # Get BM25 scores
        scores = self.bm25.get_scores(query_tokens)
        
        # Get top-k indices with scores
        indexed_scores = list(enumerate(scores))
        top_k = sorted(indexed_scores, key=lambda x: x[1], reverse=True)[:k]
        
        # Return documents with scores
        return [(self.documents[idx], score) for idx, score in top_k]
    
    @property
    def document_count(self) -> int:
        """Get the number of indexed documents."""
        return len(self.documents)
    
    def clear(self) -> None:
        """Clear all indexed documents."""
        self.documents = []
        self._corpus = []
        self.bm25 = None
