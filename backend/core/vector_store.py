"""
RAG Master Project - Vector Store
==================================
ChromaDB-based vector store with persistence and similarity search.
"""

import os
from typing import List, Optional, Tuple
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_core.vectorstores import VectorStoreRetriever

from config import settings
from core.embeddings import get_langchain_embeddings


class VectorStore:
    """
    ChromaDB vector store wrapper with persistence.
    
    Provides document storage, similarity search, and retrieval
    functionality for the RAG pipeline.
    
    Example:
        >>> store = VectorStore()
        >>> store.add_documents([Document(page_content="Hello world")])
        >>> results = store.similarity_search("greeting", k=3)
    """
    
    def __init__(
        self,
        collection_name: Optional[str] = None,
        persist_directory: Optional[str] = None
    ):
        self.collection_name = collection_name or settings.CHROMA_COLLECTION_NAME
        self.persist_directory = persist_directory or settings.CHROMA_PERSIST_DIR
        
        # Ensure persistence directory exists
        os.makedirs(self.persist_directory, exist_ok=True)
        
        self._vectorstore: Optional[Chroma] = None
        self._initialize_store()
    
    def _initialize_store(self) -> None:
        """Initialize or load the ChromaDB vector store."""
        embeddings = get_langchain_embeddings()
        
        self._vectorstore = Chroma(
            collection_name=self.collection_name,
            embedding_function=embeddings,
            persist_directory=self.persist_directory,
        )
    
    @property
    def vectorstore(self) -> Chroma:
        """Get the underlying Chroma instance."""
        if self._vectorstore is None:
            self._initialize_store()
        return self._vectorstore
    
    def add_documents(self, documents: List[Document]) -> List[str]:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of Document objects to add
            
        Returns:
            List of document IDs
        """
        if not documents:
            return []
        
        ids = self.vectorstore.add_documents(documents)
        return ids
    
    def similarity_search(
        self, 
        query: str, 
        k: Optional[int] = None
    ) -> List[Document]:
        """
        Search for similar documents.
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of similar documents
        """
        k = k or settings.TOP_K_RETRIEVAL
        return self.vectorstore.similarity_search(query, k=k)
    
    def similarity_search_with_scores(
        self, 
        query: str, 
        k: Optional[int] = None
    ) -> List[Tuple[Document, float]]:
        """
        Search for similar documents with similarity scores.
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of (Document, score) tuples
        """
        k = k or settings.TOP_K_RETRIEVAL
        return self.vectorstore.similarity_search_with_score(query, k=k)
    
    def max_marginal_relevance_search(
        self,
        query: str,
        k: Optional[int] = None,
        fetch_k: int = 20,
        lambda_mult: float = 0.5
    ) -> List[Document]:
        """
        Search using Maximum Marginal Relevance for diversity.
        
        Args:
            query: Search query
            k: Number of results to return
            fetch_k: Number of documents to fetch before MMR
            lambda_mult: Diversity factor (0 = max diversity, 1 = max relevance)
            
        Returns:
            List of diverse documents
        """
        k = k or settings.TOP_K_RETRIEVAL
        return self.vectorstore.max_marginal_relevance_search(
            query, 
            k=k, 
            fetch_k=fetch_k,
            lambda_mult=lambda_mult
        )
    
    def as_retriever(
        self, 
        search_type: str = "similarity",
        k: Optional[int] = None,
        **kwargs
    ) -> VectorStoreRetriever:
        """
        Get a LangChain retriever from this vector store.
        
        Args:
            search_type: 'similarity', 'mmr', or 'similarity_score_threshold'
            k: Number of documents to retrieve
            
        Returns:
            VectorStoreRetriever instance
        """
        k = k or settings.TOP_K_RETRIEVAL
        return self.vectorstore.as_retriever(
            search_type=search_type,
            search_kwargs={"k": k, **kwargs}
        )
    
    def get_all_documents(self) -> List[Document]:
        """
        Get all documents in the vector store.
        
        Returns:
            List of all stored documents
        """
        result = self.vectorstore.get()
        if not result or not result.get("documents"):
            return []
        
        documents = []
        for i, doc_content in enumerate(result["documents"]):
            metadata = result["metadatas"][i] if result.get("metadatas") else {}
            documents.append(Document(
                page_content=doc_content,
                metadata=metadata
            ))
        
        return documents
    
    def delete_collection(self) -> None:
        """Delete the entire collection."""
        self.vectorstore.delete_collection()
        self._vectorstore = None
    
    def reset(self) -> None:
        """Reset the collection (delete and recreate)."""
        self.delete_collection()
        self._initialize_store()
    
    @property
    def document_count(self) -> int:
        """Get the number of documents in the store."""
        try:
            result = self.vectorstore.get()
            return len(result.get("ids", []))
        except Exception:
            return 0


# Global vector store instance
_vector_store_instance: Optional[VectorStore] = None


def get_vector_store() -> VectorStore:
    """Get or create the global vector store instance."""
    global _vector_store_instance
    if _vector_store_instance is None:
        _vector_store_instance = VectorStore()
    return _vector_store_instance
