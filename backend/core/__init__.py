"""
RAG Master Project - Core Module
=================================
Core utilities for LLM, embeddings, document loading, and vector storage.
"""

from core.llm import OllamaLLM, get_llm, get_chat_model
from core.embeddings import OllamaEmbeddingsWrapper, get_embeddings, get_langchain_embeddings
from core.document_loader import DocumentLoader, load_documents
from core.vector_store import VectorStore, get_vector_store

__all__ = [
    # LLM
    "OllamaLLM",
    "get_llm",
    "get_chat_model",
    # Embeddings
    "OllamaEmbeddingsWrapper",
    "get_embeddings",
    "get_langchain_embeddings",
    # Document Loading
    "DocumentLoader",
    "load_documents",
    # Vector Store
    "VectorStore",
    "get_vector_store",
]
