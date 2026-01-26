"""
RAG Master Project - Configuration
===================================
Central configuration for Ollama models, vector store, and application settings.
Supports dynamic model switching for LLM and embeddings.
"""

import os
from typing import Literal
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # ==========================================================================
    # Ollama Configuration
    # ==========================================================================
    OLLAMA_BASE_URL: str = Field(
        default="http://localhost:11434",
        description="Ollama API base URL"
    )
    
    # Available LLM models (switchable at runtime)
    OLLAMA_LLM_MODEL: str = Field(
        default="qwen3:4b",
        description="Default Ollama LLM model"
    )
    
    # Available embedding models
    OLLAMA_EMBED_MODEL: str = Field(
        default="nomic-embed-text",
        description="Default Ollama embedding model"
    )
    
    # LLM Parameters
    LLM_TEMPERATURE: float = Field(default=0.1, ge=0.0, le=2.0)
    LLM_MAX_TOKENS: int = Field(default=2048, ge=100, le=8192)
    
    # ==========================================================================
    # Vector Store Configuration
    # ==========================================================================
    CHROMA_PERSIST_DIR: str = Field(
        default="./data/chroma_db",
        description="ChromaDB persistence directory"
    )
    CHROMA_COLLECTION_NAME: str = Field(
        default="rag_documents",
        description="ChromaDB collection name"
    )
    
    # ==========================================================================
    # Document Processing
    # ==========================================================================
    CHUNK_SIZE: int = Field(default=1000, ge=100, le=4000)
    CHUNK_OVERLAP: int = Field(default=200, ge=0, le=500)
    
    # ==========================================================================
    # Retrieval Settings
    # ==========================================================================
    TOP_K_RETRIEVAL: int = Field(default=5, ge=1, le=20)
    TOP_K_RERANK: int = Field(default=3, ge=1, le=10)
    
    # Hybrid Retrieval
    HYBRID_ALPHA: float = Field(
        default=0.5, 
        ge=0.0, 
        le=1.0,
        description="Weight for vector vs BM25 (0=BM25 only, 1=Vector only)"
    )
    
    # CRAG Settings
    CRAG_RELEVANCE_THRESHOLD: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum relevance score to consider document useful"
    )
    WEB_SEARCH_MAX_RESULTS: int = Field(default=3, ge=1, le=10)
    
    # ==========================================================================
    # API Settings
    # ==========================================================================
    API_HOST: str = Field(default="0.0.0.0")
    API_PORT: int = Field(default=8000)
    CORS_ORIGINS: list[str] = Field(
        default=["http://localhost:5173", "http://localhost:3000"]
    )
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


# Global settings instance
settings = Settings()


# ==========================================================================
# Available Models Registry
# ==========================================================================
AVAILABLE_LLM_MODELS = [
    "qwen3:4b",
    "llama3.2:3b",
    "gemma3:4b",
    "qwen2.5-coder:3b",
    "deepseek-r1:1.5b",
    "gemma3:270m",
]

AVAILABLE_EMBED_MODELS = [
    "nomic-embed-text",
    "mxbai-embed-large",
    "all-minilm",
    "snowflake-arctic-embed",
]


# ==========================================================================
# RAG Technique Types
# ==========================================================================
RAGTechnique = Literal[
    "naive",           # Basic RAG
    "hybrid",          # BM25 + Vector
    "rerank",          # With re-ranking
    "crag",            # Corrective RAG
    "combined"         # All techniques combined
]
