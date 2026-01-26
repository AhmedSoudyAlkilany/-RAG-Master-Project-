"""
RAG Master Project - API Schemas
=================================
Pydantic models for API request/response validation.
"""

from typing import List, Optional, Literal
from pydantic import BaseModel, Field


# ==========================================================================
# Common Models
# ==========================================================================

class SourceDocument(BaseModel):
    """A source document from retrieval."""
    content: str = Field(..., description="Document content (may be truncated)")
    metadata: dict = Field(default_factory=dict)
    type: str = Field(default="local", description="local or web")


class RAGResponse(BaseModel):
    """Standard RAG response."""
    answer: str = Field(..., description="Generated answer")
    technique: str = Field(..., description="RAG technique used")
    num_sources: int = Field(ge=0)
    sources: Optional[List[SourceDocument]] = None


# ==========================================================================
# Query Models
# ==========================================================================

RAGTechnique = Literal["naive", "hybrid", "rerank", "crag", "combined"]


class QueryRequest(BaseModel):
    """Request for querying the RAG system."""
    question: str = Field(..., min_length=1, max_length=2000)
    technique: RAGTechnique = Field(default="hybrid")
    return_sources: bool = Field(default=True)
    
    # Technique-specific options
    hybrid_alpha: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    reranker_type: Optional[Literal["llm", "cross_encoder"]] = None


class QueryResponse(BaseModel):
    """Response from RAG query."""
    answer: str
    technique: str
    num_sources: int | None = None
    sources: Optional[List[SourceDocument]] = None
    
    # Technique-specific fields
    hybrid_alpha: Optional[float] = None
    reranker_type: Optional[str] = None
    correction_used: Optional[bool] = None
    evaluation_result: Optional[str] = None


# ==========================================================================
# Document Models
# ==========================================================================

class DocumentUploadResponse(BaseModel):
    """Response from document upload."""
    success: bool
    message: str
    num_chunks: int = 0
    filename: Optional[str] = None


class DocumentListResponse(BaseModel):
    """List of documents in the system."""
    total_documents: int
    documents: List[dict]


# ==========================================================================
# Model Configuration
# ==========================================================================

class ModelInfo(BaseModel):
    """Information about an available model."""
    name: str
    type: Literal["llm", "embedding"]
    is_active: bool


class ModelListResponse(BaseModel):
    """List of available models."""
    llm_models: List[str]
    embedding_models: List[str]
    active_llm: str
    active_embedding: str


class ModelSwitchRequest(BaseModel):
    """Request to switch models."""
    llm_model: Optional[str] = None
    embedding_model: Optional[str] = None


class ModelSwitchResponse(BaseModel):
    """Response from model switch."""
    success: bool
    message: str
    active_llm: str
    active_embedding: str


# ==========================================================================
# Health & Status
# ==========================================================================

class HealthResponse(BaseModel):
    """API health status."""
    status: str
    ollama_connected: bool
    vector_store_ready: bool
    document_count: int


class TechniqueComparisonRequest(BaseModel):
    """Request to compare RAG techniques."""
    question: str = Field(..., min_length=1)
    techniques: List[RAGTechnique] = Field(
        default=["naive", "hybrid", "rerank", "crag"]
    )


class TechniqueComparisonResponse(BaseModel):
    """Comparison results across techniques."""
    question: str
    results: dict  # technique -> response
