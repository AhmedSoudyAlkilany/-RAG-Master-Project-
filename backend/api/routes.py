"""
RAG Master Project - API Routes
================================
FastAPI routes for the RAG system.
"""

import os
from typing import List
from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse

from api.schemas import (
    QueryRequest, QueryResponse, SourceDocument,
    DocumentUploadResponse, DocumentListResponse,
    ModelListResponse, ModelSwitchRequest, ModelSwitchResponse,
    HealthResponse, TechniqueComparisonRequest, TechniqueComparisonResponse
)
from core import get_llm, get_embeddings, get_vector_store, DocumentLoader
from config import AVAILABLE_LLM_MODELS, AVAILABLE_EMBED_MODELS


# Create router
router = APIRouter()


# ==========================================================================
# Health & Status
# ==========================================================================

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Check system health."""
    try:
        llm = get_llm()
        vector_store = get_vector_store()
        
        return HealthResponse(
            status="healthy",
            ollama_connected=True,
            vector_store_ready=True,
            document_count=vector_store.document_count
        )
    except Exception as e:
        return HealthResponse(
            status="unhealthy",
            ollama_connected=False,
            vector_store_ready=False,
            document_count=0
        )


# ==========================================================================
# Query Endpoints
# ==========================================================================

@router.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    """
    Query the RAG system using the specified technique.
    
    Techniques:
    - naive: Basic retrieval + generation
    - hybrid: BM25 + Vector fusion
    - rerank: With re-ranking step
    - crag: Corrective RAG with web fallback
    """
    try:
        # Import pipelines
        from rag_modules.naive_rag import NaiveRAG
        from rag_modules.hybrid_retrieval import HybridRAG
        from rag_modules.reranking import RerankingRAG
        from rag_modules.corrective_rag import CorrectiveRAG
        
        # Select technique
        if request.technique == "naive":
            rag = NaiveRAG()
            result = rag.query(request.question, return_sources=request.return_sources)
            
        elif request.technique == "hybrid":
            alpha = request.hybrid_alpha or 0.5
            rag = HybridRAG(alpha=alpha)
            rag.sync_with_vector_store()
            result = rag.query(request.question, return_sources=request.return_sources)
            result["hybrid_alpha"] = alpha
            
        elif request.technique == "rerank":
            reranker = request.reranker_type or "llm"
            rag = RerankingRAG(reranker_type=reranker)
            result = rag.query(request.question, return_sources=request.return_sources)
            
        elif request.technique == "crag":
            rag = CorrectiveRAG()
            result = rag.query(request.question, return_sources=request.return_sources)
            
        else:
            raise HTTPException(status_code=400, detail=f"Unknown technique: {request.technique}")
        
        # Convert sources to schema format
        sources = None
        if request.return_sources and result.get("sources"):
            sources = [
                SourceDocument(
                    content=s.get("content", ""),
                    metadata=s.get("metadata", {}),
                    type=s.get("type", "local")
                )
                for s in result["sources"]
            ]
        
        return QueryResponse(
            answer=result["answer"],
            technique=result["technique"],
            num_sources=result.get("num_sources", len(sources) if sources else 0),
            sources=sources,
            hybrid_alpha=result.get("hybrid_alpha"),
            reranker_type=result.get("reranker_type"),
            correction_used=result.get("correction_used"),
            evaluation_result=result.get("evaluation_result")
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/compare", response_model=TechniqueComparisonResponse)
async def compare_techniques(request: TechniqueComparisonRequest):
    """Compare answers from multiple RAG techniques."""
    results = {}
    
    for technique in request.techniques:
        try:
            query_req = QueryRequest(
                question=request.question,
                technique=technique,
                return_sources=False
            )
            response = await query_rag(query_req)
            results[technique] = {
                "answer": response.answer,
                "technique": response.technique
            }
        except Exception as e:
            results[technique] = {"error": str(e)}
    
    return TechniqueComparisonResponse(
        question=request.question,
        results=results
    )


# ==========================================================================
# Document Management
# ==========================================================================

@router.post("/documents/upload", response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """Upload and process a document."""
    try:
        # Validate file type
        ext = os.path.splitext(file.filename)[1].lower()
        supported = DocumentLoader.get_supported_extensions()
        
        if ext not in supported:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {ext}. Supported: {supported}"
            )
        
        # Read file
        content = await file.read()
        
        # Load and chunk document
        loader = DocumentLoader()
        documents = loader.load_from_bytes(content, file.filename)
        
        # Add to vector store
        vector_store = get_vector_store()
        vector_store.add_documents(documents)
        
        return DocumentUploadResponse(
            success=True,
            message=f"Successfully uploaded and processed {file.filename}",
            num_chunks=len(documents),
            filename=file.filename
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/documents", response_model=DocumentListResponse)
async def list_documents():
    """List all documents in the system."""
    try:
        vector_store = get_vector_store()
        docs = vector_store.get_all_documents()
        
        # Group by source
        sources = {}
        for doc in docs:
            source = doc.metadata.get("filename", doc.metadata.get("source", "Unknown"))
            if source not in sources:
                sources[source] = {"filename": source, "chunks": 0}
            sources[source]["chunks"] += 1
        
        return DocumentListResponse(
            total_documents=len(docs),
            documents=list(sources.values())
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/documents")
async def clear_documents():
    """Clear all documents from the system."""
    try:
        vector_store = get_vector_store()
        vector_store.reset()
        
        return {"success": True, "message": "All documents cleared"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==========================================================================
# Model Management
# ==========================================================================

@router.get("/models", response_model=ModelListResponse)
async def list_models():
    """List available models."""
    llm = get_llm()
    embeddings = get_embeddings()
    
    return ModelListResponse(
        llm_models=AVAILABLE_LLM_MODELS,
        embedding_models=AVAILABLE_EMBED_MODELS,
        active_llm=llm.current_model,
        active_embedding=embeddings.current_model
    )


@router.post("/models/switch", response_model=ModelSwitchResponse)
async def switch_model(request: ModelSwitchRequest):
    """Switch active models."""
    try:
        llm = get_llm()
        embeddings = get_embeddings()
        
        if request.llm_model:
            llm.switch_model(request.llm_model)
        
        if request.embedding_model:
            embeddings.switch_model(request.embedding_model)
        
        return ModelSwitchResponse(
            success=True,
            message="Models switched successfully",
            active_llm=llm.current_model,
            active_embedding=embeddings.current_model
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
