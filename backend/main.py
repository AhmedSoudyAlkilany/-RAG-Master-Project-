"""
RAG Master Project - FastAPI Application
==========================================
Main entry point for the RAG API server.

Run with:
    conda activate ASA
    cd backend
    python main.py
    
Or:
    uvicorn main:app --reload --port 8000
"""

import os
import sys

# Add backend to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from config import settings
from api.routes import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    print("=" * 60)
    print("🚀 RAG Master Project - Starting up...")
    print("=" * 60)
    
    # Initialize core components
    try:
        from core import get_llm, get_embeddings, get_vector_store
        
        print(f"📦 Initializing LLM: {settings.OLLAMA_LLM_MODEL}")
        llm = get_llm()
        
        print(f"📦 Initializing Embeddings: {settings.OLLAMA_EMBED_MODEL}")
        embeddings = get_embeddings()
        
        print(f"📦 Initializing Vector Store: {settings.CHROMA_COLLECTION_NAME}")
        vector_store = get_vector_store()
        
        print(f"✅ Vector store has {vector_store.document_count} documents")
        print("=" * 60)
        print("✅ Server ready! Visit http://localhost:8000/docs for API docs")
        print("=" * 60)
        
    except Exception as e:
        print(f"⚠️ Startup warning: {e}")
        print("Some features may not work until Ollama is properly connected.")
    
    yield
    
    # Shutdown
    print("\n👋 Shutting down RAG Master Project...")


# Create FastAPI app
app = FastAPI(
    title="RAG Master Project",
    description="""
## Advanced RAG Techniques API

This API provides access to multiple RAG (Retrieval-Augmented Generation) techniques:

### Techniques Available:
- **Naive RAG**: Basic retrieval + generation
- **Hybrid Retrieval**: Combines BM25 (keyword) and Vector (semantic) search
- **Re-ranking RAG**: Adds re-ranking step for improved relevance
- **Corrective RAG (CRAG)**: Self-correcting pipeline with web search fallback

### Features:
- Document upload and processing (PDF, DOCX, TXT, MD)
- Dynamic model switching (multiple Ollama models)
- Technique comparison endpoint
- Source attribution in responses

Built with LangChain, LangGraph, and Ollama.
""",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router, prefix="/api")


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "RAG Master Project",
        "version": "1.0.0",
        "description": "Advanced RAG Techniques API",
        "docs": "/docs",
        "techniques": ["naive", "hybrid", "rerank", "crag"],
    }


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=True
    )
