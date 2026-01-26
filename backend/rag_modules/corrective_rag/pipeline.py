"""
RAG Master Project - CRAG Pipeline
===================================
High-level Corrective RAG interface.
"""

from typing import Optional
from rag_modules.corrective_rag.crag_graph import CRAGGraph


class CorrectiveRAG:
    """
    Corrective RAG Pipeline wrapper.
    
    Provides a clean interface to the CRAG LangGraph pipeline
    with the same API as other RAG pipelines.
    """
    
    def __init__(
        self,
        relevance_threshold: float = None,
        top_k: int = None
    ):
        self.graph = CRAGGraph(
            relevance_threshold=relevance_threshold,
            top_k=top_k
        )
    
    def query(
        self, 
        question: str,
        return_sources: bool = True
    ) -> dict:
        """Run CRAG pipeline."""
        return self.graph.query(question, return_sources=return_sources)


def run_crag(question: str) -> dict:
    """Run a query through the Corrective RAG pipeline."""
    rag = CorrectiveRAG()
    return rag.query(question)
