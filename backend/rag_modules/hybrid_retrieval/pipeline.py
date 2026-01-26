"""
RAG Master Project - Hybrid RAG Pipeline
=========================================
Complete Hybrid Retrieval pipeline combining BM25 and Vector search.
"""

from typing import List, Optional
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from core.llm import get_chat_model
from core.vector_store import get_vector_store
from config import settings

from rag_modules.hybrid_retrieval.bm25_retriever import BM25Retriever
from rag_modules.hybrid_retrieval.vector_retriever import VectorRetriever
from rag_modules.hybrid_retrieval.hybrid_fusion import HybridFusion


RAG_SYSTEM_PROMPT = """You are a helpful AI assistant that answers questions based on the provided context.

**Instructions:**
1. Answer the question using ONLY the information from the provided context
2. If the context doesn't contain enough information, clearly state that
3. Cite relevant parts of the context to support your answer
4. Be concise but comprehensive

**Context:**
{context}

**Question:** {question}

**Answer:**"""


class HybridRAG:
    """
    Hybrid Retrieval RAG Pipeline.
    
    Combines BM25 (sparse/keyword) and Vector (dense/semantic) retrieval
    using Reciprocal Rank Fusion for optimal results.
    
    This technique is particularly effective for:
    - Technical documentation with specialized terminology
    - Content requiring both exact keyword and semantic matching
    - Multi-domain knowledge bases
    - Queries that mix specific terms with conceptual questions
    
    Example:
        >>> rag = HybridRAG(alpha=0.5)
        >>> rag.index_documents(documents)
        >>> response = rag.query("What is the attention mechanism in transformers?")
    """
    
    def __init__(
        self,
        alpha: float = None,
        top_k: int = None,
        system_prompt: str = None
    ):
        """
        Initialize Hybrid RAG pipeline.
        
        Args:
            alpha: Weight for vector vs BM25 (0=BM25 only, 1=Vector only)
            top_k: Number of documents to retrieve
            system_prompt: Custom system prompt for generation
        """
        self.alpha = alpha if alpha is not None else settings.HYBRID_ALPHA
        self.top_k = top_k or settings.TOP_K_RETRIEVAL
        self.system_prompt = system_prompt or RAG_SYSTEM_PROMPT
        
        # Initialize components
        self.bm25_retriever = BM25Retriever()
        self.vector_retriever = VectorRetriever()
        self.fusion = HybridFusion(alpha=self.alpha)
        
        self.llm = get_chat_model()
        self.prompt = ChatPromptTemplate.from_template(self.system_prompt)
        
        self._indexed = False
    
    def index_documents(self, documents: List[Document]) -> None:
        """
        Index documents for both BM25 and vector retrieval.
        
        Args:
            documents: List of documents to index
        """
        # Index for BM25
        self.bm25_retriever.index_documents(documents)
        
        # Vector store is already indexed (shared with other pipelines)
        # But we ensure the documents are added
        vector_store = get_vector_store()
        vector_store.add_documents(documents)
        
        self._indexed = True
    
    def _format_documents(self, documents: List[Document]) -> str:
        """Format documents into context string."""
        formatted_parts = []
        
        for i, doc in enumerate(documents, 1):
            source = doc.metadata.get("source", "Unknown")
            chunk_idx = doc.metadata.get("chunk_index", 0)
            formatted_parts.append(
                f"[Source {i}: {source} (chunk {chunk_idx})]\n{doc.page_content}"
            )
        
        return "\n\n---\n\n".join(formatted_parts)
    
    def retrieve(
        self, 
        query: str,
        fetch_k: int = None
    ) -> List[Document]:
        """
        Retrieve documents using hybrid fusion.
        
        Args:
            query: Search query
            fetch_k: Number of candidates to fetch from each retriever
            
        Returns:
            Fused list of relevant documents
        """
        fetch_k = fetch_k or (self.top_k * 2)
        
        # Get results from both retrievers
        bm25_results = self.bm25_retriever.retrieve(query, k=fetch_k)
        vector_results = self.vector_retriever.retrieve(query, k=fetch_k)
        
        # If no results from either, return what we have
        if not bm25_results and not vector_results:
            return []
        
        # If only one has results, return those
        if not bm25_results:
            return vector_results[:self.top_k]
        if not vector_results:
            return bm25_results[:self.top_k]
        
        # Fuse results
        fused = self.fusion.fuse_with_rrf(
            bm25_results,
            vector_results,
            k=self.top_k
        )
        
        return fused
    
    def retrieve_with_breakdown(
        self, 
        query: str,
        fetch_k: int = None
    ) -> dict:
        """
        Retrieve with detailed breakdown of results from each retriever.
        
        Useful for debugging and understanding hybrid retrieval behavior.
        
        Args:
            query: Search query
            fetch_k: Number of candidates to fetch
            
        Returns:
            Dictionary with BM25, vector, and fused results
        """
        fetch_k = fetch_k or (self.top_k * 2)
        
        bm25_results = self.bm25_retriever.retrieve_with_scores(query, k=fetch_k)
        vector_results = self.vector_retriever.retrieve_with_scores(query, k=fetch_k)
        
        fused = self.fusion.fuse_with_rrf(
            [doc for doc, _ in bm25_results],
            [doc for doc, _ in vector_results],
            k=self.top_k
        )
        
        return {
            "bm25_results": [
                {"content": doc.page_content[:200], "score": score}
                for doc, score in bm25_results[:5]
            ],
            "vector_results": [
                {"content": doc.page_content[:200], "score": score}
                for doc, score in vector_results[:5]
            ],
            "fused_results": [
                {"content": doc.page_content[:200]}
                for doc in fused
            ],
        }
    
    def generate(self, question: str, context: str) -> str:
        """Generate answer from context."""
        if not context.strip():
            return "I don't have any documents to answer your question. Please upload some documents first."
        chain = self.prompt | self.llm | StrOutputParser()
        return chain.invoke({"question": question, "context": context})
    
    def query(
        self, 
        question: str,
        return_sources: bool = True,
        return_breakdown: bool = False
    ) -> dict:
        """
        Run the full Hybrid RAG pipeline.
        
        Args:
            question: User's question
            return_sources: Include source documents
            return_breakdown: Include retrieval breakdown
            
        Returns:
            Response dictionary
        """
        # Retrieve
        documents = self.retrieve(question)
        
        # Format context
        context = self._format_documents(documents)
        
        # Generate
        answer = self.generate(question, context)
        
        # Build response
        response = {
            "answer": answer,
            "technique": "hybrid",
            "num_sources": len(documents),
            "alpha": self.alpha,
        }
        
        if return_sources:
            response["sources"] = [
                {
                    "content": doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content,
                    "metadata": doc.metadata,
                }
                for doc in documents
            ]
        
        if return_breakdown:
            response["retrieval_breakdown"] = self.retrieve_with_breakdown(question)
        
        return response
    
    def sync_with_vector_store(self) -> None:
        """
        Synchronize BM25 index with vector store.
        
        Call this if documents were added to the vector store externally.
        """
        vector_store = get_vector_store()
        documents = vector_store.get_all_documents()
        
        if documents:
            self.bm25_retriever.index_documents(documents)
            self._indexed = True


# Convenience function
def run_hybrid_rag(question: str, alpha: float = 0.5) -> dict:
    """Run a query through the Hybrid RAG pipeline."""
    rag = HybridRAG(alpha=alpha)
    rag.sync_with_vector_store()
    return rag.query(question)
