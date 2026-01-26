"""
RAG Master Project - Re-ranking RAG Pipeline
=============================================
Complete RAG pipeline with re-ranking for improved retrieval quality.
"""

from typing import List, Optional, Literal
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from core.llm import get_chat_model
from core.vector_store import get_vector_store
from config import settings

from rag_modules.reranking.llm_reranker import LLMReranker, BatchLLMReranker
from rag_modules.reranking.cross_encoder import CrossEncoderReranker


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


RerankerType = Literal["llm", "cross_encoder", "batch_llm"]


class RerankingRAG:
    """
    RAG Pipeline with Re-ranking.
    
    This pipeline adds a re-ranking step after initial retrieval to
    significantly improve the relevance of documents used for generation.
    
    The process:
    1. Retrieve initial candidates (fetch_k documents)
    2. Score each document with a re-ranker
    3. Select top_k best documents
    4. Generate answer from re-ranked context
    
    Re-ranking is particularly effective for:
    - Ambiguous queries that might retrieve partially relevant docs
    - High-stakes applications requiring precision
    - Improving answer quality without changing the knowledge base
    
    Example:
        >>> rag = RerankingRAG(reranker_type="cross_encoder")
        >>> response = rag.query("What is the attention mechanism?")
    """
    
    def __init__(
        self,
        reranker_type: RerankerType = "llm",
        top_k: int = None,
        fetch_k: int = None,
        system_prompt: str = None
    ):
        """
        Initialize Re-ranking RAG pipeline.
        
        Args:
            reranker_type: Type of re-ranker to use
            top_k: Final number of documents after re-ranking
            fetch_k: Number of candidates to fetch before re-ranking
            system_prompt: Custom system prompt
        """
        self.reranker_type = reranker_type
        self.top_k = top_k or settings.TOP_K_RERANK
        self.fetch_k = fetch_k or settings.TOP_K_RETRIEVAL * 2
        self.system_prompt = system_prompt or RAG_SYSTEM_PROMPT
        
        # Initialize components
        self.vector_store = get_vector_store()
        self.llm = get_chat_model()
        self.prompt = ChatPromptTemplate.from_template(self.system_prompt)
        
        # Initialize re-ranker based on type
        self._init_reranker()
    
    def _init_reranker(self):
        """Initialize the selected re-ranker."""
        if self.reranker_type == "cross_encoder":
            self.reranker = CrossEncoderReranker()
        elif self.reranker_type == "batch_llm":
            self.reranker = BatchLLMReranker()
        else:  # Default to LLM
            self.reranker = LLMReranker()
    
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
    
    def retrieve(self, query: str) -> List[Document]:
        """Initial retrieval of candidates."""
        return self.vector_store.similarity_search(query, k=self.fetch_k)
    
    def rerank(
        self, 
        query: str, 
        documents: List[Document]
    ) -> List[Document]:
        """Re-rank documents by relevance."""
        return self.reranker.rerank(query, documents, top_k=self.top_k)
    
    def retrieve_and_rerank(self, query: str) -> List[Document]:
        """
        Retrieve and re-rank in one step.
        
        Args:
            query: Search query
            
        Returns:
            Re-ranked documents
        """
        # Get candidates
        candidates = self.retrieve(query)
        
        # Re-rank
        if len(candidates) > self.top_k:
            return self.rerank(query, candidates)
        
        return candidates
    
    def retrieve_with_comparison(self, query: str) -> dict:
        """
        Retrieve with before/after re-ranking comparison.
        
        Useful for demonstrating the impact of re-ranking.
        
        Returns:
            Dictionary with before and after rankings
        """
        # Get candidates
        candidates = self.retrieve(query)
        
        # Re-rank
        reranked = self.rerank(query, candidates)
        
        return {
            "before_rerank": [
                {
                    "content": doc.page_content[:200],
                    "source": doc.metadata.get("source", "Unknown")
                }
                for doc in candidates[:self.top_k]
            ],
            "after_rerank": [
                {
                    "content": doc.page_content[:200],
                    "source": doc.metadata.get("source", "Unknown")
                }
                for doc in reranked
            ]
        }
    
    def generate(self, question: str, context: str) -> str:
        """Generate answer from context."""
        chain = self.prompt | self.llm | StrOutputParser()
        return chain.invoke({"question": question, "context": context})
    
    def query(
        self, 
        question: str,
        return_sources: bool = True,
        return_comparison: bool = False
    ) -> dict:
        """
        Run the full Re-ranking RAG pipeline.
        
        Args:
            question: User's question
            return_sources: Include source documents
            return_comparison: Include before/after comparison
            
        Returns:
            Response dictionary
        """
        # Retrieve and re-rank
        documents = self.retrieve_and_rerank(question)
        
        # Format context
        context = self._format_documents(documents)
        
        # Generate
        answer = self.generate(question, context)
        
        # Build response
        response = {
            "answer": answer,
            "technique": "rerank",
            "reranker_type": self.reranker_type,
            "num_sources": len(documents),
        }
        
        if return_sources:
            response["sources"] = [
                {
                    "content": doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content,
                    "metadata": doc.metadata,
                }
                for doc in documents
            ]
        
        if return_comparison:
            response["comparison"] = self.retrieve_with_comparison(question)
        
        return response


# Convenience function
def run_reranking_rag(
    question: str, 
    reranker_type: RerankerType = "llm"
) -> dict:
    """Run a query through the Re-ranking RAG pipeline."""
    rag = RerankingRAG(reranker_type=reranker_type)
    return rag.query(question)
