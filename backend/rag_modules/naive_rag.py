"""
RAG Master Project - Naive RAG Pipeline
========================================
Baseline RAG implementation: Index → Retrieve → Generate
This serves as the foundation for comparing advanced techniques.
"""

from typing import List, Optional, Tuple
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from core.llm import get_chat_model
from core.vector_store import get_vector_store
from config import settings


# Default RAG system prompt
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


class NaiveRAG:
    """
    Naive (Basic) RAG Pipeline.
    
    This implements the fundamental RAG pattern:
    1. User asks a question
    2. Retrieve relevant documents from vector store
    3. Combine documents as context
    4. Generate answer using LLM
    
    This serves as the baseline for comparing advanced techniques.
    
    Example:
        >>> rag = NaiveRAG()
        >>> response = rag.query("What is machine learning?")
        >>> print(response["answer"])
    """
    
    def __init__(
        self,
        system_prompt: Optional[str] = None,
        top_k: Optional[int] = None
    ):
        self.system_prompt = system_prompt or RAG_SYSTEM_PROMPT
        self.top_k = top_k or settings.TOP_K_RETRIEVAL
        
        self.vector_store = get_vector_store()
        self.llm = get_chat_model()
        
        # Build prompt template
        self.prompt = ChatPromptTemplate.from_template(self.system_prompt)
    
    def _format_documents(self, documents: List[Document]) -> str:
        """Format retrieved documents into a single context string."""
        formatted_parts = []
        
        for i, doc in enumerate(documents, 1):
            source = doc.metadata.get("source", "Unknown")
            chunk_idx = doc.metadata.get("chunk_index", 0)
            formatted_parts.append(
                f"[Source {i}: {source} (chunk {chunk_idx})]\n{doc.page_content}"
            )
        
        return "\n\n---\n\n".join(formatted_parts)
    
    def retrieve(self, query: str) -> List[Document]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: User's question
            
        Returns:
            List of relevant documents
        """
        return self.vector_store.similarity_search(query, k=self.top_k)
    
    def retrieve_with_scores(
        self, 
        query: str
    ) -> List[Tuple[Document, float]]:
        """
        Retrieve documents with similarity scores.
        
        Args:
            query: User's question
            
        Returns:
            List of (document, score) tuples
        """
        return self.vector_store.similarity_search_with_scores(query, k=self.top_k)
    
    def generate(
        self, 
        question: str, 
        context: str
    ) -> str:
        """
        Generate an answer given question and context.
        
        Args:
            question: User's question
            context: Formatted document context
            
        Returns:
            Generated answer
        """
        if not context.strip():
            return "I don't have any documents to answer your question. Please upload some documents first."
        chain = self.prompt | self.llm | StrOutputParser()
        return chain.invoke({"question": question, "context": context})
    
    def query(
        self, 
        question: str,
        return_sources: bool = True
    ) -> dict:
        """
        Run the full RAG pipeline.
        
        Args:
            question: User's question
            return_sources: Whether to include source documents in response
            
        Returns:
            Dictionary with answer and optionally source documents
        """
        # Step 1: Retrieve
        documents = self.retrieve(question)
        
        # Step 2: Format context
        context = self._format_documents(documents)
        
        # Step 3: Generate
        answer = self.generate(question, context)
        
        # Build response
        response = {
            "answer": answer,
            "technique": "naive",
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
        
        return response
    
    def build_chain(self):
        """
        Build a LangChain LCEL chain for the RAG pipeline.
        
        Returns:
            A runnable chain that takes a question and returns an answer
        """
        retriever = self.vector_store.as_retriever(k=self.top_k)
        
        def format_docs(docs: List[Document]) -> str:
            return self._format_documents(docs)
        
        chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
        
        return chain


# Convenience function
def run_naive_rag(question: str) -> dict:
    """Run a query through the Naive RAG pipeline."""
    rag = NaiveRAG()
    return rag.query(question)
