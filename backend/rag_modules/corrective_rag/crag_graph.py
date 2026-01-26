"""
RAG Master Project - CRAG LangGraph Pipeline
=============================================
Corrective RAG using LangGraph state machine.
"""

from typing import List, Literal, TypedDict, Annotated
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, START, END

from core.llm import get_chat_model
from core.vector_store import get_vector_store
from config import settings

from rag_modules.corrective_rag.evaluator import DocumentEvaluator
from rag_modules.corrective_rag.web_search import WebSearcher


# State definition for CRAG
class CRAGState(TypedDict):
    """State for Corrective RAG pipeline."""
    question: str
    documents: List[Document]
    relevant_documents: List[Document]
    web_documents: List[Document]
    evaluation_result: Literal["relevant", "not_relevant", "ambiguous"]
    generation: str
    correction_used: bool


class CRAGGraph:
    """
    Corrective RAG using LangGraph.
    
    Implements a self-correcting RAG pipeline that:
    1. Retrieves documents from vector store
    2. Evaluates document relevance
    3. If relevant: proceeds to generation
    4. If not relevant: searches the web for supplementary information
    5. Generates final answer with corrected context
    
    This pattern dramatically improves RAG reliability by
    explicitly handling retrieval failures.
    
    Example:
        >>> crag = CRAGGraph()
        >>> result = crag.invoke("What are the latest Python features?")
    """
    
    def __init__(
        self,
        relevance_threshold: float = None,
        top_k: int = None,
        web_search_results: int = None
    ):
        """
        Initialize CRAG graph.
        
        Args:
            relevance_threshold: Minimum relevance score
            top_k: Number of documents to retrieve
            web_search_results: Number of web results on fallback
        """
        self.relevance_threshold = relevance_threshold or settings.CRAG_RELEVANCE_THRESHOLD
        self.top_k = top_k or settings.TOP_K_RETRIEVAL
        self.web_search_results = web_search_results or settings.WEB_SEARCH_MAX_RESULTS
        
        # Components
        self.vector_store = get_vector_store()
        self.evaluator = DocumentEvaluator(relevance_threshold=self.relevance_threshold)
        self.web_searcher = WebSearcher(max_results=self.web_search_results)
        self.llm = get_chat_model()
        
        # Build graph
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the CRAG state graph."""
        
        # Define nodes
        def retrieve(state: CRAGState) -> dict:
            """Retrieve documents from vector store."""
            question = state["question"]
            documents = self.vector_store.similarity_search(question, k=self.top_k)
            return {"documents": documents}
        
        def evaluate_documents(state: CRAGState) -> dict:
            """Evaluate relevance of retrieved documents."""
            question = state["question"]
            documents = state["documents"]
            
            if not documents:
                return {
                    "relevant_documents": [],
                    "evaluation_result": "not_relevant"
                }
            
            # Evaluate each document
            relevant = []
            for doc in documents:
                score = self.evaluator.evaluate_with_confidence(question, doc)
                if score >= self.relevance_threshold:
                    relevant.append(doc)
            
            # Determine overall result
            if len(relevant) >= len(documents) * 0.5:
                result = "relevant"
            elif len(relevant) > 0:
                result = "ambiguous"
            else:
                result = "not_relevant"
            
            return {
                "relevant_documents": relevant,
                "evaluation_result": result
            }
        
        def web_search(state: CRAGState) -> dict:
            """Search web for supplementary information."""
            question = state["question"]
            web_docs = self.web_searcher.search(question)
            return {
                "web_documents": web_docs,
                "correction_used": True
            }
        
        def generate(state: CRAGState) -> dict:
            """Generate answer from context."""
            question = state["question"]
            
            # Combine relevant local docs with web docs
            all_docs = state.get("relevant_documents", [])
            web_docs = state.get("web_documents", [])
            all_docs.extend(web_docs)
            
            if not all_docs:
                return {
                    "generation": "I couldn't find relevant information to answer your question."
                }
            
            # Format context
            context_parts = []
            for i, doc in enumerate(all_docs[:5], 1):
                source_type = "Web" if doc.metadata.get("type") == "web_search" else "Local"
                source = doc.metadata.get("source", "Unknown")
                context_parts.append(
                    f"[{source_type} Source {i}: {source}]\n{doc.page_content}"
                )
            context = "\n\n---\n\n".join(context_parts)
            
            # Generate
            prompt = ChatPromptTemplate.from_template(
                """Answer the question based on the provided context.
                
Context:
{context}

Question: {question}

Instructions:
- Use information from the context to answer
- Indicate if information came from web search vs local documents
- Be accurate and helpful

Answer:"""
            )
            
            chain = prompt | self.llm | StrOutputParser()
            answer = chain.invoke({"question": question, "context": context})
            
            return {"generation": answer}
        
        def decide_correction(state: CRAGState) -> str:
            """Decide whether to use web search."""
            result = state["evaluation_result"]
            
            if result == "not_relevant":
                return "web_search"
            elif result == "ambiguous":
                # For ambiguous, still try web search to supplement
                return "web_search"
            else:
                return "generate"
        
        # Build graph
        workflow = StateGraph(CRAGState)
        
        # Add nodes
        workflow.add_node("retrieve", retrieve)
        workflow.add_node("evaluate", evaluate_documents)
        workflow.add_node("web_search", web_search)
        workflow.add_node("generate", generate)
        
        # Add edges
        workflow.add_edge(START, "retrieve")
        workflow.add_edge("retrieve", "evaluate")
        workflow.add_conditional_edges(
            "evaluate",
            decide_correction,
            {
                "web_search": "web_search",
                "generate": "generate"
            }
        )
        workflow.add_edge("web_search", "generate")
        workflow.add_edge("generate", END)
        
        return workflow.compile()
    
    def invoke(self, question: str) -> dict:
        """
        Run the CRAG pipeline.
        
        Args:
            question: User's question
            
        Returns:
            Final state with answer
        """
        initial_state: CRAGState = {
            "question": question,
            "documents": [],
            "relevant_documents": [],
            "web_documents": [],
            "evaluation_result": "not_relevant",
            "generation": "",
            "correction_used": False,
        }
        
        result = self.graph.invoke(initial_state)
        return result
    
    def query(
        self, 
        question: str,
        return_sources: bool = True
    ) -> dict:
        """
        Query interface matching other RAG pipelines.
        
        Args:
            question: User's question
            return_sources: Include source documents
            
        Returns:
            Response dictionary
        """
        result = self.invoke(question)
        
        response = {
            "answer": result["generation"],
            "technique": "crag",
            "correction_used": result.get("correction_used", False),
            "evaluation_result": result["evaluation_result"],
            "num_local_sources": len(result.get("relevant_documents", [])),
            "num_web_sources": len(result.get("web_documents", [])),
        }
        
        if return_sources:
            sources = []
            for doc in result.get("relevant_documents", []):
                sources.append({
                    "content": doc.page_content[:500],
                    "metadata": doc.metadata,
                    "type": "local"
                })
            for doc in result.get("web_documents", []):
                sources.append({
                    "content": doc.page_content[:500],
                    "metadata": doc.metadata,
                    "type": "web"
                })
            response["sources"] = sources
        
        return response
