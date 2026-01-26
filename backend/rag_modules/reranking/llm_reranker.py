"""
RAG Master Project - LLM Re-ranker
===================================
Re-ranks retrieved documents using LLM-based relevance scoring.
"""

from typing import List, Tuple, Optional
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from core.llm import get_chat_model
from config import settings


# Re-ranking prompt template
RERANK_PROMPT = """You are a relevance scoring assistant. Your task is to score how relevant a document is to a given query.

**Query:** {query}

**Document:** 
{document}

**Instructions:**
1. Analyze how well the document answers or relates to the query
2. Consider semantic relevance, not just keyword matching
3. Score from 1-10 where:
   - 1-3: Not relevant or barely relevant
   - 4-6: Somewhat relevant, contains related information
   - 7-10: Highly relevant, directly answers or addresses the query

**Output ONLY a single number (1-10) representing the relevance score:**"""


class LLMReranker:
    """
    LLM-based document re-ranker.
    
    Uses the LLM to score and re-rank retrieved documents based on
    their relevance to the query. This provides more nuanced ranking
    than initial retrieval methods.
    
    Advantages:
    - Captures nuanced relevance signals
    - Handles ambiguous queries better
    - Can be applied to any retrieval method
    
    Considerations:
    - Adds latency (one LLM call per document)
    - For large result sets, use batching or limit candidates
    
    Example:
        >>> reranker = LLMReranker()
        >>> ranked = reranker.rerank(query, documents, top_k=3)
    """
    
    def __init__(self, prompt_template: str = None):
        """
        Initialize LLM re-ranker.
        
        Args:
            prompt_template: Custom prompt for scoring
        """
        self.prompt = ChatPromptTemplate.from_template(
            prompt_template or RERANK_PROMPT
        )
        self.llm = get_chat_model()
    
    def _score_document(
        self, 
        query: str, 
        document: Document
    ) -> float:
        """
        Score a single document for relevance.
        
        Args:
            query: The search query
            document: Document to score
            
        Returns:
            Relevance score (1-10)
        """
        try:
            chain = self.prompt | self.llm | StrOutputParser()
            result = chain.invoke({
                "query": query,
                "document": document.page_content[:2000]  # Limit content length
            })
            
            # Parse score from response
            score = float(result.strip())
            return max(1.0, min(10.0, score))  # Clamp to 1-10
            
        except (ValueError, Exception) as e:
            # Default to middle score on parsing errors
            print(f"Scoring error: {e}")
            return 5.0
    
    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: Optional[int] = None
    ) -> List[Document]:
        """
        Re-rank documents by LLM-scored relevance.
        
        Args:
            query: Search query
            documents: Documents to re-rank
            top_k: Number of top documents to return
            
        Returns:
            Re-ranked document list
        """
        top_k = top_k or settings.TOP_K_RERANK
        
        if not documents:
            return []
        
        # Score each document
        scored_docs: List[Tuple[Document, float]] = []
        for doc in documents:
            score = self._score_document(query, doc)
            scored_docs.append((doc, score))
        
        # Sort by score (descending)
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        # Return top-k
        return [doc for doc, _ in scored_docs[:top_k]]
    
    def rerank_with_scores(
        self,
        query: str,
        documents: List[Document],
        top_k: Optional[int] = None
    ) -> List[Tuple[Document, float]]:
        """
        Re-rank and return documents with their scores.
        
        Args:
            query: Search query
            documents: Documents to re-rank
            top_k: Number of top documents to return
            
        Returns:
            List of (document, score) tuples
        """
        top_k = top_k or settings.TOP_K_RERANK
        
        if not documents:
            return []
        
        # Score each document
        scored_docs: List[Tuple[Document, float]] = []
        for doc in documents:
            score = self._score_document(query, doc)
            scored_docs.append((doc, score))
        
        # Sort by score (descending)
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        return scored_docs[:top_k]


# Batch re-ranking prompt for efficiency
BATCH_RERANK_PROMPT = """You are a relevance scoring assistant. Score the relevance of each document to the query.

**Query:** {query}

**Documents:**
{documents}

**Instructions:**
For each document, output a score from 1-10 on a separate line.
Format: "Doc N: [score]" for each document.

**Scores:**"""


class BatchLLMReranker:
    """
    Batch LLM re-ranker for efficiency.
    
    Scores multiple documents in a single LLM call to reduce latency.
    """
    
    def __init__(self):
        self.prompt = ChatPromptTemplate.from_template(BATCH_RERANK_PROMPT)
        self.llm = get_chat_model()
    
    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: Optional[int] = None
    ) -> List[Document]:
        """Batch re-rank documents."""
        top_k = top_k or settings.TOP_K_RERANK
        
        if not documents:
            return []
        
        # Format documents for batch scoring
        doc_texts = []
        for i, doc in enumerate(documents, 1):
            content = doc.page_content[:500]  # Truncate for batch
            doc_texts.append(f"Document {i}:\n{content}\n")
        
        documents_str = "\n".join(doc_texts)
        
        try:
            chain = self.prompt | self.llm | StrOutputParser()
            result = chain.invoke({
                "query": query,
                "documents": documents_str
            })
            
            # Parse scores
            scores = self._parse_scores(result, len(documents))
            
            # Sort by score
            scored_docs = list(zip(documents, scores))
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            
            return [doc for doc, _ in scored_docs[:top_k]]
            
        except Exception as e:
            print(f"Batch reranking error: {e}")
            return documents[:top_k]
    
    def _parse_scores(
        self, 
        result: str, 
        num_docs: int
    ) -> List[float]:
        """Parse scores from LLM response."""
        scores = [5.0] * num_docs  # Default scores
        
        for line in result.strip().split("\n"):
            try:
                if ":" in line:
                    parts = line.split(":")
                    doc_num = int(''.join(filter(str.isdigit, parts[0])))
                    score = float(parts[1].strip())
                    if 1 <= doc_num <= num_docs:
                        scores[doc_num - 1] = max(1.0, min(10.0, score))
            except (ValueError, IndexError):
                continue
        
        return scores
