"""
RAG Master Project - Document Relevance Evaluator
==================================================
Evaluates if retrieved documents are relevant to the query.
Core component of Corrective RAG (CRAG).
"""

from typing import List, Tuple, Literal
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from core.llm import get_chat_model
from config import settings


# Evaluation prompt
EVALUATION_PROMPT = """You are a relevance evaluation assistant. Your task is to determine if a document is relevant to answering a query.

**Query:** {query}

**Document:**
{document}

**Instructions:**
1. Analyze if the document contains information that could help answer the query
2. Consider both direct answers and supporting context
3. Be strict - only mark as RELEVANT if the document genuinely helps

**Output ONLY one of these two words:**
- RELEVANT - if the document helps answer the query
- NOT_RELEVANT - if the document does not help

**Your evaluation:**"""


# Confidence scoring prompt
CONFIDENCE_PROMPT = """You are a relevance scoring assistant. Rate how relevant a document is to a query.

**Query:** {query}

**Document:**
{document}

**Instructions:**
Score from 0.0 to 1.0 where:
- 0.0-0.3: Not relevant
- 0.3-0.5: Marginally relevant  
- 0.5-0.7: Somewhat relevant
- 0.7-0.9: Highly relevant
- 0.9-1.0: Directly answers the query

**Output ONLY a decimal number between 0.0 and 1.0:**"""


RelevanceStatus = Literal["RELEVANT", "NOT_RELEVANT", "AMBIGUOUS"]


class DocumentEvaluator:
    """
    Document relevance evaluator for Corrective RAG.
    
    Evaluates retrieved documents to determine if they are
    relevant to the query. This is the key component that
    triggers corrective actions when retrieval fails.
    
    Evaluation modes:
    - Binary: RELEVANT or NOT_RELEVANT
    - Confidence: 0.0 to 1.0 score
    - Graded: Categories based on confidence thresholds
    
    Example:
        >>> evaluator = DocumentEvaluator()
        >>> is_relevant = evaluator.evaluate(query, document)
        >>> score = evaluator.evaluate_with_confidence(query, document)
    """
    
    def __init__(
        self,
        relevance_threshold: float = None
    ):
        """
        Initialize evaluator.
        
        Args:
            relevance_threshold: Minimum score to consider relevant
        """
        self.relevance_threshold = relevance_threshold or settings.CRAG_RELEVANCE_THRESHOLD
        
        self.eval_prompt = ChatPromptTemplate.from_template(EVALUATION_PROMPT)
        self.conf_prompt = ChatPromptTemplate.from_template(CONFIDENCE_PROMPT)
        self.llm = get_chat_model()
    
    def evaluate(
        self, 
        query: str, 
        document: Document
    ) -> RelevanceStatus:
        """
        Evaluate if a document is relevant (binary).
        
        Args:
            query: The search query
            document: Document to evaluate
            
        Returns:
            'RELEVANT' or 'NOT_RELEVANT'
        """
        try:
            chain = self.eval_prompt | self.llm | StrOutputParser()
            result = chain.invoke({
                "query": query,
                "document": document.page_content[:2000]
            })
            
            result = result.strip().upper()
            
            if "RELEVANT" in result and "NOT" not in result:
                return "RELEVANT"
            elif "NOT" in result:
                return "NOT_RELEVANT"
            else:
                return "AMBIGUOUS"
                
        except Exception as e:
            print(f"Evaluation error: {e}")
            return "AMBIGUOUS"
    
    def evaluate_with_confidence(
        self, 
        query: str, 
        document: Document
    ) -> float:
        """
        Evaluate document with confidence score.
        
        Args:
            query: The search query
            document: Document to evaluate
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        try:
            chain = self.conf_prompt | self.llm | StrOutputParser()
            result = chain.invoke({
                "query": query,
                "document": document.page_content[:2000]
            })
            
            score = float(result.strip())
            return max(0.0, min(1.0, score))
            
        except (ValueError, Exception) as e:
            print(f"Confidence scoring error: {e}")
            return 0.5
    
    def evaluate_batch(
        self, 
        query: str, 
        documents: List[Document]
    ) -> List[Tuple[Document, RelevanceStatus]]:
        """
        Evaluate multiple documents.
        
        Args:
            query: The search query
            documents: Documents to evaluate
            
        Returns:
            List of (document, status) tuples
        """
        results = []
        for doc in documents:
            status = self.evaluate(query, doc)
            results.append((doc, status))
        return results
    
    def evaluate_batch_with_confidence(
        self, 
        query: str, 
        documents: List[Document]
    ) -> List[Tuple[Document, float]]:
        """
        Evaluate multiple documents with confidence scores.
        
        Args:
            query: The search query
            documents: Documents to evaluate
            
        Returns:
            List of (document, score) tuples
        """
        results = []
        for doc in documents:
            score = self.evaluate_with_confidence(query, doc)
            results.append((doc, score))
        return results
    
    def filter_relevant(
        self, 
        query: str, 
        documents: List[Document],
        use_confidence: bool = True
    ) -> List[Document]:
        """
        Filter documents to keep only relevant ones.
        
        Args:
            query: The search query
            documents: Documents to filter
            use_confidence: Use confidence scoring (more nuanced)
            
        Returns:
            List of relevant documents
        """
        if use_confidence:
            scored = self.evaluate_batch_with_confidence(query, documents)
            return [doc for doc, score in scored if score >= self.relevance_threshold]
        else:
            evaluated = self.evaluate_batch(query, documents)
            return [doc for doc, status in evaluated if status == "RELEVANT"]
    
    def get_graded_relevance(
        self, 
        query: str, 
        documents: List[Document]
    ) -> dict:
        """
        Get documents graded by relevance level.
        
        Returns:
            Dictionary with 'highly_relevant', 'somewhat_relevant', 'not_relevant' lists
        """
        scored = self.evaluate_batch_with_confidence(query, documents)
        
        return {
            "highly_relevant": [doc for doc, score in scored if score >= 0.7],
            "somewhat_relevant": [doc for doc, score in scored if 0.4 <= score < 0.7],
            "not_relevant": [doc for doc, score in scored if score < 0.4],
        }
