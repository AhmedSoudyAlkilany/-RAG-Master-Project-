"""
RAG Master Project - Web Search Fallback
=========================================
DuckDuckGo web search for when local documents are insufficient.
"""

from typing import List, Optional
from langchain_core.documents import Document

from config import settings


class WebSearcher:
    """
    Web search fallback using DuckDuckGo.
    
    When local document retrieval fails to find relevant content,
    this component searches the web to supplement the knowledge base.
    
    This is a key component of Corrective RAG that handles
    knowledge gaps gracefully.
    
    Example:
        >>> searcher = WebSearcher()
        >>> results = searcher.search("latest Python 3.13 features", max_results=3)
    """
    
    def __init__(self, max_results: int = None):
        """
        Initialize web searcher.
        
        Args:
            max_results: Maximum number of results to return
        """
        self.max_results = max_results or settings.WEB_SEARCH_MAX_RESULTS
        self._ddg = None
    
    def _get_ddg(self):
        """Lazy-load DuckDuckGo search."""
        if self._ddg is None:
            try:
                from duckduckgo_search import DDGS
                self._ddg = DDGS()
            except ImportError:
                raise ImportError(
                    "duckduckgo-search is required for web search. "
                    "Install with: pip install duckduckgo-search"
                )
        return self._ddg
    
    def search(
        self, 
        query: str, 
        max_results: int = None
    ) -> List[Document]:
        """
        Search the web and return results as documents.
        
        Args:
            query: Search query
            max_results: Override default max results
            
        Returns:
            List of Document objects from search results
        """
        max_results = max_results or self.max_results
        
        try:
            ddg = self._get_ddg()
            results = ddg.text(query, max_results=max_results)
            
            documents = []
            for result in results:
                doc = Document(
                    page_content=f"{result.get('title', '')}\n\n{result.get('body', '')}",
                    metadata={
                        "source": result.get("href", "web"),
                        "title": result.get("title", ""),
                        "type": "web_search",
                    }
                )
                documents.append(doc)
            
            return documents
            
        except Exception as e:
            print(f"Web search error: {e}")
            return []
    
    def search_with_context(
        self, 
        query: str,
        context: str = None,
        max_results: int = None
    ) -> List[Document]:
        """
        Search with additional context for better results.
        
        Args:
            query: Primary search query
            context: Additional context to refine search
            max_results: Maximum results
            
        Returns:
            List of Document objects
        """
        if context:
            enhanced_query = f"{query} {context}"
        else:
            enhanced_query = query
        
        return self.search(enhanced_query, max_results)
    
    def search_news(
        self, 
        query: str, 
        max_results: int = None
    ) -> List[Document]:
        """
        Search news articles.
        
        Args:
            query: Search query
            max_results: Maximum results
            
        Returns:
            List of Document objects from news
        """
        max_results = max_results or self.max_results
        
        try:
            ddg = self._get_ddg()
            results = ddg.news(query, max_results=max_results)
            
            documents = []
            for result in results:
                doc = Document(
                    page_content=f"{result.get('title', '')}\n\n{result.get('body', '')}",
                    metadata={
                        "source": result.get("url", "web"),
                        "title": result.get("title", ""),
                        "date": result.get("date", ""),
                        "type": "news",
                    }
                )
                documents.append(doc)
            
            return documents
            
        except Exception as e:
            print(f"News search error: {e}")
            return []


# Convenience instance
_web_searcher: Optional[WebSearcher] = None


def get_web_searcher() -> WebSearcher:
    """Get or create global web searcher."""
    global _web_searcher
    if _web_searcher is None:
        _web_searcher = WebSearcher()
    return _web_searcher


def web_search(query: str, max_results: int = 3) -> List[Document]:
    """Quick web search function."""
    return get_web_searcher().search(query, max_results)
