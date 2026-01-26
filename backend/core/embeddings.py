"""
RAG Master Project - Ollama Embeddings Wrapper
===============================================
Provides embedding functionality using Ollama with:
- Dynamic model switching
- Batch embedding support
- Caching for efficiency
"""

from typing import Optional, List
from langchain_ollama import OllamaEmbeddings
from langchain_core.embeddings import Embeddings

from config import settings, AVAILABLE_EMBED_MODELS


class OllamaEmbeddingsWrapper:
    """
    Ollama Embeddings wrapper with caching and dynamic model switching.
    
    This class provides embeddings using local Ollama models, which is
    essential for the RAG pipeline's vector similarity search.
    
    Example:
        >>> embedder = OllamaEmbeddingsWrapper()
        >>> vectors = embedder.embed_documents(["Hello world", "RAG is great"])
        >>> print(f"Embedding dimension: {len(vectors[0])}")
    """
    
    def __init__(
        self,
        model: Optional[str] = None,
        base_url: Optional[str] = None
    ):
        self.model = model or settings.OLLAMA_EMBED_MODEL
        self.base_url = base_url or settings.OLLAMA_BASE_URL
        self._embeddings: Optional[OllamaEmbeddings] = None
        self._cache: dict[str, List[float]] = {}
        self._initialize_embeddings()
    
    def _initialize_embeddings(self) -> None:
        """Initialize the OllamaEmbeddings instance."""
        self._embeddings = OllamaEmbeddings(
            model=self.model,
            base_url=self.base_url,
        )
    
    @property
    def embeddings(self) -> OllamaEmbeddings:
        """Get the underlying OllamaEmbeddings instance."""
        if self._embeddings is None:
            self._initialize_embeddings()
        return self._embeddings
    
    def switch_model(self, model_name: str) -> None:
        """
        Switch to a different embedding model.
        
        Note: Switching models clears the cache as different models
        produce different dimensional embeddings.
        
        Args:
            model_name: Name of the embedding model (e.g., 'nomic-embed-text')
        """
        if model_name not in AVAILABLE_EMBED_MODELS:
            print(f"Warning: {model_name} not in known models list, attempting anyway...")
        
        self.model = model_name
        self._cache.clear()  # Clear cache when switching models
        self._initialize_embeddings()
    
    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query text.
        
        Args:
            text: The query text to embed
            
        Returns:
            List of floats representing the embedding vector
        """
        # Check cache first
        cache_key = f"query:{text}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        embedding = self.embeddings.embed_query(text)
        self._cache[cache_key] = embedding
        return embedding
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed multiple documents.
        
        Args:
            texts: List of document texts to embed
            
        Returns:
            List of embedding vectors
        """
        # Check which texts need embedding
        results: List[List[float]] = []
        texts_to_embed: List[str] = []
        indices_to_embed: List[int] = []
        
        for i, text in enumerate(texts):
            cache_key = f"doc:{text}"
            if cache_key in self._cache:
                results.append(self._cache[cache_key])
            else:
                texts_to_embed.append(text)
                indices_to_embed.append(i)
                results.append([])  # Placeholder
        
        # Embed uncached texts
        if texts_to_embed:
            new_embeddings = self.embeddings.embed_documents(texts_to_embed)
            for idx, embedding in zip(indices_to_embed, new_embeddings):
                cache_key = f"doc:{texts[idx]}"
                self._cache[cache_key] = embedding
                results[idx] = embedding
        
        return results
    
    def get_langchain_embeddings(self) -> Embeddings:
        """Get the underlying LangChain embeddings for use in vector stores."""
        return self.embeddings
    
    @property
    def current_model(self) -> str:
        """Get the currently active embedding model name."""
        return self.model
    
    @staticmethod
    def list_available_models() -> list[str]:
        """List all configured available embedding models."""
        return AVAILABLE_EMBED_MODELS.copy()
    
    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        self._cache.clear()
    
    @property
    def cache_size(self) -> int:
        """Get the number of cached embeddings."""
        return len(self._cache)


# Global embeddings instance (lazy initialization)
_embeddings_instance: Optional[OllamaEmbeddingsWrapper] = None


def get_embeddings() -> OllamaEmbeddingsWrapper:
    """Get or create the global embeddings instance."""
    global _embeddings_instance
    if _embeddings_instance is None:
        _embeddings_instance = OllamaEmbeddingsWrapper()
    return _embeddings_instance


def get_langchain_embeddings() -> OllamaEmbeddings:
    """Get the underlying OllamaEmbeddings instance."""
    return get_embeddings().embeddings
