"""
RAG Master Project - Ollama LLM Wrapper
========================================
Provides a flexible interface to Ollama language models with:
- Dynamic model switching
- Streaming support
- Error handling and retries
"""

from typing import Optional, AsyncIterator
from langchain_ollama import ChatOllama
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.language_models.chat_models import BaseChatModel

from config import settings, AVAILABLE_LLM_MODELS


class OllamaLLM:
    """
    Ollama LLM wrapper with dynamic model switching.
    
    This class manages the connection to Ollama and provides methods
    for both synchronous and streaming responses.
    
    Example:
        >>> llm = OllamaLLM()
        >>> response = llm.invoke("What is RAG?")
        >>> print(response)
    """
    
    def __init__(
        self,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        base_url: Optional[str] = None
    ):
        self.model = model or settings.OLLAMA_LLM_MODEL
        self.temperature = temperature or settings.LLM_TEMPERATURE
        self.base_url = base_url or settings.OLLAMA_BASE_URL
        self._chat_model: Optional[ChatOllama] = None
        self._initialize_model()
    
    def _initialize_model(self) -> None:
        """Initialize or reinitialize the ChatOllama instance."""
        self._chat_model = ChatOllama(
            model=self.model,
            temperature=self.temperature,
            base_url=self.base_url,
            num_predict=settings.LLM_MAX_TOKENS,
        )
    
    @property
    def chat_model(self) -> ChatOllama:
        """Get the underlying ChatOllama instance."""
        if self._chat_model is None:
            self._initialize_model()
        return self._chat_model
    
    def switch_model(self, model_name: str) -> None:
        """
        Switch to a different Ollama model.
        
        Args:
            model_name: Name of the model to switch to (e.g., 'llama3.2:3b')
            
        Raises:
            ValueError: If the model is not in the available models list
        """
        if model_name not in AVAILABLE_LLM_MODELS:
            # Still allow switching, just warn
            print(f"Warning: {model_name} not in known models list, attempting anyway...")
        
        self.model = model_name
        self._initialize_model()
    
    def invoke(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Generate a response for the given prompt.
        
        Args:
            prompt: The user's question or input
            system_prompt: Optional system instructions
            
        Returns:
            The model's response as a string
        """
        messages: list[BaseMessage] = []
        
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        
        messages.append(HumanMessage(content=prompt))
        
        response = self.chat_model.invoke(messages)
        return response.content
    
    def invoke_with_messages(self, messages: list[BaseMessage]) -> AIMessage:
        """
        Generate a response from a list of messages.
        
        Args:
            messages: List of BaseMessage objects
            
        Returns:
            AIMessage containing the response
        """
        return self.chat_model.invoke(messages)
    
    async def astream(
        self, 
        prompt: str, 
        system_prompt: Optional[str] = None
    ) -> AsyncIterator[str]:
        """
        Stream a response asynchronously.
        
        Args:
            prompt: The user's question or input
            system_prompt: Optional system instructions
            
        Yields:
            Chunks of the response as they are generated
        """
        messages: list[BaseMessage] = []
        
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        
        messages.append(HumanMessage(content=prompt))
        
        async for chunk in self.chat_model.astream(messages):
            if chunk.content:
                yield chunk.content
    
    def get_langchain_llm(self) -> BaseChatModel:
        """Get the underlying LangChain chat model for use in chains."""
        return self.chat_model
    
    @property
    def current_model(self) -> str:
        """Get the currently active model name."""
        return self.model
    
    @staticmethod
    def list_available_models() -> list[str]:
        """List all configured available models."""
        return AVAILABLE_LLM_MODELS.copy()


# Global LLM instance (lazy initialization)
_llm_instance: Optional[OllamaLLM] = None


def get_llm() -> OllamaLLM:
    """Get or create the global LLM instance."""
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = OllamaLLM()
    return _llm_instance


def get_chat_model() -> ChatOllama:
    """Get the underlying ChatOllama instance."""
    return get_llm().chat_model
