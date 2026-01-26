"""
RAG Master Project - Document Loader
====================================
Handles loading various document formats (PDF, DOCX, TXT, MD)
with automatic chunking and metadata extraction.
"""

import os
import pathlib
from typing import List, Optional, BinaryIO
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredMarkdownLoader,
)

from config import settings


class DocumentLoader:
    """
    Multi-format document loader with automatic chunking.
    
    Supports: PDF, DOCX, DOC, TXT, MD files.
    
    Example:
        >>> loader = DocumentLoader()
        >>> docs = loader.load_file("research.pdf")
        >>> print(f"Loaded {len(docs)} chunks")
    """
    
    # Supported file extensions and their loaders
    SUPPORTED_EXTENSIONS = {
        ".pdf": PyPDFLoader,
        ".txt": TextLoader,
        ".md": UnstructuredMarkdownLoader,
        ".docx": UnstructuredWordDocumentLoader,
        ".doc": UnstructuredWordDocumentLoader,
    }
    
    def __init__(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None
    ):
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
        )
    
    def load_file(self, file_path: str) -> List[Document]:
        """
        Load a document from a file path.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            List of Document chunks
            
        Raises:
            ValueError: If file extension is not supported
            FileNotFoundError: If file doesn't exist
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        ext = pathlib.Path(file_path).suffix.lower()
        
        if ext not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file extension: {ext}. "
                f"Supported: {list(self.SUPPORTED_EXTENSIONS.keys())}"
            )
        
        # Load document
        loader_class = self.SUPPORTED_EXTENSIONS[ext]
        loader = loader_class(file_path)
        documents = loader.load()
        
        # Add source metadata
        for doc in documents:
            doc.metadata["source"] = file_path
            doc.metadata["filename"] = os.path.basename(file_path)
        
        # Split into chunks
        chunks = self.text_splitter.split_documents(documents)
        
        # Add chunk indices
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = i
            chunk.metadata["total_chunks"] = len(chunks)
        
        return chunks
    
    def load_text(
        self, 
        text: str, 
        metadata: Optional[dict] = None
    ) -> List[Document]:
        """
        Load raw text directly.
        
        Args:
            text: Raw text content
            metadata: Optional metadata dictionary
            
        Returns:
            List of Document chunks
        """
        doc = Document(
            page_content=text,
            metadata=metadata or {}
        )
        
        chunks = self.text_splitter.split_documents([doc])
        
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = i
            chunk.metadata["total_chunks"] = len(chunks)
        
        return chunks
    
    def load_from_bytes(
        self,
        file_bytes: bytes,
        filename: str,
        temp_dir: str = "./data/temp"
    ) -> List[Document]:
        """
        Load a document from bytes (for file uploads).
        
        Args:
            file_bytes: Raw file bytes
            filename: Original filename (for extension detection)
            temp_dir: Temporary directory for file storage
            
        Returns:
            List of Document chunks
        """
        os.makedirs(temp_dir, exist_ok=True)
        temp_path = os.path.join(temp_dir, filename)
        
        try:
            with open(temp_path, "wb") as f:
                f.write(file_bytes)
            
            return self.load_file(temp_path)
        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    @classmethod
    def get_supported_extensions(cls) -> List[str]:
        """Get list of supported file extensions."""
        return list(cls.SUPPORTED_EXTENSIONS.keys())


# Convenience function
def load_documents(file_path: str) -> List[Document]:
    """Load and chunk a document file."""
    loader = DocumentLoader()
    return loader.load_file(file_path)
