"""
Configuration module for RAG from Scratch.

Handles loading and validating configuration from environment variables.
Uses a frozen dataclass to ensure configuration immutability after initialization.
"""

import os
from dataclasses import dataclass
from dotenv import load_dotenv


@dataclass(frozen=True)
class Config:
    """
    Immutable configuration for the RAG system.

    Attributes:
        gemini_api_key: Google Gemini API key (required)
        qdrant_url: Qdrant Cloud cluster URL (required)
        qdrant_api_key: Qdrant Cloud API key (required)
        collection_name: Name of the Qdrant collection
        chat_model: Gemini model for chat generation
        embedding_model: Gemini model for embeddings
        chunk_size: Number of characters per chunk
        chunk_overlap: Number of overlapping characters between chunks
        top_k: Number of chunks to retrieve per query
    """
    gemini_api_key: str
    qdrant_url: str
    qdrant_api_key: str
    collection_name: str
    chat_model: str
    embedding_model: str
    chunk_size: int
    chunk_overlap: int
    top_k: int


def load_config() -> Config:
    """
    Load configuration from environment variables.

    Returns:
        Config: Validated configuration object

    Raises:
        ValueError: If required fields are missing or invalid
    """
    load_dotenv()

    gemini_api_key = os.getenv("GEMINI_API_KEY", "")
    qdrant_url = os.getenv("QDRANT_URL", "")
    qdrant_api_key = os.getenv("QDRANT_API_KEY", "")
    collection_name = os.getenv("COLLECTION_NAME", "project2")
    chat_model = os.getenv("CHAT_MODEL", "gemini-2.5-flash")
    embedding_model = os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-001")
    chunk_size = int(os.getenv("CHUNK_SIZE", "500"))
    chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "50"))
    top_k = int(os.getenv("TOP_K", "4"))

    # Validate required fields
    if not gemini_api_key or gemini_api_key == "your_gemini_api_key_here":
        raise ValueError(
            "GEMINI_API_KEY is not set or is the placeholder value.\n"
            "Get a free key at: https://aistudio.google.com/app/apikey"
        )

    if not qdrant_url or qdrant_url == "https://your-cluster-url.qdrant.io":
        raise ValueError(
            "QDRANT_URL is not set or is the placeholder value.\n"
            "Create a free cluster at: https://cloud.qdrant.io"
        )

    if not qdrant_api_key or qdrant_api_key == "your_qdrant_api_key_here":
        raise ValueError(
            "QDRANT_API_KEY is not set or is the placeholder value.\n"
            "Get your API key from your Qdrant Cloud dashboard."
        )

    return Config(
        gemini_api_key=gemini_api_key,
        qdrant_url=qdrant_url,
        qdrant_api_key=qdrant_api_key,
        collection_name=collection_name,
        chat_model=chat_model,
        embedding_model=embedding_model,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        top_k=top_k
    )
