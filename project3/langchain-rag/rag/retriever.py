"""
Loads an existing Qdrant collection and returns a LangChain retriever.

COMPARE TO PROJECT 2:
- Project 2: retrieval/vectorstore.py (100 lines) + retrieval/retriever.py (50 lines)
- Project 3: 15 lines total
- Same behaviour: embed query → cosine search → return top_k chunks
"""

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.vectorstores import VectorStoreRetriever
from qdrant_client import QdrantClient

from config import Config


def load_retriever(config: Config) -> VectorStoreRetriever:
    """
    Load an existing Qdrant collection and return a retriever.

    Args:
        config: Configuration object

    Returns:
        VectorStoreRetriever: A retriever that searches the vector store
    """
    # Create embeddings model
    embeddings = GoogleGenerativeAIEmbeddings(
        model=config.embedding_model,
        google_api_key=config.gemini_api_key
    )

    # Connect to existing Qdrant collection
    client = QdrantClient(
        url=config.qdrant_url,
        api_key=config.qdrant_api_key
    )

    vectorstore = QdrantVectorStore(
        client=client,
        collection_name=config.collection_name,
        embedding=embeddings
    )

    # Return as retriever with top_k setting
    return vectorstore.as_retriever(search_kwargs={"k": config.top_k})
