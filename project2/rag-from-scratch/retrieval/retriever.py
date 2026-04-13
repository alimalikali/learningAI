"""
Retriever for RAG from Scratch.

Combines embedding the query + searching the vector store.
This is the "R" in RAG - given a question, find the most relevant chunks.
"""

from config import Config
from ingestion.embedder import embed_query
from retrieval.vectorstore import VectorStore


class Retriever:
    """
    Retrieves relevant document chunks for a given query.

    Combines query embedding with vector search to find
    the most semantically similar chunks.
    """

    def __init__(self, config: Config, vectorstore: VectorStore) -> None:
        """
        Initialize the retriever.

        Args:
            config: Configuration with API keys and settings
            vectorstore: VectorStore instance for searching
        """
        self._config = config
        self._vectorstore = vectorstore

    def retrieve(self, query: str) -> list[dict]:
        """
        Retrieve the most relevant chunks for a query.

        Args:
            query: The user's question

        Returns:
            list[dict]: List of chunks with content, source, and score
        """
        # Embed the query using retrieval_query task type
        query_embedding = embed_query(
            text=query,
            api_key=self._config.gemini_api_key,
            model=self._config.embedding_model
        )

        # Search for similar chunks
        results = self._vectorstore.search(
            query_vector=query_embedding,
            top_k=self._config.top_k
        )

        return results
