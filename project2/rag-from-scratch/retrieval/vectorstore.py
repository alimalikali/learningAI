"""
Vector store wrapper for RAG from Scratch.

Wrapper around the Qdrant client.
Handles collection creation, upserting vectors, and similarity search.
Qdrant stores vectors + metadata (payload) together.
"""

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from rich.console import Console

from ingestion.chunker import Chunk

console = Console()


class VectorStore:
    """
    Wrapper around Qdrant Cloud for vector storage and retrieval.

    Handles collection management, vector upsertion, and similarity search.
    Uses cosine distance which is correct for Gemini embeddings (normalized).
    """

    def __init__(self, url: str, api_key: str, collection_name: str) -> None:
        """
        Initialize the Qdrant client.

        Args:
            url: Qdrant Cloud cluster URL
            api_key: Qdrant Cloud API key
            collection_name: Name of the collection to use
        """
        self._client = QdrantClient(url=url, api_key=api_key)
        self._collection_name = collection_name

    def create_collection(self, vector_size: int) -> None:
        """
        Create the collection if it doesn't exist.

        Args:
            vector_size: Dimension of the embedding vectors (768 for text-embedding-004)
        """
        if self._client.collection_exists(self._collection_name):
            console.print(
                f"[yellow]Collection '{self._collection_name}' already exists[/yellow]"
            )
            # Delete existing collection to recreate with fresh data
            self._client.delete_collection(self._collection_name)
            console.print(
                f"[yellow]Deleted existing collection for fresh ingestion[/yellow]"
            )

        self._client.create_collection(
            collection_name=self._collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE
            )
        )
        console.print(
            f"[green]Collection '{self._collection_name}' created[/green]"
        )

    def upsert(
        self,
        chunks: list[Chunk],
        embeddings: list[list[float]]
    ) -> None:
        """
        Insert or update vectors in the collection.

        Args:
            chunks: List of Chunk objects with metadata
            embeddings: List of embedding vectors corresponding to chunks
        """
        points = []

        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            point = PointStruct(
                id=i,
                vector=embedding,
                payload={
                    "source": chunk.source,
                    "content": chunk.content,
                    "chunk_index": chunk.chunk_index
                }
            )
            points.append(point)

        self._client.upsert(
            collection_name=self._collection_name,
            points=points
        )

        console.print(f"[green]Upserted {len(points)} vectors[/green]")

    def search(
        self,
        query_vector: list[float],
        top_k: int
    ) -> list[dict]:
        """
        Search for the most similar vectors.

        Args:
            query_vector: The query embedding vector
            top_k: Number of results to return

        Returns:
            list[dict]: List of results with content, source, and score
        """
        results = self._client.query_points(
            collection_name=self._collection_name,
            query=query_vector,
            limit=top_k,
            with_payload=True
        )

        return [
            {
                "content": r.payload["content"],
                "source": r.payload["source"],
                "score": r.score
            }
            for r in results.points
        ]

    def count(self) -> int:
        """
        Get the number of vectors in the collection.

        Returns:
            int: Number of vectors, or 0 if collection doesn't exist
        """
        try:
            if not self._client.collection_exists(self._collection_name):
                return 0
            return self._client.count(self._collection_name).count
        except Exception:
            return 0
