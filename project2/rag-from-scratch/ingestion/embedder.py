"""
Text embedder for RAG from Scratch.

Converts text chunks into embedding vectors using Gemini's embedding model.
Embeddings are the numerical representation of meaning - similar text = similar vectors.
This is the core of how semantic search works.
"""

import time

from google import genai
from google.genai import types
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn


def embed_texts(
    texts: list[str],
    api_key: str,
    model: str
) -> list[list[float]]:
    """
    Embed multiple texts for indexing (retrieval_document task type).

    Processes in batches of 20 to avoid rate limits on Gemini free tier.
    Adds sleep between batches to respect rate limits.

    Args:
        texts: List of text strings to embed
        api_key: Gemini API key
        model: Embedding model name (e.g., "gemini-embedding-001")

    Returns:
        list[list[float]]: List of embedding vectors (each is list[float])
    """
    client = genai.Client(api_key=api_key)

    embeddings = []
    batch_size = 20

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
    ) as progress:
        task = progress.add_task("Embedding texts...", total=len(texts))

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

            # Embed batch with task_type for indexing
            result = client.models.embed_content(
                model=model,
                contents=batch,
                config=types.EmbedContentConfig(
                    task_type="RETRIEVAL_DOCUMENT"
                )
            )

            # Extract embeddings from result
            for embedding in result.embeddings:
                embeddings.append(embedding.values)

            progress.update(task, advance=len(batch))

            # Sleep between batches to respect rate limits
            if i + batch_size < len(texts):
                time.sleep(1)

    return embeddings


def embed_query(
    text: str,
    api_key: str,
    model: str
) -> list[float]:
    """
    Embed a single query text for retrieval (retrieval_query task type).

    IMPORTANT: Uses task_type="RETRIEVAL_QUERY" which is different from
    "RETRIEVAL_DOCUMENT" used for indexing. Mixing these kills retrieval quality.

    Args:
        text: The query text to embed
        api_key: Gemini API key
        model: Embedding model name

    Returns:
        list[float]: Single embedding vector
    """
    client = genai.Client(api_key=api_key)

    result = client.models.embed_content(
        model=model,
        contents=text,
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_QUERY"
        )
    )

    return result.embeddings[0].values
