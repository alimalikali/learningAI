"""
Text chunker for RAG from Scratch.

Splits long documents into smaller overlapping chunks.
Chunking strategy matters enormously for RAG quality.
This uses character-based sliding window - simple but effective for learning.
"""

from dataclasses import dataclass

from rich.console import Console

from .loader import Document

console = Console()


@dataclass
class Chunk:
    """
    Represents a chunk of text from a document.

    Attributes:
        source: The source document filename
        content: The chunk text content
        chunk_index: The index of this chunk within its source document
    """
    source: str
    content: str
    chunk_index: int


def chunk_document(
    doc: Document,
    chunk_size: int,
    chunk_overlap: int
) -> list[Chunk]:
    """
    Split a document into overlapping chunks using sliding window.

    Args:
        doc: The document to chunk
        chunk_size: Number of characters per chunk
        chunk_overlap: Number of overlapping characters between chunks

    Returns:
        list[Chunk]: List of Chunk objects
    """
    chunks = []
    text = doc.content
    step = chunk_size - chunk_overlap
    chunk_index = 0

    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk_content = text[start:end].strip()

        # Skip chunks shorter than 50 characters (noise)
        if len(chunk_content) >= 50:
            chunks.append(Chunk(
                source=doc.source,
                content=chunk_content,
                chunk_index=chunk_index
            ))
            chunk_index += 1

        start += step

    return chunks


def chunk_documents(
    docs: list[Document],
    chunk_size: int,
    chunk_overlap: int
) -> list[Chunk]:
    """
    Split multiple documents into overlapping chunks.

    Args:
        docs: List of documents to chunk
        chunk_size: Number of characters per chunk
        chunk_overlap: Number of overlapping characters between chunks

    Returns:
        list[Chunk]: List of all Chunk objects from all documents
    """
    all_chunks = []

    for doc in docs:
        doc_chunks = chunk_document(doc, chunk_size, chunk_overlap)
        all_chunks.extend(doc_chunks)

    console.print(
        f"[bold]{len(docs)} documents → {len(all_chunks)} chunks[/bold]"
    )

    return all_chunks
