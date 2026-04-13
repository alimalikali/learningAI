"""
Ingestion pipeline using LangChain abstractions.

COMPARE TO PROJECT 2:
- Project 2: We wrote loader.py (60 lines) + chunker.py (50 lines) + embedder.py (70 lines) manually
- Project 3: LangChain collapses all of that into ~30 lines
- The logic is IDENTICAL — LangChain just packages it for you
"""

from langchain_community.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from rich.console import Console
from rich.table import Table

from config import Config

console = Console()


def load_documents(dir_path: str) -> list:
    """
    Load all PDF and TXT documents from a directory.

    Uses DirectoryLoader with appropriate loaders for each file type.

    Args:
        dir_path: Path to directory containing documents

    Returns:
        list: List of LangChain Document objects
    """
    # Load PDFs
    pdf_loader = DirectoryLoader(
        dir_path,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True
    )
    pdf_docs = pdf_loader.load()

    # Load TXT files
    txt_loader = DirectoryLoader(
        dir_path,
        glob="**/*.txt",
        loader_cls=TextLoader,
        show_progress=True
    )
    txt_docs = txt_loader.load()

    documents = pdf_docs + txt_docs

    console.print(f"[green]Loaded {len(documents)} documents[/green]")

    return documents


def split_documents(
    documents: list,
    chunk_size: int,
    chunk_overlap: int
) -> list:
    """
    Split documents into smaller chunks using RecursiveCharacterTextSplitter.

    RecursiveCharacterTextSplitter is smarter than fixed-size chunking:
    it tries to split on paragraphs first, then sentences, then words, then characters.
    Split priority: ["\\n\\n", "\\n", " ", ""]

    Args:
        documents: List of LangChain Document objects
        chunk_size: Maximum characters per chunk
        chunk_overlap: Overlapping characters between chunks

    Returns:
        list: List of chunked Document objects
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunks = splitter.split_documents(documents)

    console.print(f"[bold]{len(documents)} docs → {len(chunks)} chunks[/bold]")

    return chunks


def ingest(config: Config, dir_path: str) -> QdrantVectorStore:
    """
    Run the full ingestion pipeline.

    Steps:
    1. Load documents from directory
    2. Split into chunks
    3. Create embeddings model
    4. Store in Qdrant (embedding + upserting in one call)

    Args:
        config: Configuration object
        dir_path: Path to directory containing documents

    Returns:
        QdrantVectorStore: The vector store instance
    """
    console.print("[yellow]Step 1: Loading documents...[/yellow]")
    documents = load_documents(dir_path)

    if not documents:
        raise ValueError(f"No PDF or TXT files found in {dir_path}")

    console.print("\n[yellow]Step 2: Splitting documents...[/yellow]")
    chunks = split_documents(
        documents,
        config.chunk_size,
        config.chunk_overlap
    )

    console.print("\n[yellow]Step 3: Creating embeddings and storing in Qdrant...[/yellow]")

    # Create embeddings model
    embeddings = GoogleGenerativeAIEmbeddings(
        model=config.embedding_model,
        google_api_key=config.gemini_api_key
    )

    # QdrantVectorStore.from_documents handles embedding + upserting in one call
    # This replaces ~150 lines from Project 2's embedder.py + vectorstore.py
    vectorstore = QdrantVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        url=config.qdrant_url,
        api_key=config.qdrant_api_key,
        collection_name=config.collection_name,
        force_recreate=True  # Recreate collection for fresh ingestion
    )

    # Print summary
    table = Table(title="Ingestion Summary", show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Documents Loaded", str(len(documents)))
    table.add_row("Chunks Created", str(len(chunks)))
    table.add_row("Collection", config.collection_name)

    console.print()
    console.print(table)
    console.print()

    return vectorstore
