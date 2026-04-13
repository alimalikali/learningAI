"""
Display module for RAG from Scratch.

Terminal UI using rich library.
Handles welcome panels, streaming responses, and debug views.
"""

from typing import Generator

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

console = Console()


def print_welcome(
    chat_model: str,
    embedding_model: str,
    collection_name: str
) -> None:
    """
    Display a styled welcome panel.

    Args:
        chat_model: The Gemini chat model in use
        embedding_model: The embedding model in use
        collection_name: The Qdrant collection name
    """
    welcome_text = Text()
    welcome_text.append("Chat Model: ", style="bold")
    welcome_text.append(f"{chat_model}\n", style="cyan")
    welcome_text.append("Embedding Model: ", style="bold")
    welcome_text.append(f"{embedding_model}\n", style="cyan")
    welcome_text.append("Collection: ", style="bold")
    welcome_text.append(f"{collection_name}\n\n", style="cyan")
    welcome_text.append("Commands: ", style="bold")
    welcome_text.append("/help  /exit", style="yellow")

    panel = Panel(
        welcome_text,
        title="[bold magenta]RAG from Scratch[/bold magenta]",
        border_style="magenta",
        padding=(1, 2)
    )

    console.print(panel)


def print_ingestion_summary(
    num_files: int,
    num_chunks: int,
    num_vectors: int
) -> None:
    """
    Display ingestion statistics in a table.

    Args:
        num_files: Number of files processed
        num_chunks: Number of chunks created
        num_vectors: Number of vectors stored
    """
    table = Table(title="Ingestion Summary", show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Files Processed", str(num_files))
    table.add_row("Chunks Created", str(num_chunks))
    table.add_row("Vectors Stored", str(num_vectors))

    console.print()
    console.print(table)
    console.print()


def stream_response(
    generator: Generator[str, None, None],
    sources: list[dict]
) -> None:
    """
    Stream and display the assistant's response with sources.

    Args:
        generator: Generator yielding response chunks
        sources: List of source chunks used for the response
    """
    console.print("[bold green]Assistant:[/bold green] ", end="")

    for chunk in generator:
        print(chunk, end="", flush=True)

    print()  # Newline after stream

    # Print sources panel
    unique_sources = {}
    for source in sources:
        name = source["source"]
        score = source["score"]
        if name not in unique_sources or score > unique_sources[name]:
            unique_sources[name] = score

    if unique_sources:
        source_text = Text()
        source_text.append("Sources:\n", style="bold")
        for name, score in unique_sources.items():
            source_text.append(f"  - {name} ", style="cyan")
            source_text.append(f"(relevance: {score:.3f})\n", style="dim")

        console.print(Panel(source_text, border_style="dim", padding=(0, 1)))


def print_retrieved_chunks(chunks: list[dict]) -> None:
    """
    Debug view: show each retrieved chunk with metadata.

    Args:
        chunks: List of chunks with content, source, and score
    """
    console.print("\n[bold yellow]DEBUG: Retrieved Chunks[/bold yellow]")

    for i, chunk in enumerate(chunks, 1):
        content_preview = chunk["content"][:200]
        if len(chunk["content"]) > 200:
            content_preview += "..."

        panel_content = Text()
        panel_content.append(f"Source: {chunk['source']}\n", style="cyan")
        panel_content.append(f"Score: {chunk['score']:.4f}\n\n", style="green")
        panel_content.append(content_preview)

        panel = Panel(
            panel_content,
            title=f"[bold]Chunk {i}[/bold]",
            border_style="yellow",
            padding=(0, 1)
        )
        console.print(panel)

    console.print()


def print_error(msg: str) -> None:
    """
    Print an error message in red bold.

    Args:
        msg: The error message
    """
    console.print(f"[bold red]{msg}[/bold red]")


def print_info(msg: str) -> None:
    """
    Print an informational message in yellow.

    Args:
        msg: The info message
    """
    console.print(f"[yellow]{msg}[/yellow]")
