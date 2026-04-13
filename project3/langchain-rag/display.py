"""
Terminal UI using rich.

Handles welcome panels, streaming responses, and source display.
"""

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

console = Console()


def print_welcome(model: str, collection: str) -> None:
    """
    Display a styled welcome panel.

    Args:
        model: The Gemini chat model in use
        collection: The Qdrant collection name
    """
    welcome_text = Text()
    welcome_text.append("Chat Model: ", style="bold")
    welcome_text.append(f"{model}\n", style="cyan")
    welcome_text.append("Collection: ", style="bold")
    welcome_text.append(f"{collection}\n\n", style="cyan")
    welcome_text.append("Commands: ", style="bold")
    welcome_text.append("/help  /exit", style="yellow")

    panel = Panel(
        welcome_text,
        title="[bold magenta]LangChain RAG[/bold magenta]",
        border_style="magenta",
        padding=(1, 2)
    )

    console.print(panel)


def stream_response(chain, query: str) -> str:
    """
    Stream and display the assistant's response.

    Uses chain.stream() to get chunks as they arrive.

    Args:
        chain: The LCEL chain
        query: The user's question

    Returns:
        str: The full assembled response
    """
    console.print("[bold green]Assistant:[/bold green] ", end="")

    full_response = ""

    for chunk in chain.stream(query):
        print(chunk, end="", flush=True)
        full_response += chunk

    print()  # Newline after stream

    return full_response


def print_sources(docs: list) -> None:
    """
    Print a panel showing unique source documents.

    Args:
        docs: List of LangChain Document objects
    """
    unique_sources = set()
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        unique_sources.add(source)

    if unique_sources:
        source_text = Text()
        source_text.append("Sources:\n", style="bold")
        for source in unique_sources:
            source_text.append(f"  - {source}\n", style="cyan")

        console.print(Panel(source_text, border_style="dim", padding=(0, 1)))


def print_debug_chunks(docs: list) -> None:
    """
    Debug view: show each retrieved document with metadata.

    Args:
        docs: List of LangChain Document objects
    """
    console.print("\n[bold yellow]DEBUG: Retrieved Chunks[/bold yellow]")

    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "unknown")
        content_preview = doc.page_content[:200]
        if len(doc.page_content) > 200:
            content_preview += "..."

        panel_content = Text()
        panel_content.append(f"Source: {source}\n\n", style="cyan")
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
