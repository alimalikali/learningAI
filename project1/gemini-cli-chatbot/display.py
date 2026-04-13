"""
Display module for Gemini CLI Chatbot.

Provides rich terminal UI functions using the 'rich' library.
Handles all visual output including welcome panels, streaming responses,
history display, and styled messages.
"""

from typing import Generator

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from history import ConversationHistory


# Create a global console instance for consistent output
console = Console()


def print_welcome(model: str, system_prompt: str) -> None:
    """
    Display a styled welcome panel with chatbot information.

    Args:
        model: The Gemini model name in use
        system_prompt: The system instruction (truncated to 80 chars)
    """
    # Truncate system prompt if too long
    prompt_display = system_prompt[:80] + "..." if len(system_prompt) > 80 else system_prompt

    welcome_text = Text()
    welcome_text.append("Model: ", style="bold")
    welcome_text.append(f"{model}\n", style="cyan")
    welcome_text.append("System: ", style="bold")
    welcome_text.append(f"{prompt_display}\n\n", style="dim")
    welcome_text.append("Commands: ", style="bold")
    welcome_text.append("/reset  /history  /tokens  /help  /exit", style="yellow")

    panel = Panel(
        welcome_text,
        title="[bold magenta]Gemini CLI Chatbot[/bold magenta]",
        border_style="magenta",
        padding=(1, 2)
    )

    console.print(panel)


def print_user_label() -> None:
    """Print the 'You:' label with cyan bold styling."""
    console.print("[bold cyan]You:[/bold cyan] ", end="")


def stream_assistant_response(generator: Generator[str, None, None]) -> str:
    """
    Stream and display the assistant's response.

    Prints each chunk as it arrives for real-time output effect.

    Args:
        generator: Generator yielding response chunks

    Returns:
        str: The full assembled response
    """
    console.print("[bold green]Gemini:[/bold green] ", end="")

    full_response = ""

    for chunk in generator:
        print(chunk, end="", flush=True)
        full_response += chunk

    # Print newline after stream completes
    print()

    return full_response


def print_history(history: ConversationHistory) -> None:
    """
    Display the full conversation history in styled panels.

    Args:
        history: ConversationHistory instance containing messages
    """
    messages = history.get_messages()

    if not messages:
        print_info("No conversation history yet.")
        return

    console.print("\n[bold]Conversation History[/bold]")
    console.print("─" * 40)

    for msg in messages:
        role = msg["role"]
        text = msg["parts"][0]["text"]

        if role == "user":
            panel = Panel(
                text,
                title="[bold cyan]You[/bold cyan]",
                border_style="cyan",
                padding=(0, 1)
            )
        else:
            panel = Panel(
                text,
                title="[bold green]Gemini[/bold green]",
                border_style="green",
                padding=(0, 1)
            )

        console.print(panel)

    console.print()


def print_error(msg: str) -> None:
    """
    Print an error message in red bold.

    Args:
        msg: The error message to display
    """
    console.print(f"[bold red]{msg}[/bold red]")


def print_info(msg: str) -> None:
    """
    Print an informational message in yellow.

    Args:
        msg: The info message to display
    """
    console.print(f"[yellow]{msg}[/yellow]")
