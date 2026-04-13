"""
Terminal UI using rich.
"""

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

console = Console()


def print_welcome(model: str) -> None:
    """
    Display a styled welcome panel with agent info.

    Args:
        model: The Gemini model in use
    """
    welcome_text = Text()
    welcome_text.append("Model: ", style="bold")
    welcome_text.append(f"{model}\n\n", style="cyan")

    welcome_text.append("Available Tools:\n", style="bold")
    welcome_text.append("  - web_search    ", style="yellow")
    welcome_text.append("Search the web for current info\n", style="dim")
    welcome_text.append("  - calculator    ", style="yellow")
    welcome_text.append("Evaluate math expressions\n", style="dim")
    welcome_text.append("  - get_weather   ", style="yellow")
    welcome_text.append("Get weather for any city\n", style="dim")
    welcome_text.append("  - query_database", style="yellow")
    welcome_text.append("Query company database\n\n", style="dim")

    welcome_text.append("Commands: ", style="bold")
    welcome_text.append("/tools  /history  /reset  /help  /exit", style="yellow")

    panel = Panel(
        welcome_text,
        title="[bold magenta]LangChain Agent[/bold magenta]",
        border_style="magenta",
        padding=(1, 2)
    )

    console.print(panel)


def print_tool_use(tool_name: str, tool_input: str) -> None:
    """
    Display when agent uses a tool.

    Args:
        tool_name: Name of the tool being used
        tool_input: Input being passed to the tool
    """
    input_preview = tool_input[:50] + "..." if len(tool_input) > 50 else tool_input
    console.print(f"[yellow]Using tool: {tool_name}[/yellow]")
    console.print(f"[dim]Input: {input_preview}[/dim]")


def print_response(response: str) -> None:
    """
    Print the agent's final response.

    Args:
        response: The agent's response text
    """
    console.print("[bold green]Agent:[/bold green]", response)


def print_steps(intermediate_steps: list) -> None:
    """
    Show intermediate ReAct steps (tool calls and results).

    Args:
        intermediate_steps: List of (AgentAction, result) tuples
    """
    if not intermediate_steps:
        console.print("[dim]No tool calls made.[/dim]")
        return

    console.print("\n[bold yellow]DEBUG: Intermediate Steps[/bold yellow]")

    for i, (action, result) in enumerate(intermediate_steps, 1):
        step_text = Text()
        step_text.append(f"Tool: {action.tool}\n", style="cyan")
        step_text.append(f"Input: {action.tool_input}\n\n", style="dim")

        result_preview = str(result)[:300]
        if len(str(result)) > 300:
            result_preview += "..."
        step_text.append(f"Result:\n{result_preview}", style="green")

        panel = Panel(
            step_text,
            title=f"[bold]Step {i}[/bold]",
            border_style="yellow",
            padding=(0, 1)
        )
        console.print(panel)


def print_tools() -> None:
    """
    Display a table of all available tools.
    """
    table = Table(title="Available Tools", show_header=True)
    table.add_column("Tool", style="cyan")
    table.add_column("Description", style="white")

    table.add_row(
        "web_search",
        "Search the web for current information, news, or facts"
    )
    table.add_row(
        "calculator",
        "Evaluate math expressions (+, -, *, /, **)"
    )
    table.add_row(
        "get_weather",
        "Get current weather for any city in the world"
    )
    table.add_row(
        "query_database",
        "Query company database (employees, products, sales)"
    )

    console.print(table)


def print_history(messages: list) -> None:
    """
    Display conversation history.

    Args:
        messages: List of message objects from memory
    """
    if not messages:
        console.print("[yellow]No conversation history yet.[/yellow]")
        return

    console.print("\n[bold]Conversation History[/bold]")
    console.print("-" * 40)

    for msg in messages:
        role = msg.type
        content = msg.content[:100] + "..." if len(msg.content) > 100 else msg.content

        if role == "human":
            console.print(f"[cyan]You:[/cyan] {content}")
        else:
            console.print(f"[green]Agent:[/green] {content}")

    console.print()


def print_error(msg: str) -> None:
    """Print an error message in red bold."""
    console.print(f"[bold red]{msg}[/bold red]")


def print_info(msg: str) -> None:
    """Print an informational message in yellow."""
    console.print(f"[yellow]{msg}[/yellow]")
