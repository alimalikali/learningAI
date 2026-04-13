"""
Main entry point for Gemini CLI Chatbot.

This module wires up all components and runs the main chat loop.
Handles user input, commands, and graceful shutdown.
"""

import sys

from config import load_config
from history import ConversationHistory
from chat import GeminiChat
import display


def print_help() -> None:
    """Display available commands."""
    display.print_info("\nAvailable Commands:")
    display.print_info("  /reset    - Clear conversation history")
    display.print_info("  /history  - Show conversation history")
    display.print_info("  /tokens   - Show estimated token usage")
    display.print_info("  /help     - Show this help message")
    display.print_info("  /exit     - Exit the chatbot")
    display.print_info("  /quit     - Exit the chatbot\n")


def main() -> None:
    """
    Main function that runs the chatbot.

    Initializes all components using dependency injection pattern:
    Config -> ConversationHistory -> GeminiChat -> display functions
    """
    try:
        # Load and validate configuration
        config = load_config()
    except ValueError as e:
        display.print_error(str(e))
        sys.exit(1)

    # Initialize components
    history = ConversationHistory()
    chat = GeminiChat(config, history)

    # Display welcome panel
    display.print_welcome(config.model, config.system_prompt)

    # Main chat loop
    try:
        while True:
            # Get user input
            try:
                user_input = input("\nYou: ").strip()
            except EOFError:
                # Handle EOF (e.g., piped input ends)
                print("\nGoodbye!")
                sys.exit(0)

            # Skip empty input silently
            if not user_input:
                continue

            # Handle commands (case-insensitive)
            command = user_input.lower()

            if command in ("/exit", "/quit"):
                display.print_info("Goodbye!")
                sys.exit(0)

            elif command == "/reset":
                history.reset()
                display.print_info("Conversation reset.")
                continue

            elif command == "/history":
                display.print_history(history)
                continue

            elif command == "/tokens":
                display.print_info(f"Estimated tokens used: {history.token_estimate()}")
                continue

            elif command == "/help":
                print_help()
                continue

            # Normal message - send to Gemini
            response_generator = chat.send_message(user_input)
            display.stream_assistant_response(response_generator)

    except KeyboardInterrupt:
        # Handle Ctrl+C gracefully
        print("\nGoodbye!")
        sys.exit(0)


if __name__ == "__main__":
    main()
