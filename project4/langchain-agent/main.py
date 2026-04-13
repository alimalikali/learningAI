"""
Entry point for LangChain Agent.

Usage:
  python3 data/seed_db.py     # run once to create database
  python3 main.py             # start agent
  python3 main.py --debug     # show full ReAct reasoning steps
"""

import argparse
import sys

from config import load_config
from agent.tools.search import web_search
from agent.tools.calculator import calculator
from agent.tools.weather import get_weather
from agent.tools.database import query_database
from agent.memory import create_memory
from agent.executor import build_agent_executor
import display


def extract_response(result: dict) -> str:
    """Extract the final response text from agent result."""
    messages = result.get("messages", [])
    if messages:
        # Get the last AI message
        for msg in reversed(messages):
            if hasattr(msg, 'content') and msg.content:
                if not hasattr(msg, 'tool_calls') or not msg.tool_calls:
                    return msg.content
    return "No response generated."


def main() -> None:
    """Main function to run the agent."""
    parser = argparse.ArgumentParser(
        description="LangChain Agent - AI assistant with tools"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Show full ReAct reasoning steps"
    )

    args = parser.parse_args()

    # Load configuration
    try:
        config = load_config()
    except ValueError as e:
        display.print_error(str(e))
        sys.exit(1)

    # Setup tools
    tools = [web_search, calculator, get_weather, query_database]

    # Setup memory (for compatibility)
    memory = create_memory()

    # Build agent executor
    agent = build_agent_executor(config, tools, memory)

    # Print welcome
    display.print_welcome(config.model)
    display.print_info("\nAgent ready! Ask me anything.\n")

    # Config for agent invocation (thread_id for memory)
    agent_config = {"configurable": {"thread_id": "default"}}

    # Main loop
    try:
        while True:
            try:
                user_input = input("\nYou: ").strip()
            except EOFError:
                print("\nGoodbye!")
                sys.exit(0)

            if not user_input:
                continue

            # Handle commands
            command = user_input.lower()

            if command in ("/exit", "/quit"):
                display.print_info("Goodbye!")
                sys.exit(0)

            elif command == "/tools":
                display.print_tools()
                continue

            elif command == "/history":
                display.print_info("History is managed internally by the agent.")
                continue

            elif command == "/reset":
                # Recreate agent with fresh memory
                agent = build_agent_executor(config, tools, memory)
                agent_config = {"configurable": {"thread_id": "new_session"}}
                display.print_info("Memory cleared.")
                continue

            elif command == "/help":
                display.print_info("\nAvailable Commands:")
                display.print_info("  /tools   - Show available tools")
                display.print_info("  /history - Show conversation history")
                display.print_info("  /reset   - Clear memory")
                display.print_info("  /help    - Show this help")
                display.print_info("  /exit    - Exit the agent")
                if not args.debug:
                    display.print_info(
                        "\nTip: Run with --debug to see ReAct reasoning"
                    )
                continue

            # Process query with agent
            try:
                # Stream the response for better UX
                print()
                display.print_info("Thinking...")

                result = agent.invoke(
                    {"messages": [("user", user_input)]},
                    config=agent_config
                )

                response = extract_response(result)
                print()
                display.print_response(response)

                # Show debug info
                if args.debug:
                    print()
                    display.print_info("DEBUG: Full message flow:")
                    for msg in result.get("messages", []):
                        msg_type = type(msg).__name__
                        content = str(msg.content)[:100] if hasattr(msg, 'content') else str(msg)[:100]
                        print(f"  [{msg_type}] {content}...")

            except Exception as e:
                display.print_error(f"Agent error: {str(e)}")

    except KeyboardInterrupt:
        print("\nGoodbye!")
        sys.exit(0)


if __name__ == "__main__":
    main()
