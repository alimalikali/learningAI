"""
Entry point for LangChain RAG.

Same two modes as Project 2:
  python3 main.py --ingest ./docs
  python3 main.py
  python3 main.py --debug

NOTICE: main.py is significantly simpler than Project 2's main.py
because LangChain handles the wiring. Our job is just orchestration.
"""

import argparse
import sys

from config import load_config
from rag.ingestion import ingest
from rag.retriever import load_retriever
from rag.chain import build_rag_chain, build_chain_with_sources
import display


def run_ingestion(dir_path: str) -> None:
    """
    Run the ingestion pipeline.

    One function call replaces 5 steps from Project 2.

    Args:
        dir_path: Path to directory containing documents
    """
    display.print_info("Starting LangChain ingestion pipeline...\n")

    try:
        config = load_config()
    except ValueError as e:
        display.print_error(str(e))
        sys.exit(1)

    try:
        ingest(config, dir_path)
        display.print_info(
            "Ingestion complete! Run 'python3 main.py' to start chatting."
        )
    except Exception as e:
        display.print_error(f"Ingestion failed: {e}")
        sys.exit(1)


def run_chat(debug: bool = False) -> None:
    """
    Run the interactive chat loop.

    Args:
        debug: If True, show retrieved chunks before each answer
    """
    try:
        config = load_config()
    except ValueError as e:
        display.print_error(str(e))
        sys.exit(1)

    # Load retriever and build chain
    try:
        retriever = load_retriever(config)

        if debug:
            answer_chain, retrieval_chain = build_chain_with_sources(
                config, retriever
            )
        else:
            answer_chain = build_rag_chain(config, retriever)
            retrieval_chain = None

    except Exception as e:
        display.print_error(f"Failed to load retriever: {e}")
        display.print_info("Make sure you've run --ingest first.")
        sys.exit(1)

    # Print welcome
    display.print_welcome(config.chat_model, config.collection_name)
    display.print_info("\nReady to chat!\n")

    # Main chat loop
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

            elif command == "/help":
                display.print_info("\nAvailable Commands:")
                display.print_info("  /help   - Show this help message")
                display.print_info("  /exit   - Exit the chatbot")
                display.print_info("  /quit   - Exit the chatbot")
                if not debug:
                    display.print_info(
                        "\nTip: Run with --debug to see retrieved chunks"
                    )
                continue

            # Debug mode: show retrieved chunks first
            if debug and retrieval_chain:
                result = retrieval_chain.invoke(user_input)
                docs = result["context_docs"]
                display.print_debug_chunks(docs)

            # Stream response
            display.stream_response(answer_chain, user_input)

    except KeyboardInterrupt:
        print("\nGoodbye!")
        sys.exit(0)


def main() -> None:
    """Parse arguments and run appropriate mode."""
    parser = argparse.ArgumentParser(
        description="LangChain RAG - Chat with your documents"
    )
    parser.add_argument(
        "--ingest",
        metavar="DIR",
        help="Ingest documents from directory"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Show retrieved chunks before each answer"
    )

    args = parser.parse_args()

    if args.ingest:
        run_ingestion(args.ingest)
    else:
        run_chat(debug=args.debug)


if __name__ == "__main__":
    main()
