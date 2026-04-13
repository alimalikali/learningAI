"""
Main entry point for RAG from Scratch.

Two modes:
  1. Ingest mode:  python3 main.py --ingest ./docs
  2. Chat mode:    python3 main.py

Ingest mode processes documents and stores them in Qdrant.
Chat mode allows interactive Q&A over the indexed documents.
"""

import argparse
import sys

from config import load_config
from ingestion.loader import load_directory
from ingestion.chunker import chunk_documents
from ingestion.embedder import embed_texts
from retrieval.vectorstore import VectorStore
from retrieval.retriever import Retriever
from generation.generator import Generator
import display


def run_ingestion(dir_path: str) -> None:
    """
    Run the ingestion pipeline.

    Pipeline steps:
    1. Load documents from directory
    2. Chunk documents into smaller pieces
    3. Embed all chunks
    4. Create/recreate Qdrant collection
    5. Upsert vectors to Qdrant

    Args:
        dir_path: Path to directory containing documents
    """
    display.print_info("Starting ingestion pipeline...\n")

    # Load config
    try:
        config = load_config()
    except ValueError as e:
        display.print_error(str(e))
        sys.exit(1)

    # Step 1: Load documents
    display.print_info("Step 1: Loading documents...")
    try:
        documents = load_directory(dir_path)
    except ValueError as e:
        display.print_error(str(e))
        sys.exit(1)

    # Step 2: Chunk documents
    display.print_info("\nStep 2: Chunking documents...")
    chunks = chunk_documents(
        documents,
        config.chunk_size,
        config.chunk_overlap
    )

    if not chunks:
        display.print_error("No chunks created. Check your documents.")
        sys.exit(1)

    # Step 3: Embed chunks
    display.print_info("\nStep 3: Embedding chunks...")
    embeddings = embed_texts(
        [c.content for c in chunks],
        config.gemini_api_key,
        config.embedding_model
    )

    # Step 4: Create collection
    display.print_info("\nStep 4: Setting up vector store...")
    vectorstore = VectorStore(
        url=config.qdrant_url,
        api_key=config.qdrant_api_key,
        collection_name=config.collection_name
    )

    # Use 768 as fallback (text-embedding-004 dimension)
    vector_size = len(embeddings[0]) if embeddings else 768
    vectorstore.create_collection(vector_size)

    # Step 5: Upsert vectors
    display.print_info("\nStep 5: Storing vectors...")
    vectorstore.upsert(chunks, embeddings)

    # Print summary
    display.print_ingestion_summary(
        num_files=len(documents),
        num_chunks=len(chunks),
        num_vectors=vectorstore.count()
    )

    display.print_info(
        "Ingestion complete! Run 'python3 main.py' to start chatting."
    )


def run_chat(debug: bool = False) -> None:
    """
    Run the interactive chat loop.

    Args:
        debug: If True, show retrieved chunks before each answer
    """
    # Load config
    try:
        config = load_config()
    except ValueError as e:
        display.print_error(str(e))
        sys.exit(1)

    # Initialize components
    vectorstore = VectorStore(
        url=config.qdrant_url,
        api_key=config.qdrant_api_key,
        collection_name=config.collection_name
    )

    # Check if documents are indexed
    if vectorstore.count() == 0:
        display.print_error(
            "No documents indexed. Run with --ingest first.\n"
            "Example: python3 main.py --ingest ./docs"
        )
        sys.exit(1)

    retriever = Retriever(config, vectorstore)
    generator = Generator(config)

    # Print welcome
    display.print_welcome(
        config.chat_model,
        config.embedding_model,
        config.collection_name
    )

    display.print_info(
        f"\n{vectorstore.count()} vectors indexed. Ready to chat!\n"
    )

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

            # Retrieve relevant chunks
            chunks = retriever.retrieve(user_input)

            # Debug: show retrieved chunks
            if debug:
                display.print_retrieved_chunks(chunks)

            # Generate and stream response
            response_gen = generator.generate(user_input, chunks)
            display.stream_response(response_gen, chunks)

    except KeyboardInterrupt:
        print("\nGoodbye!")
        sys.exit(0)


def main() -> None:
    """Parse arguments and run appropriate mode."""
    parser = argparse.ArgumentParser(
        description="RAG from Scratch - Chat with your documents"
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
