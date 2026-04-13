# RAG from Scratch

A Python CLI tool for chatting with documents using RAG (Retrieval-Augmented Generation). Built from scratch using Google Gemini API and Qdrant Cloud - **NO LangChain** - raw API calls only.

This is **Project 2** of an AI Engineering learning path.

## What is RAG?

RAG (Retrieval-Augmented Generation) lets you chat with your own documents. Instead of the LLM relying on its training data, it retrieves relevant chunks from your documents and uses them as context to answer questions. This means the LLM can answer questions about information it was never trained on.

## Features

- Ingest PDF and TXT documents
- Chunk documents with configurable overlap
- Generate embeddings using Gemini's text-embedding-004
- Store vectors in Qdrant Cloud
- Semantic search to find relevant chunks
- Stream responses from Gemini
- Debug mode to see retrieved chunks

## Setup

1. **Create a virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment:**
   ```bash
   cp .env.example .env
   ```

   Edit `.env` and fill in:
   - `GEMINI_API_KEY` - Get free at https://aistudio.google.com/app/apikey
   - `QDRANT_URL` - Create free cluster at https://cloud.qdrant.io
   - `QDRANT_API_KEY` - From your Qdrant Cloud dashboard

4. **Add your documents:**
   ```bash
   mkdir docs
   # Drop your PDF or TXT files in the docs/ folder
   ```

5. **Ingest documents:**
   ```bash
   python3 main.py --ingest ./docs
   ```

6. **Start chatting:**
   ```bash
   python3 main.py
   ```

## Usage

```bash
# Ingest documents from a directory
python3 main.py --ingest ./docs

# Start chat mode
python3 main.py

# Chat with debug mode (shows retrieved chunks)
python3 main.py --debug
```

## Commands

| Command | Description |
|---------|-------------|
| `/help` | Show available commands |
| `/exit` | Exit the chatbot |
| `/quit` | Exit the chatbot |

## Project Structure

```
rag-from-scratch/
├── main.py              # Entry point (ingest/chat modes)
├── config.py            # Configuration loading
├── display.py           # Rich terminal UI
├── ingestion/
│   ├── __init__.py
│   ├── loader.py        # PDF + TXT file loading
│   ├── chunker.py       # Text splitting logic
│   └── embedder.py      # Gemini embedding API calls
├── retrieval/
│   ├── __init__.py
│   ├── vectorstore.py   # Qdrant client wrapper
│   └── retriever.py     # Query -> embed -> search
├── generation/
│   ├── __init__.py
│   └── generator.py     # Build prompt -> Gemini -> stream
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

## Key Concepts

### What are Embeddings?

Embeddings are numerical representations of meaning. Text with similar meaning produces similar vectors. This is how semantic search works - instead of matching keywords, we match meaning.

Example: "What's the weather?" and "How's the climate today?" have different words but similar embeddings.

### Why Chunk Size Matters

- **Too big**: Chunks contain irrelevant information, confusing the LLM
- **Too small**: Chunks lose context and meaning
- **Sweet spot**: 300-1000 characters depending on your documents

The overlap ensures we don't cut sentences in half and lose meaning.

### task_type: Document vs Query

Gemini embeddings use different task types:
- `retrieval_document` - For indexing documents (stored in vector DB)
- `retrieval_query` - For user queries (used during search)

**Never mix these** - using the wrong task_type destroys retrieval quality.

### The Full RAG Loop

1. **Ingest Phase:**
   - Load documents from files
   - Split into overlapping chunks
   - Embed each chunk (task_type="retrieval_document")
   - Store vectors + metadata in Qdrant

2. **Query Phase:**
   - User asks a question
   - Embed the query (task_type="retrieval_query")
   - Search Qdrant for similar vectors
   - Retrieve top-k chunks

3. **Generation Phase:**
   - Build prompt with retrieved context
   - Send to Gemini
   - Stream the response

### The LLM Doesn't "Know" Your Documents

This is crucial: the LLM never memorizes your documents. Every time you ask a question:
1. We search for relevant chunks
2. We inject them into the prompt
3. The LLM reads them fresh and generates an answer

This is why RAG works with any documents - the knowledge comes from the prompt context, not the model's training.

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `CHUNK_SIZE` | 500 | Characters per chunk |
| `CHUNK_OVERLAP` | 50 | Overlapping characters |
| `TOP_K` | 4 | Chunks to retrieve per query |
| `CHAT_MODEL` | gemini-2.5-flash | Gemini model for generation |
| `EMBEDDING_MODEL` | models/text-embedding-004 | Model for embeddings |

## Rate Limits

Gemini free tier has rate limits. The embedder automatically:
- Batches requests (20 texts per batch)
- Adds 1-second delay between batches

If you hit rate limits, wait a moment and try again.
