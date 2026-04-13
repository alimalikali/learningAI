# LangChain RAG

A Python CLI tool for chatting with documents using RAG — rebuilt with **LangChain** instead of raw API calls. This is **Project 3** of an AI Engineering learning path.

## What This Project Demonstrates

This project reimplements Project 2 (RAG from Scratch) using LangChain's abstractions. The goal is to show how LangChain simplifies what we built manually.

## Side-by-Side Comparison

| Concern           | Project 2 (raw)              | Project 3 (LangChain)                    |
|-------------------|------------------------------|------------------------------------------|
| Document loading  | loader.py — 60 lines         | DirectoryLoader — 3 lines                |
| Chunking          | chunker.py — 50 lines        | RecursiveCharacterTextSplitter — 2 lines |
| Embedding         | embedder.py — 70 lines       | GoogleGenerativeAIEmbeddings — 1 line    |
| Vector store      | vectorstore.py — 100 lines   | QdrantVectorStore — 5 lines              |
| Chain/Generation  | generator.py — 60 lines      | LCEL chain — 10 lines                    |
| **Total**         | **~350 lines**               | **~80 lines**                            |

## Setup

1. **Create virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment:**
   ```bash
   cp .env.example .env
   ```
   Edit `.env` with your API keys.

4. **Add documents:**
   ```bash
   mkdir docs
   # Drop PDF or TXT files in docs/
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
# Ingest documents
python3 main.py --ingest ./docs

# Chat mode
python3 main.py

# Debug mode (shows retrieved chunks)
python3 main.py --debug
```

## Key Concepts

### What is LCEL?

LCEL (LangChain Expression Language) uses the pipe `|` operator to compose steps into a chain. Each step is a "Runnable" — the output of one step becomes the input of the next.

```python
chain = (
    RunnableParallel({
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    })
    | RAG_PROMPT
    | llm
    | StrOutputParser()
)
```

### RunnablePassthrough

`RunnablePassthrough()` passes its input unchanged to the next step. In our chain, it passes the user's question directly to the prompt template.

### RunnableParallel

`RunnableParallel({...})` runs multiple runnables simultaneously and collects their outputs into a dictionary. In our chain:
- `"context"` runs the retriever and formats docs
- `"question"` passes the input unchanged
- Both run in parallel for efficiency

### RecursiveCharacterTextSplitter

This splitter is smarter than fixed-size chunking. It tries to split on:
1. Paragraphs (`\n\n`) first
2. Then sentences (`\n`)
3. Then words (` `)
4. Finally characters as a last resort

This preserves semantic meaning better than arbitrarily cutting at character counts.

### chain.stream() vs chain.invoke()

- `chain.invoke(query)` — Returns the full response after it's complete
- `chain.stream(query)` — Yields response chunks as they arrive (real-time streaming)

For chat interfaces, always use `.stream()` for better UX.

### Automatic Retries

LangChain's `GoogleGenerativeAIEmbeddings` handles rate limit retries automatically. In Project 2, we had to implement this manually with sleep() between batches.

## Project Structure

```
langchain-rag/
├── main.py              # Entry point (ingest/chat modes)
├── config.py            # Configuration loading
├── display.py           # Rich terminal UI
├── rag/
│   ├── __init__.py
│   ├── ingestion.py     # Document loading, chunking, storing
│   ├── chain.py         # LCEL chain definition
│   └── retriever.py     # Qdrant retriever setup
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

## When to Use LangChain vs Raw APIs

**Use LangChain when:**
- Building standard RAG pipelines
- Rapid prototyping
- You want automatic retries, batching, streaming
- Composing multiple LLMs or tools

**Use Raw APIs when:**
- You need fine-grained control
- Debugging complex issues
- Minimizing dependencies
- Learning how things work under the hood

Project 2 taught you the fundamentals. Project 3 shows how frameworks abstract them away.
