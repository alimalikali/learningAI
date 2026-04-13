"""
The LCEL chain — this is the core of Project 3.

LCEL (LangChain Expression Language) uses the pipe | operator to compose steps.
Each step is a Runnable. The output of one step becomes the input of the next.

This is what we built manually in Project 2's generator.py — but now it's
declarative, composable, and streams automatically.

THE CHAIN:
  user question
      |
  retriever          → fetches relevant chunks from Qdrant
      |
  format_docs        → formats chunks into a context string
      |
  prompt template    → injects context + question into prompt
      |
  ChatGoogleGenerativeAI  → streams response from Gemini
      |
  StrOutputParser    → extracts text from LLM response object
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_google_genai import ChatGoogleGenerativeAI

from config import Config


def format_docs(docs: list) -> str:
    """
    Format retrieved documents into a context string.

    Args:
        docs: List of LangChain Document objects

    Returns:
        str: Formatted context string with sources
    """
    formatted = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        formatted.append(f"[Source: {source}]\n{doc.page_content}")
    return "\n\n---\n\n".join(formatted)


RAG_PROMPT = ChatPromptTemplate.from_template("""You are a helpful assistant that answers questions based ONLY on the provided context.
If the answer is not in the context, say "I could not find that in the provided documents."
Always mention which document your answer came from.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
""")


def build_rag_chain(config: Config, retriever):
    """
    Build the RAG chain using LCEL.

    The chain:
    1. RunnableParallel runs retriever and passthrough simultaneously
    2. format_docs converts retrieved docs to context string
    3. RAG_PROMPT injects context + question
    4. LLM generates response
    5. StrOutputParser extracts text

    Args:
        config: Configuration object
        retriever: VectorStoreRetriever instance

    Returns:
        Runnable: The composed RAG chain
    """
    llm = ChatGoogleGenerativeAI(
        model=config.chat_model,
        google_api_key=config.gemini_api_key,
        streaming=True
    )

    chain = (
        RunnableParallel({
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        })
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )

    return chain


def build_chain_with_sources(config: Config, retriever):
    """
    Build a chain that also returns source documents alongside the answer.

    Uses RunnableParallel to run retrieval once but capture docs for display.

    Args:
        config: Configuration object
        retriever: VectorStoreRetriever instance

    Returns:
        tuple: (answer_chain, retrieval_chain) for getting answer and sources
    """
    llm = ChatGoogleGenerativeAI(
        model=config.chat_model,
        google_api_key=config.gemini_api_key,
        streaming=True
    )

    # Chain that retrieves and stores docs
    retrieval_chain = RunnableParallel({
        "context_docs": retriever,
        "question": RunnablePassthrough()
    })

    # Full answer chain
    answer_chain = (
        RunnableParallel({
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        })
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )

    return answer_chain, retrieval_chain
