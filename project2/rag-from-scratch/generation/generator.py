"""
Response generator for RAG from Scratch.

Takes the retrieved chunks + user question, builds a prompt, and streams the answer.
This is the "G" in RAG - grounded generation using retrieved context.
The key insight: the LLM doesn't "know" your documents, it reads them each time via the prompt.
"""

from typing import Generator as GenType

from google import genai
from google.genai import types

from config import Config


SYSTEM_INSTRUCTION = """You are a helpful assistant that answers questions based strictly on the provided context.
If the answer is not in the context, say 'I could not find that information in the provided documents.'
Always cite which document your answer came from."""


class Generator:
    """
    Generates responses using retrieved context and Gemini.

    Builds prompts with injected context and streams responses.
    """

    def __init__(self, config: Config) -> None:
        """
        Initialize the generator.

        Args:
            config: Configuration with API key and model settings
        """
        self._config = config
        self._client = genai.Client(api_key=config.gemini_api_key)

    def build_prompt(self, query: str, chunks: list[dict]) -> str:
        """
        Build a prompt with context from retrieved chunks.

        Args:
            query: The user's question
            chunks: List of retrieved chunks with content and source

        Returns:
            str: The formatted prompt
        """
        context_parts = []

        for chunk in chunks:
            context_parts.append(
                f"[Source: {chunk['source']}]\n{chunk['content']}"
            )

        context = "\n\n---\n\n".join(context_parts)

        prompt = f"""Answer the following question using ONLY the context below.

CONTEXT:
{context}

QUESTION: {query}

ANSWER:"""

        return prompt

    def generate(
        self,
        query: str,
        chunks: list[dict]
    ) -> GenType[str, None, None]:
        """
        Generate a response using retrieved context.

        Args:
            query: The user's question
            chunks: List of retrieved chunks

        Yields:
            str: Response chunks as they stream in
        """
        prompt = self.build_prompt(query, chunks)

        try:
            response = self._client.models.generate_content_stream(
                model=self._config.chat_model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_INSTRUCTION
                )
            )

            for chunk in response:
                if chunk.text:
                    yield chunk.text

        except Exception as e:
            yield f"[ERROR] {str(e)}"
