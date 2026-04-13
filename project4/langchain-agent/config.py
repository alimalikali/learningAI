"""
Centralized configuration for LangChain Agent.
"""

import os
from dataclasses import dataclass
from dotenv import load_dotenv


@dataclass(frozen=True)
class Config:
    """
    Immutable configuration for the agent.

    Attributes:
        gemini_api_key: Google Gemini API key (required)
        tavily_api_key: Tavily API key for web search (required)
        model: Gemini model to use
        max_iterations: Maximum tool-calling iterations
        db_path: Path to SQLite database
    """
    gemini_api_key: str
    tavily_api_key: str
    model: str
    max_iterations: int
    db_path: str


def load_config() -> Config:
    """
    Load configuration from environment variables.

    Returns:
        Config: Validated configuration object

    Raises:
        ValueError: If required fields are missing
    """
    load_dotenv()

    gemini_api_key = os.getenv("GEMINI_API_KEY", "")
    tavily_api_key = os.getenv("TAVILY_API_KEY", "")
    model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
    max_iterations = int(os.getenv("MAX_ITERATIONS", "10"))
    db_path = os.getenv("DB_PATH", "data/company.db")

    # Validate required fields
    if not gemini_api_key or gemini_api_key == "your_gemini_api_key_here":
        raise ValueError(
            "GEMINI_API_KEY is not set or is the placeholder value.\n"
            "Get a free key at: https://aistudio.google.com/app/apikey"
        )

    if not tavily_api_key or tavily_api_key == "your_tavily_api_key_here":
        raise ValueError(
            "TAVILY_API_KEY is not set or is the placeholder value.\n"
            "Get a free key at: https://tavily.com (1000 searches/month free)"
        )

    return Config(
        gemini_api_key=gemini_api_key,
        tavily_api_key=tavily_api_key,
        model=model,
        max_iterations=max_iterations,
        db_path=db_path
    )
