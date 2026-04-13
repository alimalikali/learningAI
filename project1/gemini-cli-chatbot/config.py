"""
Configuration module for Gemini CLI Chatbot.

Handles loading and validating configuration from environment variables.
Uses a frozen dataclass to ensure configuration immutability after initialization.
"""

import os
from dataclasses import dataclass
from dotenv import load_dotenv


@dataclass(frozen=True)
class Config:
    """
    Immutable configuration for the Gemini chatbot.

    Attributes:
        api_key: Google Gemini API key (required)
        model: Gemini model to use (default: gemini-2.5-flash)
        system_prompt: System instruction for the AI (default: concise assistant)
    """
    api_key: str
    model: str
    system_prompt: str


def load_config() -> Config:
    """
    Load configuration from environment variables.

    Returns:
        Config: Validated configuration object

    Raises:
        ValueError: If API key is missing or is the placeholder value
    """
    load_dotenv()

    api_key = os.getenv("GEMINI_API_KEY", "")
    model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    system_prompt = os.getenv(
        "SYSTEM_PROMPT",
        "You are a sharp, concise AI assistant."
    )

    # Validate API key
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY is not set. "
            "Please create a .env file with your API key.\n"
            "Get a free key at: https://aistudio.google.com/app/apikey"
        )

    if api_key == "your_api_key_here":
        raise ValueError(
            "GEMINI_API_KEY is still the placeholder value. "
            "Please replace it with your actual API key.\n"
            "Get a free key at: https://aistudio.google.com/app/apikey"
        )

    return Config(
        api_key=api_key,
        model=model,
        system_prompt=system_prompt
    )
