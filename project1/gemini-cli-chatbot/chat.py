"""
Chat module for Gemini CLI Chatbot.

Handles communication with the Google Gemini API using the official SDK.
Implements streaming responses for real-time output.
"""

from typing import Generator

import google.generativeai as genai
from google.api_core import exceptions as google_exceptions

from config import Config
from history import ConversationHistory


class GeminiChat:
    """
    Manages chat interactions with the Gemini API.

    Uses dependency injection for Config and ConversationHistory to allow
    for easier testing and flexibility in configuration.
    """

    def __init__(self, config: Config, history: ConversationHistory) -> None:
        """
        Initialize the Gemini chat client.

        Args:
            config: Configuration containing API key, model, and system prompt
            history: ConversationHistory instance to track messages
        """
        self._config = config
        self._history = history

        # Configure the Gemini SDK with the API key
        genai.configure(api_key=config.api_key)

        # Create the generative model with system instruction
        # Note: system_instruction is set at model level, not in message history
        self._model = genai.GenerativeModel(
            model_name=config.model,
            system_instruction=config.system_prompt
        )

    def send_message(self, user_input: str) -> Generator[str, None, None]:
        """
        Send a message to Gemini and stream the response.

        Adds the user message to history before sending, then streams
        the response chunk by chunk. After streaming completes, the full
        response is added to history.

        Args:
            user_input: The user's message text

        Yields:
            str: Response chunks as they stream in, or error message
        """
        # Add user message to history BEFORE sending
        self._history.add_user(user_input)

        try:
            # Create a new chat session with current history
            # Gemini is stateless between calls, so we rebuild history each time
            chat_session = self._model.start_chat(
                history=self._history.get_messages()[:-1]  # Exclude current message
            )

            # Send message with streaming enabled
            response = chat_session.send_message(user_input, stream=True)

            # Collect full response while streaming
            full_response = ""

            for chunk in response:
                if chunk.text:
                    full_response += chunk.text
                    yield chunk.text

            # Add complete response to history after stream ends
            self._history.add_model(full_response)

        except google_exceptions.ResourceExhausted:
            yield "[ERROR] Rate limit exceeded. Please wait a moment and try again."
            # Remove the user message since we couldn't get a response
            self._history._messages.pop()

        except google_exceptions.InvalidArgument as e:
            yield f"[ERROR] Invalid request: {str(e)}"
            self._history._messages.pop()

        except Exception as e:
            yield f"[ERROR] Unexpected error: {str(e)}"
            self._history._messages.pop()
