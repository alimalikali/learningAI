"""
Conversation history module for Gemini CLI Chatbot.

Manages the chat history in Gemini's required format.
Gemini uses {"role": "user" | "model", "parts": [{"text": str}]} format.
Note: Gemini uses "model" not "assistant" for the AI role.
"""

from copy import deepcopy


class ConversationHistory:
    """
    Manages conversation history in Gemini's format.

    Gemini SDK expects history as a list of messages where each message has:
    - role: "user" or "model" (not "assistant")
    - parts: list of content parts, each with a "text" field
    """

    def __init__(self) -> None:
        """Initialize an empty conversation history."""
        self._messages: list[dict] = []

    def add_user(self, content: str) -> None:
        """
        Add a user message to the history.

        Args:
            content: The user's message text
        """
        self._messages.append({
            "role": "user",
            "parts": [{"text": content}]
        })

    def add_model(self, content: str) -> None:
        """
        Add a model (AI) response to the history.

        Args:
            content: The model's response text
        """
        self._messages.append({
            "role": "model",
            "parts": [{"text": content}]
        })

    def get_messages(self) -> list[dict]:
        """
        Get a copy of all messages in the history.

        Returns a deep copy to prevent external modification of internal state.

        Returns:
            list[dict]: Copy of the message history
        """
        return deepcopy(self._messages)

    def reset(self) -> None:
        """Clear all messages from the history."""
        self._messages.clear()

    def count(self) -> int:
        """
        Get the number of turns (messages) in the history.

        Returns:
            int: Number of messages
        """
        return len(self._messages)

    def token_estimate(self) -> int:
        """
        Estimate the number of tokens in the history.

        Uses a rough approximation of 1 token per 4 characters.
        This is not exact but gives a reasonable estimate for display.

        Returns:
            int: Estimated token count
        """
        total_chars = sum(
            len(msg["parts"][0]["text"])
            for msg in self._messages
        )
        return total_chars // 4
