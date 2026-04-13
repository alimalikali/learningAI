"""
Conversation memory for the agent.

Without memory, the agent forgets everything between turns — like a goldfish.
With memory, it can reference earlier parts of the conversation.

CONCEPT: Memory is stored as a list of (human, ai) message pairs.
We use ConversationBufferWindowMemory to keep only the last K turns
to prevent the context window from overflowing.
"""

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory


def create_memory() -> ChatMessageHistory:
    """
    Create a chat message history store.

    Returns:
        ChatMessageHistory: Memory instance for the agent
    """
    return ChatMessageHistory()
