"""
Wires together: LLM + Tools + Memory + Prompt = Agent

CONCEPT: The agent uses ReAct (Reason + Act) pattern:
  1. THINK: "What does the user want? What tools do I have?"
  2. ACT: Choose and call a tool
  3. OBSERVE: Read the tool's output
  4. REPEAT until it has enough information to answer
  5. RESPOND: Give the final answer

The agent DECIDES which tools to use. You never hardcode "use search for this,
use calculator for that." The LLM figures that out from the tool docstrings.
This is why good docstrings on tools are critical.
"""

from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_google_genai import ChatGoogleGenerativeAI

from config import Config


SYSTEM_PROMPT = """You are a helpful AI assistant with access to tools.

You have access to the following tools:
- web_search: Search the internet for current information
- calculator: Perform mathematical calculations
- get_weather: Get current weather for any city
- query_database: Query company database for employee, product, and sales data

RULES:
- Always use tools when you need factual, real-time, or data-specific information
- Use calculator for ANY math — never do arithmetic in your head
- If a question involves company data, ALWAYS use query_database
- After using a tool, clearly explain what you found
- Be concise and direct in your final answers
- If a tool fails, try an alternative approach or explain why you cannot answer"""


def build_agent_executor(config: Config, tools: list, memory):
    """
    Build the agent executor with all components wired together.

    Args:
        config: Configuration object
        tools: List of tool functions
        memory: ChatMessageHistory instance (not used with langgraph, kept for API compat)

    Returns:
        Runnable: The configured agent
    """
    # Create LLM
    llm = ChatGoogleGenerativeAI(
        model=config.model,
        google_api_key=config.gemini_api_key
    )

    # Create memory saver for conversation persistence
    checkpointer = MemorySaver()

    # Create the ReAct agent using langgraph
    agent = create_react_agent(
        model=llm,
        tools=tools,
        prompt=SYSTEM_PROMPT,
        checkpointer=checkpointer,
    )

    return agent
