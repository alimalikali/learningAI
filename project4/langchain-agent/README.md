# LangChain Agent

A Python CLI AI Agent using LangChain that can use multiple tools to answer questions. The agent decides WHICH tools to use based on the question — this is the core concept of agentic AI.

This is **Project 4** of an AI Engineering learning path.

## What is an Agent vs a Chain?

- **Chain** = Fixed steps. Input → Step 1 → Step 2 → Output. Always the same path.
- **Agent** = LLM decides steps dynamically. It looks at the question, picks tools, uses them, and decides when it has enough info to answer.

## Available Tools

| Tool | Description |
|------|-------------|
| `web_search` | Search the web for current information (via Tavily) |
| `calculator` | Evaluate math expressions safely |
| `get_weather` | Get current weather for any city (via Open-Meteo) |
| `query_database` | Query company database using natural language |

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure environment:**
   ```bash
   cp .env.example .env
   ```
   Edit `.env` and add:
   - `GEMINI_API_KEY` - Get free at https://aistudio.google.com/app/apikey
   - `TAVILY_API_KEY` - Get free at https://tavily.com (1000 searches/month)

3. **Create the database:**
   ```bash
   python3 data/seed_db.py
   ```

4. **Start the agent:**
   ```bash
   python3 main.py
   ```

5. **Debug mode (see ReAct reasoning):**
   ```bash
   python3 main.py --debug
   ```

## Example Queries

Try these to see different tools in action:

```
# Weather tool
"What's the weather in Karachi right now?"

# Web search tool
"Search for latest AI news in 2025"

# Calculator tool
"What is 15% of 47,850?"

# Database tool
"Who are the top 3 highest paid employees?"
"Which product has the lowest stock?"
"What is the total sales amount for the Engineering department?"

# Multi-tool query (uses database + weather!)
"What's the weather in the city where our highest paid employee lives?"
```

## Commands

| Command | Description |
|---------|-------------|
| `/tools` | Show available tools |
| `/history` | Show conversation history |
| `/reset` | Clear memory |
| `/help` | Show help |
| `/exit` | Exit the agent |

## Key Concepts

### ReAct Pattern (Reason + Act)

The agent follows this loop:
1. **THINK**: "What does the user want? What tools do I have?"
2. **ACT**: Choose and call a tool
3. **OBSERVE**: Read the tool's output
4. **REPEAT** until it has enough information
5. **RESPOND**: Give the final answer

### Why Tool Docstrings Matter

The LLM reads tool docstrings to decide which tool to use. Bad docstrings = wrong tool selection. Compare:

```python
# BAD - vague
@tool
def search(q):
    """Search stuff."""

# GOOD - clear and specific
@tool
def web_search(query: str):
    """
    Search the web for current information, news, or facts.
    Use this when you need up-to-date information or facts you're not certain about.
    Input should be a clear search query string.
    """
```

### Why Calculator Tool Exists

LLMs hallucinate math. They'll confidently say 23 * 47 = 1081 (wrong). Tools fix this:

```
Without tool: "23 * 47 = 1081" (WRONG)
With tool:    calculator("23 * 47") → "Result: 1081" (CORRECT)
```

### Text-to-SQL Pattern

The `query_database` tool converts natural language to SQL:
1. User: "Who earns the most?"
2. Tool uses Gemini to generate: `SELECT name, salary FROM employees ORDER BY salary DESC LIMIT 1`
3. Executes query against SQLite
4. Returns formatted results

### Conversation Memory

`ConversationBufferWindowMemory` keeps the last K turns:
- Without memory: Agent forgets everything between questions
- With memory: "Who is the highest paid?" → "What city do they live in?" works!

The `k=10` parameter prevents context window overflow.

### verbose=True

When enabled, you see the agent's internal reasoning:
```
> Entering new AgentExecutor chain...
Invoking: `query_database` with `{'natural_language_query': 'highest paid employee'}`
...
> Finished chain.
```

## Project Structure

```
langchain-agent/
├── main.py              # Entry point
├── config.py            # Configuration
├── display.py           # Rich terminal UI
├── agent/
│   ├── __init__.py
│   ├── executor.py      # AgentExecutor setup
│   ├── memory.py        # Conversation memory
│   └── tools/
│       ├── __init__.py
│       ├── search.py    # Tavily web search
│       ├── calculator.py # Safe math eval
│       ├── weather.py   # Open-Meteo API
│       └── database.py  # Text-to-SQL
├── data/
│   └── seed_db.py       # Database seeder
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

## API Keys Required

| Service | Free Tier | Get Key |
|---------|-----------|---------|
| Gemini | Unlimited (rate limited) | https://aistudio.google.com/app/apikey |
| Tavily | 1000 searches/month | https://tavily.com |
| Open-Meteo | Unlimited | No key needed! |
