# Gemini CLI Chatbot

A Python CLI chatbot powered by the Google Gemini API. This is **Project 1** of an AI Engineering learning path.

## Features

- Interactive chat with Google's Gemini AI models
- Streaming responses for real-time output
- Beautiful terminal UI using Rich library
- Conversation history management
- Token usage estimation
- Customizable system prompts

## Setup

1. **Create a virtual environment:**
   ```bash
   python3 -m venv venv
   ```

2. **Activate the virtual environment:**
   ```bash
   # Linux/macOS
   source venv/bin/activate

   # Windows
   venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Create your configuration file:**
   ```bash
   cp .env.example .env
   ```

5. **Add your Gemini API key:**
   - Get a free key at: https://aistudio.google.com/app/apikey
   - Edit `.env` and replace `your_api_key_here` with your actual key

6. **Run the chatbot:**
   ```bash
   python main.py
   ```

## Available Commands

| Command    | Description                        |
|------------|------------------------------------|
| `/reset`   | Clear conversation history         |
| `/history` | Show conversation history          |
| `/tokens`  | Show estimated token usage         |
| `/help`    | Show available commands            |
| `/exit`    | Exit the chatbot                   |
| `/quit`    | Exit the chatbot (alias)           |

## Key Concepts Learned

### Gemini Chat History Format
- Each message uses: `{"role": "user" | "model", "parts": [{"text": str}]}`
- Gemini uses `"model"` not `"assistant"` for AI responses (different from OpenAI/Anthropic)

### Why History is Rebuilt Each Turn
- Gemini's chat sessions are stateless between API calls
- `start_chat(history=...)` must be called with full history each time
- This allows the SDK to manage context properly

### How Streaming Works
- Use `stream=True` in `send_message()` to enable streaming
- Response chunks arrive as an iterator
- Each chunk contains partial text that can be printed immediately
- Never wait for the full response when streaming is available

### System Instructions
- Set via `GenerativeModel(system_instruction=...)` at model level
- NOT included in the message history (unlike OpenAI/Anthropic)
- Persists across all messages in the conversation

### Dependency Injection Pattern
- `Config` holds all configuration (loaded from environment)
- `ConversationHistory` manages message state
- `GeminiChat` receives both via constructor injection
- Makes testing easier and components more flexible

## Project Structure

```
gemini-cli-chatbot/
├── main.py          # Entry point and main loop
├── config.py        # Configuration loading and validation
├── chat.py          # Gemini API interaction
├── history.py       # Conversation history management
├── display.py       # Rich terminal UI functions
├── requirements.txt # Python dependencies
├── .env.example     # Example configuration file
├── .gitignore       # Git ignore patterns
└── README.md        # This file
```

## Free Tier Models

- **gemini-2.5-flash** - Fast, great for chat (recommended)
- **gemini-2.5-pro** - More capable, lower rate limits on free tier

No credit card required to get started!
