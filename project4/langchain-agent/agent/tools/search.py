"""
Web search tool using Tavily API.

Tavily is purpose-built for LLM agents — returns clean, summarized results.
Free tier: 1000 searches/month at tavily.com

CONCEPT: This is how agents get real-time information beyond their training cutoff.
"""

import os
from langchain_core.tools import tool
from tavily import TavilyClient


@tool
def web_search(query: str) -> str:
    """
    Search the web for current information, news, or facts.
    Use this when you need up-to-date information or facts you're not certain about.
    Input should be a clear search query string.
    """
    try:
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            return "Search failed: TAVILY_API_KEY not configured"

        client = TavilyClient(api_key=api_key)
        response = client.search(query=query, max_results=3)

        results = []
        for r in response.get("results", []):
            results.append(
                f"Title: {r.get('title', 'N/A')}\n"
                f"URL: {r.get('url', 'N/A')}\n"
                f"Summary: {r.get('content', 'N/A')}\n"
            )

        if not results:
            return "No search results found."

        return "\n".join(results)

    except Exception as e:
        return f"Search failed: {str(e)}"
