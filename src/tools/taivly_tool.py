from tavily import TavilyClient
import os
from dotenv import load_dotenv

load_dotenv()

client = TavilyClient(
    api_key=os.getenv("TAVILY_API_KEY")
)

def tavily_search(query: str):
    """
    Search for a query using Tavily API.

    Args:
        query (str): The search query.

    Returns:
        dict: The search results.
    """
    response = client.search(
        query = query,
        max_results = 5
                             )

    result = []

    for i, r in enumerate(response["results"], start=1):
        title = r.get("title", "unknown"),
        url = r.get("url", ""),
        snippet = r.get("content", "").strip()
        # keep only the first 300 characters of the snippet

        if len(snippet) > 300:
            snippet = snippet[:300] + "..."

        result.append(f"{i}. {title}\nURL: {url}\nSnippet: {snippet}\n")

    return "\n".join(result)