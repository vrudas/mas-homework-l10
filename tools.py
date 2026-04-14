import os

import trafilatura
from ddgs import DDGS
from langchain_core.tools import tool

from config import settings
from retriever import get_retriever

retriever = get_retriever()


@tool
def web_search(query: str) -> list[dict]:
    """Run web search for given query and return a list of search results.
    Args: query: search string
    Example: [{
                "title": "Q-dance",
                "url": "https://en.wikipedia.org/wiki/Q-dance",
                "snippet": "Q-dance is a Dutch event production company",
            }]
    """
    try:
        search_results = DDGS().text(query, max_results=settings.max_search_results)
        return [
            {
                "title": search_result.get("title"),
                "url": search_result.get("href"),
                "snippet": search_result.get("body"),
            }
            for search_result in search_results
        ]
    except Exception:
        print(f"Error fetching URL: {query}")
        return []


@tool
def read_url(url: str) -> str:
    """Read url content and return it.
    Args: url: url string"""
    try:
        downloaded = trafilatura.fetch_url(url)
        text = trafilatura.extract(downloaded)
        return text[:settings.max_url_content_length]
    except Exception as e:
        return f"Content not available for {url}: {e}"


@tool
def save_report(filename: str, content: str) -> str:
    """Save a research report to a file with provided filename in the output directory and return the file path.
    Args: filename: file name string, content: content string to write in the file"""
    path = "./" + settings.output_dir + "/" + filename

    dirpath = os.path.dirname(path) or "."
    os.makedirs(dirpath, exist_ok=True)

    with open(path, 'w', encoding="utf-8") as f:
        f.write(content)

    return f"Report written to {path}"


@tool
def knowledge_search(query: str) -> str:
    """Search the local knowledge base using hybrid retrieval + reranking.
    Args: query: search string"""
    try:
        response = retriever.invoke(query)
        return str(response)
    except Exception:
        print(f"Error during finding content for: {query}")
        return f"Error during finding content for: {query}"
