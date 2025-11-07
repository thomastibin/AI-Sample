import json
import traceback
from datetime import datetime
from typing import List
from langchain_community.utilities import WikipediaAPIWrapper

from langchain_community.tools import DuckDuckGoSearchRun, DuckDuckGoSearchResults

def _format_hits(hits: List[dict], limit: int = 10) -> str:
    lines = []
    for h in (hits or [])[:limit]:
        title = h.get("title") or h.get("text") or ""
        link = h.get("href") or h.get("link") or ""
        snippet = h.get("body") or h.get("snippet") or ""
        if title or link or snippet:
            lines.append(f"- {title}\n  {link}\n  {snippet}")
    return "\n".join(lines) if lines else "(no results)"

def ddg_search_langchain(query: str, max_results: int = 10) -> str:
    """
    Prefer structured results; fall back to plain-text if needed.
    Always returns a human-readable string (never raises).
    """
    # 1) Structured results (list of dicts)
    try:
        results_tool = DuckDuckGoSearchResults(num_results=max_results)
        hits = results_tool.run(query)  # may be List[dict] or JSON string
        if isinstance(hits, str):
            try:
                hits = json.loads(hits)
            except Exception:
                hits = []
        return _format_hits(hits, limit=max_results)
    except Exception:
        # 2) Plain-text fallback
        try:
            run_tool = DuckDuckGoSearchRun()
            text_block = run_tool.run(query)
            return text_block or "(empty search text)"
        except Exception as e:
            tb = traceback.format_exc()
            return f"(Search failed: {e}\n{tb})"

def search_tool(query: str) -> str:
    """Public function: web search via DuckDuckGo, robust output string."""
    return ddg_search_langchain(query, max_results=10)


# ---- Wikipedia (LangChain community) ----------------------------------------

# Configure once; reuse
_wiki_api = WikipediaAPIWrapper(
    top_k_results=2,
    doc_content_chars_max=2000,
    lang="en",
)

def wiki_tool(query: str) -> str:
    """
    Public function: Wikipedia lookup.
    Returns a text snippet. Never raises; embeds errors in the string.
    """
    try:
        return _wiki_api.run(query)
    except Exception as e:
        tb = traceback.format_exc()
        return f"(Wikipedia lookup failed: {e}\n{tb})"


def save_to_txt(data: str, filename: str = "research_output.txt") -> str:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_text = f"--- Research Output ---\nTimestamp: {timestamp}\n\n{data}\n\n"
    with open(filename, "a", encoding="utf-8") as f:
        f.write(formatted_text)
    return f"Data successfully saved to {filename}"

def save_tool(data: str) -> str:
    """Public function: append data to a simple text file."""
    return save_to_txt(data)


__all__ = ["search_tool", "wiki_tool", "save_tool"]
