import os, json, re
from datetime import datetime
from typing import List

from dotenv import load_dotenv
load_dotenv()

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI

from tools import search_tool, wiki_tool, save_tool  


class ResearchResponse(BaseModel):
    topic: str = Field(..., description="The user-provided research topic")
    summary: str = Field(..., description="Succinct, well-structured summary of findings")
    sources: List[str] = Field(..., description="List of source URLs or citations")
    tools_used: List[str] = Field(..., description="List of tools the agent actually used")


URL_RE = re.compile(r"https?://[^\s)]+")
def extract_urls(text: str, limit: int = 8) -> List[str]:
    urls = URL_RE.findall(text or "")
    out, seen = [], set()
    for u in urls:
        if u not in seen:
            seen.add(u)
            out.append(u)
            if len(out) >= limit:
                break
    return out

def save_json(data: dict, filename: str) -> str:
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return os.path.abspath(filename)

def make_llm() -> ChatGoogleGenerativeAI:
    key = os.getenv("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError("GOOGLE_API_KEY is not set. Put it in your environment or .env")
    return ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.0)


def main():
    topic = input("What topic should I research? ").strip()
    if not topic:
        print("No topic provided. Exiting.")
        return

    tools_used: List[str] = []

    try:
        wiki_notes = wiki_tool(topic)  
        tools_used.append("Wikipedia")
    except Exception as e:
        wiki_notes = f"(Wikipedia lookup failed: {e})"

    try:
        raw_search = search_tool(topic)  
    except Exception as e:
        print("[debug] search_tool(topic) failed:", repr(e), flush=True)
        search_notes = f"(Search failed: {e})"
        search_urls = []
    else:
        print("[debug] type(raw_search) =", type(raw_search).__name__, flush=True)
        tools_used.append("DuckDuckGo")
        search_notes = raw_search if isinstance(raw_search, str) else json.dumps(raw_search, ensure_ascii=False)
        try:
            search_urls = extract_urls(search_notes, limit=10)
        except Exception as ex:
            print("[debug] extract_urls failed:", repr(ex), flush=True)
            search_urls = []

    parser = PydanticOutputParser(pydantic_object=ResearchResponse)
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a disciplined research assistant.\n"
         "You will receive:\n"
         "1) a TOPIC\n"
         "2) SEARCH_NOTES (duckduckgo results text)\n"
         "3) WIKI_NOTES (wikipedia snippet)\n\n"
         "Use them to produce a concise, neutral, accurate summary.\n"
         "Cite useful source URLs. Keep output only in the required JSON schema.\n"
         "{format_instructions}"),
        ("human",
         "TOPIC: {topic}\n\n"
         "SEARCH_NOTES:\n{search_notes}\n\n"
         "WIKI_NOTES:\n{wiki_notes}\n\n"
         "Now produce the JSON.")
    ]).partial(format_instructions=parser.get_format_instructions())

    llm = make_llm()

    try:
        response: ResearchResponse = (prompt | llm | parser).invoke({
            "topic": topic,
            "search_notes": search_notes,
            "wiki_notes": wiki_notes
        })
    except Exception as e:
        print(f"[Parser fallback] {e}")
        fb_prompt = ChatPromptTemplate.from_messages([
            ("system", "Write a concise research summary (200-300 words) using the provided material."),
            ("human", "TOPIC: {topic}\nSEARCH_NOTES: {search_notes}\nWIKI_NOTES: {wiki_notes}")
        ])
        text_summary = (fb_prompt | llm | StrOutputParser()).invoke({
            "topic": topic,
            "search_notes": search_notes,
            "wiki_notes": wiki_notes
        })
        response = ResearchResponse(
            topic=topic,
            summary=text_summary.strip(),
            sources=search_urls[:5],
            tools_used=tools_used or ["Gemini"]
        )

    if not response.sources:
        response.sources = search_urls[:5]

    slug = re.sub(r"[^a-z0-9]+", "-", topic.lower()).strip("-") or datetime.now().strftime("%Y%m%d-%H%M%S")
    path = save_json(response.model_dump(), f"research_{slug}.json")

    try:
        _ = save_tool(json.dumps(response.model_dump(), ensure_ascii=False, indent=2))
    except Exception as e:
        print("[debug] save_tool failed:", repr(e), flush=True)

    print("\nDone.")
    print(f"JSON saved to: {path}")
    print("Tools used:", ", ".join(response.tools_used))


if __name__ == "__main__":
    main()
