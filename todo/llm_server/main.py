# llm_server_gemini_autotool.py
from __future__ import annotations

import os, json
from typing import Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from google import genai
from mcp.client.streamable_http import streamablehttp_client
from mcp.client.session import ClientSession

load_dotenv()

MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

def _gemini_client() -> genai.Client:
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY is missing in environment")
    return genai.Client(api_key=key)

def _read_manifest_url(default_path: str) -> str:
    # 1) Prefer an explicit env URL (use EXACTLY as given; don't strip or rstrip)
    env_url = os.getenv("MCP_SERVER_URL")
    if env_url and env_url.strip():
        return env_url.strip()  # keep trailing slash if present

    # 2) Else load from manifest JSON (again, keep it EXACT)
    manifest_path = os.getenv("MCP_MANIFEST_PATH") or default_path
    with open(manifest_path, "r", encoding="utf-8") as f:
        m = json.load(f)
    t = (m.get("transport") or {})
    if (t.get("type") or "").lower() not in ("streamable_http", "http"):
        raise RuntimeError("Manifest transport.type must be 'streamable_http'")
    url = (t.get("url") or "")
    if not url:
        raise RuntimeError("Manifest missing transport.url")
    return url  # DO NOT rstrip("/")

def _extract_text(resp: Any) -> str:
    try:
        if hasattr(resp, "text") and isinstance(resp.text, str) and resp.text.strip():
            return resp.text.strip()
        parts = []
        for cand in getattr(resp, "candidates", []) or []:
            for part in getattr(cand, "content", {}).get("parts", []):
                if isinstance(part, dict) and part.get("text"):
                    parts.append(part["text"])
        return "\n".join(p.strip() for p in parts if isinstance(p, str) or p)
    except Exception:
        return ""

def _serialize_tools(list_tools_resp):
    return [
        {
            "name": t.name,
            "description": (t.description or "").strip(),
            "input_schema": getattr(t, "inputSchema", None),
        }
        for t in list_tools_resp.tools
    ]

class ChatIn(BaseModel):
    message: str = Field(..., min_length=1)

app = FastAPI(title="Gemini + MCP (auto tool) via streamable_http")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

@app.post("/chat")
async def chat(body: ChatIn):
    try:
        default_manifest = os.path.join(os.path.dirname(__file__), "mcp_manifest.json")
        mcp_url = _read_manifest_url(default_manifest)

        client = _gemini_client()

        # Keep the streamable-http session OPEN while Gemini runs.
        async with streamablehttp_client(mcp_url) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()

                listed = await session.list_tools()  # only ONCE
                tools_out = _serialize_tools(listed)
                if os.getenv("LOG_MCP_TOOLS", "1") == "1":
                    print("[MCP] Tools available:")
                    for i, t in enumerate(tools_out, 1):
                        print(f"  {i}. {t['name']} — {t['description']}")

                # Let Gemini auto-select and execute MCP tools
                resp = await client.aio.models.generate_content(
                    model=MODEL_NAME,
                    contents=body.message,
                    config=genai.types.GenerateContentConfig(
                        temperature=0,
                        tools=[session],   # pass the MCP session directly
                        # tool_choice defaults to AUTO; omit to reduce surface for bugs
                    ),
                )

        text = _extract_text(resp)
        if not text:
            try:
                return {"ok": True, "text": "", "raw": json.loads(resp.to_json())}
            except Exception:
                return {"ok": True, "text": "", "raw": str(resp)}

        return {"ok": True, "text": text}

    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail={"error": True, "message": f"Manifest not found: {e}"})
    except Exception as e:
        # Unwrap ExceptionGroup/TaskGroup for clearer Postman errors (Py 3.11+)
        detail = {"error": True, "message": f"{type(e).__name__}: {str(e)}"}
        # ExceptionGroup has .exceptions; include first few
        suberrs = getattr(e, "exceptions", None)
        if suberrs:
            detail["suberrors"] = [f"{type(se).__name__}: {se}" for se in suberrs][:3]
        raise HTTPException(status_code=400, detail=detail)
 

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT_LLM", "8001"))
    uvicorn.run("llm_server.main:app", host="0.0.0.0", port=port, reload=True)

