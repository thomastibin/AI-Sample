# llm_server/main.py
from __future__ import annotations

import os, json, asyncio, time
from typing import Any, Dict, List, Optional
from datetime import timedelta

import pytz
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

from mcp.client.streamable_http import streamablehttp_client
from mcp.client.session import ClientSession
from contextlib import asynccontextmanager
import httpx

from .time_utils import parse_human_time_to_utc_window

load_dotenv()
DEFAULT_TZ = os.getenv("DEFAULT_TZ", "Asia/Kolkata")


# ---------- LLM (planner) ----------
def _new_llm() -> ChatGoogleGenerativeAI:
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY missing in environment")
    return ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key=key, temperature=0)

def _intent_chain():
    schema_desc = (
        "Return ONLY valid JSON with keys: "
        "intent (schedule_meeting|search_todos|search_meetings|todo.create), "
        "title (string), attendees (array of emails), startText (string), endText (optional string). "
        "Rules: "
        "1) Always include startText. "
        "2) If duration is not explicitly provided, set endText = startText + 30 minutes. "
        "3) Use Asia/Kolkata timezone by default unless the user specifies otherwise (e.g., IST/UTC offsets). "
        '4) Respond with only RFC3339 with timezone in startText/endText (e.g., "2025-11-10T12:00:00+05:30"). '
        "5) Do NOT invent attendees; only include emails present in the message. "
        "6) Respond with only JSON, no comments."
        "7) if event is not online or no email is given, then intent is todo.create."
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You extract structured planning JSON for a meeting/todo agent.\n"
         "{schema_desc}\n\nExamples:\n{few_shot_examples}"),
        ("human", "Message: {message}\nRespond with only JSON, no commentary."),
    ])
    return prompt, schema_desc.replace("\n", " ")

def _parse_plan_json(text: str) -> Dict[str, Any]:
    start = text.find("{"); end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("LLM did not return JSON")
    return json.loads(text[start:end+1])

def _normalize_plan(plan: Dict[str, Any], original_text: str) -> Dict[str, Any]:
    if not isinstance(plan, dict):
        return {}
    if plan.get("intent"):
        return plan
    action = plan.get("action")
    typ = plan.get("type") or plan.get("subject")
    filters = plan.get("filters") or {}
    if action in ("read", "list", "get") and isinstance(typ, str) and "sched" in typ.lower():
        date = filters.get("target_date") or filters.get("date") or filters.get("targetDate")
        return {
            "intent": "search_meetings",
            "title": plan.get("title"),
            "attendees": plan.get("attendees") or [],
            "startText": date or original_text,
            "endText": None,
        }
    if isinstance(filters, dict) and any("date" in k.lower() for k in filters.keys()):
        date = filters.get("target_date") or filters.get("date") or filters.get("start")
        return {
            "intent": "search_meetings",
            "title": plan.get("title"),
            "attendees": plan.get("attendees") or [],
            "startText": date or original_text,
            "endText": None,
        }
    return plan


# ---------- MCP URL ----------
def _read_mcp_url(default_manifest_path: str) -> str:
    env_url = os.getenv("MCP_SERVER_URL")
    if env_url and env_url.strip():
        return env_url.strip()  # keep exactly as provided
    manifest_path = os.getenv("MCP_MANIFEST_PATH") or default_manifest_path
    with open(manifest_path, "r", encoding="utf-8") as f:
        m = json.load(f)
    t = (m.get("transport") or {})
    if (t.get("type") or "").lower() not in ("streamable_http", "http"):
        raise RuntimeError("Manifest transport.type must be 'streamable_http'")
    url = (t.get("url") or "")
    if not url:
        raise RuntimeError("Manifest missing transport.url")
    return url  # keep trailing slash if present


# ---------- Robust extractor ----------
def _extract_mcp_json(result: Any) -> Any:
    """
    Collect all JSON parts from an MCP ToolResult.
    - If there's exactly one JSON part: return it directly.
    - If there are multiple JSON parts: return a list of them.
    - If there are no JSON parts but there are text parts: return the concatenated text.
    - Supports pydantic v2 objects via model_dump/model_dump_json too.
    """
    try:
        content = getattr(result, "content", None)

        # If the server already gave us a plain list/dict, just return it.
        if isinstance(result, (list, dict)) and content is None:
            return result

        if isinstance(content, list):
            json_parts = []
            text_parts = []

            for item in content:
                # 1) Normal dict payload
                if isinstance(item, dict):
                    if "json" in item:
                        json_parts.append(item["json"])
                        continue
                    if "text" in item:
                        text_parts.append(item["text"])
                        continue

                # 2) Pydantic v2 objects
                if hasattr(item, "model_dump"):
                    json_parts.append(item.model_dump())
                    continue
                if hasattr(item, "model_dump_json"):
                    try:
                        json_parts.append(json.loads(item.model_dump_json()))
                    except Exception:
                        json_parts.append(item.model_dump_json())
                    continue

                # 3) Generic .json attribute/method
                if hasattr(item, "json"):
                    attr = getattr(item, "json")
                    try:
                        val = attr() if callable(attr) else attr
                    except Exception:
                        val = attr
                    if isinstance(val, str):
                        try:
                            json_parts.append(json.loads(val))
                        except Exception:
                            text_parts.append(val)
                    else:
                        json_parts.append(val)
                    continue

                # 4) Generic .text
                if hasattr(item, "text"):
                    text_parts.append(getattr(item, "text"))
                    continue

            # Prefer JSON when present
            if len(json_parts) == 1:
                return json_parts[0]
            if len(json_parts) > 1:
                return json_parts

            # Fallback to text (concat)
            if text_parts:
                return "\n".join(str(t) for t in text_parts if t is not None)

        # Final fallback
        return result
    except Exception:
        return result


# ---------- Singleton Streamable-HTTP session ----------
class MCPBus:
    """
    One long-lived streamable_http tunnel + ClientSession.
    """
    def __init__(self, manifest_path: str):
        self.manifest_path = manifest_path
        self._cm = None
        self._read = None
        self._write = None
        self.session: Optional[ClientSession] = None
        self._lock = asyncio.Lock()

    async def _preflight(self, url: str, timeout_total: float = 15.0) -> None:
        """
        Consider 200–499 a 'responsive' endpoint (FastMCP may 404 on GET /mcp).
        """
        print(f"[MCPBus] Preflight MCP URL: {url}")
        deadline = time.monotonic() + timeout_total
        last_err = None
        async with httpx.AsyncClient(timeout=2.0) as client:
            while time.monotonic() < deadline:
                try:
                    r = await client.get(url)
                    if 200 <= r.status_code < 500:
                        return
                except Exception as e:
                    last_err = e
                await asyncio.sleep(0.5)
        # If GET preflight fails, we'll still try opening the tunnel with retries
        print(f"[MCPBus] Preflight GET failed (non-fatal): {last_err}")

    async def _open_tunnel_with_retries(self, url: str, retries: int = 10, delay: float = 0.5):
        for i in range(retries):
            try:
                self._cm = streamablehttp_client(url)
                entered = await self._cm.__aenter__()
                if isinstance(entered, tuple):
                    self._read, self._write = entered[0], entered[1]
                else:
                    self._read = getattr(entered, "read", None)
                    self._write = getattr(entered, "write", None)
                    if self._read is None or self._write is None:
                        raise RuntimeError("Cannot obtain read/write from streamablehttp_client")
                return
            except Exception as e:
                if i == retries - 1:
                    raise
                await asyncio.sleep(delay)

    async def start(self):
        if self.session:
            return
        url = _read_mcp_url(os.path.join(os.path.dirname(__file__), "mcp_manifest.json"))
        print(f"[MCPBus] Using MCP URL: {url}")

        await self._preflight(url)
        await self._open_tunnel_with_retries(url)

        self.session = ClientSession(self._read, self._write)
        await self.session.__aenter__()
        await self.session.initialize()

        listed = await self.session.list_tools()
        print("[MCP] Ready. Tools:", [t.name for t in listed.tools])

    async def stop(self):
        try:
            if self.session:
                await self.session.__aexit__(None, None, None)
        finally:
            self.session = None
            if self._cm:
                await self._cm.__aexit__(None, None, None)
                self._cm = None
                self._read = self._write = None

    async def call(self, tool: str, params: Dict[str, Any]) -> Any:
        if not self.session:
            await self.start()
        async with self._lock:
            res = await self.session.call_tool(tool, params)
            return _extract_mcp_json(res)


# ---------- FastAPI wiring ----------
app = FastAPI(title="Meet+ToDo LLM Server (Streamable HTTP MCP)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

_mcp_bus: Optional[MCPBus] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _mcp_bus
    _mcp_bus = MCPBus(os.path.join(os.path.dirname(__file__), "mcp_manifest.json"))
    await _mcp_bus.start()
    try:
        yield
    finally:
        if _mcp_bus:
            await _mcp_bus.stop()
            _mcp_bus = None

app.router.lifespan_context = lifespan  # <-- ONLY lifespan; remove on_event handlers


class ChatIn(BaseModel):
    message: str


@app.post("/chat")
async def chat(body: ChatIn):
    try:
        # PHASE A: plan
        prompt, schema_desc = _intent_chain()
        chain = prompt | _new_llm() | StrOutputParser()
        few_shot_examples = (
            "Example 1:\n"
            "Input: \"3pm next week with tibin@gmail.com\"\n"
            "Output:\n"
            "{\n"
            "  \"intent\": \"schedule_meeting\",\n"
            "  \"title\": \"Meeting\",\n"
            "  \"attendees\": [\"tibin@gmail.com\"],\n"
            "  \"startText\": \"2025-11-01T00:00:00+05:30\",\n"
            "  \"endText\": \"2025-11-01T00:30:00+05:30\"\n"
            "}\n\n"
            "Example 2:\n"
            "Input: \"list online meetings next 48h\"\n"
            "Output:\n"
            "{\n"
            "  \"intent\": \"search_meetings\",\n"
            "  \"title\": \"Online meetings\",\n"
            "  \"attendees\": [],\n"
            "  \"startText\": \"2025-11-01T00:00:00+05:30\",\n"
            "  \"endText\": \"2025-11-01T00:30:00+05:30\"\n"
            "}\n\n"
            "Example 3:\n"
            "Input: \"show my todos next week\"\n"
            "Output:\n"
            "{\n"
            "  \"intent\": \"search_todos\",\n"
            "  \"title\": \"Todos\",\n"
            "  \"attendees\": [],\n"
            "  \"startText\": \"2025-11-01T00:00:00+05:30\",\n"
            "  \"endText\": \"2025-11-07T23:59:59+05:30\"\n"
            "}"
        )
        raw = await chain.ainvoke({
            "message": body.message,
            "schema_desc": schema_desc,
            "few_shot_examples": few_shot_examples
        })
        plan = _parse_plan_json(raw)
        plan = _normalize_plan(plan, body.message)
        print(f"[LLM] Raw plan JSON: {plan}")
        intent = plan.get("intent")
        print(f"[LLM] Extracted plan: intent={intent}, title={plan.get('title')}")
        if intent not in {"schedule_meeting", "search_todos", "search_meetings","todo.create"}:
            return {"error": True, "message": f"Invalid intent: {intent}"}

        title = plan.get("title") or "Meeting"
        attendees = plan.get("attendees") or []
        start_text = plan.get("startText") or body.message
        end_text = plan.get("endText")

        start_utc, end_utc, tz_name = parse_human_time_to_utc_window(start_text, DEFAULT_TZ)
        # if start_text !="":
        #    start_utc= start_text.isoformat()
        # if end_text !="":
        #    end_utc= end_text.isoformat()
        if end_text:
            e2, _, _ = parse_human_time_to_utc_window(end_text, tz_name)
            end_utc = e2
        print(f"[LLM] Parsed times: start_utc={start_utc.isoformat()}, end_utc={end_utc.isoformat()}, tz={tz_name}")

        # PHASE B: execute
        assert _mcp_bus and _mcp_bus.session, "MCP bus not initialized"
        mcp = _mcp_bus

        if intent == "schedule_meeting":
            pad = timedelta(minutes=30)
            cal_events = await mcp.call("calendar.search", {
                "dateFrom": (start_utc - pad).isoformat(),
                "dateTo":   (end_utc + pad).isoformat(),
                "attendees": attendees
            })
            if isinstance(cal_events, list) and len(cal_events) > 0:
                local_zone = pytz.timezone(tz_name)
                start_local = start_utc.astimezone(local_zone)
                alts = []
                for mins in (30, 60):
                    t = (start_local + timedelta(minutes=mins)).astimezone(local_zone)
                    alts.append(t.strftime("%I:%M %p %Z").lstrip("0"))
                return {"conflict": True, "events": cal_events, "suggestions": alts}

            cal_created = await mcp.call("calendar.schedule", {
                "title": title,
                "start": start_utc.astimezone(pytz.timezone(tz_name)).isoformat(),
                "end":   end_utc.astimezone(pytz.timezone(tz_name)).isoformat(),
                "attendees": attendees,
                "description": plan.get("description") or "",
                "tz": tz_name,
            })
            todo_created = await mcp.call("todo.create", {
                "title": title,
                "start": start_utc.isoformat(),
                "end":   end_utc.isoformat(),
                "attendees": attendees,
                "description": plan.get("description") or "",
                "tz": tz_name,
            })
            return {"ok": True, "event": cal_created or {}, "todo": todo_created or {}}

        elif intent == "search_todos":
            results = await mcp.call("todo.search", {
                "dateFrom": start_utc.isoformat(),
                "dateTo":   end_utc.isoformat(),
                "text": None
            })
            return {"ok": True, "todos": results or []}

        elif intent == "search_meetings":
            results = await mcp.call("calendar.search", {
                "dateFrom": start_utc.isoformat(),
                "dateTo":   end_utc.isoformat(),
                "attendees": attendees or []
            })
            print(f"[MCP] search_meetings results: {results}")
            return {"ok": True, "events": results or []}
        elif intent == "todo.create":
            todo_created = await mcp.call("todo.create", {
                "title": title,
                "start": start_utc.isoformat(),
                "end":   end_utc.isoformat(),
                "attendees": attendees,
                "description": plan.get("description") or "",
                "tz": tz_name,
            })
            return {"ok": True,   "todo": todo_created or {}}

        return {"error": True, "message": "Unknown state"}

    except Exception as e:
        detail = {"error": True, "message": f"{type(e).__name__}: {str(e)}"}
        subs = getattr(e, "exceptions", None)
        if subs:
            detail["suberrors"] = [f"{type(se).__name__}: {se}" for se in subs][:3]
        raise HTTPException(status_code=400, detail=detail)


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT_LLM", "8001"))
    uvicorn.run("llm_server.main:app", host="0.0.0.0", port=port, reload=False)
