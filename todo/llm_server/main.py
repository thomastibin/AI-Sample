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

from googleapiclient.discovery import build
from base64 import urlsafe_b64decode
import re


load_dotenv()
DEFAULT_TZ = os.getenv("DEFAULT_TZ", "Asia/Kolkata")

def _gmail_service_from_calendar_creds():
    # reuse calendar creds bootstrap by calling your calendar helper URL via HTTP shim
    # or—if this file can import your calendar module directly—do that instead.
    from .calendar_google import get_calendar_service
    cal = get_calendar_service()
    creds = cal._http.credentials
    return build("gmail", "v1", credentials=creds, cache_discovery=False)

async def _poll_replies_every(app_state, interval_sec=30):
    svc = _gmail_service_from_calendar_creds()
    seen_ids = set()
    while True:
        try:
            # Fetch messages containing your correlation token Ref:XXXX
            q = 'subject:"[Ref:" newer_than:7d'  # quick filter; refine if needed
            msgs = svc.users().messages().list(userId="me", q=q, maxResults=50).execute().get("messages", [])
            for m in msgs:
                if m["id"] in seen_ids: 
                    continue
                full = svc.users().messages().get(userId="me", id=m["id"], format="full").execute()
                seen_ids.add(m["id"])
                # Extract subject + body
                headers = {h["name"].lower(): h["value"] for h in full.get("payload", {}).get("headers", [])}
                subject = headers.get("subject", "")
                m = re.search(r"\[Ref:(?P<key>[A-Za-z0-9]+)\]", subject)
                if not m: 
                    continue
                key = m.group("key")

                # Decode simple/plain body part if present
                body = ""
                parts = full.get("payload", {}).get("parts", []) or []
                for p in parts:
                    if p.get("mimeType") == "text/plain" and "data" in p.get("body", {}):
                        body = urlsafe_b64decode(p["body"]["data"]).decode("utf-8", errors="ignore")
                        break

                # Store per-threadKey in memory (or Mongo) for aggregation
                app_state.setdefault("replies", {}).setdefault(key, []).append({
                    "from": headers.get("from"), "body": body, "id": full.get("id"),
                })

                # Heuristic: if all invitees replied -> finalize
                thread_info = app_state.get("threadMeta", {}).get(key)
                if thread_info:
                    invited = set(map(str.lower, thread_info.get("attendees", [])))
                    responders = set()
                    for r in app_state["replies"][key]:
                        frm = (r["from"] or "").lower()
                        # extract email inside <>
                        m2 = re.search(r"<([^>]+)>", frm)
                        if m2:
                            responders.add(m2.group(1))
                        else:
                            responders.add(frm)
                    if invited.issubset(responders):
                        # parse replies with LLM to pick a time window (not shown),
                        # then call MCP to schedule & todo.create
                        # await mcp.call("calendar.schedule", {...})
                        pass
        except Exception:
            pass
        await asyncio.sleep(interval_sec)

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
# --- DROP-IN REPLACEMENT in llm_server/main.py ---

def _extract_mcp_json(result: Any) -> Any:
    """
    Normalize MCP tool responses so the rest of the code always sees:
        {"type": "json", "data": <parsed python object>}
    Works even if the server returned a text content with JSON string.
    """
    def _maybe_json_text(txt: str) -> Any:
        s = (txt or "").strip()
        if s.startswith("{") or s.startswith("["):
            try:
                return json.loads(s)
            except Exception:
                return txt
        return txt

    def _envelope(obj: Any) -> Dict[str, Any]:
        # If it's already our envelope, keep it
        if isinstance(obj, dict) and obj.get("type") in ("json", "application/json") and "data" in obj:
            return {"type": "json", "data": obj["data"]}
        # Otherwise wrap whatever we parsed as data
        return {"type": "json", "data": obj}

    try:
        content = getattr(result, "content", None)
        if isinstance(content, list) and content:
            # 1) Prefer explicit JSON parts
            for item in content:
                if isinstance(item, dict) and item.get("type") == "application/json" and "json" in item:
                    return _envelope(item["json"])

            # 2) Pydantic-ish objects
            for item in content:
                if hasattr(item, "model_dump"):
                    return _envelope(item.model_dump())
                if hasattr(item, "model_dump_json"):
                    try:
                        return _envelope(json.loads(item.model_dump_json()))
                    except Exception:
                        pass
                if hasattr(item, "json"):
                    try:
                        val = item.json() if callable(item.json) else item.json
                    except Exception:
                        val = item
                    if isinstance(val, str):
                        return _envelope(_maybe_json_text(val))
                    return _envelope(val)

            # 3) Text fallback (common on older fastmcp)
            for item in content:
                if isinstance(item, dict) and "text" in item:
                    return _envelope(_maybe_json_text(item["text"]))
                if hasattr(item, "text"):
                    return _envelope(_maybe_json_text(getattr(item, "text")))

        # 4) Already a dict/list
        if isinstance(result, (dict, list)):
            return _envelope(result)
    except Exception:
        pass
    # Unknown shape – still envelope it so callers are consistent
    return {"type": "json", "data": result}


def mcp_data(payload: Any) -> Any:
    """
    Convenience: call this on the result of mcp.call(...) to directly get the 'data' object.
    """
    if isinstance(payload, dict) and payload.get("type") in ("json", "application/json"):
        return payload.get("data")
    return payload

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

@app.get("/_debug/replies")
async def debug_replies():
    return app.state.meet_state

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _mcp_bus
    _mcp_bus = MCPBus(os.path.join(os.path.dirname(__file__), "mcp_manifest.json"))
    await _mcp_bus.start()

    # ✅ init shared state used by the poller
    app.state.meet_state = {
        "threadMeta": {},   # key -> { "attendees": [...] , ... }
        "replies": {}       # key -> [ { from, body, id }, ... ]
    }

    # ✅ start Gmail poller (checks every 30s; adjust if needed)
    app.state.gmail_poller = asyncio.create_task(
        _poll_replies_every(app.state.meet_state, interval_sec=30)
    )

    try:
        yield
    finally:
        # ✅ stop poller cleanly
        poller = getattr(app.state, "gmail_poller", None)
        if poller:
            poller.cancel()
            try:
                await poller
            except asyncio.CancelledError:
                pass

        if _mcp_bus:
            await _mcp_bus.stop()
            _mcp_bus = None

# keep this line:
app.router.lifespan_context = lifespan  # <-- ONLY lifespan; remove on_event handlers


class ChatIn(BaseModel):
    message: str


@app.post("/chat")
async def chat(body: ChatIn):
    try:
        print(f"[LLM] chat API started: {body.message[:50]!r}...")
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
            day_start_utc = start_utc.replace(hour=0, minute=0, second=0, microsecond=0)
            day_end_utc   = end_utc.replace(hour=23, minute=59, second=59, microsecond=0)

            slots_env = await mcp.call("calendar.free", {
                "dateFrom": day_start_utc.isoformat(),
                "dateTo":   day_end_utc.isoformat(),
                "minMinutes": 30
            })
            slots = mcp_data(slots_env)
            print(f"[LLM] Found free slots: {slots}")
            if not isinstance(slots, list) or not slots:
                return {"ok": True, "message": "No free 30-min slots found on that day."}

            # 2) Build a plain-text email listing the free slots in IST for readability
            ist = pytz.timezone(tz_name)
            lines = []
            for s in slots:
                sdt = datetime.fromisoformat(s["start"]).astimezone(ist)
                edt = datetime.fromisoformat(s["end"]).astimezone(ist)
                # suggest the first 30 minutes within each gap
                lines.append(f"- {sdt.strftime('%d %b %Y %I:%M %p')} to {(sdt + timedelta(minutes=30)).strftime('%I:%M %p')} IST")

            thread_key = uuid.uuid4().hex[:8]
            subject = f"Proposed meeting times on {start_utc.astimezone(ist).strftime('%d %b %Y')}"
            body = (
                "Hi,\n\n"
                "I’m planning a Google Meet on the date above. Here are my free 30-minute windows—"
                "please reply with the option(s) you prefer (or another time on the same day):\n\n"
                + "\n".join(lines) +
                "\n\nThanks!"
            )

            email_res_env = await mcp.call("gmail.sendProposal", {
                "to": attendees,
                "subject": subject,
                "body": body,
                "threadKey": thread_key
            })
            email_res = mcp_data(email_res_env)
            # Correlate replies to this outbound proposal
            app.state.meet_state["threadMeta"][thread_key] = {
                "attendees": attendees,
                "title": title
            }

            return {
                "ok": True,
                "sent": email_res,
                "threadKey": thread_key,
                "proposedSlots": slots
            }

        elif intent == "search_todos":
            results_env = await mcp.call("todo.search", {
                "dateFrom": start_utc.isoformat(),
                "dateTo":   end_utc.isoformat(),
                "text": None
            })
            results = mcp_data(results_env)
            return {"ok": True, "todos": results or []}

        elif intent == "search_meetings":
            results_env = await mcp.call("calendar.search", {
                "dateFrom": start_utc.isoformat(),
                "dateTo":   end_utc.isoformat(),
                "attendees": attendees or []
            })
            results = mcp_data(results_env)
            print(f"[MCP] search_meetings results: {results}")
            return {"ok": True, "events": results or []}
        elif intent == "todo.create":
            todo_created_env = await mcp.call("todo.create", {
                "title": title,
                "start": start_utc.isoformat(),
                "end":   end_utc.isoformat(),
                "attendees": attendees,
                "description": plan.get("description") or "",
                "tz": tz_name,
            })
            todo_created = mcp_data(todo_created_env)
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
