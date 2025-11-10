from __future__ import annotations

import os
import json
import asyncio
import traceback
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta

from click import prompt
import pytz
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
import httpx

from .time_utils import parse_human_time_to_utc_window

 
from mcp.client.streamable_http import streamablehttp_client
 
from mcp.client.session import ClientSession

load_dotenv()


class ChatIn(BaseModel):
    message: str


class MCPToolRunner:
    """
    Minimal MCP client wrapper using Streamable HTTP via manifest.

    Manifest (place next to this file as mcp_manifest.json):
    {
      "name": "meet-todo-mcp",
      "version": "1.0.0",
      "transport": { "type": "streamable_http", "url": "http://localhost:8000/mcp" }
    }
    """

    def __init__(self, manifest_path: str):
        self.manifest_path = manifest_path
        self._cm = None
        self._session: Optional[ClientSession] = None
        self._started = False

    async def start(self):
        # load manifest
        with open(self.manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        t = (manifest.get("transport") or {})
        ttype = (t.get("type") or "").lower()
        if ttype not in ("streamable_http", "http"):
            raise RuntimeError(
                "This LLM expects 'streamable_http' in mcp_manifest.json "
                "(or 'http' if your client exposes it as an alias)."
            )
        url = t.get("url")   # <- no trailing slash
        if not url:
            raise RuntimeError("Missing 'transport.url' for Streamable HTTP MCP manifest")
        if not url.endswith("/"):
             url += "/"
        headers = t.get("headers") or {}
        print(f"[MCP START] connecting via STREAMABLE_HTTP to {url} headers={bool(headers)}")

        try:
            # Open transport tunnel. Different mcp wheels yield 2 or 3 items.
            self._cm = streamablehttp_client(url=url, headers=headers)
            entered = await asyncio.wait_for(self._cm.__aenter__(), timeout=10.0)

            if isinstance(entered, tuple):
                if len(entered) >= 2:
                    read, write = entered[0], entered[1]
                else:
                    raise RuntimeError("streamablehttp_client returned an unexpected tuple")
            else:
                # Extremely old layouts might return an object; try attributes
                read = getattr(entered, "read", None)
                write = getattr(entered, "write", None)
                if read is None or write is None:
                    raise RuntimeError("Cannot obtain read/write streams from streamablehttp_client")

            # JSON-RPC session
            self._session = ClientSession(read, write)
            await self._session.__aenter__()
            await self._session.initialize()

            # Probe tools so failures are obvious
            tools = await asyncio.wait_for(self._session.list_tools(), timeout=10.0)
            print("[MCP] list_tools OK:", [t.name for t in tools.tools])
            self._started = True

        except Exception as e:
            await self.close()
            print(f"[ERROR111] {e!r}")
            raise

        
    async def close(self):
        try:
            if self._session is not None:
                await self._session.__aexit__(None, None, None)
        finally:
            self._session = None
            if self._cm is not None:
                try:
                    await self._cm.__aexit__(None, None, None)
                finally:
                    self._cm = None
            self._started = False
    async def _ensure(self):
        if not self._started:
            await self.start()

    @staticmethod
    def _extract_result(res: Any) -> Any:
        """Best-effort extraction of JSON/text content from MCP call_tool result."""
        try:
            content = None
            if isinstance(res, dict) and "content" in res:
                content = res["content"]
            elif hasattr(res, "content"):
                content = getattr(res, "content")
            if content is not None:
                for item in content:
                    # JSON payload
                    for key in ("json", "data"):
                        if isinstance(item, dict) and key in item:
                            return item[key]
                        if hasattr(item, key):
                            return getattr(item, key)
                    # Or text containing JSON
                    txt = None
                    if isinstance(item, dict) and "text" in item:
                        txt = item["text"]
                    elif hasattr(item, "text"):
                        txt = getattr(item, "text")
                    if isinstance(txt, str):
                        t = txt.strip()
                        if (t.startswith("{") and t.endswith("}")) or (t.startswith("[") and t.endswith("]")):
                            return json.loads(t)
                        return t
            if isinstance(res, (dict, list)):
                return res
        except Exception:
            pass
        return res

    async def _call(self, tool: str, payload: Dict[str, Any]) -> Any:
        print(f"[_call] utc _call={tool}")
        try:
            await self._ensure()
            if self._session is None:
                raise RuntimeError("MCP client session not initialized")
            res = await self._session.call_tool(tool, payload)
            return self._extract_result(res)
        except Exception as e:
            print(f"[_call] exception calling tool {tool}: {repr(e)}")
            traceback.print_exc()
            try:
                s = self._session
                print("[_call] session object:", s)
                if hasattr(s, 'closed'):
                    print("[_call] session.closed:", getattr(s, 'closed'))
            except Exception:
                pass
            raise

    # Convenience tool wrappers
    async def todo_search(self, payload: Dict[str, Any]) -> Any:
        return await self._call("todo.search", payload)

    async def todo_create(self, payload: Dict[str, Any]) -> Any:
        return await self._call("todo.create", payload)

    async def calendar_search(self, payload: Dict[str, Any]) -> Any:
        return await self._call("calendar.search", payload)

    async def calendar_schedule(self, payload: Dict[str, Any]) -> Any:
        return await self._call("calendar.schedule", payload)


class HTTPMCPToolRunner:
    """HTTP-based tool runner calling the MCP server's /mcp-call endpoint.

    Set MCP_HTTP_BASE to the MCP server base URL, e.g., http://127.0.0.1:8000
    """

    def __init__(self, base_url: str):
        self.base_url = (base_url or "").rstrip("/")
        self._client: Optional[httpx.AsyncClient] = None

    async def start(self):
        self._client = httpx.AsyncClient(timeout=20.0)

    async def close(self):
        if self._client:
            await self._client.aclose()
            self._client = None

    async def _call(self, tool: str, params: Dict[str, Any]):
        if not self._client:
            await self.start()
        url = f"{self.base_url}/mcp-call"
        resp = await self._client.post(url, json={"tool": tool, "params": params})
        try:
            data = resp.json()
        except Exception:
            data = {"error": True, "message": f"Invalid response ({resp.status_code})"}
        return data

    async def todo_search(self, payload: Dict[str, Any]) -> Any:
        return await self._call("todo.search", payload)

    async def todo_create(self, payload: Dict[str, Any]) -> Any:
        return await self._call("todo.create", payload)

    async def calendar_search(self, payload: Dict[str, Any]) -> Any:
        return await self._call("calendar.search", payload)

    async def calendar_schedule(self, payload: Dict[str, Any]) -> Any:
        return await self._call("calendar.schedule", payload)


def _new_llm() -> ChatGoogleGenerativeAI:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY missing in environment")
    return ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key=api_key, temperature=0)


def _intent_chain():
    schema_desc = (
        "Return ONLY valid JSON with keys: "
        "intent (schedule_meeting|search_todos|search_meetings), "
        "title (string), attendees (array of emails), startText (string), endText (optional string). "
        "Rules: "
        "1) Always include startText. "
        "2) If duration is not explicitly provided, set endText = startText + 30 minutes. "
        "3) Use Asia/Kolkata timezone by default unless the user specifies otherwise (e.g., IST/UTC offsets). "
        "4) Respond with only RFC3339 with timezone. in startText/endText. "
        "5) Do NOT invent attendees; only include emails present in the message. "
        "6) Respond with only JSON, no comments."
    )

   

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You extract structured planning JSON for a meeting/todo agent.\n"
            "{schema_desc}\n\n"
            "Examples:\n{few_shot_examples}"
        ),
        ("human", "Message: {message}\nRespond with only JSON, no commentary."),
    ])
    return prompt, schema_desc.replace("\n", " ")


def _parse_plan_json(text: str) -> Dict[str, Any]:
    try:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            text = text[start:end+1]
        return json.loads(text)
    except Exception as e:
        raise ValueError("LLM did not return valid JSON") from e


def _normalize_plan(plan: Dict[str, Any], original_text: str) -> Dict[str, Any]:
    if not isinstance(plan, dict):
        return {}
    if plan.get("intent"):
        return plan

    # Seen in practice:
    # {"action":"read","type":"schedule","filters":{"target_date":"2025-06-11"}}
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

    if isinstance(filters, dict) and any(k.lower().count("date") for k in filters.keys()):
        date = filters.get("target_date") or filters.get("date") or filters.get("start")
        return {
            "intent": "search_meetings",
            "title": plan.get("title"),
            "attendees": plan.get("attendees") or [],
            "startText": date or original_text,
            "endText": None,
        }

    return plan


def _detect_conflicts(cal_events: List[Dict[str, Any]]) -> bool:
    return len(cal_events) > 0


def _suggest_alternatives(start_local: datetime, tz_name: str) -> List[str]:
    opts = []
    zone = pytz.timezone(tz_name)
    for mins in (30, 60):
        t = (start_local + timedelta(minutes=mins)).astimezone(zone)
        opts.append(t.strftime("%I:%M %p %Z").lstrip("0"))
    return opts


app = FastAPI(title="Meet+ToDo LLM Server (Streamable HTTP MCP)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/chat")
async def chat(body: ChatIn):
    try:
        print(f"[LLM] /chat received message: {body.message}")
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
       
        raw = await chain.ainvoke({"message": body.message, "schema_desc": schema_desc,"few_shot_examples": few_shot_examples})
        print("[LLM] raw response:", raw)
        plan = _parse_plan_json(raw)
        print("[LLM] parsed plan:", plan)
        plan = _normalize_plan(plan, body.message)
        print("[LLM] normalized plan:", plan)

        intent = plan.get("intent")
        title = plan.get("title") or "Meeting"
        attendees = plan.get("attendees") or []
        start_text = plan.get("startText") or body.message
        end_text = plan.get("endText")

        start_utc, end_utc, tz_name = parse_human_time_to_utc_window(
            start_text, os.getenv("DEFAULT_TZ", "Asia/Kolkata")
        )
        print(f"[TIME] parsed start_utc={start_utc} end_utc={end_utc} tz={tz_name}")
        if end_text:
            e2, _, _ = parse_human_time_to_utc_window(end_text, tz_name)
            end_utc = e2
            print(f"[TIME] overridden end_utc from endText={end_utc}")

        base_http = os.getenv("MCP_HTTP_BASE")
        # if base_http:
        #     runner = HTTPMCPToolRunner(base_http)
        # else:
        runner = MCPToolRunner(
            manifest_path=os.path.join(os.path.dirname(__file__), "mcp_manifest.json")
        )
        await runner.start()
        try:
            if intent == "schedule_meeting":
                pad = timedelta(minutes=30)
                window_from = (start_utc - pad).isoformat()
                window_to = (end_utc + pad).isoformat()

                cal_payload = {"dateFrom": window_from, "dateTo": window_to, "attendees": attendees}
                print("[MCP] calendar_search payload schedule_meeting:", cal_payload)
                cal_events = await runner.calendar_search(cal_payload)
                print("[MCP] calendar_search result:", cal_events)
                if isinstance(cal_events, dict) and cal_events.get("error"):
                    return cal_events

                if _detect_conflicts(cal_events):
                    local_zone = pytz.timezone(tz_name)
                    start_local = start_utc.astimezone(local_zone)
                    suggestions = _suggest_alternatives(start_local, tz_name)
                    return {"conflict": True, "events": cal_events, "suggestions": suggestions}

                cal_payload = {
                    "title": title,
                    "start": start_utc.astimezone(pytz.timezone(tz_name)).isoformat(),
                    "end": end_utc.astimezone(pytz.timezone(tz_name)).isoformat(),
                    "attendees": attendees,
                    "description": plan.get("description") or "",
                    "tz": tz_name,
                }
                print("[MCP] calendar_schedule payload:", cal_payload)
                cal_created = await runner.calendar_schedule(cal_payload)
                print("[MCP] calendar_schedule result:", cal_created)
                if isinstance(cal_created, dict) and cal_created.get("error"):
                    return cal_created

                todo_payload = {
                    "title": title,
                    "start": start_utc.isoformat(),
                    "end": end_utc.isoformat(),
                    "attendees": attendees,
                    "description": plan.get("description") or "",
                    "tz": tz_name,
                }
                print("[MCP] todo_create payload:", todo_payload)
                todo_created = await runner.todo_create(todo_payload)
                print("[MCP] todo_create result:", todo_created)
                if isinstance(todo_created, dict) and todo_created.get("error"):
                    return todo_created

                return {
                    "ok": True,
                    "meetLink": cal_created.get("meetLink"),
                    "eventId": cal_created.get("id"),
                    "todoId": todo_created.get("id"),
                }

            elif intent == "search_todos":
                payload = {"dateFrom": start_utc.isoformat(), "dateTo": end_utc.isoformat()}
                print("[MCP] todo_search payload:", payload)
                results = await runner.todo_search(payload)
                print("[MCP] todo_search result:", results)
                return {"ok": True, "todos": results}

            elif intent == "search_meetings":
                payload = {
                    "dateFrom": start_utc.isoformat(),
                    "dateTo": end_utc.isoformat(),
                    "attendees": attendees or None,
                }
                print("[MCP] calendar_search payload search_meetings:", payload)
                results = await runner.calendar_search(payload)
                print("[MCP] calendar_search result:", results)
                return {"ok": True, "events": results}

            else:
                return {"error": True, "message": f"Unknown intent: {intent}"}
        finally:
            await runner.close()

    except Exception as e:
        print("[ERROR] /chat exception:", repr(e))
        raise HTTPException(status_code=400, detail={"error": True, "message": str(e)})


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT_LLM", "8001"))
    uvicorn.run("llm_server.main:app", host="0.0.0.0", port=port, reload=False)
