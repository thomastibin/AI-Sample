from __future__ import annotations

import os
import sys
import asyncio
import json
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timedelta

import pytz
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import logging

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

from .time_utils import parse_human_time_to_utc_window


load_dotenv()

# Configure logging for the LLM server
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("llm_server")

class ChatIn(BaseModel):
    message: str


class MCPToolRunner:
    """Minimal MCP client wrapper using stdio per manifest (official SDK).

    Uses stdio_client + ClientSession; extracts JSON/text results from tool calls.
    """

    def __init__(self, manifest_path: str):
        self.manifest_path = manifest_path
        self._stdio_cm = None
        self._stdio_pair = None
        self._session = None
        self._started = False

    async def start(self):
        try:
            from mcp.client.stdio import stdio_client, StdioServerParameters  # type: ignore
            from mcp.client.session import ClientSession  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError(f"MCP client import failed: {e}") from e

        with open(self.manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        transport = manifest.get("transport", {})
        if transport.get("type") != "stdio":
            raise RuntimeError("Only stdio transport is supported in this demo")

        cmd = transport.get("command") or sys.executable
        if isinstance(cmd, str) and cmd.lower() == "python":
            cmd = sys.executable
        args = transport.get("args", []) or []

        # The mcp stdio_client expects a StdioServerParameters object (pydantic model)
        try:
            server_params = StdioServerParameters(command=cmd, args=list(args), env=transport.get("env"), cwd=transport.get("cwd"))
        except Exception:
            # Fallback: build minimal params dict if model construction fails
            server_params = StdioServerParameters(command=cmd, args=list(args))

        # Log the prepared stdio server parameters (don't log secret env values)
        try:
            logger.debug(
                "Prepared StdioServerParameters: command=%s args=%s env_set=%s cwd=%s",
                getattr(server_params, "command", None),
                getattr(server_params, "args", None),
                bool(getattr(server_params, "env", None)),
                getattr(server_params, "cwd", None),
            )
        except Exception:
            logger.debug("Prepared StdioServerParameters (unable to introspect object)")

        try:
            # Preferred usage: stdio_client(server_params) yields (read, write) streams
            self._stdio_cm = stdio_client(server_params)
            self._stdio_pair = await self._stdio_cm.__aenter__()
            # If we received read/write streams, wrap them with ClientSession
            try:
                read, write = self._stdio_pair
                self._session = ClientSession(read, write)
                await self._session.__aenter__()
                await self._session.initialize()
            except Exception:
                # If streams are not the expected pair, leave session as None and allow fallbacks
                pass
        except TypeError:
            # Older signature fallback (unlikely with modern SDKs): stdio_client(cmd, *args)
            self._stdio_cm = stdio_client(cmd, *args)
            self._stdio_pair = await self._stdio_cm.__aenter__()
            read, write = self._stdio_pair
            self._session = ClientSession(read, write)
            await self._session.__aenter__()
            await self._session.initialize()
        except Exception as e:
            # Ensure we close any partially opened contexts
            await self.close()
            raise RuntimeError(f"MCP client failed to start: {e}") from e
        self._started = True

    async def close(self):
        try:
            if self._session is not None:
                await self._session.__aexit__(None, None, None)
        finally:
            self._session = None
            if self._stdio_cm is not None:
                try:
                    await self._stdio_cm.__aexit__(None, None, None)
                finally:
                    self._stdio_cm = None
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
                    # JSON-like payloads
                    for key in ("json", "data"):
                        if isinstance(item, dict) and key in item:
                            return item[key]
                        if hasattr(item, key):
                            return getattr(item, key)
                    # Text that may contain JSON
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
            # Fallback: if res itself is serializable
            if isinstance(res, (dict, list)):
                return res
        except Exception:
            pass
        return res

    async def _call(self, tool: str, payload: Dict[str, Any]) -> Any:
        await self._ensure()
        logger.debug("Calling MCP tool %s with payload: %s", tool, payload)
        if self._session is None:
            logger.error("ClientSession is not initialized before calling tool %s", tool)
            raise RuntimeError("MCP client session not initialized")
        res = await self._session.call_tool(tool, payload)
        out = self._extract_result(res)
        logger.debug("Received raw MCP result for %s: %s", tool, out)
        return out

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
        "Return ONLY valid JSON with keys: intent (schedule_meeting|search_todos|search_meetings), "
        "title (string), attendees (array of emails), startText (string), endText (optional string)."
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You extract structured planning JSON for a meeting/todo agent. {schema_desc}"),
        ("human", "Message: {message}\nRespond with only JSON, no commentary."),
    ])
    llm = _new_llm()
    return prompt | llm | StrOutputParser()


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
    """Make the parsed plan tolerant to alternate LLM schemas.

    If the LLM returns keys like {action, type, filters} (seen in practice),
    map them to the expected {intent, startText, endText, attendees, title}.
    """
    if not isinstance(plan, dict):
        return {}
    if plan.get("intent"):
        return plan

    # Example alternate schema from LLMs:
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

    # Fallback: if there's a 'filters' with a date, assume search_meetings
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
    # Offer 2 alternatives at +30m, +60m in local tz
    opts = []
    zone = pytz.timezone(tz_name)
    for mins in (30, 60):
        t = (start_local + timedelta(minutes=mins)).astimezone(zone)
        opts.append(t.strftime("%I:%M %p %Z").lstrip("0"))
    return opts


app = FastAPI(title="Meet+ToDo LLM Server")
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
    chain = _intent_chain()
    logger.info("Invoking intent extraction LLM for message: %s", body.message)
    raw = await chain.ainvoke({"message": body.message, "schema_desc": ""})
    logger.debug("Raw LLM output: %s", raw)
    plan = _parse_plan_json(raw)
    logger.debug("Parsed plan JSON: %s", plan)
    # Normalize alternate LLM output shapes into the expected plan schema
    plan = _normalize_plan(plan, body.message)
    logger.debug("Normalized plan: %s", plan)
    # Log expected tokens/fields we want from the LLM
    logger.info("Expecting plan fields: intent, title, attendees, startText, endText")
        intent = plan.get("intent")
        title = plan.get("title") or "Meeting"
        attendees = plan.get("attendees") or []
        start_text = plan.get("startText") or body.message
        end_text = plan.get("endText")

        # Parse time windows
        start_utc, end_utc, tz_name = parse_human_time_to_utc_window(start_text, os.getenv("DEFAULT_TZ", "Asia/Kolkata"))
        if end_text:
            # override end if provided
            e2, _, _ = parse_human_time_to_utc_window(end_text, tz_name)
            end_utc = e2

    # MCP client
    runner = MCPToolRunner(manifest_path=os.path.join(os.path.dirname(__file__), "mcp_manifest.json"))
    logger.info("Starting MCPToolRunner with manifest: %s", os.path.join(os.path.dirname(__file__), "mcp_manifest.json"))
    await runner.start()
    logger.info("MCPToolRunner started")
        try:
            if intent == "schedule_meeting":
                # Search around the window for conflicts (+/-30m)
                pad = timedelta(minutes=30)
                window_from = (start_utc - pad).isoformat()
                window_to = (end_utc + pad).isoformat()

                cal_payload = {"dateFrom": window_from, "dateTo": window_to, "attendees": attendees}
                logger.info("Calling calendar.search with payload: %s", cal_payload)
                cal_events = await runner.calendar_search(cal_payload)
                if isinstance(cal_events, dict) and cal_events.get("error"):
                    return cal_events

                # If conflicts, propose alternatives
                if _detect_conflicts(cal_events):
                    local_zone = pytz.timezone(tz_name)
                    start_local = start_utc.astimezone(local_zone)
                    suggestions = _suggest_alternatives(start_local, tz_name)
                    return {"conflict": True, "events": cal_events, "suggestions": suggestions}

                # No conflict → schedule calendar + create todo
                cal_payload = {
                    "title": title,
                    "start": start_utc.astimezone(pytz.timezone(tz_name)).isoformat(),
                    "end": end_utc.astimezone(pytz.timezone(tz_name)).isoformat(),
                    "attendees": attendees,
                    "description": plan.get("description") or "",
                    "tz": tz_name,
                }
                logger.info("Calling calendar.schedule with payload: %s", cal_payload)
                cal_created = await runner.calendar_schedule(cal_payload)
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
                logger.info("Calling todo.create with payload: %s", todo_payload)
                todo_created = await runner.todo_create(todo_payload)
                if isinstance(todo_created, dict) and todo_created.get("error"):
                    return todo_created

                return {
                    "ok": True,
                    "meetLink": cal_created.get("meetLink"),
                    "eventId": cal_created.get("id"),
                    "todoId": todo_created.get("id"),
                }

            elif intent == "search_todos":
                results = await runner.todo_search({
                    "dateFrom": start_utc.isoformat(),
                    "dateTo": end_utc.isoformat(),
                })
                return {"ok": True, "todos": results}

            elif intent == "search_meetings":
                results = await runner.calendar_search({
                    "dateFrom": start_utc.isoformat(),
                    "dateTo": end_utc.isoformat(),
                    "attendees": attendees or None,
                })
                return {"ok": True, "events": results}

            else:
                return {"error": True, "message": f"Unknown intent: {intent}"}
        finally:
            await runner.close()

    except Exception as e:
        raise HTTPException(status_code=400, detail={"error": True, "message": str(e)})


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT_LLM", "8001"))
    uvicorn.run("llm_server.main:app", host="0.0.0.0", port=port, reload=False)
