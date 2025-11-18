from __future__ import annotations
import os, asyncio, traceback
from typing import Any, Dict, List, Optional
from datetime import datetime
import pytz
from pydantic import ValidationError
from dotenv import load_dotenv

from .schemas import (
    TodoSearchIn, TodoCreateIn, StatusUpdateIn,
    CalSearchIn, CalScheduleIn, ensure_tz_and_utc,
)
from .mongo import ensure_indexes_safe, search_todos, create_todo, update_status
from .models import to_todo_out
from .calendar_google import search_events, schedule_event, find_free_slots

from mcp.server.fastmcp import FastMCP

load_dotenv()
DEFAULT_TZ = os.getenv("DEFAULT_TZ", "Asia/Kolkata")
_DEBUG_LOG_PATH = os.getenv("MCP_DEBUG_LOG", ".mcp_debug.log")

def dbg(*args, **kwargs):
    try:
        with open(_DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
            print(*args, file=f, **kwargs)
    except Exception:
        pass

def _desc(s: str) -> str:
    return " ".join(s.strip().split())

host = os.getenv("MCP_HOST", "0.0.0.0")
port = int(os.getenv("MCP_PORT", "8000"))
mcp = FastMCP("meet-todo-mcp", host=host, port=port)

@mcp.tool(
    name="todo.search",
    description=_desc("""
        Search todos within a UTC time window.
        Inputs:
          - dateFrom: ISO8601 UTC datetime
          - dateTo:   ISO8601 UTC datetime
          - text?:    optional free-text filter
        Returns: TodoOut[] (array of objects).
    """),
)
async def mcp_todo_search(dateFrom: str, dateTo: str, text: Optional[str] = None) -> List[Dict[str, Any]]:
    parsed = TodoSearchIn(dateFrom=dateFrom, dateTo=dateTo, text=text)
    items = await search_todos(parsed.dateFrom, parsed.dateTo, parsed.text)
    return [to_todo_out(doc).model_dump() for doc in items]

@mcp.tool(
    name="todo.create",
    description=_desc("""
        Create a todo/event.
        Required: title, start, end
        Optional: description, tz, attendees[]
        Returns: { id }.
    """),
)
async def mcp_todo_create(
    title: str, start: str, end: str,
    description: Optional[str] = None,
    tz: Optional[str] = None,
    attendees: Optional[List[str]] = None,
) -> Dict[str, Any]:
    parsed = TodoCreateIn(
        title=title, start=start, end=end,
        description=description, tz=tz, attendees=attendees or [],
    )
    start_utc = ensure_tz_and_utc(parsed.start, parsed.tz)
    end_utc = ensure_tz_and_utc(parsed.end, parsed.tz)
    now = datetime.now(pytz.UTC)

    data = {
        "title": parsed.title,
        "description": parsed.description or "",
        "start": start_utc.isoformat(),
        "end": end_utc.isoformat(),
        "tz": parsed.tz or DEFAULT_TZ,
        "attendees": parsed.attendees or [],
        "status": "pending",
        "createdAt": now.isoformat(),
        "updatedAt": now.isoformat(),
        "source": "mcp",
    }
    new_id = await create_todo(data)
    return {"id": new_id}

@mcp.tool(
    name="todo.updateStatus",
    description=_desc("""
        Update a todo status.
        Required: id, status in {"pending","in-progress","done","cancelled"}
        Returns: { ok: boolean }.
    """),
)
async def mcp_todo_update_status(id: str, status: str) -> Dict[str, Any]:
    parsed = StatusUpdateIn(id=id, status=status)
    changed = await update_status(parsed.id, parsed.status)
    return {"ok": changed}

@mcp.tool(
    name="calendar.search",
    description=_desc("""
        Search calendar events in a UTC range.
        Inputs: dateFrom, dateTo, attendees?[]
        Returns: CalEventOut[] (array of objects).
    """),
)
async def mcp_calendar_search(dateFrom: str, dateTo: str, attendees: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    parsed = CalSearchIn(dateFrom=dateFrom, dateTo=dateTo, attendees=attendees or [])
    events = search_events(parsed.dateFrom, parsed.dateTo, parsed.attendees)
    return [e.model_dump() for e in events]

@mcp.tool(
    name="calendar.schedule",
    description=_desc("""
        Create a calendar event (and Meet link when available).
        Required: title, start, end
        Optional: attendees[], description, tz
        Returns: { id, meetLink? }.
    """),
)
async def mcp_calendar_schedule(
    title: str, start: str, end: str,
    attendees: Optional[List[str]] = None,
    description: Optional[str] = None,
    tz: Optional[str] = None,
) -> Dict[str, Any]:
    parsed = CalScheduleIn(
        title=title, start=start, end=end,
        attendees=attendees or [], description=description, tz=tz,
    )
    created = schedule_event(
        title=parsed.title, start=parsed.start, end=parsed.end,
        attendees=parsed.attendees, description=parsed.description,
        tz_name=parsed.tz or DEFAULT_TZ,
        calendar_id=os.getenv("GOOGLE_CALENDAR_ID", "primary"),
    )
    return {"id": created.get("id"), "meetLink": created.get("hangoutLink")}

@mcp.tool(
    name="calendar.free",
    description=_desc("""
        Compute free time slots (no events) between dateFrom/dateTo.
        Inputs: dateFrom (ISO), dateTo (ISO), minMinutes? (default 30)
        Returns: [{ start, end }, ...] in ISO (UTC).
    """),
)
async def mcp_calendar_free(dateFrom: str, dateTo: str, minMinutes: int = 30) -> List[Dict[str, str]]:
    parsed = CalSearchIn(dateFrom=dateFrom, dateTo=dateTo, attendees=[])
    slots = find_free_slots(parsed.dateFrom, parsed.dateTo, min_minutes=int(minMinutes or 30))
    return slots

@mcp.tool(name="ping", description="Health check: returns 'pong' string.")
async def ping() -> str:
    return "pong"

if __name__ == "__main__":
    # Safe index creation
    try:
        asyncio.run(ensure_indexes_safe())
        dbg("[STARTUP] ensure_indexes_safe() OK")
    except Exception as e:
        dbg("[STARTUP] ensure_indexes_safe() FAILED:", repr(e))
        traceback.print_exc()

    transport = os.getenv("MCP_TRANSPORT", "streamable-http").strip().lower()
    if transport == "streamable-http":
        dbg(f"[RUN] transport=streamable-http host={host} port={port}")
        mcp.run(transport="streamable-http")
    else:
        dbg("[RUN] transport=stdio")
        mcp.run(transport="stdio")
