from __future__ import annotations

import os
import asyncio
from typing import Any, Dict, List, Optional
from datetime import datetime

import pytz
from pydantic import ValidationError
from dotenv import load_dotenv

# ---- your app modules (unchanged) ----
from .schemas import (
    TodoSearchIn, TodoCreateIn, StatusUpdateIn,
    CalSearchIn, CalScheduleIn, ensure_tz_and_utc,
)
from .mongo import ensure_indexes_safe, search_todos, create_todo, update_status
from .models import to_todo_out
from .calendar_google import search_events, schedule_event

# ---- MCP core ----
from mcp.server.fastmcp import FastMCP

# -------------------------------------------------------------------
# Environment / constants / logger (kept from your original)
# -------------------------------------------------------------------
load_dotenv()
DEFAULT_TZ = os.getenv("DEFAULT_TZ", "Asia/Kolkata")
_DEBUG_LOG_PATH = os.getenv("MCP_DEBUG_LOG", ".mcp_debug.log")


def dbg(*args, **kwargs):
    """Append debug lines to file specified by MCP_DEBUG_LOG (default .mcp_debug.log)."""
    try:
        with open(_DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
            print(*args, file=f, **kwargs)
    except Exception:
        # avoid crashing the server for logging failures
        pass


def _desc(s: str) -> str:
    """Normalize multi-line descriptions into single lines for nicer tool help."""
    return " ".join(s.strip().split())


# -------------------------------------------------------------------
# MCP tools (same behavior, stdio transport)
# -------------------------------------------------------------------
host = os.getenv("MCP_HOST", "0.0.0.0")
port = int(os.getenv("MCP_PORT", "8000"))
mcp = FastMCP("meet-todo-mcp",  host=host, port=port)


@mcp.tool(
    name="todo.search",
    description=_desc("""
        Search todos within a UTC time window.
        Inputs:
          - dateFrom: ISO8601 UTC datetime (e.g., "2025-11-10T09:00:00Z")
          - dateTo:   ISO8601 UTC datetime (e.g., "2025-11-10T17:00:00Z")
          - text?:    optional free-text filter
        Returns: array of TodoOut objects.
    """),
)
async def mcp_todo_search(
    dateFrom: str,
    dateTo: str,
    text: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Search todos in the given UTC window, optionally filtering by a text query.
    """
    try:
        parsed = TodoSearchIn(dateFrom=dateFrom, dateTo=dateTo, text=text)
        items = await search_todos(parsed.dateFrom, parsed.dateTo, parsed.text)
        return [to_todo_out(doc).model_dump() for doc in items]
    except ValidationError as ve:
        raise ValueError(str(ve))


@mcp.tool(
    name="todo.create",
    description=_desc("""
        Create a todo/event which may or may not be an online event and default the meeting last for 1 hour;.
        Required:
          - title: string
          - start: ISO8601 datetime string (can include offset or local + tz)
          - end:   ISO8601 datetime string (can include offset or local + tz)
        Optional:
          - description: string
          - tz: IANA tz name (e.g., "Asia/Kolkata"). Used if start/end are naive/local.
          - attendees: array of email strings
        Returns: { id }
    """),
)
async def mcp_todo_create(
    title: str,
    start: str,
    end: str,
    description: Optional[str] = None,
    tz: Optional[str] = None,
    attendees: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Create a new todo; times are normalized to UTC before persistence
    """
    try:
        parsed = TodoCreateIn(
            title=title,
            start=start,
            end=end,
            description=description,
            tz=tz,
            attendees=attendees or [],
        )
        start_utc = ensure_tz_and_utc(parsed.start, parsed.tz)
        end_utc = ensure_tz_and_utc(parsed.end, parsed.tz)
        now = datetime.now(UTC)

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
    except ValidationError as ve:
        raise ValueError(str(ve))


@mcp.tool(
    name="todo.updateStatus",
    description=_desc("""
        Update a todo status.
        Required:
          - id: string
          - status: one of { "pending", "in-progress", "done", "cancelled" }
        Returns: { ok: boolean }
    """),
)
async def mcp_todo_update_status(
    id: str,
    status: str,
) -> Dict[str, Any]:
    """
    Update status of an existing todo.
    """
    try:
        parsed = StatusUpdateIn(id=id, status=status)
        ok = await update_status(parsed.id, parsed.status)
        return {"ok": ok}
    except ValidationError as ve:
        raise ValueError(str(ve))


@mcp.tool(
    name="calendar.search",
    description=_desc("""
        Search calendar events in a UTC range.
        Inputs:
          - dateFrom: ISO8601 UTC datetime
          - dateTo:   ISO8601 UTC datetime
          - attendees?: array of email strings to filter
        Returns: array of CalEventOut objects.
    """),
)
async def mcp_calendar_search(
    dateFrom: str,
    dateTo: str,
    attendees: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Query calendar events in the given UTC window (optionally filtered by attendees).
    """
    try:
        parsed = CalSearchIn(dateFrom=dateFrom, dateTo=dateTo, attendees=attendees or [])
        events = search_events(parsed.dateFrom, parsed.dateTo, parsed.attendees)
        return [e.model_dump() for e in events]
    except ValidationError as ve:
        raise ValueError(str(ve))


@mcp.tool(
    name="calendar.schedule",
    description=_desc("""
        Create a calendar event (and Meet link when available).
        Required:
          - title: string
          - start: ISO8601 datetime string
          - end:   ISO8601 datetime string
        Optional:
          - attendees: array of email strings
          - description: string
          - tz: IANA tz name (e.g., "Asia/Kolkata")
        Returns: { id: string, meetLink?: string }
    """),
)
async def mcp_calendar_schedule(
    title: str,
    start: str,
    end: str,
    attendees: Optional[List[str]] = None,
    description: Optional[str] = None,
    tz: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Schedule a calendar event; returns provider id and optional Meet link.
    """
    try:
        parsed = CalScheduleIn(
            title=title,
            start=start,
            end=end,
            attendees=attendees or [],
            description=description,
            tz=tz,
        )
        created = schedule_event(
            title=parsed.title,
            start=parsed.start,
            end=parsed.end,
            attendees=parsed.attendees,
            description=parsed.description,
            tz_name=parsed.tz or DEFAULT_TZ,
            calendar_id=os.getenv("GOOGLE_CALENDAR_ID", "primary"),
        )
        return {"id": created.get("id"), "meetLink": created.get("hangoutLink")}
    except ValidationError as ve:
        raise ValueError(str(ve))

@mcp.tool(
    name="time.nowIST",
    description=_desc("""
        Return the current date and time in IST (Asia/Kolkata).
        Optional 'format' can be one of: iso | rfc3339 | pretty | epoch_ms.
        Defaults to iso. Also returns a full JSON bundle when no format is given.
    """),
)
async def time_now_ist(format: Optional[str] = None) -> Dict[str, Any] | str | int:
    """
    Current time in Asia/Kolkata. If 'format' is provided, returns only that field:
      - iso:      e.g., '2025-11-11T12:34:56+05:30'
      - rfc3339:  e.g., '2025-11-11T12:34:56+0530'   (no colon in offset)
      - pretty:   e.g., 'Tue, 11 Nov 2025 12:34:56 PM IST'
      - epoch_ms: e.g., 1731318296000
    Otherwise returns a JSON object with all fields.
    """
    tz = pytz.timezone("Asia/Kolkata")
    now = datetime.now(tz)

    data: Dict[str, Any] = {
        "timezone": "Asia/Kolkata",
        "offset": "+05:30",
        "iso": now.isoformat(timespec="seconds"),            # 2025-11-11T12:34:56+05:30
        "rfc3339": now.strftime("%Y-%m-%dT%H:%M:%S%z"),      # 2025-11-11T12:34:56+0530
        "pretty": now.strftime("%a, %d %b %Y %I:%M:%S %p %Z"),
        "epoch_ms": int(now.timestamp() * 1000),
    }

    if format:
        f = format.lower()
        if f == "iso":
            return data["iso"]
        if f == "rfc3339":
            return data["rfc3339"]
        if f == "pretty":
            return data["pretty"]
        if f in ("epoch", "epoch_ms"):
            return data["epoch_ms"]
        # Unknown format -> return the full bundle so caller can choose
    return data

# (Optional but handy) a simple connectivity check
@mcp.tool(
    name="ping",
    description="Health check: returns 'pong' so the client can verify the MCP server is alive.",
)
async def ping() -> str:
    return "pong"


# -------------------------------------------------------------------
# Entry point — stdio or streamable-http (env selectable)
# -------------------------------------------------------------------
if __name__ == "__main__":
    try:
        # Run index creation using a TEMP client bound to this moment's loop,
        # then immediately close it. No globals reused.
        import asyncio
        asyncio.run(ensure_indexes_safe())
    except Exception as e:
        print("[STARTUP][WARN] ensure_indexes_safe failed:", repr(e))

    # Now start MCP. It will create its own loop; mongo client will be created
    # lazily on that loop via init_mongo() from tool calls.
    transport = os.getenv("MCP_TRANSPORT", "streamable-http").strip().lower()
    if transport == "streamable-http":
        mcp.run(transport="streamable-http")
    else:
        mcp.run(transport="stdio")