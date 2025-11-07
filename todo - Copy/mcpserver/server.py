from __future__ import annotations

import os
import sys
import json
from typing import Any, Dict, List
from datetime import datetime

import anyio
import pytz
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from dotenv import load_dotenv
import logging

from .schemas import (
    TodoSearchIn, TodoCreateIn, TodoOut, StatusUpdateIn,
    CalSearchIn, CalScheduleIn, CalEventOut, ensure_tz_and_utc,
)
from .mongo import ensure_indexes, search_todos, create_todo, update_status
from .models import to_todo_out
from .calendar_google import search_events, schedule_event


# MCP SDK
try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
except Exception:  # pragma: no cover
    Server = None  # type: ignore
    stdio_server = None  # type: ignore


load_dotenv()

logger = logging.getLogger("mcpserver")

DEFAULT_TZ = os.getenv("DEFAULT_TZ", "Asia/Kolkata")


app = FastAPI(title="Meet+ToDo MCP Debug")


def _ok(data: Any):
    return JSONResponse(content=data)


def _err(message: str):
    return JSONResponse(status_code=400, content={"error": True, "message": message})


@app.get("/health")
async def health():
    return {"ok": True}


@app.post("/todo/search")
async def rest_todo_search(payload: Dict[str, Any]):
    try:
        logger.info("HTTP /todo/search payload: %s", payload)
        parsed = TodoSearchIn(**payload)
        df = parsed.dateFrom
        dt = parsed.dateTo
        # leave as-is; search stores ISO UTC strings
        items = await search_todos(df, dt, parsed.text)
        out = [to_todo_out(doc).model_dump() for doc in items]
        logger.debug("/todo/search results count: %d", len(out))
        return _ok(out)
    except ValidationError as ve:
        return _err(str(ve))
    except Exception as e:
        return _err(str(e))


@app.post("/todo/create")
async def rest_todo_create(payload: Dict[str, Any]):
    try:
        logger.info("HTTP /todo/create payload: %s", payload)
        parsed = TodoCreateIn(**payload)
        start_utc = ensure_tz_and_utc(parsed.start, parsed.tz)
        end_utc = ensure_tz_and_utc(parsed.end, parsed.tz)

        now = datetime.utcnow().replace(tzinfo=pytz.UTC)
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
        logger.info("Created todo id: %s", new_id)
        return _ok({"id": new_id})
    except ValidationError as ve:
        return _err(str(ve))
    except Exception as e:
        return _err(str(e))


@app.post("/todo/updateStatus")
async def rest_todo_status(payload: Dict[str, Any]):
    try:
        logger.info("HTTP /todo/updateStatus payload: %s", payload)
        parsed = StatusUpdateIn(**payload)
        ok = await update_status(parsed.id, parsed.status)
        logger.info("Updated status for %s -> %s (ok=%s)", parsed.id, parsed.status, ok)
        return _ok({"ok": ok})
    except ValidationError as ve:
        return _err(str(ve))
    except Exception as e:
        return _err(str(e))


@app.post("/calendar/search")
async def rest_cal_search(payload: Dict[str, Any]):
    try:
        logger.info("HTTP /calendar/search payload: %s", payload)
        parsed = CalSearchIn(**payload)
        events = search_events(parsed.dateFrom, parsed.dateTo, parsed.attendees)
        out = [e.model_dump() for e in events]
        logger.debug("/calendar/search results count: %d", len(out))
        return _ok(out)
    except ValidationError as ve:
        return _err(str(ve))
    except Exception as e:
        return _err(str(e))


@app.post("/calendar/schedule")
async def rest_cal_schedule(payload: Dict[str, Any]):
    try:
        logger.info("HTTP /calendar/schedule payload: %s", payload)
        parsed = CalScheduleIn(**payload)
        # Keep RFC3339 with tz per Google; do not force UTC here
        created = schedule_event(
            title=parsed.title,
            start=parsed.start,
            end=parsed.end,
            attendees=parsed.attendees,
            description=parsed.description,
            tz_name=parsed.tz or DEFAULT_TZ,
            calendar_id=os.getenv("GOOGLE_CALENDAR_ID", "primary"),
        )
        logger.info("Scheduled calendar event id: %s", created.get("id"))
        return _ok({"id": created.get("id"), "meetLink": created.get("hangoutLink")})
    except ValidationError as ve:
        return _err(str(ve))
    except Exception as e:
        return _err(str(e))


# --- MCP server via stdio ---
_mcp_server = None
if Server is not None:
    _mcp_server = Server("meet-todo-mcp")

    @_mcp_server.tool(name="todo.search")
    async def mcp_todo_search(input: Dict[str, Any]) -> Any:
        try:
            logger.info("MCP tool todo.search called with input: %s", input)
            parsed = TodoSearchIn(**(input or {}))
            items = await search_todos(parsed.dateFrom, parsed.dateTo, parsed.text)
            out = [to_todo_out(doc).model_dump() for doc in items]
            logger.debug("todo.search returning %d items", len(out))
            return out
        except ValidationError as ve:
            return {"error": True, "message": str(ve)}
        except Exception as e:
            return {"error": True, "message": str(e)}

    @_mcp_server.tool(name="todo.create")
    async def mcp_todo_create(input: Dict[str, Any]) -> Any:
        try:
            logger.info("MCP tool todo.create called with input: %s", input)
            parsed = TodoCreateIn(**(input or {}))
            start_utc = ensure_tz_and_utc(parsed.start, parsed.tz)
            end_utc = ensure_tz_and_utc(parsed.end, parsed.tz)

            now = datetime.utcnow().replace(tzinfo=pytz.UTC)
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
            logger.info("todo.create inserted id: %s", new_id)
            return {"id": new_id}
        except ValidationError as ve:
            return {"error": True, "message": str(ve)}
        except Exception as e:
            return {"error": True, "message": str(e)}

    @_mcp_server.tool(name="todo.updateStatus")
    async def mcp_todo_update_status(input: Dict[str, Any]) -> Any:
        try:
            logger.info("MCP tool todo.updateStatus called with input: %s", input)
            parsed = StatusUpdateIn(**(input or {}))
            ok = await update_status(parsed.id, parsed.status)
            logger.info("todo.updateStatus finished: %s -> %s", parsed.id, parsed.status)
            return {"ok": ok}
        except ValidationError as ve:
            return {"error": True, "message": str(ve)}
        except Exception as e:
            return {"error": True, "message": str(e)}

    @_mcp_server.tool(name="calendar.search")
    async def mcp_calendar_search(input: Dict[str, Any]) -> Any:
        try:
            logger.info("MCP tool calendar.search called with input: %s", input)
            parsed = CalSearchIn(**(input or {}))
            events = search_events(parsed.dateFrom, parsed.dateTo, parsed.attendees)
            out = [e.model_dump() for e in events]
            logger.debug("calendar.search returning %d events", len(out))
            return out
        except ValidationError as ve:
            return {"error": True, "message": str(ve)}
        except Exception as e:
            return {"error": True, "message": str(e)}

    @_mcp_server.tool(name="calendar.schedule")
    async def mcp_calendar_schedule(input: Dict[str, Any]) -> Any:
        try:
            logger.info("MCP tool calendar.schedule called with input: %s", input)
            parsed = CalScheduleIn(**(input or {}))
            created = schedule_event(
                title=parsed.title,
                start=parsed.start,
                end=parsed.end,
                attendees=parsed.attendees,
                description=parsed.description,
                tz_name=parsed.tz or DEFAULT_TZ,
                calendar_id=os.getenv("GOOGLE_CALENDAR_ID", "primary"),
            )
            logger.info("calendar.schedule created event id: %s", created.get("id"))
            return {"id": created.get("id"), "meetLink": created.get("hangoutLink")}
        except ValidationError as ve:
            return {"error": True, "message": str(ve)}
        except Exception as e:
            return {"error": True, "message": str(e)}


async def _run_stdio() -> None:
    await ensure_indexes()
    if stdio_server is None or _mcp_server is None:
        print("MCP SDK not available. Please install 'mcp' package.", file=sys.stderr)
        sys.exit(1)
    await stdio_server(_mcp_server).serve()


def _run_http() -> None:
    import uvicorn

    port = int(os.getenv("PORT_MCP", "8000"))
    uvicorn.run("mcpserver.server:app", host="0.0.0.0", port=port, reload=False)


if __name__ == "__main__":
    if "--stdio" in sys.argv:
        anyio.run(_run_stdio)
    else:
        # HTTP debug server
        anyio.run(ensure_indexes)
        _run_http()

