from __future__ import annotations

import inspect
import sys, pathlib, os, uuid
from typing import Any, Dict, List, Optional, Tuple, Callable, Awaitable
from datetime import datetime

import pytz
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.encoders import jsonable_encoder
from pydantic import ValidationError
from dotenv import load_dotenv

# ---- your app modules ----
from .schemas import (
    TodoSearchIn, TodoCreateIn, StatusUpdateIn,
    CalSearchIn, CalScheduleIn, ensure_tz_and_utc,
)
from .mongo import ensure_indexes, search_todos, create_todo, update_status
from .models import to_todo_out
from .calendar_google import search_events, schedule_event

# ---- MCP core ----
from mcp.server.fastmcp import FastMCP
from starlette.responses import RedirectResponse
# from starlette.requests import Request
from mcp.server.streamable_http_manager import StreamableHTTPServerTransport

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# # ---- Transport: STREAMABLE HTTP (version-tolerant import) ----
# TransportCls = None
# for path in [
#     "mcp.transport.http",                # modern path
#     "mcp.server.transport.streamable_http",  # some older wheels
#     "mcp.server.streamable_http",        # rare older path
# ]:
#     try:
#         mod = __import__(path, fromlist=["StreamableHTTPServerTransport"])
#         TransportCls = getattr(mod, "StreamableHTTPServerTransport")
#         break
#     except Exception:
#         continue

# if TransportCls is None:
#     raise RuntimeError(
#         "StreamableHTTPServerTransport not found in your MCP install. "
#         "Try: uv pip install --upgrade mcp"
#     )
# -------------------------------------------------------------------
# Environment / constants
# -------------------------------------------------------------------
load_dotenv()
DEFAULT_TZ = os.getenv("DEFAULT_TZ", "Asia/Kolkata")
_DEBUG_LOG_PATH = os.getenv("MCP_DEBUG_LOG", ".mcp_debug.log")


def dbg(*args, **kwargs):
    try:
        with open(_DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
            print(*args, file=f, **kwargs)
    except Exception:
        pass


# -------------------------------------------------------------------
# FastAPI app
# -------------------------------------------------------------------
app = FastAPI(title="Meet+ToDo MCP (Streamable HTTP)")
# If you want Cursor / local tools to hit it without CORS pain:
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # or restrict to ["http://localhost:8000", ...]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def _startup_ensure_indexes() -> None:
    try:
        await ensure_indexes()
        dbg("[STARTUP] ensure_indexes() OK")
    except Exception as e:
        dbg("[STARTUP] ensure_indexes() FAILED:", repr(e))


def _ok(data: Any):
    return JSONResponse(content=jsonable_encoder(data))


def _err(message: str):
    return JSONResponse(status_code=400, content={"error": True, "message": message})


@app.get("/health")
async def health():
    return {"ok": True}


# ----------------------- REST: ToDo -----------------------
@app.post("/todo/search")
async def rest_todo_search(payload: Dict[str, Any]):
    try:
        dbg("[HTTP] /todo/search payload:", payload)
        parsed = TodoSearchIn(**payload)
        items = await search_todos(parsed.dateFrom, parsed.dateTo, parsed.text)
        out = [to_todo_out(doc).model_dump() for doc in items]
        dbg("[HTTP] /todo/search returning count:", len(out))
        return _ok(out)
    except ValidationError as ve:
        return _err(str(ve))
    except Exception as e:
        return _err(str(e))


@app.post("/todo/create")
async def rest_todo_create(payload: Dict[str, Any]):
    try:
        dbg("[HTTP] /todo/create payload:", payload)
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
        dbg("[HTTP] /todo/create created id:", new_id)
        return _ok({"id": new_id})
    except ValidationError as ve:
        return _err(str(ve))
    except Exception as e:
        return _err(str(e))


@app.post("/todo/updateStatus")
async def rest_todo_status(payload: Dict[str, Any]):
    try:
        dbg("[HTTP] /todo/updateStatus payload:", payload)
        parsed = StatusUpdateIn(**payload)
        ok = await update_status(parsed.id, parsed.status)
        dbg("[HTTP] /todo/updateStatus result:", ok)
        return _ok({"ok": ok})
    except ValidationError as ve:
        return _err(str(ve))
    except Exception as e:
        return _err(str(e))


# ---------------------- REST: Calendar --------------------
@app.post("/calendar/search")
async def rest_cal_search(payload: Dict[str, Any]):
    try:
        dbg("[HTTP] /calendar/search payload:", payload)
        parsed = CalSearchIn(**payload)
        events = search_events(parsed.dateFrom, parsed.dateTo, parsed.attendees)
        out = [e.model_dump() for e in events]
        dbg("[HTTP] /calendar/search returning count:", len(out))
        return _ok(out)
    except ValidationError as ve:
        return _err(str(ve))
    except Exception as e:
        return _err(str(e))


@app.post("/calendar/schedule")
async def rest_cal_schedule(payload: Dict[str, Any]):
    try:
        dbg("[HTTP] /calendar/schedule payload:", payload)
        parsed = CalScheduleIn(**payload)
        created = schedule_event(
            title=parsed.title,
            start=parsed.start,
            end=parsed.end,
            attendees=parsed.attendees,
            description=parsed.description,
            tz_name=parsed.tz or DEFAULT_TZ,
            calendar_id=os.getenv("GOOGLE_CALENDAR_ID", "primary"),
        )
        dbg("[HTTP] /calendar/schedule created id:", created.get("id"))
        return _ok({"id": created.get("id"), "meetLink": created.get("hangoutLink")})
    except ValidationError as ve:
        return _err(str(ve))
    except Exception as e:
        return _err(str(e))


# -------------------------------------------------------------------
# MCP tools (FastMCP)
# -------------------------------------------------------------------
mcp = FastMCP("meet-todo-mcp")

def _desc(s: str) -> str:
    return " ".join(s.strip().split())


@mcp.tool(
    name="todo.search",
    description=_desc("""
        Search todos in a UTC time window.
        Inputs: dateFrom (ISO8601), dateTo (ISO8601), text? (string filter).
        Returns: TodoOut[]
    """),
)
async def mcp_todo_search(
    dateFrom: str,
    dateTo: str,
    text: Optional[str] = None,
) -> List[Dict[str, Any]]:
    parsed = TodoSearchIn(dateFrom=dateFrom, dateTo=dateTo, text=text)
    items = await search_todos(parsed.dateFrom, parsed.dateTo, parsed.text)
    return [to_todo_out(doc).model_dump() for doc in items]


@mcp.tool(
    name="todo.create",
    description=_desc("""
        Create a todo/event. Required: title, start, end.
        Optional: description, tz (IANA), attendees[] (emails).
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
    return {"id": new_id}


@mcp.tool(
    name="todo.updateStatus",
    description=_desc("""
        Update a todo status. Required: id, status ∈ {pending,in-progress,done,cancelled}.
        Returns: { ok }
    """),
)
async def mcp_todo_update_status(id: str, status: str) -> Dict[str, Any]:
    parsed = StatusUpdateIn(id=id, status=status)
    ok = await update_status(parsed.id, parsed.status)
    return {"ok": ok}


@mcp.tool(
    name="calendar.search",
    description=_desc("""
        Search calendar events in a UTC range. Inputs: dateFrom, dateTo, attendees?[].
        Returns: CalEventOut[]
    """),
)
async def mcp_calendar_search(
    dateFrom: str,
    dateTo: str,
    attendees: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    parsed = CalSearchIn(dateFrom=dateFrom, dateTo=dateTo, attendees=attendees or [])
    events = search_events(parsed.dateFrom, parsed.dateTo, parsed.attendees)
    return [e.model_dump() for e in events]


@mcp.tool(
    name="calendar.schedule",
    description=_desc("""
        Create a calendar event. Required: title, start, end.
        Optional: attendees[] (emails), description, tz (IANA).
        Returns: { id, meetLink? }
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


# ------------------------------------------------------------------------------
# Mount Streamable HTTP transport
# ------------------------------------------------------------------------------
def mount_mcp_streamable_http(app: FastAPI, mcp_app: FastMCP, base_path: str = "/mcp") -> None:
    server = mcp_app._mcp_server

    # 1) Build transport (positional arg, see section A)
    session_id = os.getenv("MCP_SESSION_ID") or str(uuid.uuid4())
    transport = StreamableHTTPServerTransport(session_id)

    # 2) Stream endpoint (GET {base_path}/) — ASGI app
    async def stream_asgi(scope, receive, send):
        # quick OPTIONS handler for CORS preflight
        if scope.get("type") == "http" and scope.get("method") == "OPTIONS":
            await send({"type": "http.response.start", "status": 204, "headers": []})
            await send({"type": "http.response.body", "body": b""})
            return

        if scope.get("type") != "http":
            await send({"type": "http.response.start", "status": 404, "headers": []})
            await send({"type": "http.response.body", "body": b"Not Found"})
            return

        started = False

        async def logged_send(message):
            nonlocal started
            if message["type"] == "http.response.start":
                started = True
                print("[stream] response.start:", message.get("status"))
            elif message["type"] == "http.response.body":
                body = message.get("body") or b""
                more = message.get("more_body")
                print(f"[stream] response.body: {len(body)} bytes, more={more}")
            await send(message)

        sig = inspect.signature(transport.connect)
        params = list(sig.parameters)
        print("[stream_asgi] transport.connect params:", params)

        try:
            if len(params) == 0:
                async with transport.connect() as (read, write):
                    await server.run(read, write, server.create_initialization_options())
            else:  # len(params) == 3  (scope, receive, send)
                async with transport.connect(scope, receive, logged_send) as (read, write):
                    await server.run(read, write, server.create_initialization_options())
            print("[stream_asgi] transport.connect params2:", params)
            
        except Exception as e:
            # If transport failed before starting the response, return a 500 so client isn't left hanging.
            if not started:
                await send({"type": "http.response.start", "status": 500, "headers": []})
                await send({"type": "http.response.body", "body": str(e).encode()})
            raise


    app.mount(base_path + "/", stream_asgi)

    # 3) Write endpoint (POST {base_path}/messages) — name differs across versions
    post_handler = getattr(transport, "handle_post_message", None) \
    or getattr(transport, "post_message_app", None) \
    or getattr(transport, "asgi_post_message", None)

    # Log what we found so we know what’s mounted
    print("[mcp] transport writer handler:",
        "handle_post_message" if getattr(transport, "handle_post_message", None) else
        "post_message_app"    if getattr(transport, "post_message_app", None)    else
        "asgi_post_message"   if getattr(transport, "asgi_post_message", None)   else
        "NONE")

    if post_handler:
        # Wrap to log every hit
        async def _messages_logger(scope, receive, send):
            if scope.get("type") == "http" and scope.get("method") == "POST":
                print("[mcp] POST /mcp/messages hit")
            return await post_handler(scope, receive, send)

        # (A) Real writer endpoint
        app.mount("/mcp/messages", _messages_logger)

        # (B) Compatibility shim: POST /mcp → /mcp/messages
        from fastapi import Request
        from fastapi.responses import JSONResponse
        import json

        @app.post("/mcp", include_in_schema=False)
        async def _compat_post_mcp(request: Request):
            scope = request.scope.copy()
            scope["path"] = "/mcp/messages"
            receive = request._receive
            sent = []

            async def _send(message):
                sent.append(message)

            await post_handler(scope, receive, _send)

            status = 200
            body = b""
            for m in sent:
                if m["type"] == "http.response.start":
                    status = m.get("status", 200)
                elif m["type"] == "http.response.body":
                    body += m.get("body", b"")
            try:
                return JSONResponse(content=json.loads(body or b"{}"), status_code=status)
            except Exception:
                return JSONResponse(content={"ok": True}, status_code=status)
    else:
        print("[mcp] WARNING: transport exposes NO writer handler; "
            "streamable_http clients will hang (no /mcp/messages).")
try:
    mount_mcp_streamable_http(app, mcp, base_path="/mcp")
except Exception as e:
    # Fall back silently if the installed MCP SDK lacks HTTP writer support
    dbg("[MCP] Streamable HTTP mount skipped:", repr(e))

# Convenience redirect for GET /mcp -> /mcp/
@app.get("/mcp")
async def _mcp_redirect():
    return RedirectResponse(url="/mcp/")

# Simple HTTP shim to call tools without full MCP protocol
@app.post("/mcp-call")
async def http_mcp_call(payload: Dict[str, Any]):
    try:
        tool = payload.get("tool") or payload.get("name")
        params = payload.get("params") or payload.get("input") or {}
        if not tool:
            return _err("Missing 'tool' in payload")

        if tool == "todo.search":
            parsed = TodoSearchIn(**params)
            items = await search_todos(parsed.dateFrom, parsed.dateTo, parsed.text)
            return _ok([to_todo_out(doc).model_dump() for doc in items])

        if tool == "todo.create":
            parsed = TodoCreateIn(**params)
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
            return _ok({"id": new_id})

        if tool == "todo.updateStatus":
            parsed = StatusUpdateIn(**params)
            ok = await update_status(parsed.id, parsed.status)
            return _ok({"ok": ok})

        if tool == "calendar.search":
            parsed = CalSearchIn(**params)
            events = search_events(parsed.dateFrom, parsed.dateTo, parsed.attendees)
            return _ok([e.model_dump() for e in events])

        if tool == "calendar.schedule":
            parsed = CalScheduleIn(**params)
            created = schedule_event(
                title=parsed.title,
                start=parsed.start,
                end=parsed.end,
                attendees=parsed.attendees,
                description=parsed.description,
                tz_name=parsed.tz or DEFAULT_TZ,
                calendar_id=os.getenv("GOOGLE_CALENDAR_ID", "primary"),
            )
            return _ok({"id": created.get("id"), "meetLink": created.get("hangoutLink")})

        return _err(f"Unknown tool: {tool}")
    except ValidationError as ve:
        return _err(str(ve))
    except Exception as e:
        return _err(str(e))
# -------------------------------------------------------------------
# Dev runner
# -------------------------------------------------------------------
def _run_http() -> None:
    import uvicorn
    port = int(os.getenv("PORT_MCP", "8000"))
    uvicorn.run("mcpserver.server:app", host="0.0.0.0", port=port, reload=False)


if __name__ == "__main__":
    _run_http()
