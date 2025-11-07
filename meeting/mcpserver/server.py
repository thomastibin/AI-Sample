"""MCP server exposing tools (todo.*, calendar.*) and a FastAPI debug REST app.

This module will attempt to register MCP stdio tools when run with --stdio.
It always starts a FastAPI app (debug endpoints) on PORT_MCP so other processes can call REST for simplicity.
"""
import os
import asyncio
import os
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import json
from typing import Any, Dict

from . import mongo, models, schemas, calendar_google

PORT_MCP = int(os.getenv("PORT_MCP", "8000"))

app = FastAPI(title="MCP Meet+ToDo (debug REST)")


@app.get("/health")
async def health():
    return {"ok": True}


@app.post("/todo/search")
async def todo_search(payload: Dict[str, Any]):
    try:
        obj = schemas.TodoSearchIn.model_validate(payload)
    except Exception as e:
        return {"error": True, "message": str(e)}
    docs = await mongo.search_todos(date_from=obj.dateFrom.isoformat() if obj.dateFrom else None, date_to=obj.dateTo.isoformat() if obj.dateTo else None, text=obj.text)
    return [models.to_todo_out(d) for d in docs]


@app.post("/todo/create")
async def todo_create(payload: Dict[str, Any]):
    try:
        obj = schemas.TodoCreateIn.model_validate(payload)
    except Exception as e:
        return {"error": True, "message": str(e)}
    data = {
        "title": obj.title,
        "description": obj.description,
        "start": obj.start.isoformat(),
        "end": obj.end.isoformat(),
        "tz": obj.tz,
        "attendees": obj.attendees,
        "status": "pending",
        "source": "mcp",
    }
    tid = await mongo.create_todo(data)
    return {"ok": True, "id": tid}


@app.post("/todo/updateStatus")
async def todo_update_status(payload: Dict[str, Any]):
    try:
        obj = schemas.StatusUpdateIn.model_validate(payload)
    except Exception as e:
        return {"error": True, "message": str(e)}
    ok = await mongo.update_status(obj.id, obj.status)
    return {"ok": ok}


@app.post("/calendar/search")
async def calendar_search(payload: Dict[str, Any]):
    try:
        obj = schemas.CalSearchIn.model_validate(payload)
    except Exception as e:
        return {"error": True, "message": str(e)}
    try:
        events = calendar_google.search_events(obj.dateFrom, obj.dateTo, obj.attendees)
        return events
    except Exception as e:
        return {"error": True, "message": str(e)}


@app.post("/calendar/schedule")
async def calendar_schedule(payload: Dict[str, Any]):
    try:
        obj = schemas.CalScheduleIn.model_validate(payload)
    except Exception as e:
        return {"error": True, "message": str(e)}
    try:
        res = calendar_google.schedule_event(obj.title, obj.start, obj.end, obj.attendees, obj.description, obj.tz)
        return {"ok": True, "event": res}
    except Exception as e:
        return {"error": True, "message": str(e)}


async def _startup_indexes():
    try:
        await mongo.ensure_indexes()
    except Exception:
        pass


def _run_uvicorn():
    uvicorn.run(app, host="0.0.0.0", port=PORT_MCP)


def main_stdio_loop():
    # Try to register MCP stdio Server if available. If not, just run REST.
    try:
        import mcp
        from mcp.server import Server
        srv = Server()
        # Registering tools would typically go here; for simplicity we rely on REST endpoints as canonical
        print("MCP stdio server started (placeholder).")
        srv.serve()
    except Exception:
        print("MCP stdio not available or failed to start; continuing with REST debug server only.")


if __name__ == '__main__':
    import sys
    loop = asyncio.get_event_loop()
    loop.create_task(_startup_indexes())
    if "--stdio" in sys.argv:
        # spawn background REST and attempt stdio
        from threading import Thread

        t = Thread(target=_run_uvicorn, daemon=True)
        t.start()
        main_stdio_loop()
    else:
        _run_uvicorn()
