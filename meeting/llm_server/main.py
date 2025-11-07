"""LLM server that accepts /chat and routes to MCP tools.

For simplicity this implementation spawns the MCP server subprocess (using the manifest) so the
MCP REST debug endpoints are available. It then talks to those endpoints via HTTP.
In a production-ready version you could use the MCP stdio client to do RPC.
"""
import os
import json
import subprocess
import time
from typing import List, Dict, Any
from fastapi import FastAPI
from pydantic import BaseModel
import httpx
from llm_server.time_utils import parse_human_time_to_utc_window

PORT_LLM = int(os.getenv("PORT_LLM", "8001"))
PORT_MCP = int(os.getenv("PORT_MCP", "8000"))
MCP_MANIFEST = os.path.join(os.path.dirname(__file__), "mcp_manifest.json")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

app = FastAPI(title="LLM Meet+ToDo Agent")


class ChatIn(BaseModel):
    message: str


def spawn_mcp_from_manifest(manifest_path: str):
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            m = json.load(f)
    except Exception:
        return None
    cmd = [m["transport"]["command"]] + m["transport"].get("args", [])
    # Start as detached process; it will start REST server on PORT_MCP
    proc = subprocess.Popen(cmd)
    # give it a moment to start
    time.sleep(1.0)
    return proc


def _mcp_post(path: str, payload: Dict[str, Any]):
    url = f"http://127.0.0.1:{PORT_MCP}{path}"
    r = httpx.post(url, json=payload, timeout=30.0)
    r.raise_for_status()
    return r.json()


def simple_intent_parse(message: str) -> Dict[str, Any]:
    """Fallback simple parser to extract intent and rough fields.
    Returns: {intent: 'schedule'|'show_todos'|'list_meetings', attendees:[], startText, title}
    """
    msg = message.lower()
    if "todo" in msg or "todos" in msg or "to-do" in msg:
        return {"intent": "show_todos", "startText": "next week"}
    if "meeting" in msg or "gmeet" in msg or "google meet" in msg or "gmeet" in msg:
        # try to find emails
        import re
        emails = re.findall(r"[\w\.\-]+@[\w\.\-]+", message)
        # crude time extraction: take substring after 'at' or 'on'
        if " at " in msg:
            part = message.split(" at ", 1)[1]
        elif " on " in msg:
            part = message.split(" on ", 1)[1]
        else:
            part = "next tuesday at 3pm"
        return {"intent": "schedule", "attendees": emails, "startText": part, "title": "Meeting"}
    if "meeting" in msg and ("next" in msg or "48h" in msg or "48 h" in msg):
        return {"intent": "list_meetings", "startText": "now", "duration_hours": 48}
    return {"intent": "unknown"}


@app.on_event("startup")
def startup():
    # spawn MCP server via manifest so REST endpoints are available.
    spawn_mcp_from_manifest(MCP_MANIFEST)


@app.post("/chat")
async def chat(body: ChatIn):
    try:
        plan = simple_intent_parse(body.message)
        intent = plan.get("intent")
        if intent == "schedule":
            start_utc, end_utc, tz = parse_human_time_to_utc_window(plan.get("startText", "next tuesday at 3pm"))
            # Check conflicts: call calendar.search for window +/-30min
            from datetime import timedelta
            window_from = (start_utc - timedelta(minutes=30)).isoformat()
            window_to = (end_utc + timedelta(minutes=30)).isoformat()
            csearch = _mcp_post("/calendar/search", {"dateFrom": window_from, "dateTo": window_to, "attendees": plan.get("attendees", [])})
            if isinstance(csearch, dict) and csearch.get("error"):
                return csearch
            if csearch:
                # conflict
                return {"conflict": True, "events": csearch, "suggestions": []}
            # schedule
            schedule_payload = {
                "title": plan.get("title", "Meeting"),
                "start": start_utc.isoformat(),
                "end": end_utc.isoformat(),
                "attendees": plan.get("attendees", []),
                "description": "Scheduled via Meet+ToDo Agent",
                "tz": tz,
            }
            sched = _mcp_post("/calendar/schedule", schedule_payload)
            if sched.get("error"):
                return sched
            # create todo
            todo_payload = {"title": plan.get("title", "Meeting"), "start": start_utc.isoformat(), "end": end_utc.isoformat(), "attendees": plan.get("attendees", []), "description": "Auto-created from calendar event", "tz": tz}
            t = _mcp_post("/todo/create", todo_payload)
            return {"ok": True, "meetLink": sched.get("event", {}).get("hangoutLink"), "eventId": sched.get("event", {}).get("id"), "todoId": t.get("id")}
        elif intent == "show_todos":
            # parse timeframe
            start_utc, end_utc, tz = parse_human_time_to_utc_window(plan.get("startText", "next week"))
            res = _mcp_post("/todo/search", {"dateFrom": start_utc.isoformat(), "dateTo": end_utc.isoformat()})
            return {"ok": True, "todos": res}
        elif intent == "list_meetings":
            from datetime import datetime, timedelta
            start = datetime.utcnow().isoformat()
            to = (datetime.utcnow() + timedelta(hours=48)).isoformat()
            res = _mcp_post("/calendar/search", {"dateFrom": start, "dateTo": to, "attendees": []})
            return {"ok": True, "events": res}
        else:
            return {"error": True, "message": "Could not understand intent"}
    except Exception as e:
        return {"error": True, "message": str(e)}


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT_LLM)
