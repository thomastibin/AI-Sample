from __future__ import annotations

import os
import json
import pathlib
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

import pytz
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from google.auth.transport.requests import Request
from dotenv import load_dotenv
import sys

from .schemas import CalEventOut


load_dotenv()

# Avoid emitting to stdout/stderr when running as MCP stdio child.
RUNNING_AS_STDIO = "--stdio" in sys.argv
_DEBUG_LOG_PATH = os.getenv("MCP_DEBUG_LOG", ".mcp_debug.log")

def dbg(*args, **kwargs):
    file = kwargs.pop("file", sys.stderr)
    if RUNNING_AS_STDIO:
        try:
            with open(_DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
                print(*args, file=f, **kwargs)
        except Exception:
            pass
    else:
        print(*args, file=file, **kwargs)


SCOPES = ["https://www.googleapis.com/auth/calendar.events"]


def _ensure_parent_dir(path_str: str) -> None:
    p = pathlib.Path(path_str).expanduser().resolve()
    p.parent.mkdir(parents=True, exist_ok=True)


def get_calendar_service():
    token_path = os.getenv("GOOGLE_TOKEN_PATH", ".tokens/google_token.json")
    client_id = os.getenv("GOOGLE_CLIENT_ID")
    client_secret = os.getenv("GOOGLE_CLIENT_SECRET")

    # get_calendar_service called
    dbg(f"[CAL] get_calendar_service token_path={token_path} client_id_set={bool(client_id)}")

    creds: Optional[Credentials] = None
    if os.path.exists(token_path):
        creds = Credentials.from_authorized_user_file(token_path, SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            # Desktop flow
            flow = InstalledAppFlow.from_client_config(
                {
                    "installed": {
                        "client_id": client_id,
                        "client_secret": client_secret,
                        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                        "token_uri": "https://oauth2.googleapis.com/token",
                        "redirect_uris": [
                            "http://localhost"
                        ],
                    }
                },
                scopes=SCOPES,
            )
            creds = flow.run_local_server(port=0)
        _ensure_parent_dir(token_path)
        with open(token_path, "w", encoding="utf-8") as f:
            f.write(creds.to_json())

    service = build("calendar", "v3", credentials=creds, cache_discovery=False)
    dbg("[CAL] Google Calendar service ready")
    return service


def _rfc3339(dt: datetime) -> str:
    return dt.astimezone(pytz.UTC).isoformat()


def search_events(date_from: datetime, date_to: datetime, attendees: Optional[List[str]] = None) -> List[CalEventOut]:
    dbg(f"[CAL] search_events from={date_from} to={date_to} attendees={attendees}")
    service = get_calendar_service()
    calendar_id = os.getenv("GOOGLE_CALENDAR_ID", "primary")
    events_result = (
        service.events()
        .list(
            calendarId=calendar_id,
            timeMin=_rfc3339(date_from),
            timeMax=_rfc3339(date_to),
            singleEvents=True,
            orderBy="startTime",
        )
        .execute()
    )
    items = events_result.get("items", [])

    results: List[CalEventOut] = []
    for ev in items:
        start_dt = ev.get("start", {}).get("dateTime")
        end_dt = ev.get("end", {}).get("dateTime")
        emails = [a.get("email") for a in ev.get("attendees", []) if a.get("email")]
        if attendees:
            # Filter only events that include any of the attendees
            if not any(e in emails for e in attendees):
                continue
        # Normalize 'Z' to '+00:00' for fromisoformat compatibility
        if isinstance(start_dt, str) and start_dt.endswith('Z'):
            start_dt = start_dt.replace('Z', '+00:00')
        if isinstance(end_dt, str) and end_dt.endswith('Z'):
            end_dt = end_dt.replace('Z', '+00:00')
        results.append(
            CalEventOut(
                id=ev.get("id"),
                title=ev.get("summary", ""),
                start=datetime.fromisoformat(start_dt),
                end=datetime.fromisoformat(end_dt),
                attendees=emails,
                meetLink=ev.get("hangoutLink") or (ev.get("conferenceData", {}) or {}).get("entryPoints", [{}])[0].get("uri"),
            )
        )
    return results





def schedule_event(
    title: str,
    start: datetime,
    end: datetime,
    attendees: List[str],
    description: Optional[str],
    tz_name: str,
    calendar_id: Optional[str] = None,
) -> Dict[str, Any]:
    service = get_calendar_service()
    calendar_id = calendar_id or os.getenv("GOOGLE_CALENDAR_ID", "primary")

    body = {
        "summary": title,
        "description": description or "",
        "start": {"dateTime": start.isoformat(), "timeZone": tz_name},
        "end": {"dateTime": end.isoformat(), "timeZone": tz_name},
        "attendees": [{"email": a} for a in attendees],
        "conferenceData": {
            "createRequest": {"requestId": str(uuid.uuid4())}
        },
    }

    # scheduling event for provided details

    created = (
        service.events()
        .insert(
            calendarId=calendar_id,
            body=body,
            conferenceDataVersion=1,
            sendUpdates="all",
        )
        .execute()
    )

    return {
        "id": created.get("id"),
        "hangoutLink": created.get("hangoutLink") or (created.get("conferenceData", {}) or {}).get("entryPoints", [{}])[0].get("uri"),
    }
