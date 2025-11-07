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
import logging

from .schemas import CalEventOut


load_dotenv()

logger = logging.getLogger("calendar_google")

SCOPES = ["https://www.googleapis.com/auth/calendar.events"]


def _ensure_parent_dir(path_str: str) -> None:
    p = pathlib.Path(path_str).expanduser().resolve()
    p.parent.mkdir(parents=True, exist_ok=True)


def get_calendar_service():
    token_path = os.getenv("GOOGLE_TOKEN_PATH", ".tokens/google_token.json")
    client_id = os.getenv("GOOGLE_CLIENT_ID")
    client_secret = os.getenv("GOOGLE_CLIENT_SECRET")

    logger.debug("get_calendar_service called; token_path=%s, client_id_set=%s", token_path, bool(client_id))

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
    logger.debug("Google Calendar service built successfully")
    return service


def _rfc3339(dt: datetime) -> str:
    return dt.astimezone(pytz.UTC).isoformat()


def search_events(date_from: datetime, date_to: datetime, attendees: Optional[List[str]] = None) -> List[CalEventOut]:
    logger.info("Searching Google Calendar from %s to %s attendees=%s", date_from, date_to, attendees)
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

    logger.info("Scheduling Google Calendar event: title=%s start=%s end=%s attendees=%s tz=%s", title, start, end, attendees, tz_name)

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

    logger.info("Created calendar event id=%s", created.get("id"))
    return {
        "id": created.get("id"),
        "hangoutLink": created.get("hangoutLink") or (created.get("conferenceData", {}) or {}).get("entryPoints", [{}])[0].get("uri"),
    }
