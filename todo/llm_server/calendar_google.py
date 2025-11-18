from __future__ import annotations

import os
import json
import pathlib
import uuid
from datetime import datetime
from datetime import timedelta
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


def search_events(
    date_from: datetime,
    date_to: datetime,
    attendees: Optional[List[str]] = None
) -> List[CalEventOut]:
    dbg(f"[CAL] search_events from={date_from} to={date_to} attendees={attendees}")
    service = get_calendar_service()
    calendar_id = os.getenv("GOOGLE_CALENDAR_ID", "primary")
    default_tz = os.getenv("DEFAULT_TZ", "Asia/Kolkata")
    tz = pytz.timezone(default_tz)

    def _parse_dt(edge: Dict[str, Any]) -> datetime:
        """
        Google returns either {"dateTime": "..."} or {"date": "YYYY-MM-DD"}.
        Return timezone-aware UTC datetime for both forms.
        """
        if not edge:
            return datetime.min.replace(tzinfo=pytz.UTC)

        dt_str = edge.get("dateTime")
        if dt_str:
            # Normalize 'Z' for fromisoformat()
            if dt_str.endswith("Z"):
                dt_str = dt_str.replace("Z", "+00:00")
            return datetime.fromisoformat(dt_str).astimezone(pytz.UTC)

        d_str = edge.get("date")
        if d_str:
            # Interpret all-day as local midnight in DEFAULT_TZ, then to UTC
            local_midnight = tz.localize(datetime.fromisoformat(d_str))
            return local_midnight.astimezone(pytz.UTC)

        return datetime.min.replace(tzinfo=pytz.UTC)

    results: List[CalEventOut] = []
    page_token: Optional[str] = None

    while True:
        events_result = (
            service.events()
            .list(
                calendarId=calendar_id,
                timeMin=_rfc3339(date_from),
                timeMax=_rfc3339(date_to),   # Google treats this as exclusive upper bound
                singleEvents=True,           # expand recurrences
                orderBy="startTime",
                pageToken=page_token,
                maxResults=250,
            )
            .execute()
        )
        items = events_result.get("items", []) or []

        for ev in items:
            # Collect emails from attendees + organizer + creator
            emails: List[str] = []
            emails.extend([a.get("email") for a in ev.get("attendees", []) if a.get("email")])
            org = (ev.get("organizer") or {}).get("email")
            if org:
                emails.append(org)
            crt = (ev.get("creator") or {}).get("email")
            if crt:
                emails.append(crt)
            # de-dup + normalize
            emails = sorted({(e or "").strip() for e in emails if e})

            # Apply attendees filter if provided (match any)
            if attendees:
                target = {e.lower() for e in attendees if e}
                if not any((e or "").lower() in target for e in emails):
                    continue

            start_dt = _parse_dt(ev.get("start"))
            end_dt = _parse_dt(ev.get("end"))

            # Prefer hangoutLink, else try conferenceData.entryPoints[0].uri
            conf = ev.get("conferenceData") or {}
            entry_points = conf.get("entryPoints") or []
            meet_link = ev.get("hangoutLink") or (entry_points[0].get("uri") if entry_points else None)

            results.append(
                CalEventOut(
                    id=ev.get("id"),
                    title=ev.get("summary", "") or "",
                    start=start_dt,
                    end=end_dt,
                    attendees=emails,
                    meetLink=meet_link,
                )
            )

        page_token = events_result.get("nextPageToken")
        if not page_token:
            break

    dbg("[CAL] search_events total_count:", len(results))
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

def find_free_slots(date_from: datetime, date_to: datetime, min_minutes: int = 30) -> List[Dict[str, Any]]:
    """
    Compute free slots by subtracting busy intervals from [date_from, date_to].
    Returns list of {"start": iso, "end": iso} in local tz of date_from.
    """
    # Get busy events (reuse existing list() logic)
    events = search_events(date_from, date_to, attendees=None)
    # Build busy intervals in UTC
    busy = []
    for e in events:
        busy.append((e.start.astimezone(pytz.UTC), e.end.astimezone(pytz.UTC)))
    busy.sort(key=lambda t: t[0])

    # Merge busy intervals
    merged = []
    for s, e in busy:
        if not merged or s > merged[-1][1]:
            merged.append([s, e])
        else:
            merged[-1][1] = max(merged[-1][1], e)

    # Scan for gaps
    cur = date_from.astimezone(pytz.UTC)
    end = date_to.astimezone(pytz.UTC)
    gaps = []
    for s, e in merged:
        if cur < s:
            gaps.append((cur, s))
        cur = max(cur, e)
    if cur < end:
        gaps.append((cur, end))

    # Keep only gaps >= min_minutes, return in ISO
    min_delta = timedelta(minutes=min_minutes)
    out = []
    for g0, g1 in gaps:
        if (g1 - g0) >= min_delta:
            out.append({"start": g0.isoformat(), "end": g1.isoformat()})
    return out