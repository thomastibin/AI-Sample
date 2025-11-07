"""Google Calendar helpers: auth, search, schedule with Meet creation."""
from __future__ import annotations
import os
from typing import List, Optional, Dict, Any
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from datetime import datetime
import pytz
import uuid
from dotenv import load_dotenv

load_dotenv()

SCOPES = ["https://www.googleapis.com/auth/calendar.events"]
GOOGLE_TOKEN_PATH = os.getenv("GOOGLE_TOKEN_PATH", ".tokens/google_token.json")
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
GOOGLE_CALENDAR_ID = os.getenv("GOOGLE_CALENDAR_ID", "primary")


def _token_exists():
    return os.path.exists(GOOGLE_TOKEN_PATH)


def _load_creds() -> Optional[Credentials]:
    if not _token_exists():
        return None
    try:
        import json
        with open(GOOGLE_TOKEN_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return Credentials.from_authorized_user_info(data, SCOPES)
    except Exception:
        return None


def _save_creds(creds: Credentials):
    import json
    os.makedirs(os.path.dirname(GOOGLE_TOKEN_PATH) or ".", exist_ok=True)
    with open(GOOGLE_TOKEN_PATH, "w", encoding="utf-8") as f:
        f.write(creds.to_json())


def get_calendar_service():
    """Return an authorized Calendar v3 service, performing an InstalledAppFlow if needed."""
    creds = _load_creds()
    if not creds:
        if not GOOGLE_CLIENT_ID or not GOOGLE_CLIENT_SECRET:
            raise RuntimeError("Missing GOOGLE_CLIENT_ID/GOOGLE_CLIENT_SECRET for OAuth flow")
        flow = InstalledAppFlow.from_client_config(
            {
                "installed": {
                    "client_id": GOOGLE_CLIENT_ID,
                    "client_secret": GOOGLE_CLIENT_SECRET,
                    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                    "token_uri": "https://oauth2.googleapis.com/token",
                }
            },
            SCOPES,
        )
        creds = flow.run_local_server(port=0)
        _save_creds(creds)
    if creds.expired and creds.refresh_token:
        creds.refresh()
        _save_creds(creds)
    service = build("calendar", "v3", credentials=creds, cache_discovery=False)
    return service


def _ensure_rfc3339(dt: datetime, tz: str = "UTC") -> Dict[str, Any]:
    if dt.tzinfo is None:
        local = pytz.timezone(tz)
        dt = local.localize(dt)
    return {"dateTime": dt.isoformat(), "timeZone": tz}


def search_events(date_from: datetime, date_to: datetime, attendees: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    service = get_calendar_service()
    time_min = date_from.astimezone(pytz.UTC).isoformat()
    time_max = date_to.astimezone(pytz.UTC).isoformat()
    try:
        events_result = service.events().list(calendarId=GOOGLE_CALENDAR_ID, timeMin=time_min, timeMax=time_max, singleEvents=True, orderBy="startTime").execute()
        items = events_result.get("items", [])
        out = []
        for ev in items:
            ev_att = [a.get("email") for a in ev.get("attendees", []) if a.get("email")]
            if attendees:
                if not any(a in ev_att for a in attendees):
                    continue
            start = ev.get("start", {}).get("dateTime")
            end = ev.get("end", {}).get("dateTime")
            out.append({
                "id": ev.get("id"),
                "title": ev.get("summary"),
                "start": start,
                "end": end,
                "attendees": ev_att,
                "meetLink": ev.get("hangoutLink"),
            })
        return out
    except HttpError:
        raise


def schedule_event(title: str, start: datetime, end: datetime, attendees: List[str], description: Optional[str], tz: str = "Asia/Kolkata", calendar_id: Optional[str] = None) -> Dict[str, Any]:
    service = get_calendar_service()
    calendar_id = calendar_id or GOOGLE_CALENDAR_ID
    start_obj = _ensure_rfc3339(start, tz)
    end_obj = _ensure_rfc3339(end, tz)
    attendees_payload = [{"email": a} for a in attendees]
    body = {
        "summary": title,
        "description": description or "",
        "start": start_obj,
        "end": end_obj,
        "attendees": attendees_payload,
        "conferenceData": {
            "createRequest": {"requestId": str(uuid.uuid4())}
        },
    }
    try:
        ev = service.events().insert(calendarId=calendar_id, body=body, conferenceDataVersion=1).execute()
        return {"id": ev.get("id"), "hangoutLink": ev.get("hangoutLink")}
    except HttpError as e:
        raise
