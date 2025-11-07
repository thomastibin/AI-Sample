from __future__ import annotations

from typing import Tuple
from datetime import datetime, timedelta

import pytz
import dateparser
from dateparser.search import search_dates


def parse_human_time_to_utc_window(text: str, default_tz: str = "Asia/Kolkata") -> Tuple[datetime, datetime, str]:
    """Parse human natural language time into (start_utc, end_utc, tz_name).

    - Accepts phrases like "next tuesday at 3pm".
    - If end not specified, assume 30 minutes duration.
    - Returns timezone-aware UTC datetimes.
    """
    s = (text or "").strip()
    settings = {
        "TIMEZONE": default_tz,
        "RETURN_AS_TIMEZONE_AWARE": True,
        "PREFER_DATES_FROM": "future",
    }
    dt = dateparser.parse(s, settings=settings)
    if dt is None:
        # Fallback: search for any date/time in the string
        found = search_dates(s, settings=settings)
        if found and len(found) > 0:
            dt = found[0][1]
    if dt is None:
        raise ValueError("Could not parse time expression")

    # If parsed dt is aware, keep its tz; else apply default tz
    if dt.tzinfo is None:
        dt = pytz.timezone(default_tz).localize(dt)

    # Heuristic: 30 minutes default duration
    end_local = dt + timedelta(minutes=30)

    start_utc = dt.astimezone(pytz.UTC)
    end_utc = end_local.astimezone(pytz.UTC)
    return start_utc, end_utc, (dt.tzinfo.zone if hasattr(dt.tzinfo, 'zone') else default_tz)
