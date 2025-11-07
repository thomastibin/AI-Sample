"""Utilities to parse human time expressions into UTC windows."""
from datetime import datetime, timedelta
import dateparser
import pytz
from typing import Tuple


DEFAULT_TZ = "Asia/Kolkata"


def parse_human_time_to_utc_window(text: str, default_tz: str = DEFAULT_TZ) -> Tuple[datetime, datetime, str]:
    """Parse a natural language time to a UTC-aware start/end tuple.

    Returns (start_utc, end_utc, tz)
    Default duration is 30 minutes if end not specified.
    """
    settings = {"TIMEZONE": default_tz, "RETURN_AS_TIMEZONE_AWARE": True}
    dt = dateparser.parse(text, settings=settings)
    if not dt:
        raise ValueError(f"Could not parse time: {text}")
    # Default duration 30 minutes
    start_local = dt
    end_local = start_local + timedelta(minutes=30)
    # Convert to UTC
    start_utc = start_local.astimezone(pytz.UTC)
    end_utc = end_local.astimezone(pytz.UTC)
    return start_utc, end_utc, default_tz
