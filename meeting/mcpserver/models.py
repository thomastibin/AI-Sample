"""Mongo document shapes and helpers."""
from dataclasses import dataclass
from typing import List, Dict, Any
from datetime import datetime
import pytz


def isoutc(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = pytz.UTC.localize(dt)
    return dt.astimezone(pytz.UTC).isoformat()


def to_todo_out(doc: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": str(doc.get("_id")),
        "title": doc.get("title"),
        "start": doc.get("start"),
        "end": doc.get("end"),
        "status": doc.get("status", "pending"),
    }


def overlap(a_start: datetime, a_end: datetime, b_start: datetime, b_end: datetime) -> bool:
    """Return True if intervals overlap (a_start < b_end and b_start < a_end)."""
    return a_start < b_end and b_start < a_end
