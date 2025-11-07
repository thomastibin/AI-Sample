from __future__ import annotations

from typing import Any, Dict
from datetime import datetime

from pydantic import BaseModel

from .schemas import TodoOut


class TodoDoc(BaseModel):
    _id: Any
    title: str
    description: str | None = None
    start: str  # UTC ISO
    end: str    # UTC ISO
    tz: str
    attendees: list[str] = []
    status: str = "pending"
    createdAt: str
    updatedAt: str
    source: str | None = None


class LinkDoc(BaseModel):
    _id: Any
    todoId: Any
    calendarEventId: str
    calendarId: str
    meetLink: str | None = None


def to_todo_out(doc: Dict[str, Any]) -> TodoOut:
    return TodoOut(
        id=str(doc.get("_id")),
        title=doc.get("title"),
        start=datetime.fromisoformat(doc.get("start")),
        end=datetime.fromisoformat(doc.get("end")),
        status=doc.get("status", "pending"),
    )


def overlap(a_start: datetime, a_end: datetime, b_start: datetime, b_end: datetime) -> bool:
    """Return True if [a_start, a_end) intersects [b_start, b_end)."""
    return a_start < b_end and b_start < a_end

