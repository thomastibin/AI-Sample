from __future__ import annotations

from typing import List, Literal, Optional
from datetime import datetime

import pytz
from dateutil import tz
from pydantic import BaseModel, Field, validator


DEFAULT_TZ = "Asia/Kolkata"


def _to_tzaware(dt: datetime, tz_name: str) -> datetime:
    if dt is None:
        return dt
    if dt.tzinfo is None:
        zone = pytz.timezone(tz_name or DEFAULT_TZ)
        dt = zone.localize(dt)
    return dt


def _to_utc(dt: datetime) -> datetime:
    if dt is None:
        return dt
    return dt.astimezone(pytz.UTC)


class TodoSearchIn(BaseModel):
    dateFrom: Optional[datetime] = Field(None, description="RFC3339 start, inclusive")
    dateTo: Optional[datetime] = Field(None, description="RFC3339 end, exclusive")
    text: Optional[str] = Field(None, description="Search text in title/description")

    @validator("dateFrom", "dateTo", pre=True)
    def _parse_dt(cls, v):
        if v is None or isinstance(v, datetime):
            return v
        # Pydantic v2 parses RFC3339; leave to pydantic then ensure tz-aware in handlers
        return v


class TodoCreateIn(BaseModel):
    title: str
    start: datetime
    end: datetime
    attendees: Optional[List[str]] = Field(default_factory=list)
    description: Optional[str] = None
    tz: str = Field(default=DEFAULT_TZ)

    @validator("start", "end", pre=True)
    def _parse_start_end(cls, v):
        return v


class TodoOut(BaseModel):
    id: str
    title: str
    start: datetime
    end: datetime
    status: Literal['pending', 'done', 'cancelled']


class StatusUpdateIn(BaseModel):
    id: str
    status: Literal['pending', 'done', 'cancelled']


class CalSearchIn(BaseModel):
    dateFrom: datetime
    dateTo: datetime
    attendees: Optional[List[str]] = None

    @validator("dateFrom", "dateTo", pre=True)
    def _parse_dt(cls, v):
        return v


class CalScheduleIn(BaseModel):
    title: str
    start: datetime
    end: datetime
    attendees: List[str]
    description: Optional[str] = None
    tz: str = Field(default=DEFAULT_TZ)

    @validator("start", "end", pre=True)
    def _parse_dt(cls, v):
        return v


class CalEventOut(BaseModel):
    id: str
    title: str
    start: datetime
    end: datetime
    attendees: List[str]
    meetLink: Optional[str] = None


def ensure_tz_and_utc(dt: datetime, tz_name: str = DEFAULT_TZ) -> datetime:
    """Accept naive or aware datetimes; apply tz if naive, then convert to UTC."""
    if dt is None:
        return dt
    dt = _to_tzaware(dt, tz_name)
    return _to_utc(dt)

