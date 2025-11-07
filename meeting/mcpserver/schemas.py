"""Pydantic schemas for MCP tool I/O.

All datetimes accepted as RFC3339 strings or natural datetimes. Naive datetimes are localized to DEFAULT_TZ
and converted to UTC internally. Outputs are Python datetime objects (UTC-aware) and callers should
serialize to RFC3339 when sending to external APIs.
"""
from __future__ import annotations
from typing import Optional, List, Literal
from pydantic import BaseModel, Field, ValidationError, field_validator
from datetime import datetime
from dateutil import parser as dateutil_parser
import pytz
import os

DEFAULT_TZ = os.getenv("DEFAULT_TZ", "Asia/Kolkata")


def parse_datetime_to_utc(value) -> datetime:
    """Parse an input (str or datetime) to an aware UTC datetime.

    If input is naive, assume DEFAULT_TZ and convert to UTC.
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        dt = value
    else:
        dt = dateutil_parser.parse(value)
    if dt.tzinfo is None:
        local = pytz.timezone(DEFAULT_TZ)
        dt = local.localize(dt)
    return dt.astimezone(pytz.UTC)


class TodoSearchIn(BaseModel):
    dateFrom: Optional[datetime] = None
    dateTo: Optional[datetime] = None
    text: Optional[str] = None

    @field_validator("dateFrom", "dateTo", mode="before")
    def _parse_dates(cls, v):
        if v is None:
            return None
        return parse_datetime_to_utc(v)


class TodoCreateIn(BaseModel):
    title: str
    start: datetime
    end: datetime
    attendees: Optional[List[str]] = []
    description: Optional[str] = None
    tz: str = DEFAULT_TZ

    @field_validator("start", "end", mode="before")
    def _parse_dates(cls, v):
        return parse_datetime_to_utc(v)


class TodoOut(BaseModel):
    id: str
    title: str
    start: datetime
    end: datetime
    status: Literal["pending", "done", "cancelled"]


class StatusUpdateIn(BaseModel):
    id: str
    status: Literal["pending", "done", "cancelled"]


class CalSearchIn(BaseModel):
    dateFrom: datetime
    dateTo: datetime
    attendees: Optional[List[str]] = []

    @field_validator("dateFrom", "dateTo", mode="before")
    def _parse_dates(cls, v):
        return parse_datetime_to_utc(v)


class CalScheduleIn(BaseModel):
    title: str
    start: datetime
    end: datetime
    attendees: List[str]
    description: Optional[str] = None
    tz: str = DEFAULT_TZ

    @field_validator("start", "end", mode="before")
    def _parse_dates(cls, v):
        return parse_datetime_to_utc(v)


class CalEventOut(BaseModel):
    id: str
    title: str
    start: datetime
    end: datetime
    attendees: List[str]
    meetLink: Optional[str] = None
