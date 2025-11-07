from __future__ import annotations

import os
from typing import Any, Dict, List, Optional
from datetime import datetime

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorCollection, AsyncIOMotorDatabase
from dotenv import load_dotenv
import sys


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


_client: Optional[AsyncIOMotorClient] = None


def _get_client() -> AsyncIOMotorClient:
    global _client
    if _client is None:
        uri = os.getenv("MONGO_URI", "mongodb://localhost:27017")
        _client = AsyncIOMotorClient(uri)
    return _client


def get_db() -> AsyncIOMotorDatabase:
    name = os.getenv("MONGO_DB", "meet_todo")
    return _get_client()[name]


def todos() -> AsyncIOMotorCollection:
    return get_db()["todos"]


def links() -> AsyncIOMotorCollection:
    return get_db()["links"]


async def ensure_indexes() -> None:
    t = todos()
    dbg("[MONGO] ensuring todos indexes")
    await t.create_index([("start", 1)])
    await t.create_index([("end", 1)])
    await t.create_index([("status", 1)])


async def search_todos(date_from: Optional[datetime] = None, date_to: Optional[datetime] = None, text: Optional[str] = None) -> List[Dict[str, Any]]:
    query: Dict[str, Any] = {}
    time_filter: Dict[str, Any] = {}
    if date_from is not None:
        time_filter["$gte"] = date_from.isoformat()
    if date_to is not None:
        time_filter["$lt"] = date_to.isoformat()
    if time_filter:
        query["start"] = time_filter
    if text:
        query["$or"] = [
            {"title": {"$regex": text, "$options": "i"}},
            {"description": {"$regex": text, "$options": "i"}},
        ]
    dbg("[MONGO] search_todos query:", query)
    cursor = todos().find(query).sort("start", 1)
    results = [doc async for doc in cursor]
    dbg("[MONGO] search_todos returned count:", len(results))
    return results


async def create_todo(data: Dict[str, Any]) -> str:
    dbg("[MONGO] inserting todo:", {"title": data.get("title"), "start": data.get("start")})
    res = await todos().insert_one(data)
    inserted_id = str(res.inserted_id)
    dbg("[MONGO] inserted id:", inserted_id)
    return inserted_id


async def update_status(id: str, status: str) -> bool:
    from bson import ObjectId
    dbg(f"[MONGO] update_status id={id} -> {status}")
    res = await todos().update_one({"_id": ObjectId(id)}, {"$set": {"status": status, "updatedAt": datetime.utcnow().isoformat()}})
    ok = res.modified_count > 0
    dbg("[MONGO] update_status modified_count:", res.modified_count)
    return ok

