from __future__ import annotations

import os
from typing import Any, Dict, List, Optional
from datetime import datetime

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorCollection, AsyncIOMotorDatabase
from dotenv import load_dotenv
import logging


load_dotenv()

logger = logging.getLogger("mcp_mongo")

_client: Optional[AsyncIOMotorClient] = None


def _get_client() -> AsyncIOMotorClient:
    global _client
    if _client is None:
        uri = os.getenv("MONGO_URI", "mongodb://localhost:27017")
        # avoid logging full URI (may contain credentials); log that we're connecting
        logger.debug("Creating MongoDB client for URI (hidden)")
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
    logger.info("Ensuring indexes on todos collection")
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
    logger.debug("search_todos query: %s", query)
    cursor = todos().find(query).sort("start", 1)
    results = [doc async for doc in cursor]
    logger.debug("search_todos returned %d documents", len(results))
    return results


async def create_todo(data: Dict[str, Any]) -> str:
    logger.info("Inserting todo: %s", {"title": data.get("title"), "start": data.get("start")})
    res = await todos().insert_one(data)
    inserted_id = str(res.inserted_id)
    logger.info("Inserted todo id: %s", inserted_id)
    return inserted_id


async def update_status(id: str, status: str) -> bool:
    from bson import ObjectId

    logger.info("Updating todo %s status -> %s", id, status)
    res = await todos().update_one({"_id": ObjectId(id)}, {"$set": {"status": status, "updatedAt": datetime.utcnow().isoformat()}})
    ok = res.modified_count > 0
    logger.info("update_status modified_count=%s", res.modified_count)
    return ok

