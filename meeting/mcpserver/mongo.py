"""Mongo helpers using motor async client."""
from motor.motor_asyncio import AsyncIOMotorClient
import os
from bson import ObjectId
from typing import Optional, List
import asyncio
from datetime import datetime

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB = os.getenv("MONGO_DB", "meet_todo")

_client: Optional[AsyncIOMotorClient] = None


def get_client() -> AsyncIOMotorClient:
    global _client
    if _client is None:
        _client = AsyncIOMotorClient(MONGO_URI)
    return _client


def get_db():
    return get_client()[MONGO_DB]


def todos():
    return get_db().todos


def links():
    return get_db().links


async def ensure_indexes():
    await todos().create_index([("start", 1), ("end", 1)])
    await todos().create_index([("status", 1)])


async def search_todos(date_from: Optional[str] = None, date_to: Optional[str] = None, text: Optional[str] = None) -> List[dict]:
    q = {}
    if date_from or date_to:
        q["$and"] = []
        if date_from:
            q["$and"].append({"end": {"$gt": date_from}})
        if date_to:
            q["$and"].append({"start": {"$lt": date_to}})
    if text:
        q["title"] = {"$regex": text, "$options": "i"}
    cursor = todos().find(q).sort("start", 1)
    return [doc async for doc in cursor]


async def create_todo(data: dict) -> str:
    now = datetime.utcnow().isoformat()
    data.setdefault("createdAt", now)
    data.setdefault("updatedAt", now)
    res = await todos().insert_one(data)
    return str(res.inserted_id)


async def update_status(id: str, status: str) -> bool:
    res = await todos().update_one({"_id": ObjectId(id)}, {"$set": {"status": status, "updatedAt": datetime.utcnow().isoformat()}})
    return res.modified_count > 0
