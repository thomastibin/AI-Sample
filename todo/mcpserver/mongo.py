# mongo.py
from __future__ import annotations

import os
import traceback
from typing import Any, Dict, List, Optional
from datetime import datetime
try:
    from datetime import UTC  # py3.11+
except Exception:
    from datetime import timezone as _tz
    UTC = _tz.utc  # type: ignore

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase, AsyncIOMotorCollection

# Optional debug file logger
_DEBUG = os.getenv("MONGO_DEBUG", "1") == "1"
def dbg(*args, **kwargs):
    if _DEBUG:
        print(*args, **kwargs)

# ---- Global, initialized once on the server's running loop ----
_client: Optional[AsyncIOMotorClient] = None
_db: Optional[AsyncIOMotorDatabase] = None
_DB_NAME = os.getenv("MONGO_DB", "meet_todo")
_MONGO_URI = os.getenv("MONGO_URI", "mongodb://127.0.0.1:27017")

async def init_mongo() -> None:
    """
    Create the Motor client ON THE ACTIVE EVENT LOOP and store it globally.
    Idempotent: calling multiple times is OK.
    """
    global _client, _db
    if _client is not None and _db is not None:
        return
    try:
        dbg("[MONGO] init_mongo connecting:", _MONGO_URI)
        _client = AsyncIOMotorClient(_MONGO_URI)  # binds to current asyncio loop
        _db = _client[_DB_NAME]
        dbg("[MONGO] init_mongo OK, db:", _DB_NAME)
    except Exception as e:
        print("[MONGO][ERROR] init_mongo failed:", repr(e))
        traceback.print_exc()
        raise

async def close_mongo() -> None:
    """Close the global client (call on app shutdown)."""
    global _client, _db
    try:
        if _client is not None:
            _client.close()
            dbg("[MONGO] client closed")
    except Exception as e:
        print("[MONGO][WARN] close_mongo error:", repr(e))
        traceback.print_exc()
    finally:
        _client = None
        _db = None

def _require_db() -> AsyncIOMotorDatabase:
    if _db is None:
        raise RuntimeError("Mongo not initialized; call init_mongo() first")
    return _db

def todos() -> AsyncIOMotorCollection:
    return _require_db()["todos"]

# ---------- SAFE utility (standalone index creator using a TEMP client) ----------
async def ensure_indexes_safe() -> None:
    """
    Create indexes using a TEMPORARY client bound to the CURRENT loop.
    This avoids sharing clients across loops (root cause of your error).
    """
    temp = None
    try:
        dbg("[MONGO] ensure_indexes_safe starting")
        temp = AsyncIOMotorClient(_MONGO_URI)
        db = temp[_DB_NAME]
        col = db["todos"]
        await col.create_index("start")
        await col.create_index("status")
        dbg("[MONGO] ensure_indexes_safe OK")
    except Exception as e:
        print("[MONGO][ERROR] ensure_indexes_safe:", repr(e))
        traceback.print_exc()
    finally:
        if temp:
            temp.close()

# ---------- Your original helpers, now wrapped with try/except and loop-safe ----------
async def search_todos(
    date_from: Optional[datetime] = None,
    date_to: Optional[datetime] = None,
    text: Optional[str] = None,
) -> List[Dict[str, Any]]:
    try:
        # Ensure client is ready on THIS loop
        await init_mongo()

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
    except Exception as e:
        print("[ERROR] search_todos failed:", repr(e))
        traceback.print_exc()
        return []

async def create_todo(data: Dict[str, Any]) -> str:
    try:
        await init_mongo()
        dbg("[MONGO] inserting todo:", {"title": data.get("title"), "start": data.get("start")})
        res = await todos().insert_one(data)
        inserted_id = str(res.inserted_id)
        dbg("[MONGO] inserted id:", inserted_id)
        return inserted_id
    except Exception as e:
        print("[ERROR] create_todo failed:", repr(e), " data=", data)
        traceback.print_exc()
        return ""

async def update_status(id: str, status: str) -> bool:
    from bson import ObjectId
    try:
        await init_mongo()
        dbg(f"[MONGO] update_status id={id} -> {status}")
        res = await todos().update_one(
            {"_id": ObjectId(id)},
            {"$set": {"status": status, "updatedAt": datetime.now(UTC).isoformat()}},
        )
        ok = (res.modified_count or 0) > 0
        dbg("[MONGO] update_status modified_count:", res.modified_count)
        return ok
    except Exception as e:
        print("[ERROR] update_status failed:", repr(e), f" id={id} status={status}")
        traceback.print_exc()
        return False
