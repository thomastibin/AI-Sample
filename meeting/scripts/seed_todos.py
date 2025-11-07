"""Seed script to insert a few deterministic todos for demo/testing."""
import asyncio
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient
import pytz

load_dotenv()
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB = os.getenv("MONGO_DB", "meet_todo")

async def seed():
    client = AsyncIOMotorClient(MONGO_URI)
    db = client[MONGO_DB]
    todos = db.todos
    await todos.delete_many({})
    now = datetime.now(pytz.UTC)
    docs = [
        {
            "title": "Demo: Standup",
            "description": "Daily standup",
            "start": (now + timedelta(hours=2)).isoformat(),
            "end": (now + timedelta(hours=2, minutes=30)).isoformat(),
            "tz": "Asia/Kolkata",
            "attendees": ["alice@example.com"],
            "status": "pending",
            "createdAt": now.isoformat(),
            "updatedAt": now.isoformat(),
            "source": "seed",
        },
        {
            "title": "Demo: Customer Call",
            "description": "Call with a customer",
            "start": (now + timedelta(days=1, hours=3)).isoformat(),
            "end": (now + timedelta(days=1, hours=4)).isoformat(),
            "tz": "Asia/Kolkata",
            "attendees": ["bob@example.com"],
            "status": "pending",
            "createdAt": now.isoformat(),
            "updatedAt": now.isoformat(),
            "source": "seed",
        },
    ]
    res = await todos.insert_many(docs)
    print("Inserted IDs:", res.inserted_ids)
    await client.close()

if __name__ == '__main__':
    asyncio.run(seed())
