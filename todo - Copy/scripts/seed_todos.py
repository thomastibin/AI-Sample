from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
import pytz
from dotenv import load_dotenv

from mcpserver.mongo import create_todo, ensure_indexes


async def main():
    load_dotenv()
    await ensure_indexes()

    now = datetime.utcnow().replace(tzinfo=pytz.UTC)

    data = [
        {
            "title": "Daily planning",
            "description": "Prep tasks",
            "start": (now + timedelta(hours=2)).isoformat(),
            "end": (now + timedelta(hours=2, minutes=30)).isoformat(),
            "tz": "Asia/Kolkata",
            "attendees": ["demo@example.com"],
            "status": "pending",
            "createdAt": now.isoformat(),
            "updatedAt": now.isoformat(),
            "source": "seed",
        },
        {
            "title": "Sync with team",
            "description": "Discuss blockers",
            "start": (now + timedelta(days=1, hours=1)).isoformat(),
            "end": (now + timedelta(days=1, hours=1, minutes=30)).isoformat(),
            "tz": "Asia/Kolkata",
            "attendees": ["team@example.com"],
            "status": "pending",
            "createdAt": now.isoformat(),
            "updatedAt": now.isoformat(),
            "source": "seed",
        },
        {
            "title": "Deep work",
            "description": "Focus slot",
            "start": (now + timedelta(days=1, hours=4)).isoformat(),
            "end": (now + timedelta(days=1, hours=5)).isoformat(),
            "tz": "Asia/Kolkata",
            "attendees": [],
            "status": "pending",
            "createdAt": now.isoformat(),
            "updatedAt": now.isoformat(),
            "source": "seed",
        },
    ]

    ids = []
    for d in data:
        _id = await create_todo(d)
        ids.append(_id)
    print("Inserted IDs:", ids)


if __name__ == "__main__":
    asyncio.run(main())

