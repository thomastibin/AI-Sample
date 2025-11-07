import asyncio
from mcp.client.sse import sse_client
from mcp.client.session import ClientSession

async def main():
    url = "http://localhost:8000/mcp"
    print("[TEST] SSE connect:", url)
    async with sse_client(url=url) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            print("[TEST] tools:", [t.name for t in tools.tools])

            res = await session.call_tool("calendar.search", {
                "dateFrom": "2025-06-10T18:30:00Z",
                "dateTo":   "2025-06-10T19:30:00Z",
                "attendees": None
            })
            print("[TEST] calendar.search →", res)

asyncio.run(main())
