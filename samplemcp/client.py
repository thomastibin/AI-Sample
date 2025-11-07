import asyncio
from typing import Any, Dict


async def main() -> None:
    # Connect to the SSE MCP server and call the 'hello' tool
    # Try multiple import paths to account for SDK version differences.
    client = None
    errors: Dict[str, str] = {}

    try:
        from mcp.client.sse import SseClient  # type: ignore

        # Try base URL first; some clients append "/sse" automatically
        try:
            client = ("direct", SseClient("http://127.0.0.1:8765"))
        except Exception as inner:
            # Try explicit /sse endpoint if base URL fails
            client = ("direct", SseClient("http://127.0.0.1:8765/sse"))
    except Exception as e:
        errors["mcp.client.sse.SseClient"] = str(e)

    if client is None:
        try:
            from mcp.client.sse import connect  # type: ignore

            try:
                client = ("factory", await connect("http://127.0.0.1:8765"))
            except Exception:
                client = ("factory", await connect("http://127.0.0.1:8765/sse"))
        except Exception as e:
            errors["mcp.client.sse.connect"] = str(e)

    if client is None:  # pragma: no cover
        detail = " | ".join(f"{k}: {v}" for k, v in errors.items())
        raise RuntimeError(
            "Could not construct MCP SSE client. "
            "Ensure a recent 'mcp' package is installed. "
            f"Attempted: {detail}"
        )

    kind, handle = client

    if kind == "direct":
        # Context manager path
        async with handle as c:  # type: ignore
            tools = await c.list_tools()
            print("tools:", tools)

            result = await c.call_tool("hello", {})
            print("result:", result)
    else:
        # Factory returning an already-connected client
        c = handle
        tools = await c.list_tools()
        print("tools:", tools)

        result = await c.call_tool("hello", {})
        print("result:", result)


if __name__ == "__main__":
    asyncio.run(main())
