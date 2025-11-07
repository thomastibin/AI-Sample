from typing import Dict

try:
    from mcp.server.fastmcp import FastMCP
except Exception as e:  # pragma: no cover
    raise RuntimeError("The 'mcp' package is required. Install with `uv sync`.") from e


mcp = FastMCP("hello-sse-server")


@mcp.tool()  # FastMCP requires calling the decorator, even with no args
async def hello() -> str:
    """Return 'helloworld'"""
    return "helloworld"


# Build an ASGI app (FastAPI) for SSE transport.
app = None
_compat_errors: Dict[str, str] = {}

try:
    # Common attribute provided by FastMCP
    app = mcp.fastapi_app  # type: ignore[attr-defined]
except Exception as e:  # pragma: no cover
    _compat_errors["mcp.fastapi_app"] = str(e)

if app is None:
    try:
        # Some versions might expose a generic ASGI app
        app = mcp.asgi  # type: ignore[attr-defined]
    except Exception as e:  # pragma: no cover
        _compat_errors["mcp.asgi"] = str(e)

if app is None:  # pragma: no cover
    details = " | ".join(f"{k}: {v}" for k, v in _compat_errors.items())
    raise RuntimeError(
        "Could not construct ASGI app from FastMCP. "
        "Ensure you have a recent 'mcp' package installed. "
        f"Attempted: {details}"
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8765, log_level="info")
