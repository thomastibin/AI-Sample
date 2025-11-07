# Sample MCP SSE Server + Client

This repo contains a minimal Model Context Protocol (MCP) HTTP/SSE server that exposes a single tool `hello` which returns `helloworld`, and a Python client that connects over SSE to call that tool.

The setup is designed to run with `uv` (https://github.com/astral-sh/uv) so you can easily create an isolated environment and run both the server and client.

## Prerequisites
- Python 3.10+
- `uv` installed

## Install dependencies

Choose one of the two options below.

- Option A: use `requirements.txt` (matches your IDE flow):

```
uv pip sync requirements.txt
```

- Option B: use `pyproject.toml`:

```
uv sync
```

Either option installs the required packages: `mcp`, `uvicorn`, and `fastapi`.

## Run the MCP SSE server

You can run via the `uvicorn` module or via the module’s `__main__` guard.

Option A (recommended):

```
uv run uvicorn server.main:app --host 127.0.0.1 --port 8765 --log-level info
```

Option B (equivalent):

```
uv run python server/main.py
```

The server will listen on `http://127.0.0.1:8765` and expose an MCP SSE endpoint.

## Run the client

In a second terminal, run:

```
uv run python client.py
```

Expected output (shape depends a bit on the MCP SDK version):
- Prints the available tools (should include `hello`).
- Calls `hello` and prints the result (should include `helloworld`).

## Notes
- The code tries a couple of common import paths/APIs for the MCP Python SDK SSE transport to maximize compatibility across versions.
- If imports fail (e.g., due to an older SDK), the error message will suggest updating the `mcp` package.

## Files
- `server/main.py` — SSE MCP server with a `hello` tool.
- `client.py` — SSE MCP client that lists tools and calls `hello`.
- `pyproject.toml` — Dependencies and basic metadata for `uv`.
