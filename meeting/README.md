# Meet+ToDo Agent (MCP demo)

This demo implements a two-server setup showcasing the Model Context Protocol (MCP):

- MCP Server (FastAPI + Motor/Mongo + Google Calendar helper) exposing tools via stdio transport.
- LLM Server (FastAPI) which spawns the MCP server via stdio using an MCP manifest and routes user messages to tools.

Prereqs
- Windows 11
- MongoDB running at mongodb://localhost:27017
- Python 3.11+ recommended

Quick setup (PowerShell)

```powershell
cd .\todo
# create venv (uv is a placeholder wrapper, you may use python -m venv .venv)
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
# Fill in GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET, GEMINI_API_KEY in .env

# First-time Google Calendar auth (will open a browser)
python -c "from mcpserver.calendar_google import get_calendar_service; print('Auth OK' if get_calendar_service() else 'Auth failed')"

# Seed demo todos
python .\scripts\seed_todos.py

# Start LLM server (which will spawn MCP via stdio)
python .\llm_server\main.py

# Or run via uvicorn (auto-restart during dev)
# uvicorn llm_server.main:app --port 8001 --reload
```

API

POST http://localhost:8001/chat
Body: { "message": "..." }

Examples
- "schedule a meeting next tuesday at 3pm IST, a google meet with tibin@gmail.com and user@gmail.com"
- "show my todos next week"
- "list online meetings next 48h"

Notes
- Times are stored in Mongo as UTC ISO strings. Default timezone: Asia/Kolkata.
- Google Calendar events use RFC3339 with timeZone.
- MCP tools: todo.search, todo.create, todo.updateStatus, calendar.search, calendar.schedule
