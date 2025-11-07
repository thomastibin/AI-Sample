**Meet+ToDo Agent (MCP Demo)**

- Windows 11 demo of an MCP-based agent with a Python MCP server (FastAPI + MongoDB) and a separate LLM server (Gemini via LangChain) connecting via an MCP manifest using stdio.

**Prereqs**
- Windows 11
- MongoDB running at `mongodb://localhost:27017`
- Google OAuth app (Desktop or Web) in Google Cloud Console
- `uv` installed

**Setup (PowerShell)**
- `cd .\todo`
- `uv venv`
- `.\.venv\Scripts\Activate.ps1`
- `uv pip install -r requirements.txt`
- `copy .env.example .env` and fill `GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET`, `GEMINI_API_KEY`

First-time Google Calendar auth will trigger local OAuth:
- `python -c "from mcpserver.calendar_google import get_calendar_service; print('Auth OK' if get_calendar_service() else 'Auth failed')"`

**Seed data**
- `python .\scripts\seed_todos.py`  error user below way
- `python -m scripts.seed_todos `

**Run servers**
- You do not run the MCP server manually; the LLM server spawns it via `llm_server/mcp_manifest.json` (stdio transport).

Start the LLM server:
- `python .\llm_server\main.py` error use below
- `python -m llm_server.main`
- or `uvicorn llm_server.main:app --port 8001 --reload`

**Test from Postman**
- POST `http://localhost:8001/chat`
- Body:
  `{ "message": "i need to book a gmeet to tibin@gmail.com and user@gmail.com on indian time 3pm next week tuesday" }`

Expected success:
- `{ "ok": true, "meetLink": "...", "eventId": "...", "todoId": "..." }`

Expected conflict:
- `{ "conflict": true, "events": [...], "suggestions": ["3:30 PM IST", "4:00 PM IST"] }`

**Notes**
- Times are stored in UTC in MongoDB; default timezone is `Asia/Kolkata`.
- Google Calendar API uses RFC3339 with `timeZone`.
- MCP server tools (`todo.search`, `todo.create`, `todo.updateStatus`, `calendar.search`, `calendar.schedule`) are exposed via the Python MCP SDK (stdio) and mirrored with minimal FastAPI REST for debugging.



{
//   "mcpServers": {
//     "meet-todo-mcp": {
//       "url": "http://localhost:8000/mcp",
//       "transport": {
//         "type": "sse",
//         "url": "http://localhost:8000/mcp/"
//       }
//     }
//   }
// }