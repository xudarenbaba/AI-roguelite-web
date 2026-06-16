# AGENTS.md

## Project

Python/Flask NPC API server + vanilla JS 2D roguelite game. No build system, no test framework, no linter, no CI.

## Setup (required before first run)

```bash
pip install -r requirements.txt

# Seed ChromaDB (one-time, idempotent for world/persona)
python scripts/import_world_setting.py
python scripts/import_persona_setting.py --npc-id wuxiao_01
```

The embedding model (`BAAI/bge-small-zh-v1.5`) must be pre-downloaded to `models/`. If `local_files_only: true` in config, the server crashes on startup if it's absent. Set `local_files_only: false` to auto-download on first run.

## Running

```bash
# Terminal 1: NPC API server → http://0.0.0.0:5100
python run.py

# Terminal 2: Static game server → http://127.0.0.1:8080
python run_game.py
```

Do NOT use `flask run` — use `python run.py`. For WSGI: `gunicorn server.app:app`.

## Configuration

Copy `config.example.yaml` to `config.yaml` (gitignored). Resolution order (last wins):

1. Hardcoded defaults in `server/npc_backend/config.py`
2. `config.yaml` at project root
3. Env vars: `AI_NPC_LLM_API_KEY`, `AI_NPC_LLM_BASE_URL`, `AI_NPC_LLM_MODEL`

Config is cached after first load; restart the server to pick up changes (or call `load_config(force_reload=True)`).

There is a Windows-only fallback path (`D:\otherwise\AI-NPC\config.yaml`) in `config.py:63` — silently ignored on macOS/Linux.

## NPC ID: README is wrong

The README example says `--npc-id ember_01`. The correct ID is **`wuxiao_01`** — that is what the frontend hardcodes (`game/game.js:773`) and what the lore file describes. Always seed with `--npc-id wuxiao_01`.

## Architecture notes

- `server/npc_backend/graph.py` — `NpcConversationEngine`: main streaming dialogue orchestrator
- `server/npc_backend/memory.py` — ChromaDB long-term memory; single collection `npc_memory`, filtered by `memory_type` (`world` | `persona` | `dialogue`), `npc_id`, `dialogue_tier` (`daily` | `important`)
- `server/npc_backend/short_term.py` — in-process `deque`, keyed by `player_id:npc_id`. **Lost on server restart.**
- Memory writes after streaming happen in a daemon `threading.Thread` — they don't block the response
- `__init__.py` claims "LangGraph" — ignore it, there is no LangGraph dependency; it's plain Python `ThreadPoolExecutor`

## Emotion tags

The system prompt instructs the LLM to append `<emotion>word</emotion>` at the end of every reply. `graph.py` strips it via `_EMOTION_RE`. If malformed or missing, emotion defaults to `"neutral"`.

## Runtime artifacts (gitignored, must exist)

- `data/chroma/` — ChromaDB persistent store (created by seed scripts)
- `models/` — HuggingFace embedding model cache

## Verification (no automated tests)

```bash
curl http://127.0.0.1:5100/health

curl -X POST http://127.0.0.1:5100/api/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"player_id":"p1","npc_id":"wuxiao_01","npc_name":"乌枭","message":"这里有多少敌人？","scene_info":{"mode":"battle","floor":1}}'
```

## Frontend

`game/game.js` is ~882 lines of vanilla JS with no build step. Key hardcoded constants at the chat section (~line 773):

```js
const NPC_API = "http://127.0.0.1:5100";  // change if running API on a different host/port
const NPC_ID  = "wuxiao_01";
const NPC_NAME = "乌枭";
```

`state.playerId` is hardcoded to `"player_web_demo"` — not configurable from the UI.
