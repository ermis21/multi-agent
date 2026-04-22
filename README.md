# Phoebe — Multi-Agent Backend

Phoebe is a production-grade multi-agent orchestration system built on FastAPI and llama.cpp. Implements a **supervisor/worker loop** where worker agents complete tasks and a supervisor grades responses before they are returned. Supports multiple specialized agent roles, persistent memory, and integrations with Discord, Notion, and Exa web search.

---

## Architecture

```
                         ┌─────────────────────┐
  OpenAI-compatible ───► │   phoebe-api  :8090     │
  /v1/chat/completions   │  (FastAPI, uvicorn)  │
                         └──────┬──────┬────────┘
                                │      │
              ┌─────────────────┘      └──────────────────┐
              ▼                                            ▼
  ┌───────────────────────┐               ┌───────────────────────┐
  │  phoebe-sandbox  :9000   │               │  phoebe-discord  :4000   │
  │ (MCP tool server,     │               │  (Discord bot gateway,│
  │  Docker socket,       │               │   3 bot tokens)       │
  │  ChromaDB memory)     │               └───────────────────────┘
  └───────────────────────┘
              │
              ▼
  ┌───────────────────────┐
  │  phoebe-notion   :3000   │
  │  (Notion MCP server)  │
  └───────────────────────┘
```

All services communicate over the `phoebe-net` Docker bridge network. `phoebe-api` reaches the host's llama.cpp instance via `host.docker.internal:8080`.

---

## Prerequisites

- Docker + Docker Compose
- A running [llama.cpp](https://github.com/ggerganov/llama.cpp) server on host port `8080`
- API keys / tokens for any optional integrations (see [Environment Variables](#environment-variables))

---

## Quick Start

```bash
git clone https://github.com/ermis21/multi-agent.git
cd multi-agent
cp .env.example .env
# Edit .env — fill in your keys and tokens

make up          # Start all services
make logs        # Follow startup logs
make health      # Verify API + LLM reachability
make doctor      # Full subsystem diagnostics
```

---

## Make Targets

### Production stack

| Target | Description |
|---|---|
| `make up` | Start all services (`docker compose up -d`) |
| `make build` | Rebuild images from scratch (no cache) |
| `make down` | Stop and remove containers |
| `make restart` | Restart `phoebe-api` only |
| `make logs` | Stream all service logs |
| `make logs-api` | Stream `phoebe-api` logs |
| `make logs-sandbox` | Stream `phoebe-sandbox` logs |
| `make status` | Show container status |

### Interaction

| Target | Description |
|---|---|
| `make chat` | Interactive chat (reads stdin → POST `/v1/chat/completions`) |
| `make config-agent` | Guided CLI to read/write `config.yaml` |
| `make soul-update` | Manually trigger a soul update (`POST /internal/soul-update`) |
| `make health` | API health + LLM reachability |
| `make doctor` | Colored diagnostics across all subsystems |

### Testing

| Target | Description |
|---|---|
| `make test-up` | Start test stack (`docker-compose.test.yml`) |
| `make test-down` | Stop test stack (removes orphans) |
| `make test` | Run `pytest` on `test/` |
| `make eval` | Run `test/eval.py` |
| `make test-full` | `test-up` → `test` → `eval` in sequence |
| `make test-dream` | Focused dream-system unit suite (phrase store, model ranks, runner, mailer, state machine) |
| `make test-dream-live` | Sandbox-side live probes in `test/test_dream_diagnostics.py` |
| `make e2e-dream` | Run only the `dream_smoke` Discord e2e scenario |

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/v1/chat/completions` | OpenAI-compatible chat endpoint (supervisor/worker loop) |
| `POST` | `/v1/agents/{role}` | Call any agent role directly by name |
| `GET` | `/health` | API health + LLM reachability |
| `GET` | `/sessions` | List all sessions with turn counts |
| `GET` | `/sessions/{session_id}` | Full session history |
| `GET` | `/config` | Read current `config.yaml` |
| `PATCH` | `/config` | Live-patch config without restart |
| `POST` | `/config/agent` | Guided config update via config agent |
| `POST` | `/internal/soul-update` | Trigger soul update manually |
| `POST` | `/internal/discord-moderation` | Trigger Discord moderation job manually |
| `GET` | `/internal/diagnostics` | Deterministic subsystem health check |
| `GET` | `/models` | Proxy llama.cpp models endpoint |

**Example:**

```bash
curl -X POST http://localhost:8090/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages": [{"role": "user", "content": "Hello"}]}'
```

---

## Agent Roles

| Role | Purpose | Max Iterations |
|---|---|---|
| `worker` | Primary responder — executes tools and produces responses | 10 |
| `supervisor` | Grades worker responses (pass/score/feedback JSON) | — |
| `soul_updater` | Daily cron — rewrites `state/soul/SOUL.md` from curated context | 8 |
| `config_agent` | Guides users through config questions, writes `config.yaml` | — |
| `improvement_agent` | Self-improvement via git + test stack | 20 |
| `discord_agent` | Discord operations (send/read/manage channels) | 10 |
| `discord_moderator` | Scheduled channel organiser (every 3 days) | 30 |
| `testing_agent` | Audit and system verification | 15 |
| `research_agent` | Web research and investigation | 15 |
| `coding_agent` | Code implementation + git operations | 20 |
| `session_compactor` | Compresses old session history | — |
| `cron_creator` | Creates new scheduled cron jobs | — |
| `tool_builder` | Adds new MCP tools | — |

Invoke any role directly:

```bash
curl -X POST http://localhost:8090/v1/agents/research_agent \
  -H 'Content-Type: application/json' \
  -d '{"messages": [{"role": "user", "content": "Research X"}]}'
```

---

## Environment Variables

Copy `.env.example` to `.env` and fill in the values:

| Variable | Required | Description |
|---|---|---|
| `EXA_API_KEY` | Optional | Exa web search API key |
| `NOTION_TOKEN` | Optional | Notion integration token |
| `DISCORD_TOKEN_WORKER` | Optional | Worker bot token |
| `DISCORD_TOKEN_CONFIG` | Optional | Config bot token |
| `DISCORD_TOKEN_MOD` | Optional | Moderator bot token |
| `DISCORD_GUILD_ID` | Optional | Discord server ID (for instant slash command sync) |
| `DISCORD_WORKER_CHANNELS` | Optional | Comma-separated channel IDs (auto-created if unset) |
| `DISCORD_CONFIG_CHANNELS` | Optional | Comma-separated channel IDs |
| `DISCORD_ALLOWED_USER_IDS` | Optional | Comma-separated user IDs (empty = allow all) |
| `ANTHROPIC_API_KEY` | Optional | For Anthropic provider auth |

---

## Configuration

Runtime behaviour is controlled by two YAML files in `config/`:

### `config/config.yaml`

Key settings:

| Section | Key settings |
|---|---|
| `llm` | provider, base_url, temperature, max_tokens, thinking budget |
| `prompts` | mode (`full` / `concise`), supervisor threshold |
| `agents` | supervisor enabled, max_retries, pass_threshold, max_context turns |
| `modes` | per-mode tool allow/ask/deny lists (`plan`, `build`, `converse`) |
| `soul_update` | enabled, cron time (default 05:00), max SOUL.md chars |
| `discord_moderator` | enabled, interval days, channel auto-create settings |
| `tools` | discord/notion defaults, memory limits, web search result count |

Live-patch without restart:

```bash
curl -X PATCH http://localhost:8090/config \
  -H 'Content-Type: application/json' \
  -d '{"agents": {"max_retries": 3}}'
```

### `config/agents.yaml`

Defines each agent role: allowed tools, system prompt reference, and max iteration count. Edit this file to add new roles or modify tool permissions per role. Changes are picked up on the next request (live-reload enabled).

---

## Project layout (XDG-segregated)

Five top-level trees, one lifecycle each. Agents address any of them with a matching path prefix; no prefix = `workspace/` (scratch). Mirrors the existing `project/` prefix convention for the read-only repo mount.

| Prefix | Host dir | Mount | Lifecycle | Contents |
|---|---|---|---|---|
| `config/` | `./config/` | rw | User-edited | `identity/` (IDENTITY, USER, MEMORY files), `skills/` (playbooks), `prompts/` (templates), `*.yaml` |
| `state/` | `./state/` | rw | Agent persistent | `soul/` (SOUL.md), `memory/` (daily snapshots), `sessions/` (per-session state), `discord/`, `chroma/` (ChromaDB) |
| `cache/` | `./cache/` | rw | Regenerable | `prompts/` (assembled prompt audit trail) |
| `workspace/` | `./workspace/` | rw | Agent scratch | `venv/` + anything else agents write; safe to prune |
| `project/` | `.` | **ro** | Repo source | The checked-out tree itself (`.env*` blocked) |

Writes without a prefix default to `workspace/`. The legacy `system/` prefix is rejected with an explicit pointer to the new roots.

**Owned subpaths:** `config/identity/*` is user-edited; `state/soul/`, `state/memory/`, and `config/skills/` are rewritten only by the `soul_updater` and `skill_builder` roles.

---

## Scheduled Jobs

Two cron jobs start automatically with `phoebe-api`:

| Job | Schedule | Description |
|---|---|---|
| Soul Update | 05:00 daily | Reads curated context, rewrites `state/soul/SOUL.md` |
| Discord Moderation | 10:00 every 3 days | Organises channels, archives inactive ones |
| Dream Run | 04:00 daily (off by default) | Nightly prompt self-improvement; edits `config/prompts/` via phrase-level provenance. Requires `cfg.dream.enabled: true` and all four dream diagnostic probes green |
| Dream Digest | 10:00 daily (off by default) | Emails yesterday's dream report to `cfg.dream.email.to`; Discord fallback on SMTP failure |

---

## Live Reload

Code changes to `./app`, `./config`, and `./prompts` are reflected immediately — no container rebuild needed. Uvicorn watches these paths and reloads on change.
