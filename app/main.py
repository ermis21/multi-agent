"""
mab-api — Multi-Agent Backend API

Endpoints:
  POST  /v1/chat/completions     Main supervisor/worker loop
  POST  /config/agent            Guided configuration UI agent
  POST  /internal/soul-update    Manual trigger for soul update (testing)
  GET   /health                  Health + LLM reachability check
  GET   /sessions                List all sessions
  GET   /sessions/{session_id}   Get full session history
  GET   /config                  Read current config
  PATCH /config                  Live-patch config (no restart)
  GET   /models                  List tools / model info
"""

import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import httpx
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from app.agents import run_agent_loop, run_config_agent, run_soul_update
from app.config_loader import get_config, patch_config
from app.prompt_generator import cleanup_all_generated
from app.session_logger import get_session, list_sessions

# ── Scheduler ─────────────────────────────────────────────────────────────────

scheduler = AsyncIOScheduler()


def _parse_cron(schedule: str) -> dict:
    """Parse '0 5 * * *' into APScheduler kwargs."""
    parts = schedule.strip().split()
    if len(parts) != 5:
        return {"hour": 5, "minute": 0}
    minute, hour, day, month, weekday = parts
    kwargs: dict = {}
    if minute  != "*": kwargs["minute"]      = int(minute)
    if hour    != "*": kwargs["hour"]        = int(hour)
    if day     != "*": kwargs["day"]         = int(day)
    if month   != "*": kwargs["month"]       = int(month)
    if weekday != "*": kwargs["day_of_week"] = int(weekday)
    return kwargs


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Clean up stale generated prompt files from previous runs
    cleanup_all_generated()

    cfg = get_config()
    if cfg["soul"]["enabled"]:
        cron_kwargs = _parse_cron(cfg["soul"]["schedule"])
        scheduler.add_job(
            run_soul_update,
            "cron",
            id="soul_update",
            replace_existing=True,
            **cron_kwargs,
        )
        scheduler.start()

    yield

    if scheduler.running:
        scheduler.shutdown(wait=False)


# ── App ────────────────────────────────────────────────────────────────────────

app = FastAPI(title="mab-api", version="0.1.0", lifespan=lifespan, docs_url="/docs")


# ── Chat completions ───────────────────────────────────────────────────────────

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """
    OpenAI-compatible chat completions endpoint.
    Runs the supervisor/worker loop and returns the final response.

    Optional extra field: "session_id" — if provided, session log uses this ID.
    """
    body = await request.json()
    session_id = body.pop("session_id", None) or (
        datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:6]
    )
    result = await run_agent_loop(body, session_id)
    return JSONResponse(result)


# ── Config agent ───────────────────────────────────────────────────────────────

@app.post("/config/agent")
async def config_agent(request: Request):
    """
    Guided configuration UI.
    The agent asks strategic questions and patches config.yaml when done.
    Only read_config and write_config tools are available to this agent.

    Body: {"messages": [...], "session_id": "optional"}
    """
    body = await request.json()
    result = await run_config_agent(body)
    return JSONResponse(result)


# ── Internal triggers ──────────────────────────────────────────────────────────

@app.post("/internal/soul-update")
async def trigger_soul_update():
    """
    Manually trigger the soul update job (useful for testing without waiting for 5 AM).
    """
    await run_soul_update()
    return {"status": "ok", "message": "Soul update completed. Check workspace/SOUL.md."}


# ── Health ─────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    """
    Returns API status plus a probe of the llama.cpp server.
    """
    cfg = get_config()
    llm_status: dict = {}
    try:
        async with httpx.AsyncClient(timeout=3) as client:
            r = await client.get(f"{cfg['llm']['base_url']}/health")
            llm_status = r.json()
    except Exception as e:
        llm_status = {"error": str(e)}

    return {
        "api":     "ok",
        "llm":     llm_status,
        "mode":    cfg["prompts"]["mode"],
        "supervisor_enabled": cfg["agent"]["supervisor_enabled"],
    }


# ── Sessions ───────────────────────────────────────────────────────────────────

@app.get("/sessions")
async def sessions_list():
    """List all sessions with turn counts and last timestamp."""
    return {"sessions": list_sessions()}


@app.get("/sessions/{session_id}")
async def session_detail(session_id: str):
    """Return all turns for a specific session."""
    turns = get_session(session_id)
    if not turns:
        raise HTTPException(status_code=404, detail=f"Session {session_id!r} not found")
    return {"session_id": session_id, "turns": turns}


# ── Config endpoints ───────────────────────────────────────────────────────────

@app.get("/config")
async def config_read():
    """Return the current parsed configuration."""
    return get_config()


@app.patch("/config")
async def config_patch(request: Request):
    """
    Live-patch configuration values. Deep-merges the body into config.yaml.
    No container restart needed.

    Example body: {"prompts": {"mode": "concise"}, "agent": {"max_retries": 3}}
    """
    patch = await request.json()
    updated = patch_config(patch)
    return {"updated": True, "config": updated}


# ── Models / info ──────────────────────────────────────────────────────────────

@app.get("/models")
async def models():
    """Proxy the llama.cpp /v1/models endpoint."""
    cfg = get_config()
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            r = await client.get(f"{cfg['llm']['base_url']}/v1/models")
            return r.json()
    except Exception as e:
        return {"error": str(e)}
