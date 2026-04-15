"""
mab-api — Multi-Agent Backend API

Endpoints:
  POST  /v1/chat/completions     Main supervisor/worker loop
  POST  /config/agent            Guided configuration UI agent
  POST  /internal/soul-update    Manual trigger for soul update (testing)
  GET   /internal/diagnostics    Deterministic subsystem health check
  GET   /health                  Health + LLM reachability check
  GET   /sessions                List all sessions
  GET   /sessions/{session_id}   Get full session history
  GET   /config                  Read current config
  PATCH /config                  Live-patch config (no restart)
  GET   /models                  List tools / model info
"""

import asyncio
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import httpx
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from app.agents import run_agent_loop, run_agent_role, run_config_agent, run_soul_update
from app.config_loader import get_config, patch_config
from app.mcp_client import call_tool
from app.prompt_generator import cleanup_all_generated
from app.session_logger import get_session, list_sessions

# ── Scheduler ─────────────────────────────────────────────────────────────────

scheduler = AsyncIOScheduler()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Clean up stale generated prompt files from previous runs
    cleanup_all_generated()

    cfg = get_config()
    if cfg["soul"]["enabled"]:
        scheduler.add_job(
            run_soul_update,
            CronTrigger.from_crontab(cfg["soul"]["schedule"]),
            id="soul_update",
            replace_existing=True,
        )

    dm_cfg = cfg.get("discord_moderator", {})
    if dm_cfg.get("enabled", False):
        scheduler.add_job(
            _run_discord_moderation,
            CronTrigger.from_crontab(dm_cfg["schedule"]),
            id="discord_moderation",
            replace_existing=True,
        )

    if scheduler.get_jobs():
        scheduler.start()

    yield

    if scheduler.running:
        scheduler.shutdown(wait=False)


async def _run_discord_moderation() -> None:
    """Cron wrapper: invoke the discord_moderator agent role."""
    session_id = datetime.now(timezone.utc).strftime("discord_mod_%Y%m%d_%H%M%S")
    body = {
        "messages": [{"role": "user", "content": "Run your scheduled Discord moderation task."}],
        "mode": "build",
    }
    try:
        await run_agent_role("discord_moderator", body, session_id)
        print(f"[mab-api] discord moderation completed: {session_id}", flush=True)
    except Exception as e:
        print(f"[mab-api] discord moderation failed: {e}", flush=True)


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


# ── Direct role endpoint ───────────────────────────────────────────────────────

@app.post("/v1/agents/{role}")
async def agents_role(role: str, request: Request):
    """
    Call any agent role directly by name.

    Body: {"messages": [...], "session_id": "optional"}

    Special routes: 'soul_updater' → run_soul_update, 'config_agent' → run_config_agent.
    All other roles run the worker loop with that role's config from agents.yaml.
    """
    body = await request.json()
    session_id = body.pop("session_id", None) or (
        datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:6]
    )
    if role == "soul_updater":
        # Soul update can take minutes — dispatch as background task and return immediately
        asyncio.create_task(run_soul_update())
        return JSONResponse({"status": "accepted", "role": role})
    if role == "config_agent":
        body["session_id"] = session_id
        return JSONResponse(await run_config_agent(body))
    return JSONResponse(await run_agent_role(role, body, session_id))


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

@app.post("/internal/discord-moderation")
async def trigger_discord_moderation():
    """Manually trigger the Discord moderation job."""
    session_id = datetime.now(timezone.utc).strftime("discord_mod_%Y%m%d_%H%M%S")
    body = {
        "messages": [{"role": "user", "content": "Run Discord moderation: organize channels, delete empty ones, archive inactive ones."}],
        "mode": "build",
    }
    result = await run_agent_role("discord_moderator", body, session_id)
    return {"status": "ok", "session_id": session_id, "result": result}


@app.post("/internal/soul-update")
async def trigger_soul_update():
    """
    Manually trigger the soul update job (useful for testing without waiting for 5 AM).
    """
    await run_soul_update()
    return {"status": "ok", "message": "Soul update completed. Check workspace/SOUL.md."}


@app.get("/internal/diagnostics")
async def diagnostics():
    """
    Deterministic health check of all system components.
    Calls the sandbox diagnostic_check tool and probes the LLM in parallel.
    No LLM inference involved — safe to call at any time.
    """
    cfg = get_config()

    sandbox_result, llm_status = await asyncio.gather(
        call_tool("diagnostic_check", {}, ["diagnostic_check"]),
        _probe_llm(cfg),
    )

    fail_count = sandbox_result.get("fail_count", 0) + (1 if llm_status["status"] == "fail" else 0)
    warn_count = sandbox_result.get("warn_count", 0) + (1 if llm_status["status"] == "warn" else 0)
    pass_count = sandbox_result.get("pass_count", 0) + (1 if llm_status["status"] == "pass" else 0)
    overall    = "fail" if fail_count else ("warn" if warn_count else "pass")

    return {
        "source":         "mab-api",
        "timestamp":      datetime.now(timezone.utc).isoformat(),
        "sandbox_checks": sandbox_result,
        "api_checks":     {"llm_api_from_api": llm_status},
        "overall":        overall,
        "summary":        f"{pass_count} pass, {warn_count} warn, {fail_count} fail",
    }


async def _probe_llm(cfg: dict) -> dict:
    base_url = cfg["llm"]["base_url"]
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            r = await client.get(f"{base_url}/health")
        status = "pass" if r.status_code == 200 else "warn"
        return {"status": status, "detail": f"HTTP {r.status_code} from {base_url}"}
    except Exception as e:
        return {"status": "fail", "detail": str(e)}


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
