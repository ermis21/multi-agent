"""
phoebe-api — Phoebe (Multi-Agent Backend) API

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
import json
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import httpx
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from app.agents import run_agent_loop, run_agent_role, run_config_agent, run_soul_update
from app.config_loader import get_config, patch_config
from app.mcp_client import call_tool, resolve_approval
from app.prompt_generator import cleanup_all_generated
from app.session_logger import get_session, list_sessions
from app.session_state import SessionState

# ── Scheduler ─────────────────────────────────────────────────────────────────

scheduler = AsyncIOScheduler()


# ── Active sessions registry (for mid-flight user injections) ────────────────

# session_id → {"pending": list[dict], "cancel": asyncio.Event, "task": Task}
# A "pending" entry is {"mode": "immediate"|"not_urgent"|"clarify"|"queue"|"stop", "text": str}
# Consumed by the worker loop between iterations / after tool results.
_active_sessions: dict[str, dict] = {}
_background_tasks: set[asyncio.Task] = set()


def register_session(session_id: str, task: asyncio.Task | None) -> dict:
    """Register an in-flight session so inject/cancel endpoints can reach it.

    Hydrates `pending` from `state.pending_injections` if any survived a restart.
    """
    pending: list[dict] = []
    try:
        persisted = SessionState.load_or_create(session_id).get("pending_injections") or []
        if isinstance(persisted, list):
            pending = [p for p in persisted if isinstance(p, dict)]
    except Exception:
        pass
    rec = {"pending": pending, "cancel": asyncio.Event(), "task": task}
    _active_sessions[session_id] = rec
    return rec


def release_session(session_id: str) -> list[dict]:
    """Remove the session registration and return any remaining queued injections."""
    rec = _active_sessions.pop(session_id, None)
    if not rec:
        return []
    return [p for p in rec["pending"] if p.get("mode") == "queue"]


def get_session_state(session_id: str) -> dict | None:
    return _active_sessions.get(session_id)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Clean up stale generated prompt files from previous runs
    cleanup_all_generated()

    # Migrate flat sessions/{sid}.* layout to sessions/{sid}/{file} (idempotent).
    try:
        from app.session_migrate import migrate_flat_to_folders
        from app.session_logger import SESSIONS_DIR
        summary = migrate_flat_to_folders(SESSIONS_DIR)
        if summary["moved"]:
            print(
                f"[phoebe-api] session layout migration: moved {summary['moved']} files "
                f"across {summary['sessions']} session(s), skipped {summary['skipped_existing']} existing.",
                flush=True,
            )
    except Exception as e:
        print(f"[phoebe-api] session layout migration warning: {e}", flush=True)

    cfg = get_config()

    # Warn (don't crash) on schema drift — users may be mid-migration.
    try:
        from app.config_schema import validate_full
        issues = validate_full(cfg)
        if issues:
            print(f"[phoebe-api] config.yaml schema drift ({len(issues)} issue(s)):", flush=True)
            for line in issues[:5]:
                print(f"  - {line}", flush=True)
    except Exception as e:
        print(f"[phoebe-api] config schema check failed: {e}", flush=True)

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
        "_source_trigger": {"type": "cron", "ref": "discord_moderator"},
    }
    try:
        await run_agent_role("discord_moderator", body, session_id)
        print(f"[phoebe-api] discord moderation completed: {session_id}", flush=True)
    except Exception as e:
        print(f"[phoebe-api] discord moderation failed: {e}", flush=True)


# ── App ────────────────────────────────────────────────────────────────────────

app = FastAPI(title="phoebe-api", version="0.1.0", lifespan=lifespan, docs_url="/docs")


# ── SSE streaming helper ──────────────────────────────────────────────────────

async def _sse_generator(queue: asyncio.Queue, timeout: float = 660.0) -> AsyncIterator[str]:
    """Yield SSE-formatted events from *queue* until a ``done`` event arrives."""
    deadline = asyncio.get_event_loop().time() + timeout
    while True:
        remaining = deadline - asyncio.get_event_loop().time()
        if remaining <= 0:
            yield f"event: error\ndata: {json.dumps({'error': 'stream timeout'})}\n\n"
            return
        try:
            item = await asyncio.wait_for(queue.get(), timeout=remaining)
        except asyncio.TimeoutError:
            yield f"event: error\ndata: {json.dumps({'error': 'stream timeout'})}\n\n"
            return
        event_type = item.get("event", "unknown")
        event_data = json.dumps(item.get("data", {}), ensure_ascii=False)
        yield f"event: {event_type}\ndata: {event_data}\n\n"
        if event_type == "done":
            return


# ── Chat completions ───────────────────────────────────────────────────────────

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """
    OpenAI-compatible chat completions endpoint.
    Runs the supervisor/worker loop and returns the final response.

    Optional extra fields:
      - "session_id": session log uses this ID.
      - "stream": true → return SSE stream with real-time tool traces.
    """
    body = await request.json()
    session_id = body.pop("session_id", None) or (
        datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:6]
    )

    # Origin trigger — persisted to state on first turn (see run_agent_loop).
    ch = body.get("channel_id")
    body.setdefault(
        "_source_trigger",
        {"type": "user", "ref": f"discord:{ch}"} if ch else {"type": "user", "ref": None},
    )

    if body.pop("stream", False):
        queue: asyncio.Queue = asyncio.Queue()

        async def _run_and_cleanup(rec: dict):
            try:
                await run_agent_loop(body, session_id, trace_queue=queue, session_state=rec)
            finally:
                release_session(session_id)

        # Pre-register so inject can find the session before the task starts, and
        # so `session_state` is the SAME dict as `_active_sessions[session_id]` —
        # assigning a fresh list to `session_state["pending"]` mid-drain must not
        # orphan inject's view of it.
        rec = register_session(session_id, None)  # type: ignore[arg-type]
        task = asyncio.create_task(_run_and_cleanup(rec), name=f"stream_{session_id}")
        rec["task"] = task
        return StreamingResponse(
            _sse_generator(queue),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # Non-streaming path: still register so /inject works for long synchronous runs
    task = asyncio.current_task()
    rec = register_session(session_id, task) if task is not None else {"pending": [], "cancel": asyncio.Event(), "task": None}
    try:
        result = await run_agent_loop(body, session_id, session_state=rec)
    finally:
        release_session(session_id)
    return JSONResponse(result)


# ── Mid-flight injection ──────────────────────────────────────────────────────

_INJECT_MODES = {"immediate", "not_urgent", "clarify", "queue", "stop"}


@app.post("/v1/sessions/{session_id}/inject")
async def inject_into_session(session_id: str, request: Request):
    """Inject a mid-flight user message into an active session.

    Body: {"text": str, "mode": "immediate"|"not_urgent"|"clarify"|"queue"|"stop"}

      immediate  — appended as a user turn before the next LLM call; worker sees it ASAP.
      not_urgent — stapled onto the next tool_result as a [user_note] block.
      clarify    — same as not_urgent, with a "(this is clarification, not a new task)" suffix.
      queue      — held; delivered as a fresh turn after `done` fires (bot polls the receipt).
      stop       — sets the session's cancel event; worker exits the loop at the next boundary.
    """
    body = await request.json()
    text = (body.get("text") or "").strip()
    mode = body.get("mode") or "not_urgent"
    if mode not in _INJECT_MODES:
        raise HTTPException(400, f"mode must be one of {sorted(_INJECT_MODES)}")
    if not text and mode != "stop":
        raise HTTPException(400, "text is required (except for mode=stop)")

    rec = _active_sessions.get(session_id)
    if rec is None:
        raise HTTPException(404, "No active session with that id")

    if mode == "stop":
        rec["cancel"].set()
        # Cooperative cancel checks at iteration boundaries; a mid-LLM call may
        # take 30–120s to honor it. Schedule a hard task.cancel() as a backstop.
        task = rec.get("task")
        if task is not None:
            backstop = asyncio.create_task(_hard_cancel_after(task, delay=15.0))
            _background_tasks.add(backstop)
            backstop.add_done_callback(_background_tasks.discard)
    rec["pending"].append({"mode": mode, "text": text})
    try:
        st = SessionState.load_or_create(session_id)
        st.set("pending_injections", list(rec["pending"]))
        st.save()
    except Exception as e:
        print(f"[inject] persist failed: {e}", flush=True)
    return {"ok": True, "mode": mode, "session_id": session_id, "pending_count": len(rec["pending"])}


async def _hard_cancel_after(task: asyncio.Task, delay: float) -> None:
    """Backstop for cooperative `stop` — cancel the task if it hasn't exited yet.

    `_run_worker` / `run_agent_loop` check `rec["cancel"]` at iteration boundaries,
    but a stuck LLM call can delay that check indefinitely. CancelledError
    propagates cleanly through the loop's `except Exception:` blocks (it
    inherits from BaseException, not Exception)."""
    try:
        await asyncio.sleep(delay)
    except asyncio.CancelledError:
        return
    if not task.done():
        task.cancel()


@app.get("/v1/sessions/{session_id}/active")
async def session_is_active(session_id: str):
    """Lightweight check the Discord bot uses to decide whether to pop the dispatcher."""
    return {"active": session_id in _active_sessions}


@app.post("/v1/sessions/{session_id}/kill")
async def kill_session(session_id: str):
    """Hard-cancel an in-flight session immediately (no cooperative grace).

    Unlike `mode=stop` — which sets an event and gives the worker 15 s to exit
    at an iteration boundary — this cancels the asyncio task right away. Used
    by the discord bot's reset flow when we need to guarantee the task is gone
    before accepting a new run. Returns 404 if the session isn't active."""
    rec = _active_sessions.get(session_id)
    if rec is None:
        raise HTTPException(404, "No active session with that id")
    rec["cancel"].set()
    task = rec.get("task")
    if task is not None and not task.done():
        task.cancel()
    return {"ok": True, "session_id": session_id}


# ── Persistent session state (sessions/{sid}.state.json) ──────────────────────

@app.get("/v1/sessions/{session_id}/state")
async def get_state(session_id: str):
    """Full persistent state for a session. Creates a default file if missing."""
    from app.session_state import SessionState
    st = SessionState.load_or_create(session_id)
    return JSONResponse(st.data)


@app.patch("/v1/sessions/{session_id}/state")
async def patch_state(session_id: str, request: Request):
    """Deep-merge a patch into the session state file and persist.

    Lists are replaced wholesale; dicts merge recursively. Use an empty
    dict `{}` for a key to clear it.
    """
    from app.session_state import SessionState
    patch = await request.json()
    if not isinstance(patch, dict):
        raise HTTPException(400, "Body must be a JSON object")
    st = SessionState.load_or_create(session_id)

    def _merge(dst: dict, src: dict) -> None:
        for k, v in src.items():
            if isinstance(v, dict) and isinstance(dst.get(k), dict):
                _merge(dst[k], v)
            else:
                dst[k] = v

    _merge(st.data, patch)
    st.save()
    return JSONResponse(st.data)


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
    body.setdefault("_source_trigger", {"type": "api", "ref": f"role:{role}"})
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
        "source":         "phoebe-api",
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


# ── Tool approval callback ────────────────────────────────────────────────────

@app.post("/v1/approval_response")
async def approval_response(request: Request):
    """Called by the Discord bot when the user clicks Yes/No/Always on an approval embed."""
    body = await request.json()
    approval_id = body.get("approval_id", "")
    approved = body.get("approved", False)
    always = body.get("always", False)
    found = resolve_approval(approval_id, approved, always)
    if not found:
        raise HTTPException(404, "Unknown or expired approval_id")
    return {"ok": True}


@app.post("/v1/question_response")
async def question_response(request: Request):
    """Callback from Discord when user answers a multiple-choice question."""
    from app.ask_user import resolve_question
    data = await request.json()
    question_id = data.get("question_id", "")
    answer      = data.get("answer", "")
    answer_text = data.get("answer_text", "")
    if not question_id:
        raise HTTPException(400, "missing question_id")
    found = resolve_question(question_id, answer, answer_text)
    return {"ok": found}


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
    """List all sessions with turn counts and last timestamp, enriched with state metadata."""
    from app.session_state import SessionState
    sessions = list_sessions()
    for entry in sessions:
        sid = entry.get("session_id")
        if not sid:
            continue
        try:
            st = SessionState.load_or_create(sid)
            entry.update({
                "mode":           st.get("mode"),
                "model":          st.get("model"),
                "turn_count":     st.get("stats.turn_count"),
                "last_verdict":   st.get("supervisor.last_verdict"),
                "agent_role":     st.get("agent_role"),
                "source_trigger": st.get("source_trigger"),
                "is_active":      sid in _active_sessions,
            })
        except Exception:
            pass
    return {"sessions": sessions}


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
    from app.config_schema import ConfigPatchError
    patch = await request.json()
    try:
        updated = patch_config(patch)
    except ConfigPatchError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"updated": True, "config": updated}


@app.get("/v1/config/validate")
async def config_validate():
    """
    Validate the currently loaded config.yaml against RootConfig.
    Returns {"valid": bool, "errors": [str, ...]} — never raises.
    Consumed by diagnostic_check.
    """
    from app.config_schema import validate_full
    errors = validate_full(get_config())
    return {"valid": not errors, "errors": errors}


# ── Models / info ──────────────────────────────────────────────────────────────

@app.get("/models")
async def models():
    """Proxy the llama.cpp /v1/models endpoint."""
    cfg = get_config()
    llm = cfg.get("llm", {})
    base = llm.get("url") or llm.get("base_url", "")
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            r = await client.get(f"{base}/v1/models")
            return r.json()
    except Exception as e:
        return {"error": str(e)}
