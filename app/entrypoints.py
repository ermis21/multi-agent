"""
Role entrypoints — scheduled jobs and the generic role runner.

`run_soul_update` runs on cron (5 AM default) to rewrite SOUL.md.
`run_config_agent` and `run_agent_role` power POST /config/agent and
POST /v1/agents/{role} respectively.
"""

import os
from pathlib import Path

from app.config_loader import get_agents_config, get_config
from app.loop import _auto_store_memory, _format_response, _rebuild_session_context
from app.mode import _mode_context_string, _mode_temperature, _mode_tools
from app.prompt_generator import cleanup_generated, generate
from app.sessions.logger import SessionLogger
from app.sessions.state import SessionState
from app.worker import _run_worker

STATE = Path(os.environ.get("STATE_DIR", "/state"))


async def run_soul_update() -> None:
    """
    Reads workspace context, asks the model to rewrite SOUL.md.
    Post-write: enforces soul.max_chars as a hard character limit.
    """
    cfg        = get_config()
    agents_cfg = get_agents_config()
    soul_cfg   = agents_cfg.get("soul_updater", {})

    soul_tools   = soul_cfg.get("allowed_tools", ["file_read", "file_write"])
    soul_max_it  = soul_cfg.get("max_tool_iterations", 8)
    soul_session = "soul_update"

    try:
        _st = SessionState.load_or_create(soul_session)
        _st.set("agent_role", "soul_updater")
        _st.set("source_trigger", {"type": "cron", "ref": "soul_updater"})
        _st.save()
    except Exception:
        pass

    prompt, _ = generate(
        role="soul_updater",
        allowed_tools=soul_tools,
        session_id=soul_session,
        attempt=0,
    )

    messages = [{"role": "user", "content": "Update SOUL.md now."}]

    try:
        try:
            await _run_worker(messages, prompt, soul_tools, soul_max_it, cfg, role_cfg=soul_cfg,
                              session_id=soul_session)  # traces discarded
        except Exception:
            pass  # best-effort soul update; errors are non-fatal

        # Enforce hard char limit on SOUL.md regardless of what the model wrote
        soul_path = STATE / "soul" / "SOUL.md"
        if soul_path.exists():
            text = soul_path.read_text(encoding="utf-8")
            max_c = cfg["soul"]["max_chars"]
            if len(text) > max_c:
                soul_path.write_text(text[:max_c], encoding="utf-8")
    finally:
        cleanup_generated(soul_session)


async def run_config_agent(body: dict) -> dict:
    """Guided config UI — delegates to run_agent_role with role=config_agent."""
    session_id = body.pop("session_id", "config_agent")
    return await run_agent_role("config_agent", body, session_id)


async def run_agent_role(
    role: str,
    body: dict,
    session_id: str,
    *,
    prompts_dir: Path | None = None,
) -> dict:
    """
    Run any agent role from agents.yaml directly.
    Used by POST /v1/agents/{role} and run_config_agent.
    Includes session continuity: prior turns are reconstructed from "final" logs.

    `prompts_dir` overrides the prompt template directory — the dream simulator
    uses this to replay a conversation under a candidate prompt without mutating
    `/config/prompts/`.
    """
    cfg        = get_config()
    agents_cfg = get_agents_config()
    role_cfg   = agents_cfg.get(role, {})

    if not role_cfg:
        return {"error": f"Unknown role: {role!r}. Check agents.yaml."}

    # Per-request model override: body.model (when present) overrides the
    # default role model so the dream simulator can pin its replay to a
    # specific model without editing agents.yaml. Copy role_cfg so we don't
    # mutate the shared agents.yaml cache.
    _body_model = body.get("model")
    if _body_model:
        role_cfg = {**role_cfg, "model": _body_model}

    # Mode applies to any role that supports it (primarily worker; others use base temp)
    mode_cfg       = cfg.get("agent", {}).get("mode", {})
    mode           = body.get("mode", mode_cfg.get("default", "converse"))
    base_tools     = role_cfg.get("allowed_tools", [])
    allowed_tools  = _mode_tools(cfg, mode, base_tools)
    max_iter       = role_cfg.get("max_tool_iterations", 10)
    temperature    = _mode_temperature(cfg, mode)
    approved_tools = body.get("approved_tools", [])
    logger         = SessionLogger(session_id)
    raw_input     = body.get("messages", [])

    # Plan context for build mode
    plan_context     = body.get("plan_context", "")
    privileged_paths = body.get("privileged_paths", [])

    # Stamp agent_role + source_trigger onto persistent state (first-turn only).
    try:
        _rst = SessionState.load_or_create(session_id)
        _rst.set("agent_role", role)
        if body.get("_source_trigger") is not None \
           and _rst.get("source_trigger") == {"type": "user", "ref": None}:
            _rst.set("source_trigger", body.get("_source_trigger"))
        _rst.save()
    except Exception:
        pass

    messages = _rebuild_session_context(session_id, raw_input, cfg)

    prompt, _ = generate(
        role=role,
        allowed_tools=allowed_tools,
        session_id=session_id,
        attempt=0,
        agent_mode=mode,
        extra={
            "{{AGENT_MODE}}": _mode_context_string(mode, cfg=cfg, role_cfg=role_cfg),
            "{{PLAN_CONTEXT}}": plan_context,
        },
        prompts_dir=prompts_dir,
    )

    verbose_tools = cfg["logging"].get("verbose_tools", False)
    tool_traces: list[dict] = []

    # Dream role: inject auto-sim hook + rollback unfinalized batches on exit.
    after_iteration_hook = None
    if role == "dreamer":
        from app.dream import runner_hook as _dream_hook
        after_iteration_hook = _dream_hook.make_dream_hook(session_id, cfg)

    try:
        try:
            response, _, tool_traces = await _run_worker(
                messages, prompt, allowed_tools, max_iter, cfg, temperature, mode, approved_tools,
                role_cfg=role_cfg,
                session_id=session_id,
                extra_auto_allow_paths=privileged_paths or None,
                after_iteration_hook=after_iteration_hook,
            )
        except Exception as e:
            response = f"[{role} error: {e or type(e).__name__}]"
        logger.log_turn(0, "final", raw_input, response)
        await _auto_store_memory(raw_input, response, session_id, mode, allowed_tools)
        result = _format_response(response, session_id)
        if verbose_tools and tool_traces:
            result["tool_trace"] = tool_traces
        return result
    finally:
        cleanup_generated(session_id)
        if role == "dreamer":
            try:
                from app.dream import runner_hook as _dream_hook
                _dream_hook.rollback_if_unfinalized(session_id)
            except Exception:
                pass
