"""
Supervisor / Worker agent orchestration.

Public functions:
  run_agent_loop(body, session_id)  — main chat completions handler
  run_soul_update()                  — called by APScheduler at 5 AM
  run_config_agent(body)             — guided config UI (POST /config/agent)

Tool calls:
  Workers detect tool calls by returning raw JSON with a "tool" key.
  The loop executes the tool via mcp_client.call_tool(), injects the result
  as a user message, and continues until a final (non-JSON) answer is given
  or max_tool_iterations is reached.
"""

import json
import os
from pathlib import Path

import httpx

from app.config_loader import get_config, get_agents_config
from app.mcp_client import call_tool, _extract_tool_call, strip_json_fences
from app.prompt_generator import generate, cleanup_generated
from app.session_logger import SessionLogger, get_session

LLAMA_URL = os.environ.get("LLAMA_URL", "http://host.docker.internal:8080")
WORKSPACE = Path(os.environ.get("WORKSPACE_DIR", "/workspace"))

_llm_client = httpx.AsyncClient(timeout=180)


# ── LLM call ──────────────────────────────────────────────────────────────────

async def _llm_call(messages: list[dict], cfg: dict, temperature: float | None = None) -> dict:
    """Single call to llama.cpp via its OpenAI-compatible endpoint."""
    r = await _llm_client.post(
        f"{LLAMA_URL}/v1/chat/completions",
        json={
            "model":       cfg["llm"]["model"],
            "messages":    messages,
            "temperature": temperature if temperature is not None else cfg["llm"]["temperature"],
            "max_tokens":  cfg["llm"]["max_tokens"],
            "top_p":       cfg["llm"]["top_p"],
        },
    )
    r.raise_for_status()
    return r.json()


# ── Mode helpers ──────────────────────────────────────────────────────────────

def _mode_temperature(cfg: dict, mode: str) -> float:
    """plan = base - delta, build = base, converse = base + delta."""
    base  = cfg["llm"]["temperature"]
    delta = cfg["agent"].get("mode", {}).get("temperature_delta", 0.3)
    offset = {"plan": -delta, "build": 0.0, "converse": delta}.get(mode, 0.0)
    return max(0.0, min(2.0, base + offset))


def _mode_tools(cfg: dict, mode: str, base_tools: list[str]) -> list[str]:
    """Remove mode-specific excluded tools from the base allowed list."""
    excluded = set(cfg["agent"].get("mode", {}).get(mode, {}).get("excluded_tools", []))
    return [t for t in base_tools if t not in excluded]


def _mode_context_string(mode: str) -> str:
    return {
        "plan":     "**Mode: PLAN** — Focus on analysis, planning, and research. Do not write files or execute commands.",
        "build":    "**Mode: BUILD** — Full tool access. Execute tasks, write code, make changes.",
        "converse": "**Mode: CONVERSE** — Conversational assistant. Answer questions, explain, discuss.",
    }.get(mode, "")


def _content(llm_response: dict) -> str:
    """Extract the assistant message content from a chat completion response."""
    try:
        return llm_response["choices"][0]["message"]["content"]
    except (KeyError, IndexError):
        return str(llm_response)


def _format_response(content: str, session_id: str) -> dict:
    """Wrap a final answer in OpenAI chat completion format."""
    return {
        "id":      f"mab-{session_id}",
        "object":  "chat.completion",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
        "session_id": session_id,
    }


# ── Worker (with inner tool loop) ─────────────────────────────────────────────

async def _run_worker(
    messages:       list[dict],
    system_prompt:  str,
    allowed_tools:  list[str],
    max_iterations: int,
    cfg:            dict,
    temperature:    float | None = None,
) -> tuple[str, list[dict]]:
    """
    Run the worker agent with its inner tool-call loop.

    Returns (final_answer, updated_messages_list).
    """
    full_messages = [{"role": "system", "content": system_prompt}] + messages

    for _ in range(max_iterations):
        resp    = await _llm_call(full_messages, cfg, temperature)
        content = _content(resp)
        tc      = _extract_tool_call(content)

        if tc is None:
            # No tool call — this is the final answer
            return content, full_messages

        # Execute the tool
        tool_result = await call_tool(tc["tool"], tc.get("params", {}), allowed_tools)

        # Append tool call + result as messages and continue
        full_messages.append({"role": "assistant", "content": content})
        full_messages.append({
            "role":    "user",
            "content": f"[tool_result: {tc['tool']}]\n{json.dumps(tool_result, ensure_ascii=False)}",
        })

    # Exhausted iterations — last content is best effort
    return content, full_messages


# ── Supervisor ────────────────────────────────────────────────────────────────

async def _run_supervisor(
    worker_response:    str,
    original_messages:  list[dict],
    system_prompt:      str,
    cfg:                dict,
    include_history:    bool,
) -> dict:
    """
    Grade the worker response.
    Returns {"pass": bool, "score": float, "feedback": str, "alternative": str}.
    Falls back to pass=True on JSON parse errors to avoid infinite error loops.
    """
    context_messages = original_messages if include_history else []
    messages = (
        [{"role": "system", "content": system_prompt}]
        + context_messages
        + [{"role": "assistant", "content": worker_response}]
        + [{"role": "user", "content": "Grade the above response. Respond ONLY with JSON."}]
    )

    try:
        resp    = await _llm_call(messages, cfg)
        raw    = strip_json_fences(_content(resp))
        result = json.loads(raw)
        missing = [k for k in ("pass", "score", "feedback", "alternative") if k not in result]
        if missing:
            raise ValueError(f"supervisor response missing keys: {missing}")
        result["score"] = float(result["score"])
        return result
    except Exception:
        return {"pass": True, "score": 0.5, "feedback": "supervisor parse error — treating as pass", "alternative": ""}


# ── Main agent loop ────────────────────────────────────────────────────────────

async def run_agent_loop(body: dict, session_id: str) -> dict:
    """
    Supervisor / worker loop for a single chat completion request.

    Flow:
      1. Generate worker prompt (dynamic, role=worker)
      2. Worker produces response (with inner tool loop)
      3. If supervisor disabled → return immediately
      4. Generate supervisor prompt (dynamic, role=supervisor)
      5. Supervisor grades; if pass → return
      6. Inject feedback into messages; retry up to max_retries times
      7. Return best-scored response after exhausting retries
    """
    cfg        = get_config()
    agents_cfg = get_agents_config()
    logger     = SessionLogger(session_id)
    raw_input  = body.get("messages", [])  # unmodified user input — stored in "final" turns

    # Mode — sent by Discord bot per user preference, defaulting to config default
    mode_cfg = cfg.get("agent", {}).get("mode", {})
    mode     = body.get("mode", mode_cfg.get("default", "converse"))

    # Session continuity: reconstruct prior conversation context from "final" turns only.
    # "final" turns store the raw user input and the actual response sent to the user,
    # without accumulated supervisor feedback or retry noise, so they stack cleanly.
    max_ctx     = cfg.get("agent", {}).get("max_context_turns", 20)
    prior_turns = get_session(session_id)
    final_turns = [t for t in prior_turns if t.get("role") == "final"]
    if final_turns:
        ctx: list[dict] = []
        for turn in final_turns[-max_ctx:]:
            ctx.extend(turn.get("messages", []))
            ctx.append({"role": "assistant", "content": turn.get("response", "")})
        # Rolling message window — trim oldest messages, always starting on a user turn
        max_msgs = cfg.get("agent", {}).get("max_context_messages", 0)
        if max_msgs and len(ctx) > max_msgs:
            ctx = ctx[-max_msgs:]
            while ctx and ctx[0]["role"] != "user":
                ctx = ctx[1:]
        messages = ctx + raw_input
    else:
        messages = raw_input

    w_cfg  = agents_cfg.get("worker", {})
    s_cfg  = agents_cfg.get("supervisor", {})
    w_tools     = _mode_tools(cfg, mode, w_cfg.get("allowed_tools", []))
    w_max_iter  = w_cfg.get("max_tool_iterations", 10)
    max_retries = cfg["agent"]["max_retries"]
    sup_enabled = cfg["agent"]["supervisor_enabled"]
    temperature = _mode_temperature(cfg, mode)

    best_response = ""
    best_score    = -1.0

    total_attempts = 1 + max_retries if sup_enabled else 1

    try:
        for attempt in range(total_attempts):
            worker_prompt, _ = generate(
                role="worker",
                allowed_tools=w_tools,
                session_id=session_id,
                attempt=attempt,
                extra={"{{AGENT_MODE}}": _mode_context_string(mode)},
            )

            try:
                worker_response, _ = await _run_worker(
                    messages, worker_prompt, w_tools, w_max_iter, cfg, temperature
                )
            except httpx.HTTPError as e:
                worker_response = f"[LLM error: {e}]"

            if cfg["logging"]["log_supervisor_turns"]:
                logger.log_turn(attempt, "worker", messages, worker_response)

            if not sup_enabled:
                logger.log_turn(0, "final", raw_input, worker_response)
                return _format_response(worker_response, session_id)

            sup_prompt, _ = generate(
                role="supervisor",
                allowed_tools=s_cfg.get("allowed_tools", []),
                session_id=session_id,
                attempt=attempt,
            )

            supervisor_result = await _run_supervisor(
                worker_response,
                messages,
                sup_prompt,
                cfg,
                s_cfg.get("include_conversation_history", True),
            )

            if cfg["logging"]["log_supervisor_turns"]:
                logger.log_turn(attempt, "supervisor", messages, worker_response, supervisor_result)

            score = supervisor_result.get("score", 0.0)
            if score > best_score:
                best_score    = score
                best_response = worker_response

            if supervisor_result.get("pass", False):
                logger.log_turn(0, "final", raw_input, worker_response)
                return _format_response(worker_response, session_id)

            # Prepare retry: inject supervisor feedback as user message
            feedback = supervisor_result.get("feedback", "")
            messages = messages + [
                {"role": "assistant", "content": worker_response},
                {"role": "user",      "content": (
                    f"[supervisor_feedback] Score: {score:.2f}. {feedback} "
                    "Please revise your response."
                )},
            ]

        logger.log_turn(0, "final", raw_input, best_response)
        return _format_response(best_response, session_id)
    finally:
        cleanup_generated(session_id)


# ── Soul update (runs at 5 AM via APScheduler) ─────────────────────────────────

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

    prompt, _ = generate(
        role="soul_updater",
        allowed_tools=soul_tools,
        session_id=soul_session,
        attempt=0,
    )

    messages = [{"role": "user", "content": "Update SOUL.md now."}]

    try:
        try:
            await _run_worker(messages, prompt, soul_tools, soul_max_it, cfg)
        except Exception:
            pass  # best-effort soul update; errors are non-fatal

        # Enforce hard char limit on SOUL.md regardless of what the model wrote
        soul_path = WORKSPACE / "SOUL.md"
        if soul_path.exists():
            text = soul_path.read_text(encoding="utf-8")
            max_c = cfg["soul"]["max_chars"]
            if len(text) > max_c:
                soul_path.write_text(text[:max_c], encoding="utf-8")
    finally:
        cleanup_generated(soul_session)


# ── Config agent (POST /config/agent) ─────────────────────────────────────────

async def run_config_agent(body: dict) -> dict:
    """Guided config UI — delegates to run_agent_role with role=config_agent."""
    session_id = body.pop("session_id", "config_agent")
    return await run_agent_role("config_agent", body, session_id)


# ── Generic role runner ────────────────────────────────────────────────────────

async def run_agent_role(role: str, body: dict, session_id: str) -> dict:
    """
    Run any agent role from agents.yaml directly.
    Used by POST /v1/agents/{role} and run_config_agent.
    Includes session continuity: prior turns are reconstructed from "final" logs.
    """
    cfg        = get_config()
    agents_cfg = get_agents_config()
    role_cfg   = agents_cfg.get(role, {})

    if not role_cfg:
        return {"error": f"Unknown role: {role!r}. Check agents.yaml."}

    # Mode applies to any role that supports it (primarily worker; others use base temp)
    mode_cfg      = cfg.get("agent", {}).get("mode", {})
    mode          = body.get("mode", mode_cfg.get("default", "converse"))
    base_tools    = role_cfg.get("allowed_tools", [])
    allowed_tools = _mode_tools(cfg, mode, base_tools)
    max_iter      = role_cfg.get("max_tool_iterations", 10)
    temperature   = _mode_temperature(cfg, mode)
    logger        = SessionLogger(session_id)
    raw_input     = body.get("messages", [])

    # Session continuity + rolling window — same logic as run_agent_loop
    max_ctx     = cfg.get("agent", {}).get("max_context_turns", 20)
    prior_turns = get_session(session_id)
    final_turns = [t for t in prior_turns if t.get("role") == "final"]
    if final_turns:
        ctx: list[dict] = []
        for turn in final_turns[-max_ctx:]:
            ctx.extend(turn.get("messages", []))
            ctx.append({"role": "assistant", "content": turn.get("response", "")})
        max_msgs = cfg.get("agent", {}).get("max_context_messages", 0)
        if max_msgs and len(ctx) > max_msgs:
            ctx = ctx[-max_msgs:]
            while ctx and ctx[0]["role"] != "user":
                ctx = ctx[1:]
        messages = ctx + raw_input
    else:
        messages = raw_input

    prompt, _ = generate(
        role=role,
        allowed_tools=allowed_tools,
        session_id=session_id,
        attempt=0,
        extra={"{{AGENT_MODE}}": _mode_context_string(mode)},
    )

    try:
        try:
            response, _ = await _run_worker(messages, prompt, allowed_tools, max_iter, cfg, temperature)
        except Exception as e:
            response = f"[{role} error: {e}]"
        logger.log_turn(0, "final", raw_input, response)
        return _format_response(response, session_id)
    finally:
        cleanup_generated(session_id)
