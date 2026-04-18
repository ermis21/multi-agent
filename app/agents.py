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

import asyncio
import json
import os
import time
from pathlib import Path

import anthropic as _anthropic_sdk
import httpx

from app.config_loader import get_config, get_agents_config
from app.mcp_client import call_tool, _extract_tool_call, strip_json_fences
from app.prompt_generator import generate, cleanup_generated
from app.session_logger import SessionLogger, get_session

LLAMA_URL = os.environ.get("LLAMA_URL", "http://host.docker.internal:8080")
WORKSPACE = Path(os.environ.get("WORKSPACE_DIR", "/workspace"))

# Shell-style tools: a non-zero exit_code is promoted to a real error so the
# model sees ERROR instead of a misleading OK when commands fail.
_SHELL_TOOLS = frozenset({
    "shell_exec", "execute_command",
    "git_status", "git_commit", "git_rollback", "git_log",
    "docker_test_up", "docker_test_down", "docker_test_health",
})

# Tokens/substrings that look like an attempted tool call — used to detect
# parse failures so we can correct the model instead of accepting prose as final.
_TOOL_CALL_HINTS = ("<|tool_call", "<tool_call", '"tool":', "call:")


def _promote_shell_error(method: str, result: dict) -> dict:
    """If a shell-style tool returned exit_code != 0, surface it as an error."""
    if method not in _SHELL_TOOLS or "error" in result:
        return result
    code = result.get("exit_code", 0)
    if code == 0:
        return result
    stderr = (result.get("stderr") or "").strip()
    out = dict(result)
    out["error"] = f"exit {code}" + (f": {stderr}" if stderr else "")
    return out

_llm_client = httpx.AsyncClient(timeout=180)

# Anthropic client cache: (token_used, client_instance)
# Re-created whenever the active token changes (e.g. after OAuth refresh).
_anthropic_client: tuple[str, _anthropic_sdk.AsyncAnthropic] | None = None

# Mounted read-only from the host's ~/.claude/.credentials.json
_CLAUDE_CREDS = Path("/app/.claude_credentials.json")


def _read_oauth_token() -> str | None:
    """
    Read the Claude Code OAuth access token from the mounted credentials file.
    Returns None if the file is missing, unreadable, or the token has expired.
    """
    if not _CLAUDE_CREDS.exists():
        return None
    try:
        creds  = json.loads(_CLAUDE_CREDS.read_text())
        oauth  = creds.get("claudeAiOauth", {})
        token  = oauth.get("accessToken", "")
        # expiresAt is milliseconds; skip if expiring within 5 minutes
        if token and oauth.get("expiresAt", 0) > time.time() * 1000 + 300_000:
            return token
    except Exception:
        pass
    return None


def _get_anthropic_client() -> _anthropic_sdk.AsyncAnthropic:
    global _anthropic_client

    oauth_token = _read_oauth_token()
    if oauth_token:
        # Re-create if this is the first call or the token was refreshed
        if _anthropic_client is None or _anthropic_client[0] != oauth_token:
            _anthropic_client = (oauth_token, _anthropic_sdk.AsyncAnthropic(auth_token=oauth_token))
        return _anthropic_client[1]

    # Fall back to ANTHROPIC_API_KEY env var
    if _anthropic_client is None:
        _anthropic_client = ("", _anthropic_sdk.AsyncAnthropic())
    return _anthropic_client[1]


async def _auto_store_memory(
    raw_input: list[dict],
    response: str,
    session_id: str,
    mode: str,
    allowed_tools: list[str],
) -> None:
    """
    Fire-and-forget: store a brief turn summary in MemPalace so the agent can
    search conversation history across sessions via memory_search.
    Only runs when memory_add is in the agent's allowed tool list.
    """
    if "memory_add" not in allowed_tools:
        return
    try:
        user_text = " ".join(
            m.get("content", "") for m in raw_input if m.get("role") == "user"
        )[:300]
        summary = (
            f"Session turn [{session_id}]:\n"
            f"User: {user_text}\n"
            f"Agent: {response[:500]}"
        )
        await call_tool(
            "memory_add",
            {"content": summary, "tags": ["session_history", session_id, mode]},
            ["memory_add"],
            mode="converse",
            approved_tools=["memory_add"],
        )
    except Exception:
        pass  # non-fatal; never block the response


# ── LLM call ──────────────────────────────────────────────────────────────────

async def _llm_call_local(messages: list[dict], llm: dict, temperature: float, url: str | None = None) -> dict:
    """llama.cpp / Ollama path via OpenAI-compatible endpoint."""
    endpoint = url or LLAMA_URL
    payload: dict = {
        "model":       llm["model"],
        "messages":    messages,
        "temperature": temperature,
        "max_tokens":  llm["max_tokens"],
        "top_p":       llm["top_p"],
        "top_k":       llm.get("top_k", 40),
    }
    payload["chat_template_kwargs"] = {"enable_thinking": bool(llm.get("enable_thinking", False))}
    r = await _llm_client.post(f"{endpoint}/v1/chat/completions", json=payload)
    r.raise_for_status()
    return r.json()


async def _llm_call_anthropic(messages: list[dict], llm: dict, temperature: float) -> dict:
    """Anthropic SDK path — returns an OpenAI-compatible dict."""
    client = _get_anthropic_client()

    # Anthropic requires system as a separate param, not a message role
    system = " ".join(m["content"] for m in messages if m["role"] == "system")
    conv   = [m for m in messages if m["role"] != "system"]

    thinking = llm.get("enable_thinking", False)
    # Anthropic output token caps: 8192 normally, 16000 with extended thinking
    max_tok = min(llm["max_tokens"], 16000 if thinking else 8192)

    kwargs: dict = {
        "model":      llm["model"],
        "messages":   conv,
        "max_tokens": max_tok,
        "temperature": temperature,
    }
    if system:
        kwargs["system"] = system
    if thinking:
        budget = llm.get("thinking_budget_tokens", 8000)
        kwargs["thinking"] = {"type": "enabled", "budget_tokens": budget}
        kwargs.pop("temperature", None)  # temperature is rejected when thinking is enabled

    resp = await client.messages.create(**kwargs)

    # Normalise to OpenAI-compatible shape so the rest of agents.py is unchanged
    text = next((b.text for b in resp.content if b.type == "text"), "")
    return {
        "choices": [{"message": {"role": "assistant", "content": text}}],
        "usage": {
            "input_tokens":  resp.usage.input_tokens,
            "output_tokens": resp.usage.output_tokens,
        },
    }


async def _llm_call(messages: list[dict], cfg: dict, temperature: float | None = None, role_cfg: dict | None = None) -> dict:
    """Dispatch to the correct provider based on config.

    If role_cfg contains a 'model' key, the named model from cfg['models'] is
    merged over the default llm block — allowing per-role model selection.
    """
    llm = dict(cfg["llm"])  # copy; don't mutate the live config
    if role_cfg and (model_name := role_cfg.get("model")):
        override = cfg.get("models", {}).get(model_name, {})
        if override:
            llm.update(override)

    temp = temperature if temperature is not None else llm["temperature"]
    if llm.get("provider", "local") == "anthropic":
        return await _llm_call_anthropic(messages, llm, temp)
    return await _llm_call_local(messages, llm, temp, url=llm.get("url"))


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
    mode:           str = "converse",
    approved_tools: list[str] | None = None,
    role_cfg:       dict | None = None,
    session_id:     str = "",
    trace_queue:    asyncio.Queue | None = None,
) -> tuple[str, list[dict], list[dict]]:
    """
    Run the worker agent with its inner tool-call loop.

    Returns (final_answer, updated_messages_list, tool_traces).
    tool_traces is a list of dicts: {tool, duration_s, lines, error}.
    """
    tool_traces: list[dict] = []
    spawnable = (role_cfg or {}).get("spawnable_agents", [])
    full_messages = [{"role": "system", "content": system_prompt}] + messages

    content = ""
    for i in range(max_iterations):
        resp    = await _llm_call(full_messages, cfg, temperature, role_cfg=role_cfg)
        content = _content(resp)
        tc      = _extract_tool_call(content)

        if tc is None:
            # If the content *looks* like a malformed tool call, correct the
            # model instead of accepting it as the final answer — this is what
            # causes hallucinated "command executed successfully" cascades.
            if any(h in content for h in _TOOL_CALL_HINTS):
                full_messages.append({"role": "assistant", "content": content})
                full_messages.append({
                    "role": "user",
                    "content": (
                        "Your tool call could not be parsed. Emit exactly:\n"
                        "<|tool_call|>call: TOOL_NAME, {param_json}<|tool_call|>\n"
                        "One line. No prose, no markdown, no nesting."
                    ),
                })
                continue
            # No tool call — this is the final answer
            return content, full_messages, tool_traces

        # Execute the tool
        t0 = time.time()
        tool_result = await call_tool(tc["tool"], tc.get("params", {}), allowed_tools, mode, approved_tools,
                                      session_id=session_id, spawnable_agents=spawnable)
        tool_result = _promote_shell_error(tc["tool"], tool_result)
        duration = time.time() - t0

        # Format result clearly so the model can distinguish success from failure
        if "error" in tool_result:
            result_text = (
                f"[tool_result: {tc['tool']}] ERROR\n"
                f"{tool_result['error']}\n\n"
                "This tool call failed. Do NOT stop — keep working on the user's request. "
                "Pick one: retry with corrected parameters, try a different tool, or try a different approach. "
                "Only give a final plain-text answer if every reasonable approach has been exhausted."
            )
            tool_traces.append({"tool": tc["tool"], "duration_s": round(duration, 2), "lines": 0, "error": tool_result["error"]})
            if trace_queue is not None:
                trace_queue.put_nowait({"event": "tool_trace", "data": tool_traces[-1]})
        else:
            result_text = (
                f"[tool_result: {tc['tool']}] OK\n"
                + json.dumps(tool_result, ensure_ascii=False, indent=2)
            )
            tool_traces.append({"tool": tc["tool"], "duration_s": round(duration, 2), "lines": result_text.count("\n"), "error": None})
            if trace_queue is not None:
                trace_queue.put_nowait({"event": "tool_trace", "data": tool_traces[-1]})

        full_messages.append({"role": "assistant", "content": content})
        full_messages.append({"role": "user", "content": result_text})

    # Exhausted iterations without a final answer — ask for a plain-text summary
    full_messages.append({"role": "assistant", "content": content})
    full_messages.append({"role": "user", "content": "Summarize what happened and give your final answer in plain text."})
    resp    = await _llm_call(full_messages, cfg, temperature, role_cfg=role_cfg)
    content = _content(resp)
    return content, full_messages, tool_traces


# ── Supervisor ────────────────────────────────────────────────────────────────

async def _run_supervisor(
    worker_response:    str,
    original_messages:  list[dict],
    system_prompt:      str,
    cfg:                dict,
    include_history:    bool,
    role_cfg:           dict | None = None,
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
        resp    = await _llm_call(messages, cfg, role_cfg=role_cfg)
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

async def run_agent_loop(body: dict, session_id: str, trace_queue: asyncio.Queue | None = None) -> dict:
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
    w_tools        = _mode_tools(cfg, mode, w_cfg.get("allowed_tools", []))
    w_max_iter     = w_cfg.get("max_tool_iterations", 10)
    max_retries    = cfg["agent"]["max_retries"]
    sup_enabled    = cfg["agent"]["supervisor_enabled"]
    temperature    = _mode_temperature(cfg, mode)
    approved_tools = body.get("approved_tools", [])

    best_response = ""
    best_score    = -1.0
    tool_traces: list[dict] = []

    total_attempts = 1 + max_retries if sup_enabled else 1

    verbose_tools = cfg["logging"].get("verbose_tools", False)

    # Gate tool-trace streaming on verbose_tools; the done sentinel always uses trace_queue directly.
    _trace_q = trace_queue if (trace_queue is not None and verbose_tools) else None

    def _with_traces(resp: dict) -> dict:
        if verbose_tools and tool_traces:
            resp["tool_trace"] = tool_traces
        return resp

    result_holder: dict | None = None
    try:
        for attempt in range(total_attempts):
            worker_prompt, _ = generate(
                role="worker",
                allowed_tools=w_tools,
                session_id=session_id,
                attempt=attempt,
                agent_mode=mode,
                extra={"{{AGENT_MODE}}": _mode_context_string(mode)},
            )

            if trace_queue is not None and attempt > 0:
                trace_queue.put_nowait({"event": "retry", "data": {"attempt": attempt}})

            try:
                worker_response, _, tool_traces = await _run_worker(
                    messages, worker_prompt, w_tools, w_max_iter, cfg, temperature, mode, approved_tools,
                    role_cfg=w_cfg,
                    session_id=session_id,
                    trace_queue=_trace_q,
                )
            except httpx.HTTPError as e:
                worker_response = f"[LLM error: {e}]"

            if cfg["logging"]["log_supervisor_turns"]:
                logger.log_turn(attempt, "worker", messages, worker_response)

            if not sup_enabled:
                logger.log_turn(0, "final", raw_input, worker_response)
                await _auto_store_memory(raw_input, worker_response, session_id, mode, w_tools)
                result_holder = _with_traces(_format_response(worker_response, session_id))
                return result_holder

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
                role_cfg=s_cfg,
            )

            if cfg["logging"]["log_supervisor_turns"]:
                logger.log_turn(attempt, "supervisor", messages, worker_response, supervisor_result)

            score = supervisor_result.get("score", 0.0)
            if score > best_score:
                best_score    = score
                best_response = worker_response

            if supervisor_result.get("pass", False):
                logger.log_turn(0, "final", raw_input, worker_response)
                await _auto_store_memory(raw_input, worker_response, session_id, mode, w_tools)
                result_holder = _with_traces(_format_response(worker_response, session_id))
                return result_holder

            # Prepare retry: inject supervisor feedback as user message
            feedback = supervisor_result.get("feedback", "")
            spawn_hint = supervisor_result.get("suggest_spawn", "")
            if spawn_hint:
                feedback += f" Use `run_agent` to delegate this to `{spawn_hint}`."
            messages = messages + [
                {"role": "assistant", "content": worker_response},
                {"role": "user",      "content": (
                    f"[supervisor_feedback] Score: {score:.2f}. {feedback} "
                    "Please revise your response."
                )},
            ]

        logger.log_turn(0, "final", raw_input, best_response)
        await _auto_store_memory(raw_input, best_response, session_id, mode, w_tools)
        result_holder = _with_traces(_format_response(best_response, session_id))
        return result_holder
    finally:
        if trace_queue is not None:
            trace_queue.put_nowait({"event": "done", "data": result_holder or {"error": "agent loop failed"}})
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
            await _run_worker(messages, prompt, soul_tools, soul_max_it, cfg, role_cfg=soul_cfg,
                              session_id=soul_session)  # traces discarded
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
    mode_cfg       = cfg.get("agent", {}).get("mode", {})
    mode           = body.get("mode", mode_cfg.get("default", "converse"))
    base_tools     = role_cfg.get("allowed_tools", [])
    allowed_tools  = _mode_tools(cfg, mode, base_tools)
    max_iter       = role_cfg.get("max_tool_iterations", 10)
    temperature    = _mode_temperature(cfg, mode)
    approved_tools = body.get("approved_tools", [])
    logger         = SessionLogger(session_id)
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
        agent_mode=mode,
        extra={"{{AGENT_MODE}}": _mode_context_string(mode)},
    )

    verbose_tools = cfg["logging"].get("verbose_tools", False)
    tool_traces: list[dict] = []

    try:
        try:
            response, _, tool_traces = await _run_worker(
                messages, prompt, allowed_tools, max_iter, cfg, temperature, mode, approved_tools,
                role_cfg=role_cfg,
                session_id=session_id,
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
