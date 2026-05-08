"""
MCP client — bridges phoebe-api to phoebe-sandbox.

All tool calls flow through call_tool():
  1. Check if the method is in the agent's allowed_tools list.
  2. Route LOCAL_TOOLS to local handlers via registry — no HTTP.
  3. Everything else → POST http://phoebe-sandbox:9000/mcp

Timeouts:
  File ops (file_read, file_write, file_list): 10s
  All others (shell_exec, git_*, docker_*):    130s
"""

import json
import os
import re

import httpx
import json_repair

from app.authorizer import authorize, resolve_approval  # noqa: F401 — re-exported
from app.sessions.state import SessionState
from app.tools import registry as _tool_registry

SANDBOX_URL = os.environ.get("SANDBOX_URL", "http://phoebe-sandbox:9000")

FAST_TIMEOUT_S = 10
SLOW_TIMEOUT_S = 130

# Sandbox-side slow tools (not local; derived from sandbox HANDLERS)
_SANDBOX_SLOW_TOOLS: frozenset[str] = frozenset({
    "shell_exec", "execute_command",
    "git_status", "git_commit", "git_rollback", "git_log",
    "docker_test_up", "docker_test_down", "docker_test_health",
    "web_search", "web_fetch", "skill_install",
    "memory_add", "memory_search", "memory_list",
    "notion_search", "notion_get_page", "notion_create_page", "notion_update_page",
    "discord_send", "discord_read", "discord_set_nickname", "discord_edit_channel",
    "tts_speak",
    "diagnostic_check",
})

# Derived from registry at import time
LOCAL_TOOLS: frozenset[str] = _tool_registry.local_tool_names()
SLOW_TOOLS: frozenset[str] = _SANDBOX_SLOW_TOOLS | _tool_registry.slow_tool_names()

# Shared client — connection pool reused across tool calls in the same process.
# Per-request timeout is passed to each .post() call, overriding the client default.
_client = httpx.AsyncClient(timeout=SLOW_TIMEOUT_S)


async def call_tool(
    method: str,
    params: dict,
    allowed: list[str],
    mode: str = "converse",
    approved_tools: list[str] | None = None,
    session_id: str = "",
    spawnable_agents: list[str] | None = None,
    extra_auto_allow_paths: list[str] | None = None,
    trace_queue: "asyncio.Queue | None" = None,
) -> dict:
    """
    Execute a tool call.

    Returns a result dict on success.
    Returns {"error": "..."} on permission denial, execution failure, or user decline.
    When approval is needed, blocks until the user responds via Discord.
    Never raises — results are always returned as data.

    Wrapped in an OpenInference span (B6) so each call appears as
    `phoebe.tool.{method}` with `session_id`, `mode`, and `error` attributes.
    The downstream HTTP POST to phoebe-sandbox is auto-instrumented separately
    via the httpx instrumentor — its span attaches as a child here.
    """
    from app.observability import get_tracer, annotate
    tracer = get_tracer()
    span_ctx = tracer.start_as_current_span(f"phoebe.tool.{method}")
    with span_ctx as span:
        annotate(span, **{
            "phoebe.tool":       method,
            "phoebe.session_id": session_id,
            "phoebe.mode":       mode,
        })
        result = await _call_tool_impl(
            method, params, allowed, mode, approved_tools, session_id,
            spawnable_agents, extra_auto_allow_paths, trace_queue,
        )
        if isinstance(result, dict) and "error" in result:
            annotate(span, **{"phoebe.tool.error": str(result["error"])[:200]})
        return result


async def _call_tool_impl(
    method: str,
    params: dict,
    allowed: list[str],
    mode: str,
    approved_tools: list[str] | None,
    session_id: str,
    spawnable_agents: list[str] | None,
    extra_auto_allow_paths: list[str] | None,
    trace_queue: "asyncio.Queue | None",
) -> dict:
    """Internal: original call_tool body. Wrapped by call_tool() above for B6."""
    if method not in allowed:
        return {"error": f"Tool '{method}' is not permitted for this agent role. Allowed: {allowed}"}

    from app.config_loader import get_config
    cfg = get_config()
    state: SessionState | None = None
    if session_id:
        try:
            state = SessionState.load_or_create(session_id)
        except Exception:
            state = None

    pre_approved: set[str] = set(approved_tools or [])
    decision = await authorize(
        method=method,
        params=params,
        mode=mode,
        cfg=cfg,
        session_id=session_id,
        state=state,
        pre_approved=pre_approved,
        extra_auto_allow_paths=extra_auto_allow_paths,
        trace_queue=trace_queue,
    )
    if not decision.allowed:
        return {"error": decision.error_message}
    # Persist "Always"-approvals into the caller's session-scoped list.
    if approved_tools is not None:
        for tool in decision.always_approve:
            if tool not in approved_tools:
                approved_tools.append(tool)

    # Inject spawnable_agents into params for run_agent handler
    if spawnable_agents:
        params = {**params, "spawnable_agents": spawnable_agents}

    # Try registry dispatch (local tools)
    result = await _tool_registry.dispatch(method, params, session_id, mode, state)
    if result is not None:
        return result

    # Dream simulator: inject the `_simulate` marker so sandbox handlers route
    # writes into the per-replay overlay instead of mutating real state.
    # Contextvars task-propagate, so sub-agents spawned during a sim inherit it.
    try:
        from app.dream import sim_context as _sim_context
        _sim = _sim_context.current()
    except Exception:
        _sim = None
    if _sim is not None and "_simulate" not in params:
        params = {**params, "_simulate": _sim_context.as_sandbox_marker(_sim)}

    timeout = SLOW_TIMEOUT_S if method in SLOW_TOOLS else FAST_TIMEOUT_S
    try:
        r = await _client.post(
            f"{SANDBOX_URL}/mcp",
            json={"method": method, "params": params},
            timeout=timeout,
        )
        r.raise_for_status()
        data = r.json()
    except httpx.TimeoutException:
        return {"error": f"Tool '{method}' timed out after {timeout}s"}
    except httpx.HTTPStatusError as e:
        return {"error": f"Sandbox HTTP {e.response.status_code}: {e.response.text[:200]}"}
    except Exception as e:
        return {"error": f"Sandbox unreachable: {e}"}

    error = data.get("error")
    if error is not None:
        return {"error": error or "Tool failed with no error detail"}
    result = data.get("result")
    if not isinstance(result, dict):
        return {"error": f"Tool returned no usable result (got {type(result).__name__!r})"}
    return result


_GEMMA_TOOL_RE = re.compile(
    # Gemma / Hermes: <|tool_call|>call: NAME, {...}<|tool_call|>
    # Also tolerates the asymmetric variant the model actually emits:
    #   <|tool_call>call: NAME, {...}<tool_call|>
    # and the closing <|/tool_call|> variant.
    r"<\|?tool_call\|?>\s*call:\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*,\s*(\{.*?\})\s*<\|?/?tool_call\|?>",
    re.DOTALL,
)


def _try_repair_json(text: str, branch: str) -> object | None:
    """Best-effort JSON repair fallback for malformed tool-call payloads.

    Logs `[tool_call_repair] branch=<branch>` on success so we can measure
    which parser branches benefit most. Returns parsed object or None.
    """
    try:
        result = json_repair.loads(text)
    except Exception:
        return None
    if not result:
        return None
    print(f"[tool_call_repair] branch={branch}", flush=True)
    return result


def _extract_tool_call(content: str) -> dict | None:
    """
    Detect a tool call in the model's response.

    Returns a dict {"tool": name, "params": {...}} or None if this is a final answer.

    Handles four formats:
      0. Gemma/Hermes:         <|tool_call|>call: NAME, {...}<|tool_call|>
      1. Bare JSON:            {"tool": "...", "params": {...}}
      2. Markdown fence:       ```json\n{...}\n```
      3. JSON anywhere in prose: "Let me check:\n{...}"

    Each branch falls back to json_repair.loads on JSONDecodeError before
    giving up — catches trailing commas, unquoted keys, broken escapes, etc.
    """
    stripped = content.strip()

    # 0. Gemma / Hermes native format (the model's trained token pattern)
    m = _GEMMA_TOOL_RE.search(stripped)
    if m:
        params = None
        try:
            params = json.loads(m.group(2))
        except json.JSONDecodeError:
            params = _try_repair_json(m.group(2), "gemma_native")
        if isinstance(params, dict):
            # Tolerate the nested {"params": {...}} variant the model sometimes emits
            if set(params.keys()) == {"params"} and isinstance(params["params"], dict):
                params = params["params"]
            return {"tool": m.group(1), "params": params}

    # 1. Entire response is a JSON tool call
    if stripped.startswith("{"):
        obj = None
        try:
            obj = json.loads(stripped)
        except json.JSONDecodeError:
            obj = _try_repair_json(stripped, "bare_json")
        if isinstance(obj, dict) and "tool" in obj and isinstance(obj.get("params"), dict):
            return obj

    # 2. JSON inside a markdown code fence
    for fence in ("```json", "```"):
        start = stripped.find(fence)
        if start != -1:
            inner_start = stripped.find("\n", start) + 1
            end = stripped.find("```", inner_start)
            if end != -1:
                inner = stripped[inner_start:end].strip()
                obj = None
                try:
                    obj = json.loads(inner)
                except json.JSONDecodeError:
                    obj = _try_repair_json(inner, "fenced")
                if isinstance(obj, dict) and "tool" in obj and isinstance(obj.get("params"), dict):
                    return obj

    # 3. JSON embedded in prose — scan for any {...} containing "tool"
    depth, start_idx = 0, -1
    for i, ch in enumerate(stripped):
        if ch == "{":
            if depth == 0:
                start_idx = i
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start_idx != -1:
                candidate = stripped[start_idx:i + 1]
                obj = None
                try:
                    obj = json.loads(candidate)
                except json.JSONDecodeError:
                    obj = _try_repair_json(candidate, "in_prose")
                if isinstance(obj, dict) and "tool" in obj and isinstance(obj.get("params"), dict):
                    return obj
                start_idx = -1

    return None


def strip_json_fences(raw: str) -> str:
    """
    Strip markdown code fences from a string expected to contain raw JSON.
    Used by the supervisor parser in agents.py.
    """
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    return raw
