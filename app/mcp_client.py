"""
MCP client — bridges phoebe-api to phoebe-sandbox.

All tool calls flow through call_tool():
  1. Check if the method is in the agent's allowed_tools list.
  2. Route LOCAL_TOOLS (read_config, write_config) to local functions — no HTTP.
  3. Everything else → POST http://phoebe-sandbox:9000/mcp

Timeouts:
  File ops (file_read, file_write, file_list): 10s
  All others (shell_exec, git_*, docker_*):    130s
"""

import asyncio
import json
import os
import re
from uuid import uuid4

import httpx

from app.config_loader import get_config, patch_config
from app.session_state import SessionState, log_approval

SANDBOX_URL = os.environ.get("SANDBOX_URL", "http://phoebe-sandbox:9000")

LOCAL_TOOLS: frozenset[str] = frozenset({"read_config", "write_config", "run_agent", "deliberate", "ask_user"})

FAST_TIMEOUT_S = 10
SLOW_TIMEOUT_S = 130
SLOW_TOOLS: frozenset[str] = frozenset({
    "shell_exec", "execute_command",
    "git_status", "git_commit", "git_rollback", "git_log",
    "docker_test_up", "docker_test_down", "docker_test_health",
    "web_search", "web_fetch",
    "memory_add", "memory_search", "memory_list",
    "notion_search", "notion_get_page", "notion_create_page", "notion_update_page",
    "discord_send", "discord_read", "discord_set_nickname", "discord_edit_channel",
    "tts_speak",
    "diagnostic_check",
    "run_agent",
    "deliberate",
    "ask_user",
})

# Shared client — connection pool reused across tool calls in the same process.
# Per-request timeout is passed to each .post() call, overriding the client default.
_client = httpx.AsyncClient(timeout=SLOW_TIMEOUT_S)

# ── Inline approval gate ─────────────────────────────────────────────────────
# When a tool needs user approval, call_tool() POSTs an approval request to the
# Discord service and waits on an asyncio.Event.  The event is set when the user
# clicks a button in Discord, which triggers a callback to /v1/approval_response
# on phoebe-api, which calls resolve_approval().  call_tool() never returns
# "pending_approval" — it simply doesn't return until the tool runs or is denied.

DISCORD_URL = os.environ.get("DISCORD_URL", "http://phoebe-discord:4000")
_discord_http = httpx.AsyncClient(timeout=10)

# approval_id → {event: asyncio.Event, approved: bool, always: bool}
_pending_approvals: dict[str, dict] = {}


def _safe_log_approval(session_id: str, tool: str, status: str, extra: dict | None = None) -> None:
    """Best-effort approval audit — never raises out of the hot path."""
    if not session_id:
        return
    try:
        log_approval(session_id, tool, status, extra)
    except Exception:
        pass


def resolve_approval(approval_id: str, approved: bool, always: bool = False) -> bool:
    """Called by POST /v1/approval_response to unblock a waiting call_tool()."""
    state = _pending_approvals.get(approval_id)
    if not state:
        return False
    state["approved"] = approved
    state["always"] = always
    state["event"].set()
    return True


def _path_is_auto_allowed(method: str, params: dict, auto_allow_paths: list[str]) -> bool:
    """
    Return True if the tool's target path starts with any auto_allow path prefix.
    Used to exempt plan-file writes from ask_user in plan mode.
    """
    if not auto_allow_paths:
        return False
    path = params.get("path") or params.get("destination") or params.get("source", "")
    return any(path.startswith(prefix) for prefix in auto_allow_paths)


async def _run_agent_tool(params: dict, session_id: str, spawnable: list[str] | None) -> dict:
    """Handle the run_agent local tool — spawns a sub-agent via run_agent_role."""
    role = params.get("role", "")
    task = params.get("task", "")
    if not role or not task:
        return {"error": "run_agent requires 'role' and 'task' parameters."}
    allowed = spawnable or []
    if role not in allowed:
        return {"error": f"Cannot spawn '{role}'. Available sub-agents: {allowed}"}
    child_sid = f"{session_id or 'sub'}_{role}_{uuid4().hex[:6]}"

    parent_state = None
    if session_id:
        try:
            parent_state = SessionState.load_or_create(session_id)
            parent_state.add_sub_session(child_sid)
            parent_state.save()
        except Exception:
            parent_state = None
    try:
        child_state = SessionState.load_or_create(child_sid)
        child_state.set("parent_session_id", session_id or None)
        child_state.set("agent_role", role)
        child_state.set("source_trigger", {"type": "sub_agent", "ref": session_id or None})
        child_state.save()
    except Exception:
        pass

    try:
        from app.agents import run_agent_role  # lazy import — avoids circular
        result = await run_agent_role(role, {"messages": [{"role": "user", "content": task}]}, child_sid)
        text = result.get("choices", [{}])[0].get("message", {}).get("content", "")
        return {"role": role, "response": text, "sub_session_id": child_sid}
    except Exception as e:
        return {"error": f"Sub-agent '{role}' failed: {e}"}
    finally:
        if parent_state is not None:
            try:
                parent_state.complete_sub_session(child_sid)
                parent_state.save()
            except Exception:
                pass


async def call_tool(
    method: str,
    params: dict,
    allowed: list[str],
    mode: str = "converse",
    approved_tools: list[str] | None = None,
    session_id: str = "",
    spawnable_agents: list[str] | None = None,
    extra_auto_allow_paths: list[str] | None = None,
) -> dict:
    """
    Execute a tool call.

    Returns a result dict on success.
    Returns {"error": "..."} on permission denial, execution failure, or user decline.
    When approval is needed, blocks until the user responds via Discord.
    Never raises — results are always returned as data.
    """
    if method not in allowed:
        return {"error": f"Tool '{method}' is not permitted for this agent role. Allowed: {allowed}"}

    cfg = get_config()
    mode_approval    = cfg.get("approval", {}).get(mode, {})
    auto_fail        = set(mode_approval.get("auto_fail", []))
    ask_user         = set(mode_approval.get("ask_user", []))
    auto_allow_paths = mode_approval.get("auto_allow", {}).get("paths", [])
    if extra_auto_allow_paths:
        auto_allow_paths = auto_allow_paths + extra_auto_allow_paths
    pre_approved     = set(approved_tools or [])

    # Session-scoped deny lists (Phase 12.6) — consulted before mode-level gates.
    state = None
    if session_id:
        try:
            state = SessionState.load_or_create(session_id)
        except Exception:
            state = None
    if state is not None:
        denied = set(state.get("permissions.denied_tools", []) or [])
        if method in denied:
            _safe_log_approval(session_id, method, "auto_failed", {"reason": "denied_tools"})
            return {"error": f"Tool '{method}' is on this session's denied_tools list."}
        always_deny = state.get("permissions.always_deny_paths", []) or []
        p_path = params.get("path") or params.get("destination") or params.get("source", "")
        if p_path and any(str(p_path).startswith(prefix) for prefix in always_deny):
            _safe_log_approval(session_id, method, "auto_failed",
                               {"reason": "always_deny_paths", "path": str(p_path)})
            return {"error": f"Path '{p_path}' is on this session's always_deny_paths list."}

    # Hard-block — no LLM or user approval can bypass this
    if method in auto_fail:
        _safe_log_approval(session_id, method, "auto_failed", {"reason": "mode_auto_fail", "mode": mode})
        return {"error": f"Tool '{method}' is permanently blocked in {mode!r} mode by system policy."}

    # Ask-user gate — skip if pre-approved or if path is auto-allowed.
    # When approval is needed, POST to Discord and wait for the user's response.
    # call_tool() does NOT return until the tool has been approved+executed or denied.
    needs_ask = method in ask_user
    if needs_ask and (method in pre_approved or _path_is_auto_allowed(method, params, auto_allow_paths)):
        reason = "pre_approved" if method in pre_approved else "auto_allow_path"
        _safe_log_approval(session_id, method, "auto_allowed", {"reason": reason})
    if (needs_ask
            and method not in pre_approved
            and not _path_is_auto_allowed(method, params, auto_allow_paths)):
        approval_id = uuid4().hex[:12]
        event = asyncio.Event()
        _pending_approvals[approval_id] = {"event": event, "approved": False, "always": False}
        _safe_log_approval(session_id, method, "requested", {"approval_id": approval_id, "mode": mode})
        try:
            await _discord_http.post(f"{DISCORD_URL}/discord/request_approval", json={
                "tool": method, "params": params,
                "approval_id": approval_id, "session_id": session_id,
            })
        except Exception as e:
            _pending_approvals.pop(approval_id, None)
            return {"error": f"Could not request approval: {e}"}
        try:
            await asyncio.wait_for(event.wait(), timeout=660)
        except asyncio.TimeoutError:
            _pending_approvals.pop(approval_id, None)
            _safe_log_approval(session_id, method, "timeout", {"approval_id": approval_id})
            return {"error": f"Approval for '{method}' timed out."}
        result = _pending_approvals.pop(approval_id)
        _safe_log_approval(session_id, method,
                           "approved" if result["approved"] else "denied",
                           {"always": bool(result.get("always")), "approval_id": approval_id})
        if not result["approved"]:
            return {"error": f"User declined to run '{method}'."}
        if result["always"] and approved_tools is not None:
            approved_tools.append(method)
        # Fall through → execute the tool normally

    if method == "read_config":
        return {"config": get_config()}

    if method == "write_config":
        from app.config_schema import ConfigPatchError
        try:
            return {"updated": True, "config": patch_config(params)}
        except ConfigPatchError as e:
            return {"error": str(e)}
        except Exception as e:
            return {"error": f"Config write failed: {e}"}

    if method == "run_agent":
        return await _run_agent_tool(params, session_id, spawnable_agents)

    if method == "deliberate":
        from app.debate import run_debate
        result = await run_debate(
            question=params.get("question", ""),
            context=params.get("context", ""),
            position_a=params.get("position_a", ""),
            position_b=params.get("position_b", ""),
            session_id=session_id,
            debate_id=params.get("debate_id", ""),
            max_exchanges=params.get("max_exchanges", 0),
        )
        if session_id and isinstance(result, dict) and result.get("debate_id"):
            try:
                dst = SessionState.load_or_create(session_id)
                dst.set("debate_id", result["debate_id"])
                dst.save()
            except Exception:
                pass
        return result

    if method == "ask_user":
        from app.ask_user import ask_user_question
        return await ask_user_question(
            question=params.get("question", ""),
            options=params.get("options", []),
            context=params.get("context", ""),
            session_id=session_id,
        )

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


def _extract_tool_call(content: str) -> dict | None:
    """
    Detect a tool call in the model's response.

    Returns a dict {"tool": name, "params": {...}} or None if this is a final answer.

    Handles four formats:
      0. Gemma/Hermes:         <|tool_call|>call: NAME, {...}<|tool_call|>
      1. Bare JSON:            {"tool": "...", "params": {...}}
      2. Markdown fence:       ```json\n{...}\n```
      3. JSON anywhere in prose: "Let me check:\n{...}"
    """
    stripped = content.strip()

    # 0. Gemma / Hermes native format (the model's trained token pattern)
    m = _GEMMA_TOOL_RE.search(stripped)
    if m:
        try:
            params = json.loads(m.group(2))
            if isinstance(params, dict):
                # Tolerate the nested {"params": {...}} variant the model sometimes emits
                if set(params.keys()) == {"params"} and isinstance(params["params"], dict):
                    params = params["params"]
                return {"tool": m.group(1), "params": params}
        except json.JSONDecodeError:
            pass

    # 1. Entire response is a JSON tool call
    if stripped.startswith("{"):
        try:
            obj = json.loads(stripped)
            if "tool" in obj and isinstance(obj.get("params"), dict):
                return obj
        except json.JSONDecodeError:
            pass

    # 2. JSON inside a markdown code fence
    for fence in ("```json", "```"):
        start = stripped.find(fence)
        if start != -1:
            inner_start = stripped.find("\n", start) + 1
            end = stripped.find("```", inner_start)
            if end != -1:
                try:
                    obj = json.loads(stripped[inner_start:end].strip())
                    if "tool" in obj and isinstance(obj.get("params"), dict):
                        return obj
                except json.JSONDecodeError:
                    pass

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
                try:
                    obj = json.loads(stripped[start_idx:i + 1])
                    if "tool" in obj and isinstance(obj.get("params"), dict):
                        return obj
                except json.JSONDecodeError:
                    pass
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
