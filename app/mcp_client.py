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

import json
import os
import re
from uuid import uuid4

import httpx

from app.authorizer import authorize, resolve_approval  # noqa: F401 — re-exported
from app.config_loader import get_config, patch_config
from app.sessions.state import SessionState

SANDBOX_URL = os.environ.get("SANDBOX_URL", "http://phoebe-sandbox:9000")

LOCAL_TOOLS: frozenset[str] = frozenset({
    "read_config", "write_config", "run_agent", "deliberate", "ask_user",
    "phrase_history_recall",
    "dream_submit", "edit_revise", "dream_finalize", "recal_historical_prompt",
})

FAST_TIMEOUT_S = 10
SLOW_TIMEOUT_S = 130
SLOW_TOOLS: frozenset[str] = frozenset({
    "shell_exec", "execute_command",
    "git_status", "git_commit", "git_rollback", "git_log",
    "docker_test_up", "docker_test_down", "docker_test_health",
    "web_search", "web_fetch", "skill_install",
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


async def _run_agent_tool(params: dict, session_id: str, spawnable: list[str] | None) -> dict:
    """Handle the run_agent local tool — spawns a sub-agent via run_agent_role."""
    # Coerce common misspellings silently — weak models reach for `agent_name`
    # because `run_agent` reads naturally with it. Intent is unambiguous.
    if not params.get("role"):
        for alt in ("agent_name", "agent", "sub_agent", "name"):
            if params.get(alt):
                params["role"] = params[alt]
                break
    role = params.get("role", "")
    task = params.get("task", "")
    allowed = spawnable or []
    if not role or not task:
        return {"error": (
            f"run_agent needs 'role' (options: {allowed}) and 'task' (full instruction string). "
            'Example: {"role": "' + (allowed[0] if allowed else "coding_agent") + '", '
            '"task": "Draft …"}'
        )}
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
    trace_queue: "asyncio.Queue | None" = None,
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

    if method == "phrase_history_recall":
        from app.dream import phrase_store
        pid = str(params.get("phrase_id") or "").strip()
        if not pid:
            return {"error": "phrase_history_recall requires 'phrase_id'"}
        try:
            k = int(params.get("k") or 3)
        except (TypeError, ValueError):
            k = 3
        try:
            rec = phrase_store._read_index(pid)
            excerpt = phrase_store.get_history_excerpt(pid, k=k)
            return {
                "phrase_id": pid,
                "current_text": rec.get("current_text", ""),
                "rev": rec.get("rev", 0),
                "section_path": rec.get("section_path", ""),
                "path": rec.get("path", ""),
                "history": excerpt,
                "history_total": len(phrase_store.get_history(pid)),
            }
        except phrase_store.LocateFailure as e:
            return {"error": str(e), "unknown_phrase_id": True}
        except Exception as e:
            return {"error": f"phrase_history_recall failed: {e}"}

    # Dream runs target a conversation different from the dreamer's own session.
    # Entrypoint stamps `_dream_conversation_sid` on the dreamer's state; read
    # it here so dream_submit/finalize operate on the right pending batch.
    def _conv_sid_from_state() -> str:
        try:
            st = SessionState.load_or_create(session_id)
            conv = st.get("_dream_conversation_sid")
            if conv:
                return str(conv)
        except Exception:
            pass
        return session_id

    if method == "dream_submit":
        from app.dream import dream_tools
        targets = params.get("targets")
        rationale = str(params.get("rationale") or "").strip()
        # Legacy single-target params (path + new_full_text) still accepted —
        # the dreamer role may call either shape depending on how its prompt
        # has been updated. If `targets` is set, use it; else wrap legacy.
        if targets is None:
            path = str(params.get("path") or "").strip()
            new_full_text = params.get("new_full_text")
            if not path or not isinstance(new_full_text, str):
                return {"error": "dream_submit requires `targets=[{path, new_full_text}, ...]` (or legacy `path` + `new_full_text`)"}
            targets = [{"path": path, "new_full_text": new_full_text}]
        try:
            return await dream_tools.dream_submit(
                targets=targets,
                rationale=rationale,
                conversation_sid=_conv_sid_from_state(),
                session_id=session_id,
                cfg=cfg,
            )
        except Exception as e:
            return {"error": f"dream_submit failed: {e}"}

    if method == "edit_revise":
        from app.dream import dream_tools
        pid = str(params.get("phrase_id") or "").strip()
        new_text = params.get("new_text")
        rationale = str(params.get("rationale") or "").strip()
        if not pid or not isinstance(new_text, str):
            return {"error": "edit_revise requires 'phrase_id' and 'new_text' (str)"}
        try:
            return await dream_tools.edit_revise(
                phrase_id=pid,
                new_text=new_text,
                rationale=rationale,
                conversation_sid=_conv_sid_from_state(),
                session_id=session_id,
                cfg=cfg,
            )
        except Exception as e:
            return {"error": f"edit_revise failed: {e}"}

    if method == "dream_finalize":
        from app.dream import dream_tools
        keep = params.get("keep") or []
        drop = params.get("drop") or []
        rationale = params.get("rationale")
        if not isinstance(keep, list) or not isinstance(drop, list):
            return {"error": "dream_finalize requires 'keep' and 'drop' as lists of phrase_ids"}
        try:
            return await dream_tools.dream_finalize(
                keep=[str(p) for p in keep],
                drop=[str(p) for p in drop],
                conversation_sid=_conv_sid_from_state(),
                session_id=session_id,
                cfg=cfg,
                rationale=(str(rationale) if rationale is not None else None),
            )
        except Exception as e:
            return {"error": f"dream_finalize failed: {e}"}

    if method == "recal_historical_prompt":
        from app.dream import dream_tools
        ts = str(params.get("timestamp") or "").strip()
        prompt_name = str(params.get("prompt_name") or "").strip()
        if not ts or not prompt_name:
            return {"error": "recal_historical_prompt requires 'timestamp' and 'prompt_name'"}
        try:
            return await dream_tools.recal_historical_prompt(ts, prompt_name)
        except Exception as e:
            return {"error": f"recal_historical_prompt failed: {e}"}

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
