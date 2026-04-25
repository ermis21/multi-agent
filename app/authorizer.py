"""
Authorizer — approval policy for tool calls.

`call_tool()` asks `authorize()` whether a tool may run; `authorize()` collapses
five sub-decisions (session deny lists, mode `auto_fail`, plan-mode write scope,
ask-user gate with auto-allow exemptions, ask-user Discord roundtrip) behind a
single seam. The pre-allowlist filter (`method in role.allowed_tools`) is
*not* part of policy — it stays in the caller as a separate concern.

Decision precedence is invariant (see CLAUDE.md §16):

    1. state.permissions.denied_tools           → deny  (auto_failed)
    2. state.permissions.always_deny_paths      → deny  (auto_failed)
    3. cfg.approval[mode].auto_fail             → deny  (auto_failed)
    4. plan-mode write-scope (mode == "plan")   → deny on path mismatch,
                                                  bypass ask-user on match
    5. ask-user gate (cfg.approval[mode].ask_user)
        a. method in pre_approved               → allow (auto_allowed)
        b. params.path in auto_allow.paths      → allow (auto_allowed)
        c. Discord roundtrip                    → allow / deny / timeout
    6. default                                  → allow

Approval audit (`state/sessions/<sid>/approvals.jsonl`) is written inline at
each gate — best-effort, never raises out of the hot path. SessionAudit
unification is a future refactor; it does not belong in this file.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field
from uuid import uuid4

import httpx

from app.sessions.state import SessionState, log_approval

# ── Discord ask-user wiring ──────────────────────────────────────────────────
# When a tool needs user approval, authorize() POSTs an approval request to the
# Discord service and waits on an asyncio.Event. The event is set when the user
# clicks a button in Discord, which triggers a callback to /v1/approval_response
# on phoebe-api, which calls resolve_approval(). authorize() never returns
# "pending_approval" — it simply doesn't return until the tool is approved or
# denied (660s hard timeout).

DISCORD_URL = os.environ.get("DISCORD_URL", "http://phoebe-discord:4000")
_discord_http = httpx.AsyncClient(timeout=10)

# approval_id → {event: asyncio.Event, approved: bool, always: bool}
_pending_approvals: dict[str, dict] = {}

# Plan-mode write scoping — in plan mode, the only writable target is the
# session plan file. Other writes auto-fail so the worker stops asking for
# them mid-plan. See RC3 in /home/homelab/.claude/plans/20-58-lets-recursive-owl.md.
_PLAN_MODE_WRITE_TOOLS: frozenset[str] = frozenset({
    "file_write", "file_edit", "create_dir", "file_move", "write_config",
})


@dataclass
class AuthDecision:
    allowed: bool
    error_message: str | None = None
    # Tools the caller should append to its session-scoped `approved_tools`
    # list. Populated only when the Discord user ticked "Always" — plan-mode
    # bypass is per-call and is *not* propagated.
    always_approve: tuple[str, ...] = field(default_factory=tuple)


def resolve_approval(approval_id: str, approved: bool, always: bool = False) -> bool:
    """Called by POST /v1/approval_response to unblock a waiting authorize()."""
    state = _pending_approvals.get(approval_id)
    if not state:
        return False
    state["approved"] = approved
    state["always"] = always
    state["event"].set()
    return True


def _safe_log_approval(session_id: str, tool: str, status: str, extra: dict | None = None) -> None:
    """Best-effort approval audit — never raises out of the hot path."""
    if not session_id:
        return
    try:
        log_approval(session_id, tool, status, extra)
    except Exception:
        pass


def _path_is_auto_allowed(method: str, params: dict, auto_allow_paths: list[str]) -> bool:
    """True if the tool's target path starts with any auto_allow path prefix."""
    if not auto_allow_paths:
        return False
    path = params.get("path") or params.get("destination") or params.get("source", "")
    return any(path.startswith(prefix) for prefix in auto_allow_paths)


def _session_plan_path(session_id: str) -> str:
    """Canonical plan-file path for a session, in the prefix-form tools expect."""
    sid = session_id or "unknown"
    return f"state/sessions/{sid}/plan.md"


def _normalize_rel_path(path: str) -> str:
    """Strip leading slash + normalize separators for prefix comparison."""
    return str(path or "").lstrip("/").replace("\\", "/")


async def _approval_heartbeat(trace_queue: "asyncio.Queue | None", tool: str,
                               approval_id: str, interval: float = 90.0) -> None:
    """Emit periodic `approval_waiting` events into the trace queue so the SSE
    stream's idle timer stays armed and the Discord bot can show progress.
    Cancelled by the caller once `event.wait()` resolves.
    """
    if trace_queue is None:
        return
    elapsed = 0.0
    try:
        while True:
            await asyncio.sleep(interval)
            elapsed += interval
            try:
                trace_queue.put_nowait({
                    "event": "approval_waiting",
                    "data": {"tool": tool, "approval_id": approval_id,
                             "elapsed_s": int(elapsed)},
                })
            except Exception:
                return
    except asyncio.CancelledError:
        return


async def _ask_user_via_discord(
    method: str,
    params: dict,
    session_id: str,
    pre_approved: set[str],
    trace_queue: "asyncio.Queue | None",
) -> AuthDecision:
    """Request approval through Discord and block until the user responds."""
    approval_id = uuid4().hex[:12]
    event = asyncio.Event()
    _pending_approvals[approval_id] = {"event": event, "approved": False, "always": False}
    _safe_log_approval(session_id, method, "requested",
                       {"approval_id": approval_id})
    try:
        await _discord_http.post(f"{DISCORD_URL}/discord/request_approval", json={
            "tool": method, "params": params,
            "approval_id": approval_id, "session_id": session_id,
        })
    except Exception as e:
        _pending_approvals.pop(approval_id, None)
        return AuthDecision(False, f"Could not request approval: {e}")

    hb_task = asyncio.create_task(_approval_heartbeat(trace_queue, method, approval_id))
    try:
        try:
            await asyncio.wait_for(event.wait(), timeout=660)
        except asyncio.TimeoutError:
            _pending_approvals.pop(approval_id, None)
            _safe_log_approval(session_id, method, "timeout",
                               {"approval_id": approval_id})
            return AuthDecision(False, f"Approval for '{method}' timed out.")
    finally:
        hb_task.cancel()
        try:
            await hb_task
        except (asyncio.CancelledError, Exception):
            pass

    result = _pending_approvals.pop(approval_id)
    _safe_log_approval(session_id, method,
                       "approved" if result["approved"] else "denied",
                       {"always": bool(result.get("always")),
                        "approval_id": approval_id})
    if not result["approved"]:
        return AuthDecision(False, f"User declined to run '{method}'.")
    always = bool(result.get("always"))
    if always:
        pre_approved.add(method)
    return AuthDecision(True, always_approve=((method,) if always else ()))


async def authorize(
    method: str,
    params: dict,
    mode: str,
    cfg: dict,
    session_id: str,
    state: SessionState | None,
    pre_approved: set[str],
    extra_auto_allow_paths: list[str] | None,
    trace_queue: "asyncio.Queue | None",
) -> AuthDecision:
    """Decide whether `method` may run and (if approval is needed) ask the user.

    Returns an AuthDecision; on `allowed=False`, `error_message` is the user-
    facing string the caller should surface as `{"error": ...}`.
    """
    mode_approval    = cfg.get("approval", {}).get(mode, {})
    auto_fail        = set(mode_approval.get("auto_fail", []))
    ask_user_tools   = set(mode_approval.get("ask_user", []))
    auto_allow_paths = list(mode_approval.get("auto_allow", {}).get("paths", []) or [])
    if extra_auto_allow_paths:
        auto_allow_paths = auto_allow_paths + list(extra_auto_allow_paths)

    # 1. Session-scoped deny lists
    if state is not None:
        denied = set(state.get("permissions.denied_tools", []) or [])
        if method in denied:
            _safe_log_approval(session_id, method, "auto_failed",
                               {"reason": "denied_tools"})
            return AuthDecision(False,
                                f"Tool '{method}' is on this session's denied_tools list.")
        always_deny = state.get("permissions.always_deny_paths", []) or []
        p_path = params.get("path") or params.get("destination") or params.get("source", "")
        if p_path and any(str(p_path).startswith(prefix) for prefix in always_deny):
            _safe_log_approval(session_id, method, "auto_failed",
                               {"reason": "always_deny_paths", "path": str(p_path)})
            return AuthDecision(False,
                                f"Path '{p_path}' is on this session's always_deny_paths list.")

    # 2. Mode auto_fail — hard block, no LLM or user approval can bypass
    if method in auto_fail:
        _safe_log_approval(session_id, method, "auto_failed",
                           {"reason": "mode_auto_fail", "mode": mode})
        return AuthDecision(False,
                            f"Tool '{method}' is permanently blocked in {mode!r} mode by system policy.")

    # 3. Plan-mode write scoping
    plan_mode_bypass = False
    if mode == "plan" and method in _PLAN_MODE_WRITE_TOOLS:
        plan_file = _session_plan_path(session_id)
        target = params.get("path") or params.get("destination") or ""
        if _normalize_rel_path(target) != _normalize_rel_path(plan_file):
            _safe_log_approval(session_id, method, "auto_failed",
                               {"reason": "plan_mode_write_scope",
                                "attempted_path": str(target),
                                "plan_file": plan_file})
            return AuthDecision(False, (
                f"Plan mode only writes to '{plan_file}'. "
                f"Put your plan there, or switch to build mode for general writes."
            ))
        plan_mode_bypass = True
        _safe_log_approval(session_id, method, "auto_allowed",
                           {"reason": "plan_mode_plan_file", "path": str(target)})

    # 4. Ask-user gate
    needs_ask = method in ask_user_tools
    if needs_ask:
        if plan_mode_bypass or method in pre_approved:
            reason = ("plan_mode_plan_file" if plan_mode_bypass
                      else "pre_approved")
            # Plan-mode already logged its auto_allowed above; only log the
            # pre_approved-shortcut case here.
            if not plan_mode_bypass:
                _safe_log_approval(session_id, method, "auto_allowed",
                                   {"reason": reason})
            return AuthDecision(True)
        if _path_is_auto_allowed(method, params, auto_allow_paths):
            _safe_log_approval(session_id, method, "auto_allowed",
                               {"reason": "auto_allow_path"})
            return AuthDecision(True)
        return await _ask_user_via_discord(method, params, session_id,
                                           pre_approved, trace_queue)

    return AuthDecision(True)
