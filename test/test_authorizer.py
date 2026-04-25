"""Unit tests for app.authorizer — table-driven approval policy.

Every gate covered without spinning up the tool-calling stack: state denies,
mode auto_fail, plan-mode write scope, ask-user (pre_approved /
auto_allow_paths / extra_auto_allow_paths / Discord roundtrip / 'always'
propagation / timeout), default allow, and the heartbeat.
"""

from __future__ import annotations

import asyncio

import pytest

from app import authorizer
from app.authorizer import AuthDecision, _pending_approvals, authorize, resolve_approval


def _run(coro):
    return asyncio.run(coro)


class _FakeState:
    """Minimal SessionState stand-in — only `.get()` is used."""
    def __init__(self, **values):
        self._values = values

    def get(self, key, default=None):
        return self._values.get(key, default)


# ── Pure-decision gates (no Discord) ─────────────────────────────────────────

def test_default_allow_when_no_gate_matches():
    cfg = {"approval": {"build": {"auto_fail": [], "ask_user": [],
                                   "auto_allow": {"paths": []}}}}
    decision = _run(authorize("file_read", {"path": "p"}, "build", cfg,
                              "sid_default", None, set(), None, None))
    assert decision.allowed and decision.error_message is None


def test_denied_tools_short_circuits(monkeypatch):
    seen = []
    monkeypatch.setattr(authorizer, "_safe_log_approval",
                        lambda sid, t, s, extra=None: seen.append((sid, t, s, extra)))
    state = _FakeState(**{"permissions.denied_tools": ["shell_exec"]})
    cfg = {"approval": {"build": {"auto_fail": ["shell_exec"], "ask_user": [],
                                   "auto_allow": {"paths": []}}}}
    decision = _run(authorize("shell_exec", {"command": "ls"}, "build", cfg,
                              "sid_d", state, set(), None, None))
    assert not decision.allowed
    assert "denied_tools list" in decision.error_message
    # denied_tools log entry came first; auto_fail did not run.
    assert seen[0][2] == "auto_failed"
    assert seen[0][3] == {"reason": "denied_tools"}


def test_always_deny_paths_match_path_destination_source():
    state = _FakeState(**{"permissions.always_deny_paths": ["state/secrets/"]})
    cfg = {"approval": {"build": {"auto_fail": [], "ask_user": [],
                                   "auto_allow": {"paths": []}}}}
    for key in ("path", "destination", "source"):
        decision = _run(authorize("file_read", {key: "state/secrets/api.key"},
                                  "build", cfg, "sid_dp", state, set(), None, None))
        assert not decision.allowed, f"{key} should match always_deny_paths"
        assert "always_deny_paths" in decision.error_message


def test_mode_auto_fail_blocks():
    cfg = {"approval": {"build": {"auto_fail": ["docker_test_up"], "ask_user": [],
                                   "auto_allow": {"paths": []}}}}
    decision = _run(authorize("docker_test_up", {}, "build", cfg,
                              "sid_af", None, set(), None, None))
    assert not decision.allowed
    assert "permanently blocked" in decision.error_message
    assert "'build'" in decision.error_message


# ── Plan-mode write scoping ─────────────────────────────────────────────────

@pytest.mark.parametrize("method", ["file_write", "file_edit", "create_dir",
                                      "file_move", "write_config"])
def test_plan_mode_rejects_non_plan_paths(method):
    cfg = {"approval": {"plan": {"auto_fail": [], "ask_user": [],
                                  "auto_allow": {"paths": []}}}}
    decision = _run(authorize(method, {"path": "config/skills/foo"}, "plan", cfg,
                              "sid_p", None, set(), None, None))
    assert not decision.allowed
    assert "Plan mode" in decision.error_message and "plan.md" in decision.error_message


def test_plan_mode_allows_session_plan_file_and_skips_ask_user(monkeypatch):
    cfg = {"approval": {"plan": {"auto_fail": [], "ask_user": ["file_write"],
                                  "auto_allow": {"paths": []}}}}
    sid = "sid_planok"
    pre_approved: set[str] = set()
    decision = _run(authorize(
        "file_write",
        {"path": f"state/sessions/{sid}/plan.md", "content": "# plan"},
        "plan", cfg, sid, None, pre_approved, None, None,
    ))
    assert decision.allowed
    # plan-mode bypass is per-call; not propagated to caller's approved_tools.
    assert decision.always_approve == ()


def test_plan_mode_does_not_apply_to_non_write_tools():
    cfg = {"approval": {"plan": {"auto_fail": [], "ask_user": [],
                                  "auto_allow": {"paths": []}}}}
    decision = _run(authorize("file_read", {"path": "anywhere"}, "plan", cfg,
                              "sid", None, set(), None, None))
    assert decision.allowed


# ── Ask-user gate: bypass paths ─────────────────────────────────────────────

def test_pre_approved_skips_discord(monkeypatch):
    async def _boom(*a, **kw): raise AssertionError("Discord must not be hit")
    monkeypatch.setattr(authorizer._discord_http, "post", _boom)
    cfg = {"approval": {"build": {"auto_fail": [], "ask_user": ["shell_exec"],
                                   "auto_allow": {"paths": []}}}}
    decision = _run(authorize("shell_exec", {"command": "ls"}, "build", cfg,
                              "sid_pa", None, {"shell_exec"}, None, None))
    assert decision.allowed


def test_auto_allow_path_skips_discord(monkeypatch):
    async def _boom(*a, **kw): raise AssertionError("Discord must not be hit")
    monkeypatch.setattr(authorizer._discord_http, "post", _boom)
    cfg = {"approval": {"build": {"auto_fail": [], "ask_user": ["file_write"],
                                   "auto_allow": {"paths": ["workspace/"]}}}}
    decision = _run(authorize("file_write", {"path": "workspace/notes.md"},
                              "build", cfg, "sid_aap", None, set(), None, None))
    assert decision.allowed


def test_extra_auto_allow_paths_augment_config(monkeypatch):
    async def _boom(*a, **kw): raise AssertionError("Discord must not be hit")
    monkeypatch.setattr(authorizer._discord_http, "post", _boom)
    cfg = {"approval": {"build": {"auto_fail": [], "ask_user": ["file_write"],
                                   "auto_allow": {"paths": []}}}}
    decision = _run(authorize("file_write", {"path": "project/src/a.py"},
                              "build", cfg, "sid_eap", None, set(),
                              ["project/src/"], None))
    assert decision.allowed


# ── Ask-user gate: Discord roundtrip ────────────────────────────────────────

def _stub_discord(monkeypatch, captured: list):
    async def _post(url, json=None, **kw):
        captured.append({"url": url, "json": json})
        class _R:
            status_code = 200
            def json(self_inner): return {}
            def raise_for_status(self_inner): pass
        return _R()
    monkeypatch.setattr(authorizer._discord_http, "post", _post)


def test_user_approves_decision_allowed(monkeypatch):
    captured: list = []
    _stub_discord(monkeypatch, captured)
    cfg = {"approval": {"build": {"auto_fail": [], "ask_user": ["shell_exec"],
                                   "auto_allow": {"paths": []}}}}

    async def _flow():
        # Resolve the approval as soon as one is registered.
        async def _resolver():
            for _ in range(50):
                if _pending_approvals:
                    aid = next(iter(_pending_approvals))
                    resolve_approval(aid, approved=True, always=False)
                    return
                await asyncio.sleep(0.01)
        task = asyncio.create_task(_resolver())
        decision = await authorize("shell_exec", {"command": "ls"}, "build",
                                    cfg, "sid_ok", None, set(), None, None)
        await task
        return decision

    decision = _run(_flow())
    assert decision.allowed
    assert decision.always_approve == ()
    assert captured and captured[0]["url"].endswith("/discord/request_approval")


def test_user_approves_with_always_propagates(monkeypatch):
    _stub_discord(monkeypatch, [])
    cfg = {"approval": {"build": {"auto_fail": [], "ask_user": ["shell_exec"],
                                   "auto_allow": {"paths": []}}}}
    pre_approved: set[str] = set()

    async def _flow():
        async def _resolver():
            for _ in range(50):
                if _pending_approvals:
                    aid = next(iter(_pending_approvals))
                    resolve_approval(aid, approved=True, always=True)
                    return
                await asyncio.sleep(0.01)
        task = asyncio.create_task(_resolver())
        d = await authorize("shell_exec", {"command": "ls"}, "build", cfg,
                             "sid_always", None, pre_approved, None, None)
        await task
        return d

    decision = _run(_flow())
    assert decision.allowed
    assert decision.always_approve == ("shell_exec",)
    assert "shell_exec" in pre_approved   # mutated in place


def test_user_denies_decision_rejected(monkeypatch):
    _stub_discord(monkeypatch, [])
    cfg = {"approval": {"build": {"auto_fail": [], "ask_user": ["shell_exec"],
                                   "auto_allow": {"paths": []}}}}

    async def _flow():
        async def _resolver():
            for _ in range(50):
                if _pending_approvals:
                    aid = next(iter(_pending_approvals))
                    resolve_approval(aid, approved=False, always=False)
                    return
                await asyncio.sleep(0.01)
        task = asyncio.create_task(_resolver())
        d = await authorize("shell_exec", {"command": "ls"}, "build", cfg,
                             "sid_no", None, set(), None, None)
        await task
        return d

    decision = _run(_flow())
    assert not decision.allowed
    assert "User declined" in decision.error_message


def test_discord_post_failure_returns_error(monkeypatch):
    async def _broken(*a, **kw): raise RuntimeError("network is down")
    monkeypatch.setattr(authorizer._discord_http, "post", _broken)
    cfg = {"approval": {"build": {"auto_fail": [], "ask_user": ["shell_exec"],
                                   "auto_allow": {"paths": []}}}}
    decision = _run(authorize("shell_exec", {}, "build", cfg, "sid_brk",
                              None, set(), None, None))
    assert not decision.allowed
    assert "Could not request approval" in decision.error_message
    # Pending entry is cleaned up so the next request gets a fresh slot.
    assert all("sid_brk" not in v for v in _pending_approvals.values())


def test_timeout_returns_error_and_cleans_up(monkeypatch):
    _stub_discord(monkeypatch, [])
    async def _timeout(coro, timeout):
        try:
            coro.close()
        except Exception:
            pass
        raise asyncio.TimeoutError()
    monkeypatch.setattr(asyncio, "wait_for", _timeout)
    cfg = {"approval": {"build": {"auto_fail": [], "ask_user": ["shell_exec"],
                                   "auto_allow": {"paths": []}}}}
    before = len(_pending_approvals)
    decision = _run(authorize("shell_exec", {}, "build", cfg, "sid_to",
                              None, set(), None, None))
    assert not decision.allowed
    assert "timed out" in decision.error_message
    assert len(_pending_approvals) == before  # cleaned up


# ── Heartbeat ───────────────────────────────────────────────────────────────

def test_heartbeat_emits_into_trace_queue(monkeypatch):
    """Patch interval down so a single tick lands before the resolver fires."""
    _stub_discord(monkeypatch, [])
    cfg = {"approval": {"build": {"auto_fail": [], "ask_user": ["shell_exec"],
                                   "auto_allow": {"paths": []}}}}

    real_heartbeat = authorizer._approval_heartbeat
    async def _fast_heartbeat(trace_queue, tool, approval_id, interval=90.0):
        await real_heartbeat(trace_queue, tool, approval_id, interval=0.02)
    monkeypatch.setattr(authorizer, "_approval_heartbeat", _fast_heartbeat)

    queue: asyncio.Queue = asyncio.Queue()

    async def _flow():
        async def _resolver():
            await asyncio.sleep(0.05)
            if _pending_approvals:
                aid = next(iter(_pending_approvals))
                resolve_approval(aid, approved=True, always=False)
        task = asyncio.create_task(_resolver())
        d = await authorize("shell_exec", {}, "build", cfg, "sid_hb",
                             None, set(), None, queue)
        await task
        return d

    decision = _run(_flow())
    assert decision.allowed
    # At least one heartbeat event landed in the queue.
    saw_heartbeat = False
    while not queue.empty():
        ev = queue.get_nowait()
        if ev.get("event") == "approval_waiting":
            saw_heartbeat = True
            break
    assert saw_heartbeat


# ── Decision dataclass ──────────────────────────────────────────────────────

def test_auth_decision_defaults():
    d = AuthDecision(allowed=True)
    assert d.error_message is None
    assert d.always_approve == ()
