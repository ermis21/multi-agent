"""Unit tests for plan-mode write scoping in app.mcp_client.call_tool.

In plan mode the only writable target is state/sessions/{sid}/plan.md. All
other write-class tool attempts auto-fail with a clear message — no Discord
approval prompt, no retry loop.
"""

from __future__ import annotations

import asyncio

from app.mcp_client import (
    _PLAN_MODE_WRITE_TOOLS,
    _normalize_rel_path,
    _session_plan_path,
    call_tool,
)


def _run(coro):
    return asyncio.run(coro)


def test_plan_path_prefix_form():
    assert _session_plan_path("abc123") == "state/sessions/abc123/plan.md"


def test_normalize_rel_strips_leading_slash():
    assert _normalize_rel_path("/state/sessions/x/plan.md") == "state/sessions/x/plan.md"
    assert _normalize_rel_path("state/sessions/x/plan.md") == "state/sessions/x/plan.md"


def test_plan_mode_rejects_write_to_other_paths(monkeypatch):
    monkeypatch.setattr(
        "app.mcp_client.get_config",
        lambda: {"approval": {"plan": {"auto_allow": {"tools": [], "paths": []},
                                         "ask_user": [], "auto_fail": []}}},
    )
    async def _boom(*a, **k): raise AssertionError("should not POST to Discord")
    monkeypatch.setattr("app.mcp_client._discord_http.post", _boom, raising=False)

    result = _run(call_tool(
        "create_dir",
        {"path": "config/skills/commitment_tracker"},
        allowed=["create_dir"],
        mode="plan",
        session_id="sid_plan_reject",
    ))
    assert "error" in result
    assert "plan.md" in result["error"]
    assert "Plan mode" in result["error"]


def test_plan_mode_allows_write_to_session_plan_file(monkeypatch):
    """file_write to the session plan file auto-allows (no Discord hit, no ask_user)."""
    monkeypatch.setattr(
        "app.mcp_client.get_config",
        lambda: {"approval": {"plan": {"auto_allow": {"tools": [], "paths": []},
                                         "ask_user": ["file_write"], "auto_fail": []}}},
    )
    async def _boom(*a, **k): raise AssertionError("should not POST to Discord for plan file")
    monkeypatch.setattr("app.mcp_client._discord_http.post", _boom, raising=False)

    async def _fake_post(url, json=None, **kw):
        class _R:
            status_code = 200
            def json(self_inner): return {"result": {"bytes_written": 10}}
            def raise_for_status(self_inner): pass
        return _R()
    monkeypatch.setattr("app.mcp_client._client.post", _fake_post, raising=False)

    sid = "sid_plan_allow"
    result = _run(call_tool(
        "file_write",
        {"path": _session_plan_path(sid), "content": "# plan"},
        allowed=["file_write"],
        mode="plan",
        session_id=sid,
    ))
    assert "error" not in result, result


def test_plan_mode_rejects_write_config(monkeypatch):
    """write_config has no `path` — auto-fails with the scoping message."""
    monkeypatch.setattr(
        "app.mcp_client.get_config",
        lambda: {"approval": {"plan": {"auto_allow": {"tools": [], "paths": []},
                                         "ask_user": ["write_config"], "auto_fail": []}}},
    )
    async def _boom(*a, **k): raise AssertionError("should not POST to Discord")
    monkeypatch.setattr("app.mcp_client._discord_http.post", _boom, raising=False)

    result = _run(call_tool(
        "write_config",
        {"agent.supervisor_enabled": False},
        allowed=["write_config"],
        mode="plan",
        session_id="sid_writecfg",
    ))
    assert "error" in result
    assert "Plan mode" in result["error"]


def test_plan_mode_write_scope_does_not_affect_build_mode(monkeypatch):
    """Build mode is unaffected by the plan-mode gate (still uses ask_user / auto_allow)."""
    monkeypatch.setattr(
        "app.mcp_client.get_config",
        lambda: {"approval": {"build": {"auto_allow": {"tools": ["create_dir"], "paths": []},
                                          "ask_user": ["create_dir"], "auto_fail": []}}},
    )
    async def _boom(*a, **k): raise AssertionError("should not POST to Discord when pre-approved")
    monkeypatch.setattr("app.mcp_client._discord_http.post", _boom, raising=False)

    async def _fake_post(url, json=None, **kw):
        class _R:
            status_code = 200
            def json(self_inner): return {"result": {"created": True}}
            def raise_for_status(self_inner): pass
        return _R()
    monkeypatch.setattr("app.mcp_client._client.post", _fake_post, raising=False)

    result = _run(call_tool(
        "create_dir",
        {"path": "config/skills/foo"},
        allowed=["create_dir"],
        mode="build",
        approved_tools=["create_dir"],
        session_id="sid_build",
    ))
    assert "error" not in result, result


def test_plan_mode_write_tools_constant_covers_expected_set():
    """If we add a new write tool, this test catches forgetting to gate it."""
    assert {"file_write", "file_edit", "create_dir", "file_move",
            "write_config"} <= _PLAN_MODE_WRITE_TOOLS
