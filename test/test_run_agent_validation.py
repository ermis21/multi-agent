"""Unit tests for the run_agent local tool's parameter coercion + errors.

Weak workers routinely call it with `agent_name` instead of `role`. The handler
coerces the alias silently and emits a richer error when both keys are missing.
"""

from __future__ import annotations

import asyncio

from app.mcp_client import _run_agent_tool


def _run(coro):
    return asyncio.run(coro)


def test_missing_role_error_lists_available_agents():
    result = _run(_run_agent_tool({"task": "x"}, "sid_x", ["skill_builder", "coding_agent"]))
    assert "error" in result
    err = result["error"]
    assert "role" in err
    assert "skill_builder" in err
    assert "coding_agent" in err
    # Example should be present — a concrete call template.
    assert '"role"' in err or "'role'" in err


def test_missing_task_error_lists_available_agents():
    result = _run(_run_agent_tool({"role": "skill_builder"}, "sid_x", ["skill_builder"]))
    assert "error" in result
    assert "task" in result["error"]


def test_unknown_role_error_lists_alternatives():
    result = _run(_run_agent_tool({"role": "skill_author", "task": "x"}, "sid_x", ["skill_builder"]))
    assert "error" in result
    assert "skill_builder" in result["error"]


def test_agent_name_alias_coerces_to_role(monkeypatch):
    """Coercion exists to spare weak models from the retry loop."""
    captured: dict = {}

    async def _fake_run_agent_role(role, body, child_sid):
        captured["role"] = role
        captured["child_sid"] = child_sid
        return {"choices": [{"message": {"role": "assistant", "content": "done"}}]}

    import app.agents as agents_mod
    monkeypatch.setattr(agents_mod, "run_agent_role", _fake_run_agent_role, raising=False)

    result = _run(_run_agent_tool(
        {"agent_name": "skill_builder", "task": "draft a skill"},
        "sid_coerce", ["skill_builder"],
    ))
    assert "error" not in result, result
    assert captured["role"] == "skill_builder"
    assert result.get("role") == "skill_builder"


def test_other_aliases_also_coerce(monkeypatch):
    import app.agents as agents_mod

    async def _fake(role, body, child_sid):
        return {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}

    monkeypatch.setattr(agents_mod, "run_agent_role", _fake, raising=False)

    for alias in ("agent", "sub_agent", "name"):
        result = _run(_run_agent_tool(
            {alias: "coding_agent", "task": "x"}, "sid", ["coding_agent"],
        ))
        assert "error" not in result, f"alias {alias!r} failed: {result}"
