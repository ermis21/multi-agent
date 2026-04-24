"""Integration tests for app.mcp_client.call_tool dispatch of the four
dreamer-facing tools.

Scope: verify LOCAL_TOOLS membership, parameter validation, and that each tool
lands in `app.dream.dream_tools` (not the sandbox). The tool logic itself is
covered by test_dream_tools.py — these tests stub the target functions and
assert call-through + shape of the error payloads for invalid inputs.
"""

from __future__ import annotations

import asyncio

import pytest

from app import mcp_client
from app.dream import dream_tools


DREAM_TOOLS = ("dream_submit", "edit_revise", "dream_finalize", "recal_historical_prompt")


@pytest.fixture
def dispatch_env(monkeypatch):
    """Neutral cfg + network-free sandbox guard. Any leaked HTTP call to the
    sandbox raises loudly so we catch a routing regression immediately."""
    monkeypatch.setattr(mcp_client, "get_config", lambda: {
        "approval": {"build": {"auto_fail": [], "ask_user": [], "auto_allow": {"paths": []}}},
    })

    async def no_sandbox(*args, **kwargs):
        raise AssertionError("sandbox HTTP path must not be reached for local tools")
    monkeypatch.setattr(mcp_client._client, "post", no_sandbox)
    return {}


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro) if False else asyncio.run(coro)


# ── LOCAL_TOOLS membership ───────────────────────────────────────────────────

def test_local_tools_contains_dream_tools():
    for name in DREAM_TOOLS:
        assert name in mcp_client.LOCAL_TOOLS, f"{name!r} missing from LOCAL_TOOLS"


# ── Permission gate still applies ────────────────────────────────────────────

def test_dream_submit_blocked_when_not_in_allowed(dispatch_env):
    result = _run(mcp_client.call_tool(
        "dream_submit", {"path": "worker_full", "new_full_text": "x", "rationale": "y"},
        allowed=["file_read"], mode="build",
    ))
    assert "error" in result and "not permitted" in result["error"]


# ── dream_submit dispatch ────────────────────────────────────────────────────

def test_dream_submit_missing_path_returns_validation_error(dispatch_env):
    result = _run(mcp_client.call_tool(
        "dream_submit", {"new_full_text": "x"},
        allowed=["dream_submit"], mode="build",
    ))
    assert "error" in result and "targets" in result["error"]


def test_dream_submit_non_string_text_returns_validation_error(dispatch_env):
    result = _run(mcp_client.call_tool(
        "dream_submit", {"path": "worker_full", "new_full_text": None},
        allowed=["dream_submit"], mode="build",
    ))
    assert "error" in result and "new_full_text" in result["error"]


def test_dream_submit_routes_to_dream_tools(dispatch_env, monkeypatch):
    """Legacy path+new_full_text call shape still reaches dream_tools
    wrapped into the new `targets=[...]` list form by the mcp_client shim."""
    seen: dict = {}

    async def fake_submit(targets=None, rationale="", *, conversation_sid,
                          session_id, cfg, **_kwargs):
        seen["targets"] = targets
        seen["rationale"] = rationale
        seen["conversation_sid"] = conversation_sid
        seen["session_id"] = session_id
        seen["cfg"] = cfg
        return {"pending_batch_id": "pb-stub", "ok": True}

    monkeypatch.setattr(dream_tools, "dream_submit", fake_submit)
    result = _run(mcp_client.call_tool(
        "dream_submit",
        {"path": "worker_full", "new_full_text": "# hi\n\nbody\n", "rationale": "r"},
        allowed=["dream_submit"], mode="build", session_id="conv-123",
    ))
    assert result == {"pending_batch_id": "pb-stub", "ok": True}
    assert seen["targets"] == [{"path": "worker_full", "new_full_text": "# hi\n\nbody\n"}]
    assert seen["rationale"] == "r"
    assert seen["conversation_sid"] == "conv-123"
    assert seen["session_id"] == "conv-123"
    assert isinstance(seen["cfg"], dict)


def test_dream_submit_multi_target_passthrough(dispatch_env, monkeypatch):
    """Native multi-target call: body carries `targets=[...]` directly."""
    seen: dict = {}

    async def fake_submit(targets=None, rationale="", *, conversation_sid,
                          session_id, cfg, **_kwargs):
        seen["targets"] = targets
        return {"pending_batch_id": "pb-stub", "ok": True}

    monkeypatch.setattr(dream_tools, "dream_submit", fake_submit)
    _run(mcp_client.call_tool(
        "dream_submit",
        {
            "targets": [
                {"path": "worker_full", "new_full_text": "a"},
                {"path": "supervisor_full", "new_full_text": "b"},
            ],
            "rationale": "r",
        },
        allowed=["dream_submit"], mode="build", session_id="conv-999",
    ))
    assert len(seen["targets"]) == 2
    assert seen["targets"][0]["path"] == "worker_full"
    assert seen["targets"][1]["path"] == "supervisor_full"


def test_dream_submit_exception_converted_to_error(dispatch_env, monkeypatch):
    async def boom(**kwargs):
        raise RuntimeError("narrator exploded")
    monkeypatch.setattr(dream_tools, "dream_submit", boom)
    result = _run(mcp_client.call_tool(
        "dream_submit",
        {"path": "worker_full", "new_full_text": "x", "rationale": "r"},
        allowed=["dream_submit"], mode="build",
    ))
    assert "error" in result and "narrator exploded" in result["error"]


# ── edit_revise dispatch ─────────────────────────────────────────────────────

def test_edit_revise_missing_fields_returns_validation_error(dispatch_env):
    result = _run(mcp_client.call_tool(
        "edit_revise", {"phrase_id": "ph-aaa"},
        allowed=["edit_revise"], mode="build",
    ))
    assert "error" in result and "new_text" in result["error"]


def test_edit_revise_routes_to_dream_tools(dispatch_env, monkeypatch):
    seen: dict = {}

    async def fake_revise(*, phrase_id, new_text, rationale, conversation_sid, session_id, cfg):
        seen.update(locals())
        return {"pending_batch_id": "pb-stub", "edit": {"phrase_id": phrase_id}}

    monkeypatch.setattr(dream_tools, "edit_revise", fake_revise)
    result = _run(mcp_client.call_tool(
        "edit_revise",
        {"phrase_id": "ph-abc", "new_text": "replacement", "rationale": "r"},
        allowed=["edit_revise"], mode="build", session_id="conv-9",
    ))
    assert result["edit"]["phrase_id"] == "ph-abc"
    assert seen["phrase_id"] == "ph-abc"
    assert seen["new_text"] == "replacement"
    assert seen["conversation_sid"] == "conv-9"


# ── dream_finalize dispatch ──────────────────────────────────────────────────

def test_dream_finalize_non_list_params_return_validation_error(dispatch_env):
    result = _run(mcp_client.call_tool(
        "dream_finalize", {"keep": "ph-a", "drop": []},
        allowed=["dream_finalize"], mode="build",
    ))
    assert "error" in result and "lists of phrase_ids" in result["error"]


def test_dream_finalize_routes_and_coerces_to_str(dispatch_env, monkeypatch):
    seen: dict = {}

    async def fake_finalize(*, keep, drop, conversation_sid, session_id, cfg, rationale=None):
        seen.update(locals())
        return {"committed": [], "dropped": []}

    monkeypatch.setattr(dream_tools, "dream_finalize", fake_finalize)
    result = _run(mcp_client.call_tool(
        "dream_finalize",
        {"keep": ["ph-a", 42], "drop": ["ph-b"]},
        allowed=["dream_finalize"], mode="build", session_id="conv-x",
    ))
    assert result == {"committed": [], "dropped": []}
    assert seen["keep"] == ["ph-a", "42"]
    assert seen["drop"] == ["ph-b"]
    assert seen["conversation_sid"] == "conv-x"


def test_dream_finalize_missing_params_defaults_to_empty_lists(dispatch_env, monkeypatch):
    seen: dict = {}

    async def fake_finalize(*, keep, drop, conversation_sid, session_id, cfg, rationale=None):
        seen.update(locals())
        return {"committed": [], "dropped": []}

    monkeypatch.setattr(dream_tools, "dream_finalize", fake_finalize)
    _run(mcp_client.call_tool(
        "dream_finalize", {}, allowed=["dream_finalize"], mode="build",
    ))
    assert seen["keep"] == []
    assert seen["drop"] == []


# ── recal_historical_prompt dispatch ─────────────────────────────────────────

def test_recal_historical_prompt_missing_fields_returns_error(dispatch_env):
    result = _run(mcp_client.call_tool(
        "recal_historical_prompt", {"timestamp": "2026-04-01T00:00:00+00:00"},
        allowed=["recal_historical_prompt"], mode="build",
    ))
    assert "error" in result and "prompt_name" in result["error"]


def test_recal_historical_prompt_routes_to_dream_tools(dispatch_env, monkeypatch):
    seen: dict = {}

    async def fake_recal(ts, prompt_name):
        seen["ts"] = ts
        seen["prompt_name"] = prompt_name
        return {"text": "reconstructed", "warnings": []}

    monkeypatch.setattr(dream_tools, "recal_historical_prompt", fake_recal)
    result = _run(mcp_client.call_tool(
        "recal_historical_prompt",
        {"timestamp": "2026-04-01T00:00:00+00:00", "prompt_name": "worker_full"},
        allowed=["recal_historical_prompt"], mode="build",
    ))
    assert result == {"text": "reconstructed", "warnings": []}
    assert seen == {"ts": "2026-04-01T00:00:00+00:00", "prompt_name": "worker_full"}
