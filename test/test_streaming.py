"""Tests for SSE streaming helpers + ordering invariant."""

import time

import pytest

from app.agents import _short_params


# ── _short_params (pure-unit) ─────────────────────────────────────────────────

def test_short_params_empty():
    assert _short_params({}) == ""


def test_short_params_query_key_preferred():
    assert _short_params({"query": "foo bar"}) == 'query="foo bar"'


def test_short_params_truncates_long_value():
    out = _short_params({"query": "x" * 100}, max_chars=20)
    assert len(out) <= 20 + len('query=""')
    assert out.endswith('…"')


def test_short_params_strips_newlines():
    out = _short_params({"command": "line1\nline2"})
    assert "\n" not in out


def test_short_params_fallback_to_json():
    out = _short_params({"unknown_key": "value", "another": 42})
    # Neither key is in the preferred list → falls back to JSON dump
    assert "unknown_key" in out or "another" in out


def test_short_params_prefers_command_key():
    # command is in the preferred list
    out = _short_params({"command": "ls /workspace"})
    assert out.startswith('command="')


# ── Live: tool_started precedes worker text ───────────────────────────────────

@pytest.mark.live
def test_tool_started_precedes_tool_trace(client):
    """For any tool that completes, its tool_started event MUST arrive before
    its tool_trace event — the call_id pairing is what lets the Discord bot
    edit the placeholder in place.
    """
    from conftest import stream_chat  # noqa

    result = stream_chat(
        client,
        "search the web for fastapi server-sent events",
        mode="build",
        timeout=120,
    )
    started_ids: list[str] = []
    trace_ids: list[str] = []
    for ev in result["events"]:
        if ev["event"] == "tool_started":
            cid = ev["data"].get("call_id")
            if cid:
                started_ids.append(cid)
        elif ev["event"] == "tool_trace":
            cid = ev["data"].get("call_id")
            if cid:
                trace_ids.append(cid)

    if not started_ids:
        pytest.skip("worker emitted no tool calls for this prompt")

    # Every trace has a matching started that came first
    for cid in trace_ids:
        started_idx = next(
            (i for i, e in enumerate(result["events"])
             if e["event"] == "tool_started" and e["data"].get("call_id") == cid),
            None,
        )
        trace_idx = next(
            (i for i, e in enumerate(result["events"])
             if e["event"] == "tool_trace" and e["data"].get("call_id") == cid),
            None,
        )
        assert started_idx is not None, f"tool_trace without tool_started for call_id={cid}"
        assert trace_idx is not None
        assert started_idx < trace_idx, f"tool_started must precede tool_trace for call_id={cid}"


@pytest.mark.live
def test_tool_started_carries_params_preview(client):
    from conftest import stream_chat  # noqa

    result = stream_chat(client, "search the web for qwen3", mode="build", timeout=90)
    started = [e for e in result["events"] if e["event"] == "tool_started"]
    if not started:
        pytest.skip("no tool calls in this run")
    # At least one tool_started should carry a non-empty params_preview
    assert any(e["data"].get("params_preview") for e in started)
