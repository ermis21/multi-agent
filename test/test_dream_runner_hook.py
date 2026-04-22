"""Tests for `app.dream.runner_hook` — auto-sim state-machine hook.

Covers:
  - _last_tool_name parses the most recent `[tool_result: NAME]` user message
    (ignores prose, non-user messages, malformed lines, trailing whitespace).
  - make_dream_hook:
      * returns None when no pending batch.
      * returns None when pending batch is not in `submit` phase.
      * returns None when latest tool was dream_submit / edit_revise.
      * fires simulator.run_simulation and returns a synthetic
        `[tool_result: simulate_conversation] OK ...` body when firing.
      * formats SimulatorError as an ERROR body (tool-result shape).
      * catches unexpected exceptions and still yields an ERROR body.
  - rollback_if_unfinalized deletes the pending batch when one exists and
    returns True; returns False when there's nothing to drop.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass

import pytest

from app.dream import dream_state, phrase_store, runner_hook
from app.dream import simulator as dream_simulator


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro) \
        if asyncio.get_event_loop().is_running() is False else asyncio.run(coro)


# ── Fixture: isolated state roots + stub simulator ───────────────────────────

@pytest.fixture
def env(tmp_path, monkeypatch):
    state = tmp_path / "state"
    runs = state / "dream" / "runs"
    runs.mkdir(parents=True)
    monkeypatch.setattr(phrase_store, "STATE_DIR", state)
    monkeypatch.setattr(dream_state, "DREAM_RUNS_ROOT", runs)
    return {"state": state, "runs": runs}


def _seed_pending(conv_sid: str, phase: str = dream_state.PHASE_SUBMIT,
                  simulations_run: int = 0) -> dream_state.PendingBatch:
    batch = dream_state.create_or_replace_pending(
        conversation_sid=conv_sid,
        target_prompt="worker_full",
        new_prompt_text="body",
        rationale="why",
        edits=[{"idx": 0, "phrase_id": "ph-aaaaaaaaaa", "old_text": "a",
                "new_text": "b", "section_path": "Root", "status": "ok"}],
    )
    batch.data["phase"] = phase
    batch.data["simulations_run"] = simulations_run
    dream_state.save_pending(batch)
    return batch


# ── _last_tool_name ──────────────────────────────────────────────────────────

def test_last_tool_name_picks_most_recent():
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "[tool_result: file_read] OK\nbody"},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "[tool_result: dream_submit] OK\n{...}"},
        {"role": "assistant", "content": "thinking"},
    ]
    assert runner_hook._last_tool_name(msgs) == "dream_submit"


def test_last_tool_name_skips_non_tool_user_messages():
    msgs = [
        {"role": "user", "content": "[tool_result: memory_search] OK\n..."},
        {"role": "assistant", "content": "scratch"},
        {"role": "user", "content": "hello, please continue"},
    ]
    assert runner_hook._last_tool_name(msgs) == "memory_search"


def test_last_tool_name_returns_none_when_no_tool_result():
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "just some user text"},
        {"role": "assistant", "content": "reply"},
    ]
    assert runner_hook._last_tool_name(msgs) is None


def test_last_tool_name_tolerates_malformed_line():
    msgs = [
        {"role": "user", "content": "[tool_result: WILDLY MALFORMED"},
        {"role": "user", "content": "[tool_result: good_tool] OK\nbody"},
    ]
    assert runner_hook._last_tool_name(msgs) == "good_tool"


# ── hook: no pending batch ───────────────────────────────────────────────────

def test_hook_returns_none_without_pending_batch(env):
    hook = runner_hook.make_dream_hook("conv-x", cfg={})
    assert asyncio.run(hook([])) is None


# ── hook: phase gating ───────────────────────────────────────────────────────

def test_hook_returns_none_when_phase_is_post_sim(env):
    _seed_pending("conv-1", phase=dream_state.PHASE_POST_SIM)
    hook = runner_hook.make_dream_hook("conv-1", cfg={})
    msgs = [{"role": "user", "content": "[tool_result: file_read] OK\n..."}]
    assert asyncio.run(hook(msgs)) is None


def test_hook_returns_none_when_phase_is_finalize_only(env):
    _seed_pending("conv-2", phase=dream_state.PHASE_FINALIZE_ONLY)
    hook = runner_hook.make_dream_hook("conv-2", cfg={})
    msgs = [{"role": "user", "content": "[tool_result: file_read] OK\n..."}]
    assert asyncio.run(hook(msgs)) is None


# ── hook: latest-tool gating ─────────────────────────────────────────────────

def test_hook_returns_none_when_last_tool_is_dream_submit(env):
    _seed_pending("conv-3")
    hook = runner_hook.make_dream_hook("conv-3", cfg={})
    msgs = [{"role": "user", "content": "[tool_result: dream_submit] OK\n{...}"}]
    assert asyncio.run(hook(msgs)) is None


def test_hook_returns_none_when_last_tool_is_edit_revise(env):
    _seed_pending("conv-4")
    hook = runner_hook.make_dream_hook("conv-4", cfg={})
    msgs = [{"role": "user", "content": "[tool_result: edit_revise] OK\n{...}"}]
    assert asyncio.run(hook(msgs)) is None


# ── hook: firing path ────────────────────────────────────────────────────────

@dataclass
class _FakeResult:
    payload: dict

    def to_payload(self) -> dict:
        return self.payload


def test_hook_fires_sim_when_last_tool_is_unrelated(env, monkeypatch):
    _seed_pending("conv-5")
    captured: dict = {}

    async def fake_run_simulation(sid, cfg):
        captured["sid"] = sid
        captured["cfg"] = cfg
        return _FakeResult(payload={
            "session_id": sid, "model_match": True, "can_iterate": True,
            "simulations_remaining": 2,
            "before": {"transcript": [], "tool_calls": []},
            "after": {"transcript": [{"role": "assistant", "content": "after"}], "tool_calls": []},
        })

    monkeypatch.setattr(dream_simulator, "run_simulation", fake_run_simulation)
    hook = runner_hook.make_dream_hook("conv-5", cfg={"x": 1})
    msgs = [{"role": "user", "content": "[tool_result: file_read] OK\nbody"}]
    out = asyncio.run(hook(msgs))
    assert out is not None
    assert out.startswith("[tool_result: simulate_conversation] OK\n")
    # Body is JSON-formatted payload — parse and assert.
    body = out.split("\n", 1)[1]
    parsed = json.loads(body)
    assert parsed["session_id"] == "conv-5"
    assert parsed["model_match"] is True
    assert captured == {"sid": "conv-5", "cfg": {"x": 1}}


def test_hook_fires_sim_on_end_marker_path_no_prior_tool(env, monkeypatch):
    """After the first dream_submit completes and the dreamer sends plain
    prose + <|end|>, full_messages' last tool message IS dream_submit — so
    hook should NOT fire. We cover the reverse case: an earlier unrelated
    tool followed by no further tool call; hook fires."""
    _seed_pending("conv-6")

    async def fake_run_simulation(sid, cfg):
        return _FakeResult(payload={"session_id": sid, "model_match": False})

    monkeypatch.setattr(dream_simulator, "run_simulation", fake_run_simulation)
    hook = runner_hook.make_dream_hook("conv-6", cfg={})
    # No tool_result in messages at all → last_tool_name is None, which is
    # not in _REVISE_TOOLS, so sim fires.
    out = asyncio.run(hook([{"role": "user", "content": "hello"}]))
    assert out is not None and "simulate_conversation] OK" in out


def test_hook_surfaces_simulator_error_as_tool_result(env, monkeypatch):
    _seed_pending("conv-7")

    async def boom(sid, cfg):
        raise dream_simulator.SimulatorError("cap exceeded")

    monkeypatch.setattr(dream_simulator, "run_simulation", boom)
    hook = runner_hook.make_dream_hook("conv-7", cfg={})
    out = asyncio.run(hook([{"role": "user", "content": "[tool_result: file_read] OK\nx"}]))
    assert out is not None
    assert out.startswith("[tool_result: simulate_conversation] ERROR")
    assert "cap exceeded" in out
    assert "dream_finalize" in out  # guidance line present


def test_hook_catches_unexpected_exception(env, monkeypatch):
    _seed_pending("conv-8")

    async def kablooey(sid, cfg):
        raise RuntimeError("bad things")

    monkeypatch.setattr(dream_simulator, "run_simulation", kablooey)
    hook = runner_hook.make_dream_hook("conv-8", cfg={})
    out = asyncio.run(hook([{"role": "user", "content": "[tool_result: memory_search] OK\nx"}]))
    assert out is not None
    assert "[tool_result: simulate_conversation] ERROR" in out
    assert "RuntimeError" in out
    assert "bad things" in out


# ── rollback_if_unfinalized ──────────────────────────────────────────────────

def test_rollback_deletes_pending(env):
    _seed_pending("conv-r1")
    assert dream_state.has_pending_batch("conv-r1") is True
    assert runner_hook.rollback_if_unfinalized("conv-r1") is True
    assert dream_state.has_pending_batch("conv-r1") is False


def test_rollback_noop_when_nothing_pending(env):
    assert runner_hook.rollback_if_unfinalized("conv-nonexistent") is False
