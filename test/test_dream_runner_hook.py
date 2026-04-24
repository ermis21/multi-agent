"""Tests for `app.dream.runner_hook` — auto-sim state-machine hook.

Covers:
  - make_dream_hook:
      * returns None when no pending batch.
      * returns None when pending batch is not in `submit` phase.
      * returns None when the worker tells it the just-completed iteration
        was a dream_submit / edit_revise call (`just_revised=True`).
      * fires simulator.run_simulation and returns a synthetic
        `[tool_result: simulate_conversation] OK ...` body when firing.
      * formats SimulatorError as an ERROR body (tool-result shape).
      * catches unexpected exceptions and still yields an ERROR body.
  - rollback_if_unfinalized deletes the pending batch when one exists and
    returns True; returns False when there's nothing to drop.

The hook signature changed from `hook(messages)` (which inferred
`just_revised` from message history — broken on prose-without-end paths
where the last tool_result is stale) to `hook(messages, just_revised)`
which receives the flag explicitly from the worker. These tests exercise
both values of the flag.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass

import pytest

from app.dream import dream_state, phrase_store, runner_hook
from app.dream import simulator as dream_simulator


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


# ── hook: no pending batch ───────────────────────────────────────────────────

def test_hook_returns_none_without_pending_batch(env):
    hook = runner_hook.make_dream_hook("conv-x", cfg={})
    assert asyncio.run(hook([], False)) is None


# ── hook: phase gating ───────────────────────────────────────────────────────

def test_hook_returns_none_when_phase_is_post_sim(env):
    _seed_pending("conv-1", phase=dream_state.PHASE_POST_SIM)
    hook = runner_hook.make_dream_hook("conv-1", cfg={})
    assert asyncio.run(hook([], False)) is None


def test_hook_returns_none_when_phase_is_finalize_only(env):
    _seed_pending("conv-2", phase=dream_state.PHASE_FINALIZE_ONLY)
    hook = runner_hook.make_dream_hook("conv-2", cfg={})
    assert asyncio.run(hook([], False)) is None


# ── hook: just_revised gating ────────────────────────────────────────────────

def test_hook_returns_none_when_just_revised_flag_is_true(env):
    """Caller tells us the iteration that just completed WAS a revise tool
    → skip auto-sim, let the dreamer wait for the next iteration."""
    _seed_pending("conv-3")
    hook = runner_hook.make_dream_hook("conv-3", cfg={})
    assert asyncio.run(hook([], True)) is None


# ── hook: firing path ────────────────────────────────────────────────────────

@dataclass
class _FakeResult:
    payload: dict

    def to_payload(self) -> dict:
        return self.payload


def test_hook_fires_sim_when_not_revising(env, monkeypatch):
    """The dreamer did something other than submit/revise (ran a read-only
    tool, or emitted prose) → auto-sim fires."""
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
    out = asyncio.run(hook([], False))
    assert out is not None
    assert out.startswith("[tool_result: simulate_conversation] OK\n")
    body = out.split("\n", 1)[1]
    parsed = json.loads(body)
    assert parsed["session_id"] == "conv-5"
    assert parsed["model_match"] is True
    assert captured == {"sid": "conv-5", "cfg": {"x": 1}}


def test_hook_fires_sim_on_prose_without_end_branch(env, monkeypatch):
    """Regression for the bug where the dreamer emitted prose without `<|end|>`
    (e.g. "Now waiting for simulation") and the hook stayed silent because
    the last tool_result in message history was still dream_submit. The new
    flag-based API fires the sim whenever `just_revised=False`, regardless of
    what's in the message log."""
    _seed_pending("conv-6")

    async def fake_run_simulation(sid, cfg):
        return _FakeResult(payload={"session_id": sid, "model_match": False})

    monkeypatch.setattr(dream_simulator, "run_simulation", fake_run_simulation)
    hook = runner_hook.make_dream_hook("conv-6", cfg={})
    # Message history still contains a stale `[tool_result: dream_submit]`
    # from the prior iteration — the old inference-based hook would've
    # skipped auto-sim here. The new flag-based API ignores history.
    stale_msgs = [
        {"role": "assistant", "content": "..."},
        {"role": "user", "content": "[tool_result: dream_submit] OK\n{}"},
        {"role": "assistant", "content": "Now waiting for simulation."},
    ]
    out = asyncio.run(hook(stale_msgs, False))
    assert out is not None and "simulate_conversation] OK" in out


def test_hook_surfaces_simulator_error_as_tool_result(env, monkeypatch):
    _seed_pending("conv-7")

    async def boom(sid, cfg):
        raise dream_simulator.SimulatorError("cap exceeded")

    monkeypatch.setattr(dream_simulator, "run_simulation", boom)
    hook = runner_hook.make_dream_hook("conv-7", cfg={})
    out = asyncio.run(hook([], False))
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
    out = asyncio.run(hook([], False))
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
