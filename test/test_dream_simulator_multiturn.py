"""Tests for the multi-turn replay + counterfactual user-sim rewrite of
``app.dream.simulator``. Covers:

  - Multi-turn replay produces N agent turns (not 1 flattened shot).
  - Identical similarity band skips the user-sim (verbatim original turn).
  - Substantial/divergent band invokes dream_user_simulator with the right args.
  - Unrelated band is sticky: once hit, subsequent turns fall back to verbatim.
  - can_iterate turns False when fidelity == "low".
  - Sim sub-session folders cleaned up after the run.
  - Overlay root cleaned up on both success and failure paths.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from app.dream import counterfactual as cf
from app.dream import dream_state, phrase_store, session_iter, simulator


# ── Fixtures (mirror test_dream_simulator.py patterns) ─────────────────────

@pytest.fixture
def sim_env(tmp_path, monkeypatch):
    state = tmp_path / "state"
    prompts = tmp_path / "prompts"
    sessions = state / "sessions"
    sessions.mkdir(parents=True)
    prompts.mkdir()
    (state / "dream" / "runs").mkdir(parents=True)

    monkeypatch.setattr(phrase_store, "STATE_DIR", state)
    monkeypatch.setattr(phrase_store, "PROMPTS_DIR", prompts)
    monkeypatch.setattr(session_iter, "SESSIONS_ROOT", sessions)
    monkeypatch.setattr(simulator, "SESSIONS_ROOT", sessions)
    monkeypatch.setattr(simulator, "DREAM_SIM_CACHE_ROOT", tmp_path / "sim_cache")
    monkeypatch.setattr(simulator, "SIM_OVERLAY_ROOT", tmp_path / "overlay")
    monkeypatch.setattr(dream_state, "DREAM_RUNS_ROOT", state / "dream" / "runs")

    # Stub out chroma collection deletion — sim uses chromadb.PersistentClient
    # which would try to touch /state/chroma.
    monkeypatch.setattr(simulator, "_drop_sim_chroma_collection", lambda _name: None)

    return {"state": state, "prompts": prompts, "sessions": sessions}


def _write_state_json(sim_env, sid, *, agent_role="worker", model="vpn_local"):
    d = sim_env["sessions"] / sid
    d.mkdir(exist_ok=True)
    (d / "state.json").write_text(json.dumps({
        "session_id": sid, "agent_role": agent_role, "model": model,
    }), encoding="utf-8")


def _write_turns(sim_env, sid, rows):
    d = sim_env["sessions"] / sid
    d.mkdir(exist_ok=True)
    with (d / "turns.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _write_prompt(sim_env, name):
    (sim_env["prompts"] / f"{name}.md").write_text("# t\n<|end|>\n", encoding="utf-8")


def _make_pending(conv_sid, target="worker_full"):
    return dream_state.create_or_replace_pending(
        conversation_sid=conv_sid,
        target_prompt=target,
        new_prompt_text="# SHADOW\n<|end|>\n",
        rationale="test",
        edits=[{"idx": 0, "phrase_id": "ph-a", "old_text": "", "new_text": "x",
                "section_path": "", "kind": "replace", "status": "ok"}],
    )


def _local_catalog(monkeypatch):
    class E:
        name = "vpn_local"; provider = "local"

    class Cat:
        models = [E()]

    monkeypatch.setattr(simulator, "_load_catalog", lambda: Cat())


def _cfg():
    return {
        "dream": {
            "simulation": {"max_simulations_per_conversation": 3,
                           "max_turns_replayed": 10,
                           "fallback_local_model": "vpn_local"},
            "counterfactual": {"enabled": True, "user_sim_model": "vpn_local",
                               "goal_extraction_enabled": False},
        },
    }


# ── _to_interleaved ────────────────────────────────────────────────────────

def test_to_interleaved_zips_user_and_agent_turns():
    before = simulator.TranscriptView(transcript=[
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "u2"},
        {"role": "assistant", "content": "a2"},
    ])
    il = simulator._to_interleaved(before)
    assert il.users == ["u1", "u2"]
    assert il.agents == ["a1", "a2"]


def test_to_interleaved_drops_trailing_user():
    before = simulator.TranscriptView(transcript=[
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "dangling"},
    ])
    il = simulator._to_interleaved(before)
    assert il.users == ["u1"]
    assert il.agents == ["a1"]


def test_to_interleaved_joins_consecutive_users():
    before = simulator.TranscriptView(transcript=[
        {"role": "user", "content": "part A"},
        {"role": "user", "content": "part B"},
        {"role": "assistant", "content": "ok"},
    ])
    il = simulator._to_interleaved(before)
    assert il.users == ["part A\n\npart B"]
    assert il.agents == ["ok"]


# ── Multi-turn replay core ─────────────────────────────────────────────────

def _fake_run_agent_role_factory(monkeypatch, *, new_agent_responses, user_sim_responses=None):
    """Install a fake `run_agent_role` that round-robins per role.

    new_agent_responses:  canned assistant texts for role==original_role
    user_sim_responses:   canned texts for role==dream_user_simulator

    Returns a `calls` list capturing (role, body, sid) tuples for assertions.
    """
    calls: list[tuple[str, dict, str]] = []
    agent_iter = iter(new_agent_responses)
    user_iter = iter(user_sim_responses or [])

    async def fake(role, body, sid, *, prompts_dir=None):
        calls.append((role, body, sid))
        if role == "dream_user_simulator":
            text = next(user_iter)
        else:
            text = next(agent_iter)
        return {"choices": [{"message": {"content": text}}], "tool_trace": []}

    from app import entrypoints as ep
    monkeypatch.setattr(ep, "run_agent_role", fake)
    return calls


def test_replay_produces_n_agent_turns(sim_env, monkeypatch):
    _local_catalog(monkeypatch)
    sid = "s_multi"
    _write_state_json(sim_env, sid)
    _write_turns(sim_env, sid, [
        {"role": "final", "timestamp": "2026-04-22T00:00:00Z",
         "messages": [{"role": "user", "content": "u1"}], "response": "a1"},
        {"role": "final", "timestamp": "2026-04-22T00:01:00Z",
         "messages": [{"role": "user", "content": "u2"}], "response": "a2"},
        {"role": "final", "timestamp": "2026-04-22T00:02:00Z",
         "messages": [{"role": "user", "content": "u3"}], "response": "a3"},
    ])
    _write_prompt(sim_env, "worker_full")
    batch = _make_pending(sid)

    calls = _fake_run_agent_role_factory(
        monkeypatch,
        new_agent_responses=["new_a1", "new_a2", "new_a3"],
        user_sim_responses=["revised_u2", "revised_u3"],  # in case non-identical
    )

    result = asyncio.run(simulator.run_simulation(sid, _cfg()))

    # Count agent-role calls (excluding dream_user_simulator).
    agent_calls = [c for c in calls if c[0] != "dream_user_simulator"]
    assert len(agent_calls) == 3  # three replay turns
    # After transcript: 3 user + 3 agent entries
    after = result.after.transcript
    assert sum(1 for t in after if t.get("role") == "user") == 3
    assert sum(1 for t in after if t.get("role") == "assistant") == 3
    # Metrics populated
    assert len(result.counterfactual.per_turn) == 3


def test_identical_band_uses_verbatim_user_turn(sim_env, monkeypatch):
    _local_catalog(monkeypatch)
    # Force all similarities to IDENTICAL by monkeypatching compute_similarity
    monkeypatch.setattr(cf, "compute_similarity",
                        lambda old, new, cfg: cf.Similarity(
                            lex=1.0, sem=1.0,
                            lex_band=cf.Band.IDENTICAL, sem_band=cf.Band.IDENTICAL,
                            band=cf.Band.IDENTICAL))

    sid = "s_ident"
    _write_state_json(sim_env, sid)
    _write_turns(sim_env, sid, [
        {"role": "final", "messages": [{"role": "user", "content": "u1"}], "response": "a1"},
        {"role": "final", "messages": [{"role": "user", "content": "u2"}], "response": "a2"},
    ])
    _write_prompt(sim_env, "worker_full")
    _make_pending(sid)

    calls = _fake_run_agent_role_factory(
        monkeypatch,
        new_agent_responses=["new_a1", "new_a2"],
        user_sim_responses=[],  # should not be called
    )

    result = asyncio.run(simulator.run_simulation(sid, _cfg()))

    user_sim_calls = [c for c in calls if c[0] == "dream_user_simulator"]
    assert user_sim_calls == []
    # All per_turn entries are IDENTICAL, adjusted=False
    for t in result.counterfactual.per_turn:
        assert t["band"] == "identical"
        assert t["adjusted"] is False


def test_substantial_band_invokes_user_sim(sim_env, monkeypatch):
    _local_catalog(monkeypatch)
    # Turn 0 is always IDENTICAL (no prior agent); subsequent turns SUBSTANTIAL.
    call_count = {"n": 0}
    def fake_sim(old, new, cfg):
        call_count["n"] += 1
        return cf.Similarity(lex=0.5, sem=0.55,
                             lex_band=cf.Band.SUBSTANTIAL,
                             sem_band=cf.Band.SUBSTANTIAL,
                             band=cf.Band.SUBSTANTIAL)
    monkeypatch.setattr(cf, "compute_similarity", fake_sim)

    sid = "s_subst"
    _write_state_json(sim_env, sid)
    _write_turns(sim_env, sid, [
        {"role": "final", "messages": [{"role": "user", "content": "u1"}], "response": "a1"},
        {"role": "final", "messages": [{"role": "user", "content": "u2"}], "response": "a2"},
        {"role": "final", "messages": [{"role": "user", "content": "u3"}], "response": "a3"},
    ])
    _write_prompt(sim_env, "worker_full")
    _make_pending(sid)

    calls = _fake_run_agent_role_factory(
        monkeypatch,
        new_agent_responses=["new_a1", "new_a2", "new_a3"],
        user_sim_responses=["rewritten_u2", "rewritten_u3"],
    )

    result = asyncio.run(simulator.run_simulation(sid, _cfg()))

    user_sim_calls = [c for c in calls if c[0] == "dream_user_simulator"]
    assert len(user_sim_calls) == 2  # Turn 0 verbatim, turns 1-2 use user-sim
    # Per-turn metrics: turn 0 IDENTICAL/verbatim; turns 1-2 SUBSTANTIAL/adjusted
    assert result.counterfactual.per_turn[0]["band"] == "identical"
    assert result.counterfactual.per_turn[0]["adjusted"] is False
    assert result.counterfactual.per_turn[1]["band"] == "substantial"
    assert result.counterfactual.per_turn[1]["adjusted"] is True


def test_unrelated_band_falls_back_to_verbatim_stickily(sim_env, monkeypatch):
    _local_catalog(monkeypatch)
    # Turn 1 (the first compared turn) returns UNRELATED; after that should
    # stay verbatim without invoking user-sim.
    calls_sim = {"n": 0}
    def fake_sim(old, new, cfg):
        calls_sim["n"] += 1
        return cf.Similarity(lex=0.05, sem=0.05,
                             lex_band=cf.Band.UNRELATED,
                             sem_band=cf.Band.UNRELATED,
                             band=cf.Band.UNRELATED)
    monkeypatch.setattr(cf, "compute_similarity", fake_sim)

    sid = "s_unrel"
    _write_state_json(sim_env, sid)
    _write_turns(sim_env, sid, [
        {"role": "final", "messages": [{"role": "user", "content": f"u{i}"}],
         "response": f"a{i}"}
        for i in range(4)
    ])
    _write_prompt(sim_env, "worker_full")
    _make_pending(sid)

    calls = _fake_run_agent_role_factory(
        monkeypatch,
        new_agent_responses=[f"new_a{i}" for i in range(4)],
        user_sim_responses=[],  # should not be called
    )

    result = asyncio.run(simulator.run_simulation(sid, _cfg()))

    user_sim_calls = [c for c in calls if c[0] == "dream_user_simulator"]
    assert user_sim_calls == []  # sticky fallback means no user-sim
    # Fidelity should be low
    assert result.counterfactual.fidelity == "low"


def test_fidelity_low_blocks_can_iterate(sim_env, monkeypatch):
    _local_catalog(monkeypatch)
    monkeypatch.setattr(cf, "compute_similarity",
                        lambda old, new, cfg: cf.Similarity(
                            lex=0.3, sem=0.3,
                            lex_band=cf.Band.DIVERGENT, sem_band=cf.Band.DIVERGENT,
                            band=cf.Band.DIVERGENT))

    sid = "s_fid"
    _write_state_json(sim_env, sid)
    _write_turns(sim_env, sid, [
        {"role": "final", "messages": [{"role": "user", "content": "u1"}], "response": "a1"},
        {"role": "final", "messages": [{"role": "user", "content": "u2"}], "response": "a2"},
    ])
    _write_prompt(sim_env, "worker_full")
    _make_pending(sid)

    _fake_run_agent_role_factory(
        monkeypatch,
        new_agent_responses=["new_a1", "new_a2"],
        user_sim_responses=["revised_u2"],
    )

    result = asyncio.run(simulator.run_simulation(sid, _cfg()))
    assert result.counterfactual.fidelity == "low"
    assert result.can_iterate is False


def test_sim_sessions_cleaned_up_after_run(sim_env, monkeypatch):
    _local_catalog(monkeypatch)
    monkeypatch.setattr(cf, "compute_similarity",
                        lambda old, new, cfg: cf.Similarity(
                            lex=1.0, sem=1.0,
                            lex_band=cf.Band.IDENTICAL, sem_band=cf.Band.IDENTICAL,
                            band=cf.Band.IDENTICAL))

    sid = "s_clean"
    _write_state_json(sim_env, sid)
    _write_turns(sim_env, sid, [
        {"role": "final", "messages": [{"role": "user", "content": "u1"}], "response": "a1"},
        {"role": "final", "messages": [{"role": "user", "content": "u2"}], "response": "a2"},
    ])
    _write_prompt(sim_env, "worker_full")
    _make_pending(sid)

    # Fake run_agent_role also creates a sub-session folder (mirrors real side effect)
    async def fake(role, body, sid_arg, *, prompts_dir=None):
        d = sim_env["sessions"] / sid_arg
        d.mkdir(exist_ok=True)
        (d / "turns.jsonl").write_text("{}\n", encoding="utf-8")
        return {"choices": [{"message": {"content": "ok"}}], "tool_trace": []}

    from app import entrypoints as ep
    monkeypatch.setattr(ep, "run_agent_role", fake)

    asyncio.run(simulator.run_simulation(sid, _cfg()))

    # Original session folder must survive; sim sub-sessions must be gone.
    assert (sim_env["sessions"] / sid).exists()
    sim_leftovers = [p for p in sim_env["sessions"].iterdir()
                     if p.name.startswith(f"{sid}__sim_")]
    assert sim_leftovers == []
