"""Pure-unit tests for app.dream.simulator — shadow replay + model-match gate.

Covers every testable surface without running an LLM:

  select_simulation_model   — local / remote / fallback pathways
  _resolve_original_model   — state.json parsing + missing/invalid files
  _truncate_turns           — head/tail truncation invariant
  load_before_transcript    — final-only filter + user/assistant interleave
  materialize_shadow        — shadow dir shape + overwrite semantics
  run_simulation            — full loop with run_agent_role monkeypatched:
                              model-match true path, model-match false path,
                              cap-exhausted rejection, finalize-only rejection
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.dream import dream_state, phrase_store, session_iter, simulator


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def sim_env(tmp_path, monkeypatch):
    """Redirect STATE_DIR, SESSIONS_ROOT, PROMPTS_DIR, and the dream-runs root
    into tmp_path. Also monkeypatch the sim-cache root to tmp_path so
    tempfile.TemporaryDirectory lands in the fixture tree."""
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
    monkeypatch.setattr(dream_state, "DREAM_RUNS_ROOT", state / "dream" / "runs")

    return {
        "state": state,
        "prompts": prompts,
        "sessions": sessions,
    }


def _cfg() -> dict:
    return {
        "dream": {
            "simulation": {
                "max_simulations_per_conversation": 3,
                "max_turns_replayed": 5,
                "fallback_local_model": "vpn_local",
            },
            "counterfactual": {
                # Skip the one-shot goal-extraction LLM call so tests see only
                # the replay invocations in their captured-calls list.
                "goal_extraction_enabled": False,
            },
        },
    }


def _write_state_json(sim_env, sid: str, *, agent_role="worker", model=None) -> None:
    d = sim_env["sessions"] / sid
    d.mkdir(exist_ok=True)
    (d / "state.json").write_text(json.dumps({
        "session_id": sid,
        "agent_role": agent_role,
        "model": model,
        "parent_session_id": None,
        "source_trigger": {"type": "user", "ref": None},
    }), encoding="utf-8")


def _write_turns(sim_env, sid: str, rows: list[dict]) -> None:
    d = sim_env["sessions"] / sid
    d.mkdir(exist_ok=True)
    with (d / "turns.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _write_prompt_template(sim_env, name: str, body: str = "# template\n\n<|end|>\n") -> None:
    (sim_env["prompts"] / f"{name}.md").write_text(body, encoding="utf-8")


def _make_pending(sim_env, conv_sid: str, target_prompt: str = "worker_full",
                  new_text: str = "# SHADOW\n\nrewritten.\n") -> dream_state.PendingBatch:
    return dream_state.create_or_replace_pending(
        conversation_sid=conv_sid,
        target_prompt=target_prompt,
        new_prompt_text=new_text,
        rationale="test",
        edits=[{
            "idx": 0, "phrase_id": "ph-a", "old_text": "", "new_text": "x",
            "section_path": "", "kind": "replace", "status": "ok",
        }],
    )


# ── select_simulation_model ──────────────────────────────────────────────────

def test_select_simulation_uses_original_when_local(monkeypatch):
    class E:
        name = "vpn_local"
        provider = "local"

    class Cat:
        models = [E()]

    monkeypatch.setattr(simulator, "_load_catalog", lambda: Cat())
    model, match = simulator.select_simulation_model("vpn_local", _cfg())
    assert model == "vpn_local" and match is True


def test_select_simulation_falls_back_when_remote(monkeypatch):
    class E:
        name = "claude-opus-4-7"
        provider = "anthropic"

    class Cat:
        models = [E()]

    monkeypatch.setattr(simulator, "_load_catalog", lambda: Cat())
    model, match = simulator.select_simulation_model("claude-opus-4-7", _cfg())
    assert model == "vpn_local" and match is False


def test_select_simulation_handles_none(monkeypatch):
    monkeypatch.setattr(simulator, "_load_catalog", lambda: type("C", (), {"models": []})())
    model, match = simulator.select_simulation_model(None, _cfg())
    assert match is False


def test_select_simulation_uses_config_override(monkeypatch):
    monkeypatch.setattr(simulator, "_load_catalog", lambda: type("C", (), {"models": []})())
    cfg = {"dream": {"simulation": {"fallback_local_model": "my_local_model"}}}
    model, _ = simulator.select_simulation_model("claude-opus-4-7", cfg)
    assert model == "my_local_model"


# ── _resolve_original_model ──────────────────────────────────────────────────

def test_resolve_original_model_reads_state_json(sim_env):
    _write_state_json(sim_env, "s1", model="claude-opus-4-7")
    assert simulator._resolve_original_model("s1") == "claude-opus-4-7"


def test_resolve_original_model_returns_none_for_missing(sim_env):
    assert simulator._resolve_original_model("does-not-exist") is None


def test_resolve_original_model_returns_none_for_malformed_json(sim_env):
    d = sim_env["sessions"] / "s_bad"
    d.mkdir()
    (d / "state.json").write_text("{not json", encoding="utf-8")
    assert simulator._resolve_original_model("s_bad") is None


def test_resolve_original_model_returns_none_when_model_absent(sim_env):
    _write_state_json(sim_env, "s1", model=None)
    assert simulator._resolve_original_model("s1") is None


# ── _truncate_turns ──────────────────────────────────────────────────────────

def test_truncate_turns_noop_when_under_budget():
    turns = [{"n": i} for i in range(4)]
    assert simulator._truncate_turns(turns, 10) == turns


def test_truncate_turns_head_tail_with_elision_marker():
    turns = [{"n": i} for i in range(20)]
    out = simulator._truncate_turns(turns, 6)
    # head=3 + marker + tail=3 → 7 items, marker reports 14 elided.
    assert len(out) == 7
    assert out[3]["__elided__"] == 14
    assert out[:3] == turns[:3]
    assert out[-3:] == turns[-3:]


def test_truncate_turns_zero_budget_returns_copy():
    turns = [{"n": i} for i in range(4)]
    out = simulator._truncate_turns(turns, 0)
    assert out == turns


# ── load_before_transcript ───────────────────────────────────────────────────

def test_load_before_transcript_shapes_final_turns_only(sim_env):
    _write_turns(sim_env, "s1", [
        {"role": "worker",    "timestamp": "2026-04-20T10:00:00+00:00",
         "messages": [{"role": "user", "content": "hi"}], "response": "noise"},
        {"role": "final",     "timestamp": "2026-04-20T10:01:00+00:00",
         "messages": [{"role": "user", "content": "what's 2+2?"}], "response": "4"},
        {"role": "supervisor", "timestamp": "2026-04-20T10:01:10+00:00",
         "messages": [], "response": "graded"},
        {"role": "final",     "timestamp": "2026-04-20T10:05:00+00:00",
         "messages": [{"role": "user", "content": "again?"}], "response": "still 4"},
    ])
    view = simulator.load_before_transcript("s1", max_turns=10)
    roles = [t.get("role") for t in view.transcript]
    assert roles == ["user", "assistant", "user", "assistant"]
    assert view.transcript[0]["content"] == "what's 2+2?"
    assert view.transcript[1]["content"] == "4"


def test_load_before_transcript_empty_when_no_final_turns(sim_env):
    _write_turns(sim_env, "s1", [
        {"role": "worker", "messages": [], "response": "x", "timestamp": "2026-04-20T10:00:00+00:00"},
    ])
    view = simulator.load_before_transcript("s1", max_turns=5)
    assert view.transcript == []


def test_load_before_transcript_missing_file_returns_empty(sim_env):
    view = simulator.load_before_transcript("ghost", max_turns=5)
    assert view.transcript == [] and view.tool_calls == []


# ── materialize_shadow ───────────────────────────────────────────────────────

def test_materialize_shadow_copies_tree_and_overwrites_target(sim_env, tmp_path):
    _write_prompt_template(sim_env, "worker_full", "# LIVE\n")
    _write_prompt_template(sim_env, "supervisor_full", "# SUPER\n")
    batch = _make_pending(sim_env, "s1", target_prompt="worker_full",
                          new_text="# SHADOW OVERRIDE\n")
    shadow = simulator.materialize_shadow(batch, root=tmp_path / "shadows")

    # Target got rewritten.
    assert (shadow / "worker_full.md").read_text() == "# SHADOW OVERRIDE\n"
    # Siblings carried over unchanged.
    assert (shadow / "supervisor_full.md").read_text() == "# SUPER\n"


def test_materialize_shadow_missing_target_raises(sim_env, tmp_path):
    _write_prompt_template(sim_env, "worker_full", "# LIVE\n")
    batch = _make_pending(sim_env, "s1", target_prompt="nonexistent")
    with pytest.raises(simulator.SimulatorError, match="target_prompt"):
        simulator.materialize_shadow(batch, root=tmp_path / "shadows")


def test_materialize_shadow_missing_live_dir_raises(sim_env, tmp_path, monkeypatch):
    monkeypatch.setattr(phrase_store, "PROMPTS_DIR", tmp_path / "ghost")
    batch = _make_pending(sim_env, "s1", target_prompt="worker_full")
    with pytest.raises(simulator.SimulatorError, match="live prompts dir"):
        simulator.materialize_shadow(batch, root=tmp_path / "shadows")


# ── run_simulation (end-to-end with stubbed replay) ──────────────────────────

@pytest.fixture
def patched_replay(monkeypatch):
    """Replace the heavy run_agent_role call with a deterministic stub.

    Captures every invocation so tests can assert on (session_id, role,
    prompts_dir, model) that the simulator hands to the runner."""
    calls: list[dict] = []

    async def fake_run_agent_role(role, body, session_id, *, prompts_dir=None):
        calls.append({
            "role": role,
            "session_id": session_id,
            "prompts_dir": str(prompts_dir) if prompts_dir else None,
            "model": body.get("model"),
            "messages": list(body.get("messages", [])),
        })
        return {"choices": [{"message": {"content": "simulated reply"}}]}

    import app.entrypoints as _entry
    monkeypatch.setattr(_entry, "run_agent_role", fake_run_agent_role)
    return calls


async def _run(coro):
    return await coro


def test_run_simulation_end_to_end_model_match(sim_env, patched_replay, monkeypatch):
    """Local original model → sim uses same model, model_match=true, phase→post_sim."""
    _write_prompt_template(sim_env, "worker_full", "# LIVE WORKER\n")
    _write_state_json(sim_env, "s1", agent_role="worker", model="vpn_local")
    _write_turns(sim_env, "s1", [
        {"role": "final", "timestamp": "2026-04-20T10:00:00+00:00",
         "messages": [{"role": "user", "content": "hello"}], "response": "hi"},
    ])
    batch = _make_pending(sim_env, "s1")

    class E:
        name = "vpn_local"
        provider = "local"

    monkeypatch.setattr(simulator, "_load_catalog",
                        lambda: type("C", (), {"models": [E()]})())

    import asyncio
    result = asyncio.run(simulator.run_simulation("s1", _cfg()))

    assert result.model_match is True
    assert result.simulation_model == "vpn_local"
    assert result.original_model == "vpn_local"
    # Replay fired with the overlay prompts dir.
    assert len(patched_replay) == 1
    assert patched_replay[0]["role"] == "worker"
    assert patched_replay[0]["prompts_dir"] is not None
    assert patched_replay[0]["model"] == "vpn_local"
    # Only user turns should be replayed.
    assert all(m["role"] == "user" for m in patched_replay[0]["messages"])
    # State-machine advanced.
    post = dream_state.load_pending("s1")
    assert post.phase == dream_state.PHASE_POST_SIM
    assert post.simulations_run == 1
    assert result.can_iterate is True
    assert result.simulations_remaining == 2


def test_run_simulation_falls_back_and_blocks_iteration(sim_env, patched_replay, monkeypatch):
    """Remote original → fallback, model_match=false, phase→finalize_only."""
    _write_prompt_template(sim_env, "worker_full", "# LIVE WORKER\n")
    _write_state_json(sim_env, "s1", agent_role="worker", model="claude-opus-4-7")
    _write_turns(sim_env, "s1", [
        {"role": "final", "timestamp": "2026-04-20T10:00:00+00:00",
         "messages": [{"role": "user", "content": "remote"}], "response": "done"},
    ])
    _make_pending(sim_env, "s1")

    class E:
        name = "claude-opus-4-7"
        provider = "anthropic"

    monkeypatch.setattr(simulator, "_load_catalog",
                        lambda: type("C", (), {"models": [E()]})())

    import asyncio
    result = asyncio.run(simulator.run_simulation("s1", _cfg()))

    assert result.model_match is False
    assert result.simulation_model == "vpn_local"
    assert result.can_iterate is False
    post = dream_state.load_pending("s1")
    assert post.phase == dream_state.PHASE_FINALIZE_ONLY


def test_run_simulation_rejects_finalize_only_batch(sim_env, patched_replay, monkeypatch):
    _write_prompt_template(sim_env, "worker_full", "# LIVE\n")
    _write_state_json(sim_env, "s1", model="vpn_local")
    _write_turns(sim_env, "s1", [
        {"role": "final", "timestamp": "2026-04-20T10:00:00+00:00",
         "messages": [{"role": "user", "content": "x"}], "response": "y"},
    ])
    batch = _make_pending(sim_env, "s1")
    batch.data["phase"] = dream_state.PHASE_FINALIZE_ONLY
    dream_state.save_pending(batch)

    import asyncio
    with pytest.raises(simulator.SimulatorError, match="finalize_only"):
        asyncio.run(simulator.run_simulation("s1", _cfg()))


def test_run_simulation_rejects_when_cap_reached(sim_env, patched_replay, monkeypatch):
    _write_prompt_template(sim_env, "worker_full", "# LIVE\n")
    _write_state_json(sim_env, "s1", model="vpn_local")
    _write_turns(sim_env, "s1", [
        {"role": "final", "timestamp": "2026-04-20T10:00:00+00:00",
         "messages": [{"role": "user", "content": "x"}], "response": "y"},
    ])
    batch = _make_pending(sim_env, "s1")
    batch.data["simulations_run"] = 3
    dream_state.save_pending(batch)

    import asyncio
    with pytest.raises(simulator.SimulatorError, match="cap"):
        asyncio.run(simulator.run_simulation("s1", _cfg()))


def test_run_simulation_uses_original_role_from_state(sim_env, patched_replay, monkeypatch):
    """Replay fires with the role recorded in the session's state.json."""
    _write_prompt_template(sim_env, "worker_full", "# LIVE\n")
    _write_state_json(sim_env, "s1", agent_role="discord_agent", model="vpn_local")
    _write_turns(sim_env, "s1", [
        {"role": "final", "timestamp": "2026-04-20T10:00:00+00:00",
         "messages": [{"role": "user", "content": "hi"}], "response": "hello"},
    ])
    _make_pending(sim_env, "s1")

    class E:
        name = "vpn_local"
        provider = "local"

    monkeypatch.setattr(simulator, "_load_catalog",
                        lambda: type("C", (), {"models": [E()]})())

    import asyncio
    asyncio.run(simulator.run_simulation("s1", _cfg()))
    assert patched_replay[0]["role"] == "discord_agent"


def test_run_simulation_defaults_to_worker_without_state_file(sim_env, patched_replay, monkeypatch):
    _write_prompt_template(sim_env, "worker_full", "# LIVE\n")
    # No state.json written — but turns.jsonl has final turns.
    _write_turns(sim_env, "s1", [
        {"role": "final", "timestamp": "2026-04-20T10:00:00+00:00",
         "messages": [{"role": "user", "content": "hi"}], "response": "hello"},
    ])
    _make_pending(sim_env, "s1")
    monkeypatch.setattr(simulator, "_load_catalog",
                        lambda: type("C", (), {"models": []})())

    import asyncio
    asyncio.run(simulator.run_simulation("s1", _cfg()))
    assert patched_replay[0]["role"] == "worker"


def test_sim_result_to_payload_shape():
    before = simulator.TranscriptView(transcript=[{"role": "user", "content": "x"}])
    after  = simulator.TranscriptView(transcript=[{"role": "assistant", "content": "y"}])
    cf_metrics = simulator.CounterfactualMetrics(
        per_turn=[],
        avg_lex=0.0, avg_sem=0.0,
        turns_adjusted=0, turns_verbatim=0,
        max_band="identical", fidelity="high",
        cf_aborts=0, goal="",
    )
    r = simulator.SimResult(
        session_id="s1", original_model="m", simulation_model="n",
        model_match=True, before=before, after=after,
        simulations_remaining=2, can_iterate=True,
        counterfactual=cf_metrics,
    )
    p = r.to_payload()
    assert p["session_id"] == "s1"
    assert p["model_match"] is True
    assert p["before"]["transcript"] == [{"role": "user", "content": "x"}]
    assert p["after"]["transcript"] == [{"role": "assistant", "content": "y"}]
    assert p["simulations_remaining"] == 2 and p["can_iterate"] is True
    assert p["counterfactual"]["fidelity"] == "high"
