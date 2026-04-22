"""Pure-unit tests for app.dream.dream_state — pending-batch I/O + phase FSM.

Covers:
  - create_or_replace_pending → fresh id, phase=submit, counter=0
  - `dream_submit` on a prior batch replaces wholesale (new id, reset counter)
  - load_pending / has_pending_batch / delete_pending round-trip
  - can_accept_submit_or_revise: allowed in submit + post_sim, blocked in finalize_only
  - should_auto_sim: only in submit, only when dreamer didn't just call submit/revise
  - on_simulation_complete: advances counter, respects cap, flips to finalize_only on !model_match
  - can_iterate + simulations_remaining
  - validate_finalize_coverage: ok / uncovered / unknown / overlap
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.dream import dream_state
from app.dream.dream_state import (
    ERR_MODEL_MISMATCH,
    PHASE_FINALIZE_ONLY,
    PHASE_POST_SIM,
    PHASE_SUBMIT,
)


@pytest.fixture
def state_root(tmp_path, monkeypatch):
    runs = tmp_path / "state" / "dream" / "runs"
    monkeypatch.setattr(dream_state, "DREAM_RUNS_ROOT", runs)
    return runs


def _edit(phrase_id: str, status: str = "ok", **extra) -> dict:
    base = {"idx": 0, "phrase_id": phrase_id, "old_text": "old",
            "new_text": "new", "section_path": "Root / R", "status": status}
    base.update(extra)
    return base


def _create(state_root, conv_sid: str = "c1", edits: list[dict] | None = None):
    return dream_state.create_or_replace_pending(
        conversation_sid=conv_sid,
        target_prompt="worker_full",
        new_prompt_text="# rewritten\n\nbody\n",
        rationale="test",
        edits=edits if edits is not None else [_edit("ph-a"), _edit("ph-b", "possible_conflict")],
    )


# ── Create / load / delete ───────────────────────────────────────────────────

def test_create_pending_writes_fresh_id_and_phase_submit(state_root):
    batch = _create(state_root)
    assert batch.pending_batch_id.startswith("pb-") and len(batch.pending_batch_id) == 11
    assert batch.phase == PHASE_SUBMIT
    assert batch.simulations_run == 0
    assert batch.last_sim_model_match is None
    # File landed at /state/dream/runs/<today>/pending_c1.json.
    loaded = dream_state.load_pending("c1")
    assert loaded.pending_batch_id == batch.pending_batch_id
    assert dream_state.has_pending_batch("c1") is True


def test_fresh_submit_replaces_prior_batch_wholesale(state_root):
    first = _create(state_root)
    second = _create(state_root)
    assert second.pending_batch_id != first.pending_batch_id
    # Only one file present.
    files = list(state_root.rglob("pending_c1.json"))
    assert len(files) == 1
    assert dream_state.load_pending("c1").pending_batch_id == second.pending_batch_id


def test_fresh_submit_resets_sim_counter(state_root):
    batch = _create(state_root)
    dream_state.on_simulation_complete(batch, model_match=True, simulations_cap=3)
    dream_state.save_pending(batch)
    assert batch.simulations_run == 1
    # Resubmit → counter goes back to 0.
    fresh = _create(state_root)
    assert fresh.simulations_run == 0
    assert fresh.last_sim_model_match is None
    assert fresh.phase == PHASE_SUBMIT


def test_delete_pending_returns_bool(state_root):
    assert dream_state.delete_pending("c1") is False
    _create(state_root)
    assert dream_state.delete_pending("c1") is True
    assert dream_state.has_pending_batch("c1") is False


def test_load_pending_missing_raises(state_root):
    with pytest.raises(dream_state.NoPendingBatch):
        dream_state.load_pending("does-not-exist")


# ── Summary + edit lookup ────────────────────────────────────────────────────

def test_summary_counts_and_edit_by_phrase_id(state_root):
    batch = _create(state_root, edits=[
        _edit("ph-a", "ok"),
        _edit("ph-b", "possible_conflict"),
        _edit("ph-c", "possible_loop"),
        _edit("ph-d", "possible_conflict"),
    ])
    c = batch.summary_counts()
    assert c == {"ok": 1, "possible_conflict": 2, "possible_loop": 1}
    assert "1 ok, 2 possible_conflict, 1 possible_loop" in batch.summary_line()
    assert batch.has_any_flag() is True
    assert batch.edit_by_phrase_id("ph-c")["status"] == "possible_loop"
    assert batch.edit_by_phrase_id("ph-zz") is None


def test_summary_all_ok(state_root):
    batch = _create(state_root, edits=[_edit("ph-a"), _edit("ph-b")])
    assert batch.has_any_flag() is False


# ── can_accept_submit_or_revise ──────────────────────────────────────────────

def test_submit_allowed_in_submit_phase(state_root):
    batch = _create(state_root)
    assert batch.phase == PHASE_SUBMIT
    ok, reason = dream_state.can_accept_submit_or_revise(batch)
    assert ok is True and reason is None


def test_submit_allowed_in_post_sim_phase(state_root):
    batch = _create(state_root)
    dream_state.on_simulation_complete(batch, model_match=True, simulations_cap=3)
    assert batch.phase == PHASE_POST_SIM
    ok, reason = dream_state.can_accept_submit_or_revise(batch)
    assert ok is True and reason is None


def test_submit_blocked_in_finalize_only(state_root):
    batch = _create(state_root)
    dream_state.on_simulation_complete(batch, model_match=False, simulations_cap=3)
    assert batch.phase == PHASE_FINALIZE_ONLY
    ok, reason = dream_state.can_accept_submit_or_revise(batch)
    assert ok is False and reason == ERR_MODEL_MISMATCH


# ── should_auto_sim ──────────────────────────────────────────────────────────

def test_auto_sim_fires_when_dreamer_idle_in_submit(state_root):
    batch = _create(state_root)
    assert dream_state.should_auto_sim(batch, dreamer_just_called_submit_or_revise=False) is True


def test_auto_sim_suppressed_when_dreamer_still_revising(state_root):
    batch = _create(state_root)
    assert dream_state.should_auto_sim(batch, dreamer_just_called_submit_or_revise=True) is False


def test_auto_sim_suppressed_in_post_sim_phase(state_root):
    batch = _create(state_root)
    dream_state.on_simulation_complete(batch, model_match=True, simulations_cap=3)
    # Even if the dreamer is idle, post_sim phase has already simulated —
    # the runner shouldn't sim again until a new submit bumps us back to submit.
    assert dream_state.should_auto_sim(batch, dreamer_just_called_submit_or_revise=False) is False


def test_auto_sim_suppressed_in_finalize_only(state_root):
    batch = _create(state_root)
    dream_state.on_simulation_complete(batch, model_match=False, simulations_cap=3)
    assert dream_state.should_auto_sim(batch, dreamer_just_called_submit_or_revise=False) is False


# ── on_simulation_complete + cap ─────────────────────────────────────────────

def test_simulation_complete_advances_counter_and_sets_post_sim(state_root):
    batch = _create(state_root)
    dream_state.on_simulation_complete(batch, model_match=True, simulations_cap=3)
    assert batch.simulations_run == 1
    assert batch.last_sim_model_match is True
    assert batch.data["last_sim_at"] is not None
    assert batch.phase == PHASE_POST_SIM


def test_simulation_cap_flips_to_finalize_only(state_root):
    batch = _create(state_root)
    for _ in range(3):
        dream_state.on_simulation_complete(batch, model_match=True, simulations_cap=3)
    assert batch.simulations_run == 3
    assert batch.phase == PHASE_FINALIZE_ONLY
    assert dream_state.simulations_remaining(batch, 3) == 0
    assert dream_state.can_iterate(batch, 3) is False


def test_model_mismatch_flips_immediately(state_root):
    batch = _create(state_root)
    dream_state.on_simulation_complete(batch, model_match=False, simulations_cap=3)
    assert batch.simulations_run == 1
    assert batch.phase == PHASE_FINALIZE_ONLY
    assert dream_state.can_iterate(batch, 3) is False


def test_can_iterate_true_when_under_cap_and_model_matches(state_root):
    batch = _create(state_root)
    dream_state.on_simulation_complete(batch, model_match=True, simulations_cap=3)
    assert dream_state.simulations_remaining(batch, 3) == 2
    assert dream_state.can_iterate(batch, 3) is True


def test_on_submit_resets_phase_back_to_submit(state_root):
    """After post_sim, a new `dream_submit` puts us back in submit phase."""
    batch = _create(state_root)
    dream_state.on_simulation_complete(batch, model_match=True, simulations_cap=3)
    assert batch.phase == PHASE_POST_SIM
    dream_state.on_submit_resets_phase(batch)
    assert batch.phase == PHASE_SUBMIT


# ── validate_finalize_coverage ───────────────────────────────────────────────

def test_finalize_coverage_ok(state_root):
    batch = _create(state_root, edits=[_edit("ph-a"), _edit("ph-b"), _edit("ph-c")])
    cov = dream_state.validate_finalize_coverage(batch, keep=["ph-a"], drop=["ph-b", "ph-c"])
    assert cov.ok and cov.uncovered == [] and cov.unknown == []


def test_finalize_coverage_abandon_all(state_root):
    batch = _create(state_root, edits=[_edit("ph-a"), _edit("ph-b")])
    cov = dream_state.validate_finalize_coverage(batch, keep=[], drop=["ph-a", "ph-b"])
    assert cov.ok


def test_finalize_coverage_uncovered(state_root):
    batch = _create(state_root, edits=[_edit("ph-a"), _edit("ph-b")])
    cov = dream_state.validate_finalize_coverage(batch, keep=["ph-a"], drop=[])
    assert cov.ok is False
    assert cov.uncovered == ["ph-b"]
    assert "uncovered" in cov.reason


def test_finalize_coverage_unknown_ids(state_root):
    batch = _create(state_root, edits=[_edit("ph-a")])
    cov = dream_state.validate_finalize_coverage(
        batch, keep=["ph-a"], drop=["ph-nope"],
    )
    assert cov.ok is False
    assert cov.unknown == ["ph-nope"]


def test_finalize_coverage_overlap_rejected(state_root):
    batch = _create(state_root, edits=[_edit("ph-a"), _edit("ph-b")])
    cov = dream_state.validate_finalize_coverage(
        batch, keep=["ph-a", "ph-b"], drop=["ph-b"],
    )
    assert cov.ok is False
    assert "both keep and drop" in cov.reason
