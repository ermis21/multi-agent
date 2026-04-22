"""Pure-unit tests for app.dream.session_iter — yesterday's session picker.

Seeds a fake STATE_DIR with multiple session folders containing turns.jsonl +
state.json and verifies filtering (date, excluded roles, sub-agents) and
ordering (most-recent first).
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from app.dream import phrase_store, session_iter


@pytest.fixture
def sessions_root(tmp_path, monkeypatch):
    """Redirect both phrase_store.STATE_DIR and session_iter.SESSIONS_ROOT into
    tmp_path. session_iter imports SESSIONS_ROOT at module load time so we
    monkeypatch it directly rather than rely on STATE_DIR."""
    root = tmp_path / "sessions"
    root.mkdir(parents=True)
    monkeypatch.setattr(phrase_store, "STATE_DIR", tmp_path)
    monkeypatch.setattr(session_iter, "SESSIONS_ROOT", root)
    return root


def _write_state(sid_dir: Path, **kwargs) -> None:
    state = {
        "session_id": sid_dir.name,
        "parent_session_id": None,
        "agent_role": "worker",
        "mode": "converse",
        "model": None,
        "source_trigger": {"type": "user", "ref": None},
    }
    state.update(kwargs)
    (sid_dir / "state.json").write_text(json.dumps(state), encoding="utf-8")


def _write_turns(sid_dir: Path, rows: list[dict]) -> None:
    with (sid_dir / "turns.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _final_turn(ts: str, role: str = "final") -> dict:
    return {"role": role, "timestamp": ts, "messages": [], "response": "hi"}


def _mk_session(root: Path, sid: str, *, state_kwargs: dict | None = None,
                turns: list[dict] | None = None) -> Path:
    sid_dir = root / sid
    sid_dir.mkdir()
    _write_state(sid_dir, **(state_kwargs or {}))
    _write_turns(sid_dir, turns or [])
    return sid_dir


# ── Date filter ──────────────────────────────────────────────────────────────

def test_returns_sessions_with_final_turns_on_target_date(sessions_root):
    _mk_session(sessions_root, "s1", turns=[
        _final_turn("2026-04-20T12:00:00+00:00"),
        _final_turn("2026-04-20T18:30:00+00:00"),
    ])
    _mk_session(sessions_root, "s2", turns=[
        _final_turn("2026-04-19T09:00:00+00:00"),
    ])
    out = session_iter.iter_sessions_for_date("2026-04-20")
    assert [c.session_id for c in out] == ["s1"]
    assert out[0].final_turn_count == 2
    assert out[0].last_final_ts == "2026-04-20T18:30:00+00:00"


def test_filters_non_final_turn_roles(sessions_root):
    _mk_session(sessions_root, "s1", turns=[
        _final_turn("2026-04-20T10:00:00+00:00", role="worker"),
        _final_turn("2026-04-20T11:00:00+00:00", role="supervisor"),
        _final_turn("2026-04-20T12:00:00+00:00", role="final"),
    ])
    out = session_iter.iter_sessions_for_date("2026-04-20")
    assert len(out) == 1 and out[0].final_turn_count == 1


def test_empty_or_missing_turns_file_is_skipped(sessions_root):
    sid_dir = sessions_root / "s_empty"
    sid_dir.mkdir()
    _write_state(sid_dir)
    # turns.jsonl missing
    out = session_iter.iter_sessions_for_date("2026-04-20")
    assert out == []


def test_malformed_jsonl_lines_are_tolerated(sessions_root):
    sid_dir = _mk_session(sessions_root, "s1", turns=[
        _final_turn("2026-04-20T10:00:00+00:00"),
    ])
    with (sid_dir / "turns.jsonl").open("a", encoding="utf-8") as f:
        f.write("{not json}\n")
        f.write("\n")
    out = session_iter.iter_sessions_for_date("2026-04-20")
    assert [c.session_id for c in out] == ["s1"]


# ── Excluded roles / sub-agents ──────────────────────────────────────────────

@pytest.mark.parametrize("role", [
    "dreamer", "meta_dreamer", "soul_updater", "session_compactor",
    "prompt_suggester", "webfetch_summarizer", "supervisor",
])
def test_excluded_roles_filtered(sessions_root, role):
    _mk_session(sessions_root, f"s_{role}",
                state_kwargs={"agent_role": role},
                turns=[_final_turn("2026-04-20T10:00:00+00:00")])
    assert session_iter.iter_sessions_for_date("2026-04-20") == []


def test_sub_agent_sessions_filtered_out(sessions_root):
    _mk_session(sessions_root, "child",
                state_kwargs={"parent_session_id": "parent-sid"},
                turns=[_final_turn("2026-04-20T10:00:00+00:00")])
    assert session_iter.iter_sessions_for_date("2026-04-20") == []


def test_dream_triggered_sessions_filtered_by_source_trigger(sessions_root):
    _mk_session(sessions_root, "s_dream",
                state_kwargs={"source_trigger": {"type": "cron", "ref": "dream_run"}},
                turns=[_final_turn("2026-04-20T10:00:00+00:00")])
    assert session_iter.iter_sessions_for_date("2026-04-20") == []


def test_session_without_state_file_still_included(sessions_root):
    sid_dir = sessions_root / "s_no_state"
    sid_dir.mkdir()
    _write_turns(sid_dir, [_final_turn("2026-04-20T10:00:00+00:00")])
    out = session_iter.iter_sessions_for_date("2026-04-20")
    assert [c.session_id for c in out] == ["s_no_state"]
    # agent_role defaults to 'worker' when state.json is missing.
    assert out[0].agent_role == "worker"


# ── Ordering ─────────────────────────────────────────────────────────────────

def test_results_sorted_by_latest_final_desc(sessions_root):
    _mk_session(sessions_root, "s_early", turns=[
        _final_turn("2026-04-20T06:00:00+00:00"),
    ])
    _mk_session(sessions_root, "s_late", turns=[
        _final_turn("2026-04-20T23:00:00+00:00"),
    ])
    _mk_session(sessions_root, "s_mid", turns=[
        _final_turn("2026-04-20T14:00:00+00:00"),
    ])
    out = session_iter.iter_sessions_for_date("2026-04-20")
    assert [c.session_id for c in out] == ["s_late", "s_mid", "s_early"]


# ── Metadata propagation ─────────────────────────────────────────────────────

def test_candidate_carries_role_mode_model(sessions_root):
    _mk_session(sessions_root, "s1",
                state_kwargs={"agent_role": "worker", "mode": "plan",
                              "model": "gemma3n-27b"},
                turns=[_final_turn("2026-04-20T10:00:00+00:00")])
    out = session_iter.iter_sessions_for_date("2026-04-20")
    assert out[0].agent_role == "worker"
    assert out[0].mode == "plan"
    assert out[0].model == "gemma3n-27b"
    assert out[0].turns_path == sessions_root / "s1" / "turns.jsonl"


# ── iter_yesterday_sessions convenience wrapper ──────────────────────────────

def test_iter_yesterday_resolves_to_utc_minus_one_day(sessions_root, monkeypatch):
    today = datetime.now(timezone.utc).date()
    yesterday = (today - timedelta(days=1)).strftime("%Y-%m-%d")
    _mk_session(sessions_root, "s_y", turns=[
        _final_turn(f"{yesterday}T10:00:00+00:00"),
    ])
    _mk_session(sessions_root, "s_t", turns=[
        _final_turn(f"{today.strftime('%Y-%m-%d')}T10:00:00+00:00"),
    ])
    out = session_iter.iter_yesterday_sessions()
    assert [c.session_id for c in out] == ["s_y"]


# ── Non-existent SESSIONS_ROOT ───────────────────────────────────────────────

def test_missing_sessions_root_returns_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(session_iter, "SESSIONS_ROOT", tmp_path / "does-not-exist")
    assert session_iter.iter_sessions_for_date("2026-04-20") == []
