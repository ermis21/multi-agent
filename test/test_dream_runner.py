"""Tests for app.dream.runner — nightly orchestration.

Covers:
  - run_dream with zero candidates writes run.json with empty lists.
  - run_dream iterates candidates, dispatches run_agent_role per session,
    collects outcomes, and calls meta-dreamer.
  - interrupt_event set between sessions short-circuits the loop and
    stamps interrupted_at.
  - Per-session run_agent_role exception surfaces as status="error" and
    does not stop the loop.
  - _collect_outcome returns rolled_back_unfinalized when a pending batch
    still exists after the agent call.
  - _committed_today collects phrase_history rows matching session_id.
  - run.json is written even when meta-dreamer fails.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from app.dream import dream_state, meta_dreamer, phrase_store, runner, session_iter


@pytest.fixture
def rootfs(tmp_path, monkeypatch):
    state = tmp_path / "state"
    runs = state / "dream" / "runs"
    index = state / "dream" / "phrase_index"
    history = state / "dream" / "phrase_history"
    sessions = state / "sessions"
    for p in (runs, index, history, sessions):
        p.mkdir(parents=True)

    monkeypatch.setattr(phrase_store, "STATE_DIR", state)
    monkeypatch.setattr(phrase_store, "INDEX_DIR", index)
    monkeypatch.setattr(phrase_store, "HISTORY_DIR", history)
    monkeypatch.setattr(runner, "DREAM_RUNS_ROOT", runs)
    monkeypatch.setattr(dream_state, "DREAM_RUNS_ROOT", runs)
    monkeypatch.setattr(session_iter, "SESSIONS_ROOT", sessions)

    return {"state": state, "runs": runs, "history": history, "sessions": sessions}


def _seed_session(rootfs, sid: str, *, role: str = "worker",
                  turns: list[dict] | None = None,
                  state_stamp: dict | None = None) -> None:
    d = rootfs["sessions"] / sid
    d.mkdir()
    state = {"agent_role": role, "source_trigger": {"type": "user", "ref": None}}
    if state_stamp:
        state.update(state_stamp)
    (d / "state.json").write_text(json.dumps(state), encoding="utf-8")
    turns = turns or [{
        "role": "final",
        "timestamp": "2026-04-20T12:00:00+00:00",
        "messages": [{"role": "user", "content": "hello"}],
        "response": "hi",
    }]
    (d / "turns.jsonl").write_text(
        "\n".join(json.dumps(t) for t in turns) + "\n", encoding="utf-8",
    )


# ── run_dream behavior ───────────────────────────────────────────────────────

def test_run_dream_no_candidates_writes_empty_run_json(rootfs, monkeypatch):
    async def _noop_agent(role, body, session_id, *, prompts_dir=None):
        return {"choices": [{"message": {"content": "ok"}}]}

    async def _noop_meta(record, *, cfg=None, session_id=None, top_k=3):
        return {"status": "no_conflicts"}

    import app.entrypoints as ep_mod
    monkeypatch.setattr(ep_mod, "run_agent_role", _noop_agent)
    monkeypatch.setattr(meta_dreamer, "run_meta_dreamer", _noop_meta)

    record = asyncio.run(runner.run_dream("2026-04-20"))
    assert record["session_ids_seen"] == []
    assert record["conversations"] == []
    assert record["interrupted_at"] is None
    run_json = rootfs["runs"] / "2026-04-20" / "run.json"
    assert run_json.exists()
    saved = json.loads(run_json.read_text(encoding="utf-8"))
    assert saved["date"] == "2026-04-20"


def test_run_dream_dispatches_per_candidate(rootfs, monkeypatch):
    _seed_session(rootfs, "conv-1")
    _seed_session(rootfs, "conv-2")

    calls: list[dict] = []

    async def fake_agent(role, body, session_id, *, prompts_dir=None):
        calls.append({"role": role, "sid": session_id, "body": body})
        return {"choices": [{"message": {"content": "ok"}}]}

    async def fake_meta(record, *, cfg=None, session_id=None, top_k=3):
        return {"status": "no_conflicts", "seen": len(record.get("conversations", []))}

    import app.entrypoints as ep_mod
    monkeypatch.setattr(ep_mod, "run_agent_role", fake_agent)
    monkeypatch.setattr(meta_dreamer, "run_meta_dreamer", fake_meta)

    record = asyncio.run(runner.run_dream("2026-04-20"))
    assert len(calls) == 2
    assert all(c["role"] == "dreamer" for c in calls)
    # dreamer session ids are prefixed with the run date and original sid.
    assert all(c["sid"].startswith("dreamer_2026-04-20_") for c in calls)
    assert set(record["session_ids_completed"]) == {"conv-1", "conv-2"}
    assert record["meta"]["status"] == "no_conflicts"
    assert record["meta"]["seen"] == 2


def test_run_dream_interrupt_short_circuits(rootfs, monkeypatch):
    _seed_session(rootfs, "conv-a")
    _seed_session(rootfs, "conv-b")

    ev = asyncio.Event()

    async def fake_agent(role, body, session_id, *, prompts_dir=None):
        # After the first session, flip the interrupt.
        ev.set()
        return {"choices": [{"message": {"content": "ok"}}]}

    called_meta = {"n": 0}

    async def fake_meta(record, *, cfg=None, session_id=None, top_k=3):
        called_meta["n"] += 1
        return {"status": "no_conflicts"}

    import app.entrypoints as ep_mod
    monkeypatch.setattr(ep_mod, "run_agent_role", fake_agent)
    monkeypatch.setattr(meta_dreamer, "run_meta_dreamer", fake_meta)

    record = asyncio.run(runner.run_dream("2026-04-20", interrupt_event=ev))
    # First session ran, second got skipped.
    assert len(record["session_ids_completed"]) == 1
    assert record["interrupted_at"] is not None
    # Meta-dreamer must NOT run when interrupted.
    assert called_meta["n"] == 0


def test_run_dream_surfaces_per_session_error_without_aborting(rootfs, monkeypatch):
    _seed_session(rootfs, "conv-good")
    _seed_session(rootfs, "conv-bad")

    async def fake_agent(role, body, session_id, *, prompts_dir=None):
        if "conv-bad" in session_id:
            raise RuntimeError("kaboom")
        return {"choices": [{"message": {"content": "ok"}}]}

    async def fake_meta(record, **_):
        return {"status": "no_conflicts"}

    import app.entrypoints as ep_mod
    monkeypatch.setattr(ep_mod, "run_agent_role", fake_agent)
    monkeypatch.setattr(meta_dreamer, "run_meta_dreamer", fake_meta)

    record = asyncio.run(runner.run_dream("2026-04-20"))
    # Find the error row.
    statuses = {c["conversation_sid"]: c["status"] for c in record["conversations"]}
    assert statuses.get("conv-bad") == "error"
    # Good session still completes — session_ids_completed includes it.
    assert "conv-good" in record["session_ids_completed"]
    assert "conv-bad" not in record["session_ids_completed"]


def test_run_dream_survives_meta_dreamer_failure(rootfs, monkeypatch):
    _seed_session(rootfs, "conv-x")

    async def fake_agent(role, body, session_id, *, prompts_dir=None):
        return {"choices": [{"message": {"content": "ok"}}]}

    async def explode_meta(record, **_):
        raise RuntimeError("meta kaboom")

    import app.entrypoints as ep_mod
    monkeypatch.setattr(ep_mod, "run_agent_role", fake_agent)
    monkeypatch.setattr(meta_dreamer, "run_meta_dreamer", explode_meta)

    record = asyncio.run(runner.run_dream("2026-04-20"))
    assert record["meta"]["status"] == "error"
    assert "meta kaboom" in record["meta"]["error"]
    # run.json still written.
    assert (rootfs["runs"] / "2026-04-20" / "run.json").exists()


# ── _collect_outcome ─────────────────────────────────────────────────────────

def test_collect_outcome_reports_unfinalized_pending(rootfs):
    _seed_session(rootfs, "conv-pending")
    # Write a pending batch directly.
    dream_state.create_or_replace_pending(
        conversation_sid="conv-pending",
        target_prompt="worker_full",
        new_prompt_text="...",
        rationale="test",
        edits=[{"phrase_id": "ph-a", "status": "ok", "old_text": "a",
                "new_text": "b", "section_path": "R", "idx": 0}],
    )
    c = session_iter.SessionCandidate(
        session_id="conv-pending", agent_role="worker", mode=None, model=None,
        final_turn_count=1, last_final_ts="", turns_path=Path("/dev/null"),
    )
    out = runner._collect_outcome(c)
    assert out["status"] == "rolled_back_unfinalized"
    assert out["committed"] == []


def test_collect_outcome_reads_committed_from_phrase_history(rootfs):
    _seed_session(rootfs, "conv-finalized")
    # Append a phrase_history row whose session_id matches.
    hf = rootfs["history"] / "ph-aaaaaaaaaa.jsonl"
    hf.write_text(json.dumps({
        "rev": 1,
        "session_id": "conv-finalized",
        "role_template_name": "worker_full",
        "section_breadcrumb": "Rules",
        "old_text": "x", "new_text": "y",
        "run_date": "2026-04-20T05:00:00+00:00",
        "applied": True,
    }) + "\n", encoding="utf-8")

    c = session_iter.SessionCandidate(
        session_id="conv-finalized", agent_role="worker", mode=None, model=None,
        final_turn_count=1, last_final_ts="", turns_path=Path("/dev/null"),
    )
    out = runner._collect_outcome(c)
    assert out["status"] == "finalized"
    assert len(out["committed"]) == 1
    assert out["committed"][0]["phrase_id"] == "ph-aaaaaaaaaa"
    assert out["committed"][0]["prompt_name"] == "worker_full"


def test_collect_outcome_no_submission_when_clean(rootfs):
    _seed_session(rootfs, "conv-clean")
    c = session_iter.SessionCandidate(
        session_id="conv-clean", agent_role="worker", mode=None, model=None,
        final_turn_count=1, last_final_ts="", turns_path=Path("/dev/null"),
    )
    out = runner._collect_outcome(c)
    assert out["status"] == "no_submission"
    assert out["committed"] == []
    assert out["flagged"] == []


# ── _committed_today dedup / filters ────────────────────────────────────────

def test_committed_today_filters_by_session_id(rootfs):
    hf = rootfs["history"] / "ph-zzzzzzzzzz.jsonl"
    hf.write_text(
        json.dumps({"session_id": "other_sid", "applied": True,
                    "rev": 1, "role_template_name": "worker_full",
                    "section_breadcrumb": "R", "old_text": "a", "new_text": "b"})
        + "\n" +
        json.dumps({"session_id": "target_sid", "applied": True,
                    "rev": 2, "role_template_name": "worker_full",
                    "section_breadcrumb": "R", "old_text": "b", "new_text": "c"})
        + "\n",
        encoding="utf-8",
    )
    rows = runner._committed_today("target_sid")
    assert len(rows) == 1
    assert rows[0]["phrase_id"] == "ph-zzzzzzzzzz"
    assert rows[0]["rev"] == 2


def test_committed_today_respects_applied_flag(rootfs):
    hf = rootfs["history"] / "ph-yyyyyyyyyy.jsonl"
    hf.write_text(json.dumps({
        "session_id": "target_sid", "applied": False, "rev": 1,
        "role_template_name": "worker_full", "section_breadcrumb": "R",
        "old_text": "a", "new_text": "b",
    }) + "\n", encoding="utf-8")
    rows = runner._committed_today("target_sid")
    assert rows == []
