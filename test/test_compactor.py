"""Unit tests for the rolling session compactor (app/compactor.py).

These tests stay pure-unit — the `session_compactor` role invocation is
stubbed so nothing hits the LLM. The goal is to pin down the trigger
contract and the state-rotation sequence; the integration test for real
compaction is the e2e `long_conversation` scenario.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

import app.compactor as compactor
import app.sessions.logger as slog
import app.sessions.state as ss
from app.sessions.state import SessionState


@pytest.fixture(autouse=True)
def _isolate_state(tmp_path, monkeypatch):
    monkeypatch.setattr(ss, "SESSIONS_DIR", tmp_path)
    monkeypatch.setattr(slog, "SESSIONS_DIR", tmp_path)
    monkeypatch.setattr(compactor, "SESSIONS_DIR", tmp_path)
    ss._CACHE.clear()
    compactor._compactor_locks.clear()
    yield


def _cfg(budget_history: int = 100, interval: int = 6, churn: float = 60.0) -> dict:
    return {
        "context": {
            "enabled": True,
            "budgets": {"history": budget_history},
            "compaction_interval_turns": interval,
            "compaction_churn_seconds": churn,
        }
    }


# ── should_trigger ───────────────────────────────────────────────────────────

def test_no_trigger_below_interval():
    st = SessionState.load_or_create("sid-a")
    st.set("stats.turn_count", 3)                  # < K
    st.set("context_stats.section_tokens", {"HISTORY": 9999})
    assert compactor.should_trigger(st, _cfg(interval=6)) is False


def test_no_trigger_under_history_budget():
    st = SessionState.load_or_create("sid-b")
    st.set("stats.turn_count", 6)
    st.set("context_stats.section_tokens", {"HISTORY": 50})   # well under 1.5× budget
    assert compactor.should_trigger(st, _cfg(budget_history=100)) is False


def test_triggers_when_all_conditions_met():
    st = SessionState.load_or_create("sid-c")
    st.set("stats.turn_count", 6)
    st.set("context_stats.section_tokens", {"HISTORY": 151})  # ≥ 1.5 × 100
    assert compactor.should_trigger(st, _cfg(budget_history=100, interval=6)) is True


def test_skips_retrigger_immediately_after_previous():
    st = SessionState.load_or_create("sid-d")
    st.set("stats.turn_count", 12)
    st.set("history.compaction_covers_up_to_turn", 7)         # only 5 new (< K=6)
    st.set("context_stats.section_tokens", {"HISTORY": 9999})
    assert compactor.should_trigger(st, _cfg(interval=6)) is False


def test_churn_guard_holds_fresh_compaction():
    st = SessionState.load_or_create("sid-e")
    st.set("stats.turn_count", 12)
    st.set("history.compaction_covers_up_to_turn", 0)
    st.set("context_stats.section_tokens", {"HISTORY": 9999})
    # Just finished compacting — churn guard must block.
    st.data["history"]["last_compaction_ts"] = ss._now()
    assert compactor.should_trigger(st, _cfg(churn=60.0)) is False


def test_respects_feature_flag():
    st = SessionState.load_or_create("sid-f")
    st.set("stats.turn_count", 999)
    st.set("context_stats.section_tokens", {"HISTORY": 999999})
    cfg = _cfg()
    cfg["context"]["enabled"] = False
    assert compactor.should_trigger(st, cfg) is False


# ── run_compaction (with stubbed role) ────────────────────────────────────────

def _write_turns(tmp_path: Path, sid: str, n: int) -> None:
    sdir = tmp_path / sid
    sdir.mkdir(parents=True, exist_ok=True)
    p = sdir / "turns.jsonl"
    with p.open("w", encoding="utf-8") as f:
        for i in range(n):
            f.write(json.dumps({
                "role": "final",
                "messages": [{"role": "user", "content": f"user turn {i}"}],
                "response": f"assistant turn {i}",
            }) + "\n")


def test_run_compaction_writes_active_and_rotates_pointer(tmp_path, monkeypatch):
    sid = "sid-run"
    _write_turns(tmp_path, sid, n=5)

    async def _stub_role(role, body, sub_sid):
        assert role == "session_compactor"
        assert "user turn 0" in body["plan_context"]
        assert "assistant turn 4" in body["plan_context"]
        return {
            "choices": [{"message": {"role": "assistant", "content":
                "## RUNNING_SUMMARY\n\n### Primary Intent\nexample\n\n"
                "## RECENT_DELTA\n\n### Next Step\nAWAIT_USER\n"}}],
        }

    import app.entrypoints as entrypoints
    monkeypatch.setattr(entrypoints, "run_agent_role", _stub_role)

    out = asyncio.run(compactor.run_compaction(sid))
    assert out["covers_up_to_turn"] == 5
    assert out["body_chars"] > 0

    st = SessionState.load_or_create(sid)
    assert st.get("history.active") == f"sessions/{sid}/active.jsonl"
    assert st.get("history.compaction_covers_up_to_turn") == 5
    assert st.get("history.last_compaction_ts") is not None

    active_path = tmp_path / sid / "active.jsonl"
    assert active_path.exists()
    line = json.loads(active_path.read_text().strip())
    assert line["role"] == "final"
    assert "## RUNNING_SUMMARY" in line["response"]


def test_run_compaction_rejects_malformed_output(tmp_path, monkeypatch):
    sid = "sid-malformed"
    _write_turns(tmp_path, sid, n=3)

    async def _stub_bad(role, body, sub_sid):
        return {"choices": [{"message": {"role": "assistant", "content": "no header here"}}]}

    import app.entrypoints as entrypoints
    monkeypatch.setattr(entrypoints, "run_agent_role", _stub_bad)

    out = asyncio.run(compactor.run_compaction(sid))
    assert out.get("error") == "malformed_output"
    st = SessionState.load_or_create(sid)
    assert st.get("history.active") is None


def test_concurrent_triggers_coalesce(tmp_path, monkeypatch):
    """Two simultaneous run_compaction calls must not both emit output."""
    sid = "sid-concurrent"
    _write_turns(tmp_path, sid, n=3)

    call_count = 0

    async def _stub_slow(role, body, sub_sid):
        nonlocal call_count
        call_count += 1
        await asyncio.sleep(0.05)
        return {
            "choices": [{"message": {"role": "assistant", "content":
                "## RUNNING_SUMMARY\nok\n## RECENT_DELTA\nok\n"}}],
        }

    import app.entrypoints as entrypoints
    monkeypatch.setattr(entrypoints, "run_agent_role", _stub_slow)

    async def _drive():
        return await asyncio.gather(
            compactor.run_compaction(sid),
            compactor.run_compaction(sid),
        )

    r1, r2 = asyncio.run(_drive())
    statuses = sorted([r1.get("skipped") or "ran", r2.get("skipped") or "ran"])
    assert statuses == ["already_locked", "ran"]
    assert call_count == 1


# ── maybe_spawn (thin wrapper) ────────────────────────────────────────────────

def test_maybe_spawn_returns_none_when_no_trigger():
    # Fresh state + empty config → nothing to do.
    SessionState.load_or_create("sid-idle")
    assert compactor.maybe_spawn("sid-idle", _cfg()) is None


def test_maybe_spawn_returns_none_without_event_loop():
    """Called from sync context (no running loop) — must not raise."""
    st = SessionState.load_or_create("sid-noloop")
    st.set("stats.turn_count", 6)
    st.set("context_stats.section_tokens", {"HISTORY": 999})
    # No running loop here, so create_task would raise RuntimeError — we expect None.
    assert compactor.maybe_spawn("sid-noloop", _cfg(budget_history=100)) is None
