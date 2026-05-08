"""Pure-unit tests for the Phase 0 telemetry fields added for the prompting
overhaul. Each new field must default correctly and survive a save/load
round-trip — they're load-bearing for diagnostic probes added in later phases.
"""

from __future__ import annotations

import pytest

import app.sessions.state as ss
from app.sessions.state import SessionState


@pytest.fixture(autouse=True)
def _isolate_sessions_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(ss, "SESSIONS_DIR", tmp_path)
    ss._CACHE.clear()
    yield


def test_supervisor_parse_failures_defaults_zero():
    st = SessionState.load_or_create("sid-sp")
    assert st.get("supervisor")["parse_failures"] == 0


def test_anthropic_native_fallbacks_defaults_zero():
    st = SessionState.load_or_create("sid-af")
    assert st.get("stats")["anthropic_native_fallbacks"] == 0


def test_playbook_entries_added_defaults_zero():
    st = SessionState.load_or_create("sid-pb")
    assert st.get("playbook") == {"entries_added": 0}


def test_telemetry_round_trips():
    st = SessionState.load_or_create("sid-rt")
    sup = st.get("supervisor")
    sup["parse_failures"] = 3
    st.set("supervisor", sup)
    stats = st.get("stats")
    stats["anthropic_native_fallbacks"] = 2
    st.set("stats", stats)
    st.set("playbook", {"entries_added": 5})
    st.save()

    ss._CACHE.clear()
    reloaded = SessionState.load_or_create("sid-rt")
    assert reloaded.get("supervisor")["parse_failures"] == 3
    assert reloaded.get("stats")["anthropic_native_fallbacks"] == 2
    assert reloaded.get("playbook")["entries_added"] == 5
