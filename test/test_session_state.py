"""Pure-unit tests for the Phase 12 session-state integration.

Covers:
  - `SessionState.load_or_create` round-trip + `pending_injections` key present by default.
  - Sub-session graph (`add_sub_session` / `complete_sub_session`).
  - `TurnAccumulator.record_tool` auto-detects skill-file reads.
  - `log_approval` appends to `.approvals.jsonl` sidecar.
  - `_rebuild_session_context` prefers `state.history.active` over the full JSONL.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import app.sessions.state as ss
from app.sessions.state import SessionState, TurnAccumulator, log_approval


@pytest.fixture(autouse=True)
def _isolate_sessions_dir(tmp_path, monkeypatch):
    """Redirect SESSIONS_DIR so tests don't touch /sessions."""
    monkeypatch.setattr(ss, "SESSIONS_DIR", tmp_path)
    # Clear the global SessionState cache between tests.
    ss._CACHE.clear()
    yield


def test_default_state_has_pending_injections():
    st = SessionState.load_or_create("sid-default")
    # New key added in Phase 12.10 — migration backfills existing files.
    assert st.get("pending_injections") == []


def test_round_trip_save_load(tmp_path):
    st = SessionState.load_or_create("sid-rt")
    st.set("mode", "build")
    st.set("model", "local-gemma")
    st.set("supervisor.threshold", 0.73)
    st.save()

    # Drop from cache to force a disk read.
    ss._CACHE.clear()
    st2 = SessionState.load_or_create("sid-rt")
    assert st2.get("mode") == "build"
    assert st2.get("model") == "local-gemma"
    assert st2.get("supervisor.threshold") == 0.73


def test_sub_session_graph():
    parent = SessionState.load_or_create("sid-parent")
    parent.add_sub_session("sid-child-1")
    parent.add_sub_session("sid-child-2")
    assert parent.get("sub_sessions.active") == ["sid-child-1", "sid-child-2"]
    parent.complete_sub_session("sid-child-1")
    assert parent.get("sub_sessions.active") == ["sid-child-2"]
    assert parent.get("sub_sessions.completed") == ["sid-child-1"]


def test_sub_session_add_dedupes():
    parent = SessionState.load_or_create("sid-parent-dedupe")
    parent.add_sub_session("sid-a")
    parent.add_sub_session("sid-a")
    assert parent.get("sub_sessions.active") == ["sid-a"]


def test_record_tool_detects_skill_file_read():
    acc = TurnAccumulator()
    acc.record_tool("file_read", {"path": "/config/skills/log-triage/SKILL.md"}, error=False)
    acc.record_tool("file_read", {"path": "config/skills/other/SKILL.md"}, error=False)
    acc.record_tool("file_read", {"path": "/project/app/main.py"}, error=False)
    assert acc.skills_invoked == ["log-triage", "other"]


def test_record_tool_skill_is_set_semantic():
    acc = TurnAccumulator()
    acc.record_tool("file_read", {"path": "/config/skills/foo/SKILL.md"}, error=False)
    acc.record_tool("file_read", {"path": "/config/skills/foo/SKILL.md"}, error=False)
    assert acc.skills_invoked == ["foo"]


def test_flush_turn_merges_skills_invoked():
    st = SessionState.load_or_create("sid-skills")
    acc = TurnAccumulator()
    acc.record_tool("file_read", {"path": "/config/skills/first/SKILL.md"}, error=False)
    st.flush_turn(acc, verdict=None)

    acc2 = TurnAccumulator()
    acc2.record_tool("file_read", {"path": "/config/skills/second/SKILL.md"}, error=False)
    acc2.record_tool("file_read", {"path": "/config/skills/first/SKILL.md"}, error=False)  # dup
    st.flush_turn(acc2, verdict=None)

    inv = st.get("skills.invoked")
    assert inv == ["first", "second"]


def test_log_approval_appends_sidecar(tmp_path):
    log_approval("sid-approvals", "shell_exec", "auto_failed", {"reason": "denied_tools"})
    log_approval("sid-approvals", "shell_exec", "requested", {"approval_id": "abc"})
    p = tmp_path / "sid-approvals" / "approvals.jsonl"
    assert p.exists()
    lines = [json.loads(ln) for ln in p.read_text().strip().splitlines()]
    assert [ln["status"] for ln in lines] == ["auto_failed", "requested"]
    assert lines[0]["reason"] == "denied_tools"
    # Timestamps should be stamped automatically.
    assert all("ts" in ln for ln in lines)


def test_migrate_in_place_backfills_pending_injections(tmp_path):
    # Simulate an old state file missing the pending_injections key.
    old = {
        "session_id": "sid-old",
        "stats": {"turn_count": 1},
        "history": {"full": "sessions/sid-old/turns.jsonl"},
    }
    sdir = tmp_path / "sid-old"
    sdir.mkdir(parents=True, exist_ok=True)
    p = sdir / "state.json"
    p.write_text(json.dumps(old))
    ss._CACHE.clear()
    st = SessionState.load_or_create("sid-old")
    assert st.get("pending_injections") == []
    # Existing stats key is preserved.
    assert st.get("stats.turn_count") == 1


def test_default_state_has_context_stats_and_tool_results():
    """PR 1 keys must be present on fresh state."""
    st = SessionState.load_or_create("sid-ctx-default")
    assert st.get("context_stats.last_prompt_tokens") == 0
    assert st.get("context_stats.last_kv_prefix_hash") is None
    assert st.get("context_stats.soft_cap_exceeded") is False
    assert st.get("context_stats.section_tokens") == {}
    # Compression triggers dict is present with all slots at 0.
    trig = st.get("context_stats.compression_triggers")
    assert isinstance(trig, dict)
    for slot in ("soul", "memory", "user", "identity", "tool_docs", "skills", "history", "tool_result"):
        assert trig[slot] == 0
    # Handle index is empty dict (not list).
    assert st.get("tool_results") == {}


def test_migrate_in_place_backfills_context_stats(tmp_path):
    """An old state file missing context_stats/tool_results must get them on load."""
    old = {
        "session_id": "sid-ctx-old",
        "stats": {"turn_count": 3},
        "history": {"full": "sessions/sid-ctx-old/turns.jsonl"},
        # deliberately absent: context_stats, tool_results
    }
    sdir = tmp_path / "sid-ctx-old"
    sdir.mkdir(parents=True, exist_ok=True)
    (sdir / "state.json").write_text(json.dumps(old))
    ss._CACHE.clear()
    st = SessionState.load_or_create("sid-ctx-old")

    # New keys backfilled with defaults.
    assert st.get("context_stats.last_prompt_tokens") == 0
    assert st.get("context_stats.compression_triggers.soul") == 0
    assert st.get("tool_results") == {}
    # Pre-existing data untouched.
    assert st.get("stats.turn_count") == 3


def test_context_stats_round_trip():
    """Writing telemetry via dotted setters survives save/reload."""
    st = SessionState.load_or_create("sid-ctx-rt")
    st.set("context_stats.last_prompt_tokens", 7845)
    st.set("context_stats.last_kv_prefix_hash", "sha1:8c2f")
    st.set("context_stats.section_tokens", {"SOUL": 487, "USER": 214})
    st.set("context_stats.soft_cap_exceeded", True)
    st.save()

    ss._CACHE.clear()
    st2 = SessionState.load_or_create("sid-ctx-rt")
    assert st2.get("context_stats.last_prompt_tokens") == 7845
    assert st2.get("context_stats.last_kv_prefix_hash") == "sha1:8c2f"
    assert st2.get("context_stats.section_tokens") == {"SOUL": 487, "USER": 214}
    assert st2.get("context_stats.soft_cap_exceeded") is True


def test_flush_turn_persists_stats_and_verdict():
    st = SessionState.load_or_create("sid-stats")
    acc = TurnAccumulator(llm_call_count=3, token_usage={"input": 100, "output": 20, "thinking": 5}, duration_ms=1500)
    acc.record_tool("web_search", {"query": "hello"}, error=False)
    acc.record_tool("web_fetch", {"url": "https://example.com"}, error=True)
    verdict = {"pass": False, "score": 0.4, "feedback": "thin", "attempt": 2}
    st.flush_turn(acc, verdict=verdict)

    assert st.get("stats.turn_count") == 1
    assert st.get("stats.llm_call_count") == 3
    assert st.get("stats.tool_error_count") == 1
    assert st.get("stats.token_usage.input") == 100
    assert st.get("tools.invoked.web_search") == 1
    assert st.get("tools.invoked.web_fetch") == 1
    assert st.get("context_audit.web_searches") == ["hello"]
    assert st.get("context_audit.web_fetches") == ["https://example.com"]
    lv = st.get("supervisor.last_verdict")
    assert lv["pass"] is False
    assert lv["score"] == 0.4
    assert st.get("supervisor.fail_count") == 1


def test_history_active_preferred_for_rebuild(tmp_path, monkeypatch):
    """`_rebuild_session_context` reads from `state.history.active` when it exists."""
    import app.sessions.logger as slog
    import app.agents as agents

    monkeypatch.setattr(slog, "SESSIONS_DIR", tmp_path)

    sid = "sid-compacted"
    sdir = tmp_path / sid
    sdir.mkdir(parents=True, exist_ok=True)
    full_jsonl = sdir / "turns.jsonl"
    active_jsonl = sdir / "active.jsonl"

    def _turn(user_text: str, response: str) -> str:
        return json.dumps({
            "role": "final",
            "messages": [{"role": "user", "content": user_text}],
            "response": response,
        }) + "\n"

    full_jsonl.write_text(_turn("question A", "answer A") + _turn("question B", "answer B"))
    active_jsonl.write_text(_turn("compacted question", "compacted answer"))

    st = SessionState.load_or_create(sid)
    st.set("history.active", str(active_jsonl))
    st.save()

    ctx = agents._rebuild_session_context(sid, [{"role": "user", "content": "new"}], {})
    contents = [m.get("content") for m in ctx]
    assert "compacted question" in contents
    assert "compacted answer" in contents
    # Full-log entries must not leak through.
    assert "question A" not in contents
    assert "question B" not in contents


def test_history_active_falls_back_to_full(tmp_path, monkeypatch):
    """Without `history.active`, replay uses the full JSONL."""
    import app.sessions.logger as slog
    import app.agents as agents

    monkeypatch.setattr(slog, "SESSIONS_DIR", tmp_path)

    sid = "sid-full-only"
    sdir = tmp_path / sid
    sdir.mkdir(parents=True, exist_ok=True)
    full_jsonl = sdir / "turns.jsonl"
    full_jsonl.write_text(json.dumps({
        "role": "final",
        "messages": [{"role": "user", "content": "only question"}],
        "response": "only answer",
    }) + "\n")

    # Create the state file but leave history.active null.
    st = SessionState.load_or_create(sid)
    st.save()

    ctx = agents._rebuild_session_context(sid, [{"role": "user", "content": "new"}], {})
    contents = [m.get("content") for m in ctx]
    assert "only question" in contents
    assert "only answer" in contents
