"""Tests for message editing / session rewind.

Covers:
  - `SessionState.append_user_msg` / `append_bot_msgs` are idempotent and order-preserving.
  - `SessionLogger.truncate_to_final` drops the target final + its attempt chain + everything after.
  - `POST /v1/sessions/{sid}/rewind` truncates log, recomputes stats, invalidates compaction.
  - `POST /v1/sessions/{sid}/message_index/bot_msgs` appends+dedupes.
  - Rewind is rejected with 409 when the session is in-flight.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import app.sessions.logger as sl
import app.sessions.state as ss
from app.sessions.logger import SessionLogger
from app.sessions.state import SessionState


# ── Fixtures: isolate on-disk session dirs ────────────────────────────────────

@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    monkeypatch.setattr(ss, "SESSIONS_DIR", tmp_path)
    monkeypatch.setattr(sl, "SESSIONS_DIR", tmp_path)
    ss._CACHE.clear()
    yield
    ss._CACHE.clear()


def _log_turn_group(logger: SessionLogger, msgs, response, attempts: int = 1) -> None:
    """Append `attempts-1` worker/supervisor pairs then a clean final."""
    for a in range(attempts - 1):
        logger.log_turn(a, "worker", msgs, f"draft_{a}")
        logger.log_turn(a, "supervisor", msgs, f"draft_{a}",
                        supervisor={"pass": False, "score": 0.4, "feedback": "retry"})
    logger.log_turn(0, "final", msgs, response)


# ── SessionState.append_user_msg / append_bot_msgs ───────────────────────────

def test_append_user_msg_idempotent_on_discord_msg_id():
    st = SessionState.load_or_create("sid-u1")
    st.append_user_msg(turn_index=0, discord_msg_id="d-100", channel_id="c-1")
    st.append_user_msg(turn_index=0, discord_msg_id="d-100", channel_id="c-1")
    st.append_user_msg(turn_index=1, discord_msg_id="d-101", channel_id="c-1")
    msgs = st.get("message_index.user_msgs")
    assert len(msgs) == 2
    assert msgs[0]["turn_index"] == 0 and msgs[0]["discord_msg_id"] == "d-100"
    assert msgs[1]["turn_index"] == 1 and msgs[1]["discord_msg_id"] == "d-101"


def test_append_bot_msgs_dedupes_per_turn():
    st = SessionState.load_or_create("sid-b1")
    st.append_bot_msgs(turn_index=0, discord_msg_ids=["b-1", "b-2"], channel_id="c-1")
    st.append_bot_msgs(turn_index=0, discord_msg_ids=["b-2", "b-3"], channel_id="c-1")  # b-2 dup
    st.append_bot_msgs(turn_index=1, discord_msg_ids=["b-4"], channel_id="c-1")
    bots = st.get("message_index.bot_msgs")
    assert [m["discord_msg_id"] for m in bots] == ["b-1", "b-2", "b-3", "b-4"]
    assert [m["turn_index"] for m in bots] == [0, 0, 0, 1]


# ── SessionLogger.truncate_to_final ──────────────────────────────────────────

def test_truncate_drops_target_final_and_everything_after():
    logger = SessionLogger("sid-trunc-1")
    _log_turn_group(logger, [{"role": "user", "content": "q0"}], "a0")
    _log_turn_group(logger, [{"role": "user", "content": "q1"}], "a1")
    _log_turn_group(logger, [{"role": "user", "content": "q2"}], "a2")

    surviving, dropped = logger.truncate_to_final(1)

    assert [t["role"] for t in surviving] == ["final"]
    assert [t.get("response") for t in surviving] == ["a0"]
    assert [t["role"] for t in dropped].count("final") == 2
    # The on-disk log now has exactly one line (only the first final survived).
    lines = logger.path.read_text().strip().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["response"] == "a0"


def test_truncate_drops_worker_supervisor_attempts_of_target_final():
    """The attempt chain (worker+supervisor lines) preceding the target final
    belongs to the target's retry history and must be dropped too."""
    logger = SessionLogger("sid-trunc-2")
    _log_turn_group(logger, [{"role": "user", "content": "q0"}], "a0")
    _log_turn_group(logger, [{"role": "user", "content": "q1"}], "a1", attempts=3)

    surviving, dropped = logger.truncate_to_final(1)

    # Only the first final (attempts=1, one line) should survive.
    assert len(surviving) == 1
    assert surviving[0]["role"] == "final"
    # Dropped: 2 worker + 2 supervisor + 1 final for the retried turn.
    roles = [t["role"] for t in dropped]
    assert roles.count("worker") == 2
    assert roles.count("supervisor") == 2
    assert roles.count("final") == 1


def test_truncate_rejects_out_of_range():
    logger = SessionLogger("sid-trunc-3")
    _log_turn_group(logger, [{"role": "user", "content": "q"}], "a")
    with pytest.raises(ValueError):
        logger.truncate_to_final(5)
    with pytest.raises(ValueError):
        logger.truncate_to_final(-1)


def test_truncate_is_atomic_on_failure(monkeypatch):
    """If os.replace raises, the original turns.jsonl must be unchanged."""
    logger = SessionLogger("sid-trunc-4")
    _log_turn_group(logger, [{"role": "user", "content": "q0"}], "a0")
    _log_turn_group(logger, [{"role": "user", "content": "q1"}], "a1")
    original = logger.path.read_text()

    def boom(*_a, **_kw):
        raise OSError("simulated rename failure")
    monkeypatch.setattr("os.replace", boom)

    with pytest.raises(OSError):
        logger.truncate_to_final(0)

    assert logger.path.read_text() == original


# ── /v1/sessions/{sid}/rewind endpoint ────────────────────────────────────────

def _make_client(monkeypatch, tmp_path):
    """Build a TestClient against app.main with monkeypatched SESSIONS_DIRs
    so the endpoint reads/writes under tmp_path."""
    from app import main as api
    monkeypatch.setattr(ss, "SESSIONS_DIR", tmp_path)
    monkeypatch.setattr(sl, "SESSIONS_DIR", tmp_path)
    ss._CACHE.clear()
    api._active_sessions.clear()
    return TestClient(api.app), api


def test_rewind_truncates_turns_and_stats(monkeypatch, tmp_path):
    client, api = _make_client(monkeypatch, tmp_path)
    sid = "sid-rw-1"
    # Seed three finals via the logger + state (bypasses LLM).
    logger = SessionLogger(sid)
    _log_turn_group(logger, [{"role": "user", "content": "q0"}], "a0")
    _log_turn_group(logger, [{"role": "user", "content": "q1"}], "a1")
    _log_turn_group(logger, [{"role": "user", "content": "q2"}], "a2")

    st = SessionState.load_or_create(sid)
    st.set("stats.turn_count", 3)
    st.set("stats.llm_call_count", 9)
    st.set("stats.token_usage", {"input": 100, "output": 200, "thinking": 0})
    st.append_user_msg(0, "u0", "c")
    st.append_user_msg(1, "u1", "c")
    st.append_user_msg(2, "u2", "c")
    st.append_bot_msgs(0, ["b0a", "b0b"], "c")
    st.append_bot_msgs(1, ["b1a"], "c")
    st.append_bot_msgs(2, ["b2a", "b2b"], "c")
    st.save()

    resp = client.post(f"/v1/sessions/{sid}/rewind", json={"target_turn_index": 1})
    assert resp.status_code == 200, resp.text
    body = resp.json()

    assert body["truncated_to_turn"] == 1
    assert body["dropped_user_msg_ids"] == ["u1", "u2"]
    assert body["dropped_bot_msg_ids"]  == ["b1a", "b2a", "b2b"]
    assert body["compaction_invalidated"] is False

    # State reloaded should reflect rewind.
    ss._CACHE.clear()
    st2 = SessionState.load_or_create(sid)
    assert st2.get("stats.turn_count") == 1
    assert st2.get("stats.token_usage") == {"input": 0, "output": 0, "thinking": 0}
    assert [m["discord_msg_id"] for m in st2.get("message_index.user_msgs")] == ["u0"]
    assert [m["discord_msg_id"] for m in st2.get("message_index.bot_msgs")]  == ["b0a", "b0b"]

    # turns.jsonl has only one line.
    lines = (tmp_path / sid / "turns.jsonl").read_text().strip().splitlines()
    assert len(lines) == 1


def test_rewind_invalidates_compaction(monkeypatch, tmp_path):
    client, api = _make_client(monkeypatch, tmp_path)
    sid = "sid-rw-2"
    logger = SessionLogger(sid)
    for i in range(5):
        _log_turn_group(logger, [{"role": "user", "content": f"q{i}"}], f"a{i}")
    st = SessionState.load_or_create(sid)
    st.set("stats.turn_count", 5)
    # Pretend compactor covered turns 0..3 and wrote active.jsonl.
    active = tmp_path / sid / "active.jsonl"
    active.write_text('{"role":"final","response":"[compacted]"}\n')
    st.set("history.active", f"sessions/{sid}/active.jsonl")
    st.set("history.compaction_covers_up_to_turn", 3)
    st.save()

    resp = client.post(f"/v1/sessions/{sid}/rewind", json={"target_turn_index": 2})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["compaction_invalidated"] is True
    assert not active.exists()

    ss._CACHE.clear()
    st2 = SessionState.load_or_create(sid)
    assert st2.get("history.active") is None
    assert st2.get("history.compaction_covers_up_to_turn") is None


def test_rewind_preserves_non_turn_state(monkeypatch, tmp_path):
    client, api = _make_client(monkeypatch, tmp_path)
    sid = "sid-rw-3"
    logger = SessionLogger(sid)
    _log_turn_group(logger, [{"role": "user", "content": "q0"}], "a0")
    _log_turn_group(logger, [{"role": "user", "content": "q1"}], "a1")
    st = SessionState.load_or_create(sid)
    st.set("stats.turn_count", 2)
    st.set("mode", "build")
    st.set("model", "preset-x")
    st.set("plan", {"path": "plans/x.md", "context": "ctx"})
    st.set("permissions.approved_tools", ["memory_add"])
    st.set("permissions.privileged_paths", ["/workspace/a"])
    st.add_sub_session("child-1")
    st.save()

    resp = client.post(f"/v1/sessions/{sid}/rewind", json={"target_turn_index": 0})
    assert resp.status_code == 200

    ss._CACHE.clear()
    st2 = SessionState.load_or_create(sid)
    assert st2.get("mode") == "build"
    assert st2.get("model") == "preset-x"
    assert st2.get("plan") == {"path": "plans/x.md", "context": "ctx"}
    assert st2.get("permissions.approved_tools") == ["memory_add"]
    assert st2.get("permissions.privileged_paths") == ["/workspace/a"]
    assert st2.get("sub_sessions.active") == ["child-1"]


def test_rewind_rejects_inflight(monkeypatch, tmp_path):
    client, api = _make_client(monkeypatch, tmp_path)
    sid = "sid-rw-4"
    logger = SessionLogger(sid)
    _log_turn_group(logger, [{"role": "user", "content": "q"}], "a")
    st = SessionState.load_or_create(sid)
    st.set("stats.turn_count", 1)
    st.save()

    api._active_sessions[sid] = {"pending": [], "cancel": asyncio.Event(), "task": None}
    try:
        resp = client.post(f"/v1/sessions/{sid}/rewind", json={"target_turn_index": 0})
        assert resp.status_code == 409
    finally:
        api._active_sessions.clear()


def test_rewind_out_of_range(monkeypatch, tmp_path):
    client, api = _make_client(monkeypatch, tmp_path)
    sid = "sid-rw-5"
    logger = SessionLogger(sid)
    _log_turn_group(logger, [{"role": "user", "content": "q"}], "a")
    st = SessionState.load_or_create(sid)
    st.set("stats.turn_count", 1)
    st.save()

    resp = client.post(f"/v1/sessions/{sid}/rewind", json={"target_turn_index": 5})
    assert resp.status_code == 400


# ── /v1/sessions/{sid}/message_index/bot_msgs endpoint ───────────────────────

def test_append_bot_msgs_endpoint(monkeypatch, tmp_path):
    client, api = _make_client(monkeypatch, tmp_path)
    sid = "sid-bm-1"
    r1 = client.post(
        f"/v1/sessions/{sid}/message_index/bot_msgs",
        json={"turn_index": 0, "discord_msg_ids": ["b1", "b2"], "channel_id": "c"},
    )
    assert r1.status_code == 200
    r2 = client.post(
        f"/v1/sessions/{sid}/message_index/bot_msgs",
        json={"turn_index": 0, "discord_msg_ids": ["b2", "b3"]},  # b2 dup
    )
    assert r2.status_code == 200

    ss._CACHE.clear()
    st = SessionState.load_or_create(sid)
    assert [m["discord_msg_id"] for m in st.get("message_index.bot_msgs")] == ["b1", "b2", "b3"]


def test_append_bot_msgs_endpoint_validates_payload(monkeypatch, tmp_path):
    client, api = _make_client(monkeypatch, tmp_path)
    sid = "sid-bm-2"
    # Missing turn_index.
    r = client.post(f"/v1/sessions/{sid}/message_index/bot_msgs",
                    json={"discord_msg_ids": ["x"]})
    assert r.status_code == 400
    # discord_msg_ids not a list.
    r = client.post(f"/v1/sessions/{sid}/message_index/bot_msgs",
                    json={"turn_index": 0, "discord_msg_ids": "nope"})
    assert r.status_code == 400
