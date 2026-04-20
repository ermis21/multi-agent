"""End-to-end test for the tool-result handle store + recall.

compact_tool_result + store_tool_result write to a session directory under
STATE_DIR/sessions/{sid}/tool_results/{hid}.txt with a sidecar index.jsonl.
The sandbox's tool_result_recall handler reads that file back.
"""

import json

import pytest

from app import context_compressor as cc


@pytest.fixture
def tmp_state(tmp_path, monkeypatch):
    monkeypatch.setattr(cc, "_STATE_DIR", tmp_path)
    yield tmp_path


def test_store_then_read_file(tmp_state):
    sid = "sess_01"
    body = "hello world, " * 50
    inline, hid = cc.compact_tool_result("web_fetch", {"url": "x"}, body, 5)
    assert hid is not None
    cc.store_tool_result(sid, hid, "web_fetch", {"url": "x"}, body)

    target = tmp_state / "sessions" / sid / "tool_results" / f"{hid}.txt"
    assert target.exists()
    assert target.read_text() == body


def test_index_jsonl_appended(tmp_state):
    sid = "sess_02"
    body = "x" * 500
    _, hid = cc.compact_tool_result("file_read", {"path": "a"}, body, 10)
    cc.store_tool_result(sid, hid, "file_read", {"path": "a"}, body)

    idx = tmp_state / "sessions" / sid / "tool_results" / "index.jsonl"
    assert idx.exists()
    lines = [json.loads(l) for l in idx.read_text().splitlines() if l.strip()]
    assert len(lines) == 1
    entry = lines[0]
    assert entry["handle_id"] == hid
    assert entry["tool"] == "file_read"
    assert entry["bytes"] == len(body.encode("utf-8"))


def test_store_is_idempotent_same_handle(tmp_state):
    sid = "sess_03"
    body = "identical body"
    hid = "rf-abc123"
    cc.store_tool_result(sid, hid, "x", {}, body)
    cc.store_tool_result(sid, hid, "x", {}, body)
    target = tmp_state / "sessions" / sid / "tool_results" / f"{hid}.txt"
    # Body file should exist exactly once with the same content
    assert target.read_text() == body


def test_store_no_op_without_session_id(tmp_state):
    # Defensive: missing sid should neither crash nor write
    cc.store_tool_result("", "rf-000000", "x", {}, "body")
    assert not any(tmp_state.glob("sessions/*/tool_results/*.txt"))
