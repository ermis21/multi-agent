"""Tests for app.dream.meta_dreamer — conflict ranking + briefing + dispatch.

Covers:
  - top_conflict_phrases counts flagged edits across conversations;
    highest-count first, tie-breaking respects Counter.most_common order.
  - top_conflict_phrases enriches entries with current_text + section_path
    from phrase_store when the phrase index exists.
  - top_conflict_phrases tolerates missing phrase_index files gracefully
    (no crash; current_text empty).
  - top_conflict_phrases respects top_k.
  - build_meta_briefing yields a short "no action" message on empty input.
  - build_meta_briefing mentions each phrase_id + count + prompt + status.
  - run_meta_dreamer returns `no_conflicts` when there's nothing flagged.
  - run_meta_dreamer dispatches run_agent_role("dreamer", ...) with the
    briefing as the user message and stamps source_trigger=cron/meta_dreamer.
  - run_meta_dreamer surfaces dispatch errors as status="error".
"""

from __future__ import annotations

import asyncio
import json

import pytest

from app.dream import meta_dreamer, phrase_store


@pytest.fixture
def phrase_state(tmp_path, monkeypatch):
    state = tmp_path / "state"
    prompts = tmp_path / "prompts"
    (state / "dream" / "phrase_index").mkdir(parents=True)
    (state / "dream" / "phrase_history").mkdir(parents=True)
    prompts.mkdir()
    monkeypatch.setattr(phrase_store, "STATE_DIR", state)
    monkeypatch.setattr(phrase_store, "INDEX_DIR", state / "dream" / "phrase_index")
    monkeypatch.setattr(phrase_store, "HISTORY_DIR", state / "dream" / "phrase_history")
    monkeypatch.setattr(phrase_store, "PROMPTS_DIR", prompts)
    return {"state": state, "prompts": prompts}


def _write_index(phrase_id: str, text: str, breadcrumb: str = "Root", role: str = "worker_full"):
    (phrase_store.INDEX_DIR / f"{phrase_id}.json").write_text(
        json.dumps({
            "phrase_id": phrase_id,
            "role_template_name": role,
            "section_breadcrumb": breadcrumb,
            "current_text": text,
        }),
        encoding="utf-8",
    )


def _run_record(conversations: list[dict]) -> dict:
    return {"date": "2026-04-21", "conversations": conversations}


# ── top_conflict_phrases ─────────────────────────────────────────────────────

def test_top_conflict_phrases_ranks_by_count(phrase_state):
    _write_index("ph-aaaaaaaaaa", "AAA")
    _write_index("ph-bbbbbbbbbb", "BBB")
    _write_index("ph-cccccccccc", "CCC")

    rec = _run_record([
        {"conversation_sid": "s1", "flagged": [
            {"phrase_id": "ph-aaaaaaaaaa", "status": "possible_conflict", "prompt_name": "worker_full"},
            {"phrase_id": "ph-bbbbbbbbbb", "status": "possible_loop", "prompt_name": "worker_full"},
        ]},
        {"conversation_sid": "s2", "flagged": [
            {"phrase_id": "ph-aaaaaaaaaa", "status": "possible_conflict", "prompt_name": "worker_full"},
        ]},
        {"conversation_sid": "s3", "flagged": [
            {"phrase_id": "ph-aaaaaaaaaa", "status": "possible_loop", "prompt_name": "worker_full"},
            {"phrase_id": "ph-cccccccccc", "status": "possible_conflict", "prompt_name": "worker_full"},
        ]},
    ])
    ranked = meta_dreamer.top_conflict_phrases(rec, top_k=3)
    assert len(ranked) == 3
    assert ranked[0]["phrase_id"] == "ph-aaaaaaaaaa"
    assert ranked[0]["count"] == 3
    # The other two both have count=1; order is insertion — acceptable.
    other_ids = {ranked[1]["phrase_id"], ranked[2]["phrase_id"]}
    assert other_ids == {"ph-bbbbbbbbbb", "ph-cccccccccc"}


def test_top_conflict_phrases_enriches_with_current_text(phrase_state):
    _write_index("ph-aaaaaaaaaa", "the canonical text", breadcrumb="Hard rules")
    rec = _run_record([
        {"flagged": [{"phrase_id": "ph-aaaaaaaaaa", "status": "possible_conflict",
                      "prompt_name": "worker_full"}]},
    ])
    ranked = meta_dreamer.top_conflict_phrases(rec, top_k=1)
    assert ranked[0]["current_text"] == "the canonical text"
    assert ranked[0]["section_path"] == "Hard rules"
    assert ranked[0]["last_status"] == "possible_conflict"


def test_top_conflict_phrases_tolerates_missing_phrase_index(phrase_state):
    # No write_index call — the index file for ph-xxx doesn't exist.
    rec = _run_record([
        {"flagged": [{"phrase_id": "ph-xxxxxxxxxx", "status": "possible_loop",
                      "prompt_name": "worker_full"}]},
    ])
    ranked = meta_dreamer.top_conflict_phrases(rec, top_k=1)
    assert len(ranked) == 1
    assert ranked[0]["phrase_id"] == "ph-xxxxxxxxxx"
    assert ranked[0]["current_text"] == ""
    assert ranked[0]["section_path"] == ""


def test_top_conflict_phrases_respects_top_k(phrase_state):
    for pid in ("ph-aaaaaaaaaa", "ph-bbbbbbbbbb", "ph-cccccccccc", "ph-dddddddddd"):
        _write_index(pid, pid)
    rec = _run_record([
        {"flagged": [{"phrase_id": pid, "prompt_name": "worker_full"} for pid in
                     ("ph-aaaaaaaaaa", "ph-bbbbbbbbbb", "ph-cccccccccc", "ph-dddddddddd")]},
    ])
    ranked = meta_dreamer.top_conflict_phrases(rec, top_k=2)
    assert len(ranked) == 2


def test_top_conflict_phrases_empty_run(phrase_state):
    assert meta_dreamer.top_conflict_phrases(_run_record([])) == []


def test_top_conflict_phrases_ignores_committed_and_dropped(phrase_state):
    _write_index("ph-aaaaaaaaaa", "AAA")
    rec = _run_record([
        {"flagged": [{"phrase_id": "ph-aaaaaaaaaa", "prompt_name": "worker_full"}]},
        {"committed": [{"phrase_id": "ph-zzzzzzzzzz"}], "dropped": [{"phrase_id": "ph-yyyyyyyyyy"}]},
    ])
    ranked = meta_dreamer.top_conflict_phrases(rec, top_k=5)
    ids = {r["phrase_id"] for r in ranked}
    assert ids == {"ph-aaaaaaaaaa"}


# ── build_meta_briefing ──────────────────────────────────────────────────────

def test_briefing_empty_input_says_no_action():
    msg = meta_dreamer.build_meta_briefing([])
    assert "no conflict-heavy phrases" in msg
    assert "<|end|>" in msg


def test_briefing_names_each_phrase_with_details():
    top = [
        {"phrase_id": "ph-aaa", "count": 4, "prompt_name": "worker_full",
         "last_status": "possible_loop", "section_path": "Behavior / Hard",
         "current_text": "avoid writing code when asked for plans"},
        {"phrase_id": "ph-bbb", "count": 2, "prompt_name": "dreamer",
         "last_status": "possible_conflict", "section_path": "Rules",
         "current_text": "explain your reasoning"},
    ]
    msg = meta_dreamer.build_meta_briefing(top)
    for field in ("ph-aaa", "count=4", "worker_full", "possible_loop",
                  "Behavior / Hard", "avoid writing code",
                  "ph-bbb", "count=2", "dreamer", "possible_conflict",
                  "dream_submit", "dream_finalize"):
        assert field in msg, f"expected {field!r} in briefing, got:\n{msg}"


def test_briefing_truncates_overlong_current_text():
    long_txt = "x" * 900
    top = [{"phrase_id": "ph-z", "count": 1, "prompt_name": "worker_full",
            "last_status": "possible_conflict", "section_path": "Root",
            "current_text": long_txt}]
    msg = meta_dreamer.build_meta_briefing(top)
    # Truncated to ~400 + ellipsis — original 900-char string can't appear verbatim.
    assert long_txt not in msg
    assert "…" in msg


# ── run_meta_dreamer ─────────────────────────────────────────────────────────

def test_run_meta_dreamer_no_conflicts_returns_early(phrase_state):
    rec = _run_record([])
    out = asyncio.run(meta_dreamer.run_meta_dreamer(rec))
    assert out == {"status": "no_conflicts", "top_phrases": []}


def test_run_meta_dreamer_dispatches_to_run_agent_role(phrase_state, monkeypatch):
    _write_index("ph-aaaaaaaaaa", "canonical")
    captured: dict = {}

    async def fake_run_agent_role(role, body, session_id, *, prompts_dir=None, **_kwargs):
        captured["role"] = role
        captured["body"] = body
        captured["session_id"] = session_id
        captured["prompts_dir"] = prompts_dir
        return {"choices": [{"message": {"content": "meta done"}}]}

    # Patch at the import site — run_meta_dreamer does a lazy
    # `from app.entrypoints import run_agent_role`, so we patch that module.
    import app.entrypoints as entrypoints_mod
    monkeypatch.setattr(entrypoints_mod, "run_agent_role", fake_run_agent_role)

    rec = _run_record([
        {"flagged": [{"phrase_id": "ph-aaaaaaaaaa", "status": "possible_loop",
                      "prompt_name": "worker_full"}]},
    ])
    out = asyncio.run(meta_dreamer.run_meta_dreamer(rec, top_k=1))
    assert out["status"] == "ok"
    assert captured["role"] == "dreamer"
    assert captured["session_id"] == out["session_id"]
    assert captured["session_id"].startswith(meta_dreamer.META_SESSION_PREFIX)
    msgs = captured["body"]["messages"]
    assert len(msgs) == 1
    assert msgs[0]["role"] == "user"
    assert "ph-aaaaaaaaaa" in msgs[0]["content"]
    assert captured["body"]["_source_trigger"] == {"type": "cron", "ref": "meta_dreamer"}


def test_run_meta_dreamer_surfaces_dispatch_error(phrase_state, monkeypatch):
    _write_index("ph-aaaaaaaaaa", "txt")

    async def boom(role, body, session_id, *, prompts_dir=None, **_kwargs):
        raise RuntimeError("api down")

    import app.entrypoints as entrypoints_mod
    monkeypatch.setattr(entrypoints_mod, "run_agent_role", boom)

    rec = _run_record([
        {"flagged": [{"phrase_id": "ph-aaaaaaaaaa", "prompt_name": "worker_full"}]},
    ])
    out = asyncio.run(meta_dreamer.run_meta_dreamer(rec, top_k=1))
    assert out["status"] == "error"
    assert "api down" in out["error"]
    assert "RuntimeError" in out["error"]


def test_run_meta_dreamer_uses_custom_session_id(phrase_state, monkeypatch):
    _write_index("ph-aaaaaaaaaa", "txt")

    async def ok(role, body, session_id, *, prompts_dir=None, **_kwargs):
        return {"choices": [{"message": {"content": "ok"}}]}

    import app.entrypoints as entrypoints_mod
    monkeypatch.setattr(entrypoints_mod, "run_agent_role", ok)

    rec = _run_record([
        {"flagged": [{"phrase_id": "ph-aaaaaaaaaa", "prompt_name": "worker_full"}]},
    ])
    out = asyncio.run(meta_dreamer.run_meta_dreamer(rec, session_id="fixed_sid"))
    assert out["session_id"] == "fixed_sid"
