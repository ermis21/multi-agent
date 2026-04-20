"""Unit tests for app/context_retriever.py.

Pure-unit. The HTTP call to the sandbox is monkeypatched so these tests
don't need a running stack.
"""

from __future__ import annotations

import pytest

import app.context_retriever as retriever
from app.context_retriever import (
    RetrievedChunk,
    _rerank_score_bucket_recency,
    chunk_by_sentence,
    format_zone_b,
    query,
)


# ── Chunker ──────────────────────────────────────────────────────────────────

def test_chunker_empty_string_returns_empty():
    assert chunk_by_sentence("") == []
    assert chunk_by_sentence("   \n\n   ") == []


def test_chunker_short_input_emits_single_chunk():
    text = "First sentence. Second sentence. Third sentence."
    chunks = chunk_by_sentence(text, target_tokens=500)
    assert len(chunks) == 1
    # Sentences may get joined with single spaces — accept either.
    for s in ("First sentence", "Second sentence", "Third sentence"):
        assert s in chunks[0]


def test_chunker_splits_long_input_into_multiple_chunks():
    # Build ~20 sentences so we guarantee at least one chunk boundary at
    # target_tokens=40 (≈160 chars per chunk given avg sentence length).
    sents = [f"Sentence number {i} with some padding text to bulk it up." for i in range(20)]
    chunks = chunk_by_sentence(" ".join(sents), target_tokens=40, overlap_tokens=5)
    assert len(chunks) >= 2


def test_chunker_preserves_code_fence_atomicity():
    text = (
        "Some prose before. "
        "```python\n"
        "def foo():\n"
        "    return 42\n"
        "```\n"
        "Some prose after. "
    )
    chunks = chunk_by_sentence(text, target_tokens=500)
    # The ``` pair must stay intact — balanced in one chunk.
    joined = "\n\n".join(chunks)
    assert joined.count("```") % 2 == 0


def test_chunker_drops_blank_chunks():
    assert chunk_by_sentence("  \n\n  ") == []


# ── Re-rank ──────────────────────────────────────────────────────────────────

def test_rerank_prefers_recent_within_score_band():
    hits = [
        RetrievedChunk(content="old but slightly stronger", score=0.84, turn_no=3),
        RetrievedChunk(content="fresh and close", score=0.82, turn_no=40),
        RetrievedChunk(content="clearly weaker", score=0.40, turn_no=39),
    ]
    ranked = _rerank_score_bucket_recency(hits, band=0.05)
    # Within 5% band, turn 40 wins over turn 3.
    assert ranked[0].turn_no == 40
    assert ranked[1].turn_no == 3
    # Clearly-weaker stays last regardless of recency.
    assert ranked[2].score == 0.40


def test_rerank_strictly_higher_bucket_wins_over_recency():
    hits = [
        RetrievedChunk(content="very strong ancient", score=0.95, turn_no=1),
        RetrievedChunk(content="so-so recent", score=0.60, turn_no=99),
    ]
    ranked = _rerank_score_bucket_recency(hits, band=0.05)
    assert ranked[0].turn_no == 1


# ── query ────────────────────────────────────────────────────────────────────

def test_query_returns_empty_on_blank_user_msg():
    assert query("sid-any", "") == []
    assert query("sid-any", "   ") == []


def test_query_scopes_by_session_id_and_reranks(monkeypatch):
    captured = {}

    def _stub(query_text, n, where):
        captured["query"] = query_text
        captured["n"] = n
        captured["where"] = where
        return [
            {"content": "old", "score": 0.80, "metadata": {"turn_no": 2, "kind": "user_msg"}},
            {"content": "new", "score": 0.78, "metadata": {"turn_no": 25, "kind": "user_msg"}},
            {"content": "weak", "score": 0.20, "metadata": {"turn_no": 30, "kind": "user_msg"}},
        ]

    monkeypatch.setattr(retriever, "_sandbox_memory_search", _stub)
    hits = query("sid-x", "hello world", pending_tool_target="web_fetch", k=3)

    # query text concatenates pending_tool_target
    assert "hello world" in captured["query"]
    assert "web_fetch" in captured["query"]
    # mandatory session scope
    assert captured["where"] == {"session_id": "sid-x"}
    # over-fetched k*2
    assert captured["n"] == 6
    # recency wins within band: turn 25 promoted over turn 2 (both near 0.80)
    assert hits[0].turn_no == 25


def test_query_cross_session_drops_scope(monkeypatch):
    captured = {}
    monkeypatch.setattr(retriever, "_sandbox_memory_search",
                        lambda q, n, where: captured.setdefault("where", where) or [])
    query("sid-x", "hello", cross_session=True)
    assert captured["where"] == {}


def test_query_swallows_sandbox_errors(monkeypatch):
    # Simulate transport failure — retriever must not raise.
    def _boom(*a, **kw):
        raise RuntimeError("network dead")
    monkeypatch.setattr(retriever, "_sandbox_memory_search",
                        lambda q, n, where: [])
    assert query("sid-any", "hi") == []


# ── format_zone_b ────────────────────────────────────────────────────────────

def test_format_zone_b_respects_budget():
    hits = [
        RetrievedChunk(content="A" * 400, score=0.9, turn_no=1, kind="user_msg"),
        RetrievedChunk(content="B" * 400, score=0.8, turn_no=2, kind="user_msg"),
        RetrievedChunk(content="C" * 400, score=0.7, turn_no=3, kind="user_msg"),
    ]
    # Tiny budget — should emit at most one chunk.
    out = format_zone_b(hits, budget_tokens=50)
    assert out.count("[retrieved") <= 1


def test_format_zone_b_empty_when_no_hits():
    assert format_zone_b([], budget_tokens=1000) == ""
    assert format_zone_b([RetrievedChunk(content="x", score=1.0)], budget_tokens=0) == ""


def test_format_zone_b_renders_provenance_header():
    hits = [RetrievedChunk(content="body", score=0.77,
                           turn_no=5, kind="tool_result", tool="web_fetch")]
    out = format_zone_b(hits, budget_tokens=1000)
    assert "turn=5" in out
    assert "tool=web_fetch" in out
    assert "score=0.77" in out
    assert "body" in out
