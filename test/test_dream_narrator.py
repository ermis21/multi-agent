"""Pure-unit tests for app.dream.narrator.

Stubs the LLM seam with a counting echo so we can assert:
  - prompt construction carries phrase_id, section_path, history excerpt,
    proposed text, and (for loop) sibling ids + period
  - NarratorCache returns cached text without re-calling the LLM
"""

from __future__ import annotations

import asyncio

import pytest

from app.dream import narrator


@pytest.fixture
def stub_llm(monkeypatch):
    """Replace narrator._llm_call with a deterministic recorder."""
    calls: list[tuple[str, dict]] = []

    async def fake(prompt: str, cfg: dict) -> str:
        calls.append((prompt, cfg))
        return f"STUB[{len(calls)}]"

    narrator._set_llm_call(fake)
    try:
        yield calls
    finally:
        narrator._set_llm_call(narrator._default_llm_call)


def _run(coro):
    return asyncio.run(coro)


# ── Conflict narrative ──────────────────────────────────────────────────────

def test_conflict_prompt_carries_inputs(stub_llm):
    text = _run(narrator.narrate_conflict(
        phrase_id="ph-abc",
        section_path="Root / Rules",
        history_excerpt=[
            {"rev": 4, "new_text": "second newest version", "rationale": "r4"},
            {"rev": 5, "new_text": "newest version", "rationale": "r5"},
        ],
        new_text="proposed replacement body",
        cfg={},
    ))
    assert text == "STUB[1]"
    assert len(stub_llm) == 1
    prompt = stub_llm[0][0]
    assert "ph-abc" in prompt
    assert "Root / Rules" in prompt
    assert "second newest version" in prompt
    assert "newest version" in prompt
    assert "proposed replacement body" in prompt
    # No recommend-verb.
    assert "accept" not in prompt.lower() or "Do NOT recommend" in prompt


def test_conflict_cache_hits_return_without_llm_call(stub_llm):
    cache = narrator.NarratorCache()
    first = _run(narrator.narrate_conflict(
        phrase_id="ph-abc", section_path="S",
        history_excerpt=[{"rev": 1, "new_text": "x"}],
        new_text="y", cfg={}, cache=cache,
    ))
    second = _run(narrator.narrate_conflict(
        phrase_id="ph-abc", section_path="S",
        history_excerpt=[{"rev": 1, "new_text": "x"}],
        new_text="y", cfg={}, cache=cache,
    ))
    assert first == second
    assert len(stub_llm) == 1  # second call served from cache


# ── Loop narrative ──────────────────────────────────────────────────────────

def test_loop_prompt_names_siblings_and_period(stub_llm):
    _run(narrator.narrate_loop(
        phrase_id="ph-x",
        section_path="Root / Rules",
        history_excerpt=[
            {"rev": 1, "new_text": "v1 body"},
            {"rev": 5, "new_text": "v5 body"},
        ],
        sibling_phrase_ids=["ph-y", "ph-z"],
        period_lag=2,
        new_text="repeat of v1 body",
        cfg={},
    ))
    prompt = stub_llm[0][0]
    assert "ph-x" in prompt
    assert "ph-y" in prompt and "ph-z" in prompt
    assert "lag=2" in prompt
    assert "v1 body" in prompt and "v5 body" in prompt


def test_loop_prompt_without_siblings_says_so(stub_llm):
    _run(narrator.narrate_loop(
        phrase_id="ph-x", section_path="",
        history_excerpt=[],
        sibling_phrase_ids=[], period_lag=None,
        new_text="proposal", cfg={},
    ))
    prompt = stub_llm[0][0]
    assert "No co-oscillating sibling phrases" in prompt
    # period_lag=None → narrator falls back to the churn-cap framing.
    assert "churn cap" in prompt.lower() or "rewritten many times" in prompt


def test_loop_cache_shared_with_conflict_is_independent(stub_llm):
    """Same phrase_id stored under kind='conflict' and kind='loop' → two entries."""
    cache = narrator.NarratorCache()
    _run(narrator.narrate_conflict(
        phrase_id="ph-abc", section_path="S",
        history_excerpt=[{"rev": 1, "new_text": "x"}],
        new_text="y", cfg={}, cache=cache,
    ))
    _run(narrator.narrate_loop(
        phrase_id="ph-abc", section_path="S",
        history_excerpt=[{"rev": 1, "new_text": "x"}],
        sibling_phrase_ids=[], period_lag=2,
        new_text="y", cfg={}, cache=cache,
    ))
    assert len(stub_llm) == 2  # both calls hit the LLM (different kinds)
    assert len(cache.entries) == 2
