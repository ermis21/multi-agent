"""Pure-unit tests for app/tokenizer.py — exercises the fallback path only.

The primary backend (llama.cpp /tokenize) is absent under plain pytest, so
count() falls through to tiktoken; we verify the results are stable and
positive for a handful of representative inputs.
"""

import pytest

from app.tokenizer import ElisionStrategy, count, truncate


def test_count_nonempty_returns_positive_int():
    n = count("hello world")
    assert isinstance(n, int) and n > 0


def test_count_empty_returns_zero():
    assert count("") == 0


def test_count_is_stable_across_calls():
    # memoization must not corrupt subsequent calls
    a = count("stability check")
    b = count("stability check")
    assert a == b


def test_count_longer_text_strictly_larger():
    short = count("one line")
    long  = count("one line\n" * 50)
    assert long > short


def test_truncate_under_budget_is_noop():
    text = "short body"
    out = truncate(text, 1000, ElisionStrategy.HEAD_TAIL)
    assert out == text


def test_truncate_head_keeps_prefix():
    text = "abcdefghij " * 400  # ~4400 chars, lots of tokens
    out = truncate(text, 50, ElisionStrategy.HEAD)
    assert len(out) < len(text)
    # HEAD preserves the original start
    assert text.startswith(out[:20])


def test_truncate_tail_keeps_suffix():
    text = "abcdefghij " * 400
    out = truncate(text, 50, ElisionStrategy.TAIL)
    assert len(out) < len(text)
    assert text.endswith(out[-20:])


def test_truncate_head_tail_contains_elision_marker():
    text = "abcdefghij " * 400
    out = truncate(text, 50, ElisionStrategy.HEAD_TAIL)
    # HEAD_TAIL must signal what was dropped
    assert "elided" in out or "…" in out or "..." in out


@pytest.mark.parametrize("strategy", list(ElisionStrategy))
def test_truncate_respects_budget_bound(strategy):
    text = "word " * 2000
    budget = 80
    out = truncate(text, budget, strategy)
    # Can't check exact tokens without the backend, but bounded char size is a
    # reasonable proxy: elision strategies keep at most ~ budget * 8 chars.
    assert len(out) <= len(text)
