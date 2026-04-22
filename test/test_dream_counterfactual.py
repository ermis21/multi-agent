"""Unit tests for app.dream.counterfactual — similarity, banding, sanitizer,
fidelity aggregation, briefing builder. No LLM, no chroma calls (embedding
function is monkey-patched to a deterministic fake)."""

from __future__ import annotations

import math

import pytest

from app.dream import counterfactual as cf


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _reset_embed_cache():
    cf._embed_cache.clear()
    yield
    cf._embed_cache.clear()


class _FakeEmbed:
    """Deterministic text→vector mapping for similarity tests.

    Each unique character becomes a dimension; char count is the coordinate.
    Two strings with identical char-histograms get cosine=1.0.
    """

    def __init__(self, dim: int = 64):
        self.dim = dim

    def __call__(self, inputs):
        out = []
        for text in inputs:
            vec = [0.0] * self.dim
            for ch in text:
                vec[ord(ch) % self.dim] += 1.0
            out.append(vec)
        return out


@pytest.fixture
def fake_embed(monkeypatch):
    ef = _FakeEmbed()
    monkeypatch.setattr(cf, "_embed_fn", ef)
    return ef


def _cfg() -> dict:
    return {"dream": {"counterfactual": {}}}  # use defaults


# ── Similarity + band ───────────────────────────────────────────────────────

def test_identical_strings_short_circuit_to_band_identical(fake_embed):
    s = cf.compute_similarity("hello world", "hello world", _cfg())
    assert s.band == cf.Band.IDENTICAL
    assert s.lex == 1.0
    assert s.sem == 1.0


def test_empty_string_on_either_side_is_unrelated(fake_embed):
    s = cf.compute_similarity("", "anything", _cfg())
    assert s.band == cf.Band.UNRELATED
    s = cf.compute_similarity("anything", "", _cfg())
    assert s.band == cf.Band.UNRELATED


def test_near_identical_lex_and_sem_classifies_minor_or_identical(fake_embed):
    a = "The deployment succeeded. Five containers are healthy."
    b = "The deployment succeeded! Five containers are healthy."
    s = cf.compute_similarity(a, b, _cfg())
    # Near-identical text; should land identical or minor, definitely not worse
    assert s.band in (cf.Band.IDENTICAL, cf.Band.MINOR)


def test_unrelated_content_classifies_substantial_or_worse(fake_embed):
    a = "I committed the change to main and the tests pass."
    b = "Elephants are large mammals found in Africa and Asia."
    s = cf.compute_similarity(a, b, _cfg())
    # FakeEmbed is char-histogram so we don't expect full unrelated here,
    # but the band should definitely not be identical or minor.
    assert s.band not in (cf.Band.IDENTICAL, cf.Band.MINOR)


def test_band_uses_stricter_of_two_metrics():
    # High lex, low sem → band tracks sem (stricter)
    lb, sb, band = cf.classify_band(lex=0.95, sem=0.30, cfg=_cfg())
    assert lb == cf.Band.IDENTICAL
    assert sb == cf.Band.DIVERGENT
    assert band == cf.Band.DIVERGENT


def test_band_max_ordinal_picks_more_divergent_side():
    lb, sb, band = cf.classify_band(lex=0.10, sem=0.95, cfg=_cfg())
    assert lb == cf.Band.UNRELATED
    assert sb == cf.Band.IDENTICAL
    # stricter = more divergent = unrelated
    assert band == cf.Band.UNRELATED


def test_classify_band_uses_default_thresholds_when_cfg_missing():
    lb, sb, band = cf.classify_band(lex=0.95, sem=0.95, cfg=None)
    assert lb == cf.Band.IDENTICAL
    assert sb == cf.Band.IDENTICAL
    assert band == cf.Band.IDENTICAL


# ── Embedding cache ─────────────────────────────────────────────────────────

def test_embedding_cache_hits_on_same_text(fake_embed, monkeypatch):
    calls = {"n": 0}

    class _CountingEF:
        def __call__(self, inputs):
            calls["n"] += 1
            return [[float(len(t))] * 4 for t in inputs]

    monkeypatch.setattr(cf, "_embed_fn", _CountingEF())
    cf._embed_cache.clear()

    cf._embed("foo bar")
    cf._embed("foo bar")  # cache hit
    cf._embed("different text")
    assert calls["n"] == 2  # two distinct embeddings; second "foo bar" cached


# ── Cosine ──────────────────────────────────────────────────────────────────

def test_cosine_basic():
    assert cf._cosine([1.0, 0.0], [1.0, 0.0]) == 1.0
    assert cf._cosine([1.0, 0.0], [0.0, 1.0]) == 0.0
    assert abs(cf._cosine([1.0, 1.0], [1.0, 0.0]) - 1.0 / math.sqrt(2)) < 1e-6


def test_cosine_handles_zero_vectors():
    assert cf._cosine([0.0, 0.0], [1.0, 1.0]) == 0.0
    assert cf._cosine([], [1.0]) == 0.0


# ── Sanitizer ───────────────────────────────────────────────────────────────

def test_sanitize_strips_tool_call_markers():
    raw = "<|tool_call|>call: file_write, {\"path\":\"x\"}<|tool_call|>\nhello there"
    r = cf.sanitize_user_sim_output(raw)
    assert r.aborted is False
    assert "tool_call" not in r.text
    assert "hello there" in r.text


def test_sanitize_strips_multiple_tool_calls():
    raw = "<|tool_call|>a<|tool_call|> text <|tool_call|>b<|tool_call|> end"
    r = cf.sanitize_user_sim_output(raw)
    assert r.aborted is False
    assert "<|tool_call|>" not in r.text
    assert "text" in r.text and "end" in r.text


def test_sanitize_detects_cf_abort_sentinel():
    raw = "stuff before [[CF_ABORT: I cannot produce a plausible turn]] stuff after"
    r = cf.sanitize_user_sim_output(raw)
    assert r.aborted is True
    assert "cannot produce" in r.abort_reason


def test_sanitize_abort_without_reason_uses_unspecified():
    r = cf.sanitize_user_sim_output("[[CF_ABORT: ]]")
    assert r.aborted is True
    assert r.abort_reason == "unspecified"


def test_sanitize_strips_leading_role_labels():
    r = cf.sanitize_user_sim_output("user: can you do X?")
    assert r.aborted is False
    assert r.text == "can you do X?"


def test_sanitize_empty_output_returns_abort():
    r = cf.sanitize_user_sim_output("")
    assert r.aborted is True
    assert r.text is None


def test_sanitize_whitespace_only_returns_abort():
    r = cf.sanitize_user_sim_output("   \n\n  \t ")
    assert r.aborted is True


def test_sanitize_caps_length():
    r = cf.sanitize_user_sim_output("x" * 10000)
    assert r.aborted is False
    assert len(r.text) == cf._MAX_USER_TURN_CHARS


def test_sanitize_none_input_returns_abort():
    r = cf.sanitize_user_sim_output(None)  # type: ignore[arg-type]
    assert r.aborted is True


# ── Briefing ────────────────────────────────────────────────────────────────

def test_build_cf_briefing_contains_all_sections():
    sim = cf.Similarity(lex=0.5, sem=0.6, lex_band=cf.Band.SUBSTANTIAL,
                        sem_band=cf.Band.SUBSTANTIAL, band=cf.Band.SUBSTANTIAL)
    b = cf.build_cf_briefing(
        original_user="please run the tests",
        old_agent="ok, tests pass",
        new_agent="I can't find the test suite",
        goal="get CI green before Friday",
        replay_so_far=[{"role": "user", "content": "hi"},
                       {"role": "assistant", "content": "hello"}],
        similarity=sim,
    )
    assert "get CI green before Friday" in b
    assert "please run the tests" in b
    assert "ok, tests pass" in b
    assert "I can't find the test suite" in b
    assert "band=substantial" in b
    assert "CF_ABORT" in b  # abort sentinel instructions included


def test_build_cf_briefing_renders_replay_window():
    sim = cf.Similarity(lex=0.5, sem=0.6, lex_band=cf.Band.SUBSTANTIAL,
                        sem_band=cf.Band.SUBSTANTIAL, band=cf.Band.SUBSTANTIAL)
    many = [{"role": "user" if i % 2 == 0 else "assistant",
             "content": f"msg {i}"} for i in range(20)]
    b = cf.build_cf_briefing(
        original_user="x", old_agent="y", new_agent="z",
        goal="g", replay_so_far=many, similarity=sim,
    )
    assert "msg 19" in b   # tail of window
    assert "msg 0" not in b  # far tail truncated (window=6)


def test_build_cf_briefing_empty_replay_uses_placeholder():
    sim = cf.Similarity(lex=0.5, sem=0.6, lex_band=cf.Band.SUBSTANTIAL,
                        sem_band=cf.Band.SUBSTANTIAL, band=cf.Band.SUBSTANTIAL)
    b = cf.build_cf_briefing(
        original_user="x", old_agent="y", new_agent="z",
        goal="g", replay_so_far=[], similarity=sim,
    )
    assert "no prior simulated turns" in b


# ── Fidelity ────────────────────────────────────────────────────────────────

def test_fidelity_high_when_only_identical_or_minor():
    per_turn = [
        {"band": "identical", "adjusted": False, "lex": 1.0, "sem": 1.0},
        {"band": "minor_variation", "adjusted": True, "lex": 0.8, "sem": 0.85},
    ]
    assert cf.compute_fidelity(per_turn, cf_aborts=0) == "high"


def test_fidelity_moderate_when_substantial_present():
    per_turn = [
        {"band": "identical", "adjusted": False, "lex": 1.0, "sem": 1.0},
        {"band": "substantial", "adjusted": True, "lex": 0.5, "sem": 0.55},
    ]
    assert cf.compute_fidelity(per_turn, cf_aborts=0) == "moderate"


def test_fidelity_low_on_divergent():
    per_turn = [{"band": "divergent", "adjusted": True, "lex": 0.2, "sem": 0.3}]
    assert cf.compute_fidelity(per_turn, cf_aborts=0) == "low"


def test_fidelity_low_on_unrelated():
    per_turn = [{"band": "unrelated", "adjusted": True, "lex": 0.05, "sem": 0.1}]
    assert cf.compute_fidelity(per_turn, cf_aborts=0) == "low"


def test_fidelity_low_on_two_or_more_aborts():
    per_turn = [{"band": "identical", "adjusted": False, "lex": 1.0, "sem": 1.0}]
    assert cf.compute_fidelity(per_turn, cf_aborts=2) == "low"


def test_fidelity_high_on_empty_turns():
    assert cf.compute_fidelity([], cf_aborts=0) == "high"


# ── summarize_metrics ──────────────────────────────────────────────────────

def test_summarize_metrics_produces_expected_shape():
    per_turn = [
        {"band": "identical", "adjusted": False, "lex": 1.0, "sem": 1.0, "i": 0},
        {"band": "substantial", "adjusted": True, "lex": 0.5, "sem": 0.5, "i": 1},
        {"band": "minor_variation", "adjusted": True, "lex": 0.8, "sem": 0.8, "i": 2},
    ]
    out = cf.summarize_metrics(per_turn, cf_aborts=0, goal="ship X")
    assert out["turns_adjusted"] == 2
    assert out["turns_verbatim"] == 1
    assert out["max_band"] == "substantial"
    assert abs(out["avg_lex"] - (1.0 + 0.5 + 0.8) / 3) < 1e-9
    assert out["fidelity"] == "moderate"
    assert out["goal"] == "ship X"
    assert out["per_turn"] == per_turn
