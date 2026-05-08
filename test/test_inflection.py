"""Unit tests for the rebuilt inflection module (B3).

The module didn't exist on disk before B3 — the import in worker.py was
silently failing under try/except. These tests pin the contract for all
three engines (logprobs / linguistic / uqlm_whitebox) plus the resolver.
"""

import math

from app import inflection


# ── Logprobs detector ───────────────────────────────────────────────────────

def _lp(token: str, alts: list[tuple[str, float]]) -> dict:
    """Build a logprobs entry in the OpenAI-compatible shape worker.py emits."""
    return {
        "token": token,
        "logprob": alts[0][1] if alts else 0.0,
        "top_logprobs": [{"token": t, "logprob": lp} for t, lp in alts],
    }


def test_inflection_quiet_when_high_confidence():
    # All probability concentrated on the chosen token — no inflection.
    payload = [_lp("hi", [("hi", math.log(0.99)), ("hello", math.log(0.005))])]
    assert inflection.detect_inflection_points("hi", payload) == []


def test_inflection_fires_on_high_entropy():
    # Three near-equally-likely tokens → high entropy on top-K, fires.
    payload = [_lp("a", [
        ("a", math.log(0.34)),
        ("b", math.log(0.33)),
        ("c", math.log(0.33)),
    ])]
    out = inflection.detect_inflection_points("a", payload, entropy_threshold=0.5)
    assert len(out) == 1
    assert out[0]["token"] == "a"
    assert out[0]["entropy"] > 0.5


def test_inflection_fires_on_small_gap():
    # Top-1 vs top-2 gap is tiny → fires regardless of entropy.
    payload = [_lp("a", [("a", math.log(0.51)), ("b", math.log(0.49))])]
    out = inflection.detect_inflection_points(
        "a", payload, entropy_threshold=10.0, logprob_gap_threshold=0.5,
    )
    assert len(out) == 1
    assert out[0]["gap"] < 0.5


def test_inflection_handles_empty_payload():
    assert inflection.detect_inflection_points("x", []) == []


def test_format_inflection_nudge_renders():
    out = inflection.format_inflection_nudge([{
        "position": 3, "token": "maybe", "entropy": 1.7, "gap": 0.1,
        "alternatives": ["maybe", "perhaps", "possibly"],
    }])
    assert "[inflection_nudge]" in out
    assert "maybe" in out


# ── Linguistic detector ─────────────────────────────────────────────────────

def test_linguistic_strong_marker_fires():
    signals, should_nudge = inflection.detect_linguistic_markers(
        "I'm not sure about this answer", strong_threshold=1, weak_threshold=99,
    )
    assert should_nudge
    assert signals  # contains the matched marker


def test_linguistic_weak_markers_only_when_above_threshold():
    text = "Maybe this could possibly be the right approach, perhaps"
    signals, should_nudge = inflection.detect_linguistic_markers(
        text, strong_threshold=99, weak_threshold=3,
    )
    assert should_nudge


def test_linguistic_quiet_on_assertive_text():
    _, should_nudge = inflection.detect_linguistic_markers(
        "The answer is 42. The script ran in 1.2 seconds.",
    )
    assert not should_nudge


def test_format_linguistic_nudge_renders():
    out = inflection.format_linguistic_nudge(["i'm not sure"])
    assert "[linguistic_nudge]" in out
    assert "i'm not sure" in out


# ── UQLM whitebox detector ──────────────────────────────────────────────────

def test_uqlm_quiet_on_high_confidence():
    # All tokens at logprob=-0.05 → norm_prob ≈ 0.95, well above default 0.4.
    payload = [_lp(t, [(t, -0.05)]) for t in ("the", "cat", "sat")]
    should_nudge, scores = inflection.detect_uqlm_whitebox("the cat sat", payload)
    assert not should_nudge
    assert scores["length_normalized_prob"] > 0.9


def test_uqlm_fires_on_low_confidence():
    # Tokens at logprob=-2.0 → norm_prob ≈ 0.135, below default 0.4.
    payload = [_lp(t, [(t, -2.0)]) for t in ("um", "maybe", "uh")]
    should_nudge, scores = inflection.detect_uqlm_whitebox("um maybe uh", payload)
    assert should_nudge
    assert scores["length_normalized_prob"] < 0.4


def test_uqlm_threshold_respects_config():
    # Same payload but a stricter threshold should flip the decision.
    payload = [_lp(t, [(t, -0.5)]) for t in ("ok", "fine")]
    cfg = {"inflection": {"uqlm_min_norm_prob": 0.7}}
    should_nudge, scores = inflection.detect_uqlm_whitebox("ok fine", payload, cfg)
    assert should_nudge
    assert scores["threshold"] == 0.7


def test_uqlm_handles_empty_payload():
    should_nudge, scores = inflection.detect_uqlm_whitebox("x", [])
    assert not should_nudge
    assert scores == {"min_prob": None, "length_normalized_prob": None}


def test_format_uqlm_nudge_renders():
    out = inflection.format_uqlm_nudge({
        "min_prob": 0.05, "length_normalized_prob": 0.2,
        "threshold": 0.4, "n_tokens": 12,
    })
    assert "[uqlm_nudge]" in out
    assert "0.2" in out


# ── Engine resolver ─────────────────────────────────────────────────────────

def test_resolve_engine_prefers_new_field():
    cfg = {"inflection": {"engine": "uqlm_whitebox", "inflection_mode": "logprobs"}}
    assert inflection.resolve_engine(cfg) == "uqlm_whitebox"


def test_resolve_engine_falls_back_to_legacy_field():
    cfg = {"inflection": {"engine": "off", "inflection_mode": "linguistic"}}
    assert inflection.resolve_engine(cfg) == "linguistic"


def test_resolve_engine_returns_off_when_unset():
    assert inflection.resolve_engine({}) == "off"
    assert inflection.resolve_engine({"inflection": {}}) == "off"


def test_resolve_engine_falls_back_to_agent_inflection_mode():
    """Legacy: inflection_mode used to live under cfg.agent.* — preserve."""
    cfg = {"inflection": {}, "agent": {"inflection_mode": "both"}}
    assert inflection.resolve_engine(cfg) == "both"
