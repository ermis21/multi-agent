"""Pure-unit tests for app.dream.loop_guard.

Two triggers to exercise:
  - churn cap (`max_history`) — length alone flips the verdict
  - period detection (lag=2 / lag=3 over last `window` samples)

Sibling detection: two phrases oscillating in phase at the same lag both come
back `loop_suspected` and each sees the other in `sibling_phrase_ids`.
"""

from __future__ import annotations

from app.dream.loop_guard import check_loop, find_siblings


def _cfg(**overrides) -> dict:
    base = {
        "similarity_backend": "fuzzy",
        "similarity_threshold": 0.85,
        "max_history": 8,
        "period_detection_window": 6,
    }
    base.update(overrides)
    return {"dream": {"loop_guard": base}}


def _hist(*pairs: tuple[str, str]) -> list[dict]:
    """Build history rows from (old_text, new_text) pairs. Rev is 1-based."""
    return [
        {"rev": i + 1, "old_text": o, "new_text": n}
        for i, (o, n) in enumerate(pairs)
    ]


# ── Virgin phrase → never a loop ─────────────────────────────────────────────

def test_virgin_history_is_not_loop():
    v = check_loop("ph-x", "any new text", history=[], cfg=_cfg())
    assert v.loop_suspected is False


# ── Churn cap trigger ────────────────────────────────────────────────────────

def test_churn_cap_triggers_at_max_history():
    # 8 entries, random-looking content — would NOT trip period detection, but
    # the churn cap flips it regardless.
    hist = _hist(*[(f"v{i}", f"w{i}") for i in range(8)])
    v = check_loop("ph-x", "w7-proposed", history=hist, cfg=_cfg(max_history=8))
    assert v.loop_suspected is True
    assert v.period_lag is None
    assert "max_history" in v.reason


def test_churn_cap_does_not_trigger_below_threshold():
    hist = _hist(*[(f"v{i}", f"w{i}") for i in range(3)])
    v = check_loop("ph-x", "proposed", history=hist, cfg=_cfg(max_history=8))
    # 3 entries < cap AND no period echo → not a loop.
    assert v.loop_suspected is False


# ── Period detection (lag=2) ────────────────────────────────────────────────

def test_period_lag2_oscillation():
    # Timeline: original → A → B → A → B → A → B. Proposed "A again" = clear lag=2 echo.
    A = "alpha text body here"
    B = "bravo text body here"
    hist = _hist(("original", A), (A, B), (B, A), (A, B), (B, A))
    v = check_loop("ph-x", A, history=hist, cfg=_cfg(max_history=99))
    assert v.loop_suspected is True
    assert v.period_lag == 2
    assert v.similarity >= 0.85


def test_period_lag3_oscillation():
    A = "alpha text body here"
    B = "bravo text body here"
    C = "charlie text body here"
    hist = _hist(("original", A), (A, B), (B, C), (C, A), (A, B), (B, C))
    v = check_loop("ph-x", A, history=hist, cfg=_cfg(max_history=99))
    assert v.loop_suspected is True
    assert v.period_lag == 3


def test_unique_versions_not_a_loop():
    hist = _hist(
        ("v0", "completely different one"),
        ("completely different one", "another new phrasing entirely"),
        ("another new phrasing entirely", "yet a third fresh take"),
    )
    v = check_loop("ph-x", "a fourth distinct rewrite", history=hist, cfg=_cfg(max_history=99))
    assert v.loop_suspected is False


# ── Sibling detection ───────────────────────────────────────────────────────

def test_siblings_cooscillate_at_same_lag():
    A, B = "alpha body here", "bravo body here"
    C, D = "gamma body here", "delta body here"
    hist_x = _hist(("o", A), (A, B), (B, A), (A, B), (B, A))
    hist_y = _hist(("o", C), (C, D), (D, C), (C, D), (D, C))
    verdict_x = check_loop("ph-x", A, history=hist_x, cfg=_cfg(max_history=99))
    assert verdict_x.loop_suspected and verdict_x.period_lag == 2
    siblings = find_siblings(
        "ph-x", verdict_x,
        batch_histories={"ph-x": hist_x, "ph-y": hist_y},
        batch_candidates={"ph-x": A, "ph-y": C},
        cfg=_cfg(max_history=99),
    )
    assert "ph-y" in siblings
    assert "ph-x" not in siblings  # never self


def test_siblings_empty_when_no_cooscillation():
    A, B = "alpha body here", "bravo body here"
    hist_x = _hist(("o", A), (A, B), (B, A), (A, B), (B, A))
    # ph-y has a flat history — no oscillation.
    hist_y = _hist(("o", "fresh 1"), ("fresh 1", "fresh 2"))
    verdict_x = check_loop("ph-x", A, history=hist_x, cfg=_cfg(max_history=99))
    siblings = find_siblings(
        "ph-x", verdict_x,
        batch_histories={"ph-x": hist_x, "ph-y": hist_y},
        batch_candidates={"ph-x": A, "ph-y": "fresh 3"},
        cfg=_cfg(max_history=99),
    )
    assert siblings == []


def test_churn_cap_trigger_returns_no_siblings():
    """When the trigger is churn-cap (period_lag=None), siblings are meaningless."""
    hist = _hist(*[(f"v{i}", f"w{i}") for i in range(8)])
    v = check_loop("ph-x", "prop", history=hist, cfg=_cfg(max_history=8))
    assert v.loop_suspected is True and v.period_lag is None
    siblings = find_siblings(
        "ph-x", v,
        batch_histories={"ph-x": hist, "ph-y": hist},
        batch_candidates={"ph-x": "prop", "ph-y": "prop"},
        cfg=_cfg(max_history=8),
    )
    assert siblings == []


# ── Backend fallback ─────────────────────────────────────────────────────────

def test_unknown_backend_degrades_to_fuzzy():
    """An unrecognized similarity_backend shouldn't crash — fuzzy is the safe default."""
    A, B = "alpha body", "bravo body"
    hist = _hist(("o", A), (A, B), (B, A), (A, B), (B, A))
    cfg = _cfg(similarity_backend="embedding", max_history=99)
    v = check_loop("ph-x", A, history=hist, cfg=cfg)
    # Still detects the lag=2 loop.
    assert v.loop_suspected and v.period_lag == 2
