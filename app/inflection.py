"""Inflection-nudging detection (B3).

Until this module landed, every `from app.inflection import …` in
`app/worker.py` silently failed (the import was wrapped in try/except).
Inflection nudging has been off the entire time despite the config keys
existing. This module restores the three intended detectors:

  * `detect_inflection_points`  — token-level entropy + top-2 logprob gap
  * `detect_linguistic_markers` — hedge-word counts on the response text
  * `detect_uqlm_whitebox`      — UQLM MinTokenProbability +
                                  LengthNormalizedProbability on the same
                                  logprobs payload (no extra LLM call)

All three return data the worker can use to inject a "you sound uncertain
here, double-check" nudge before the next iteration.

Configured by `cfg.inflection.engine`:
  off            → no nudging (default; same behavior as broken-import era)
  logprobs       → logprobs detector only
  linguistic     → linguistic markers only
  both           → logprobs + linguistic combined
  uqlm_whitebox  → UQLM whitebox scorer (logprobs-based)
"""

from __future__ import annotations

import math
import re
from typing import Any


# ── Logprobs-based detection ─────────────────────────────────────────────────

def detect_inflection_points(
    content: str,
    logprobs_data: list[dict],
    entropy_threshold: float = 1.5,
    logprob_gap_threshold: float = 0.5,
) -> list[dict]:
    """Scan per-token logprobs for points where the model was uncertain.

    A "point" fires when EITHER:
      * Shannon entropy of top-K tokens at that position ≥ entropy_threshold, OR
      * top-1/top-2 logprob gap ≤ logprob_gap_threshold

    Returns a list of {position, token, entropy, gap, alternatives} dicts —
    empty if no inflection. Caller passes the list to `format_inflection_nudge`.
    """
    points: list[dict] = []
    for i, tok_data in enumerate(logprobs_data or []):
        # Tolerate both shapes: {"token": str, "top_logprobs": [{token, logprob}, ...]}
        # and OpenAI-style {token, logprob, top_logprobs: [...]}.
        token = tok_data.get("token") or tok_data.get("text") or ""
        top = tok_data.get("top_logprobs") or []
        if len(top) < 2:
            continue
        # Compute probabilities from logprobs
        probs = []
        for opt in top:
            lp = opt.get("logprob") if isinstance(opt, dict) else None
            if lp is None:
                continue
            probs.append(math.exp(lp))
        if len(probs) < 2:
            continue
        # Normalize over the top-K (so entropy is bounded by log(K))
        s = sum(probs)
        if s <= 0:
            continue
        normed = [p / s for p in probs]
        entropy = -sum(p * math.log(p) for p in normed if p > 0)
        # Top-2 gap on raw logprobs (the model's actual uncertainty)
        sorted_lps = sorted(
            (opt.get("logprob") for opt in top if isinstance(opt, dict) and opt.get("logprob") is not None),
            reverse=True,
        )
        gap = abs(sorted_lps[0] - sorted_lps[1]) if len(sorted_lps) >= 2 else 0.0
        if entropy >= entropy_threshold or gap <= logprob_gap_threshold:
            points.append({
                "position":     i,
                "token":        token,
                "entropy":      round(entropy, 3),
                "gap":          round(gap, 3),
                "alternatives": [opt.get("token") or opt.get("text") or "" for opt in top[:3]],
            })
    return points


def format_inflection_nudge(inflections: list[dict]) -> str:
    """Render an inflection-points list as a worker nudge string."""
    if not inflections:
        return ""
    n = len(inflections)
    sample = inflections[0]
    alts = ", ".join(repr(a) for a in (sample.get("alternatives") or [])[:3])
    return (
        f"\n[inflection_nudge] You sounded uncertain in {n} place(s) "
        f"(e.g. token {sample.get('token')!r} at pos {sample.get('position')}, "
        f"entropy={sample.get('entropy')}, alternatives=[{alts}]). "
        f"Double-check those decisions before continuing."
    )


# ── Linguistic-marker detection ──────────────────────────────────────────────

# "Strong" markers indicate explicit hedging that the model literally signals.
# Collected from common LLM hedging vocab; not exhaustive but covers most.
_STRONG_MARKERS = (
    "i'm not sure", "i am not sure", "i don't know", "i do not know",
    "i'm uncertain", "uncertain about", "let me check", "let me verify",
    "double-check", "double check", "i could be wrong", "may be wrong",
    "not entirely sure", "i think this might",
)

# "Weak" markers: hedging vocabulary that's noisy when alone but signals
# uncertainty in clusters. Counted; if frequency exceeds weak_threshold,
# fire a nudge.
_WEAK_MARKER_RE = re.compile(
    r"\b(?:maybe|perhaps|possibly|probably|likely|might|could|seems?|appears?|"
    r"i\s+think|i\s+believe|i\s+suspect|i\s+guess|sort\s+of|kind\s+of|"
    r"approximately|roughly|around|about(?=\s+\d))\b",
    re.IGNORECASE,
)


def detect_linguistic_markers(
    content: str,
    strong_threshold: int = 1,
    weak_threshold: int = 3,
) -> tuple[list[str], bool]:
    """Count hedging markers in the response text.

    Returns (signals, should_nudge). `signals` lists the strong markers
    encountered (sample for the nudge text). `should_nudge` is True iff
    strong_count >= strong_threshold OR weak_count >= weak_threshold.
    """
    if not content:
        return [], False
    lowered = content.lower()
    strong_hits = [m for m in _STRONG_MARKERS if m in lowered]
    weak_hits = _WEAK_MARKER_RE.findall(content)
    should_nudge = (len(strong_hits) >= strong_threshold) or (len(weak_hits) >= weak_threshold)
    return strong_hits or [m.group(0) for m in _WEAK_MARKER_RE.finditer(content)][:3], should_nudge


def format_linguistic_nudge(signals: list[str]) -> str:
    """Render a linguistic-markers list as a worker nudge string."""
    if not signals:
        return ""
    sample = ", ".join(repr(s) for s in signals[:3])
    return (
        f"\n[linguistic_nudge] Your response contained hedging signals ({sample}). "
        f"If you're uncertain, run a tool to verify before answering."
    )


# ── UQLM whitebox detection ──────────────────────────────────────────────────

def detect_uqlm_whitebox(
    content: str,
    logprobs_data: list[dict],
    cfg: dict | None = None,
) -> tuple[bool, dict]:
    """UQLM-style whitebox confidence scoring on the existing logprobs payload.

    Computes two scorers from the cvs-health/uqlm paper:
      * MinTokenProbability      — min(prob across all tokens)
      * LengthNormalizedProbability — geometric mean of token probabilities,
                                      i.e. exp(mean_logprob)

    Both are CHEAP — no extra LLM call, just math on data we already have.
    Returns (should_nudge, scores_dict). `should_nudge` fires when the
    length-normalized probability falls below `cfg.inflection.uqlm_min_norm_prob`
    (default 0.4 — calibrated as a starting point; A/B against logged sessions
    to tune).
    """
    cfg = cfg or {}
    inf_cfg = (cfg.get("inflection") or {}) if isinstance(cfg, dict) else {}
    threshold = float(inf_cfg.get("uqlm_min_norm_prob", 0.4))

    if not logprobs_data:
        return False, {"min_prob": None, "length_normalized_prob": None}

    chosen_logprobs: list[float] = []
    for tok_data in logprobs_data:
        # Prefer the per-token "logprob" field (the actually-sampled token).
        # Fall back to the highest-ranked top_logprobs entry.
        lp = tok_data.get("logprob")
        if lp is None and tok_data.get("top_logprobs"):
            top = tok_data["top_logprobs"][0]
            lp = top.get("logprob") if isinstance(top, dict) else None
        if lp is not None:
            chosen_logprobs.append(float(lp))

    if not chosen_logprobs:
        return False, {"min_prob": None, "length_normalized_prob": None}

    probs = [math.exp(lp) for lp in chosen_logprobs]
    min_prob = min(probs)
    norm_prob = math.exp(sum(chosen_logprobs) / len(chosen_logprobs))

    scores = {
        "min_prob":               round(min_prob, 4),
        "length_normalized_prob": round(norm_prob, 4),
        "threshold":              threshold,
        "n_tokens":               len(chosen_logprobs),
    }
    should_nudge = norm_prob < threshold
    return should_nudge, scores


def format_uqlm_nudge(scores: dict) -> str:
    """Render a UQLM whitebox score dict as a worker nudge string."""
    if not scores or scores.get("length_normalized_prob") is None:
        return ""
    return (
        f"\n[uqlm_nudge] Confidence is low (length-normalized probability = "
        f"{scores['length_normalized_prob']}, threshold = {scores['threshold']}). "
        f"Min single-token probability was {scores['min_prob']} across "
        f"{scores['n_tokens']} tokens. Verify before continuing."
    )


# ── Resolver: which engine to actually use ───────────────────────────────────

def resolve_engine(cfg: dict | None) -> str:
    """Return the active engine name from cfg, with back-compat for the
    legacy `inflection_mode` field. New `engine` field wins when both
    are present and not 'off'."""
    cfg = cfg or {}
    inf = (cfg.get("inflection") or {}) if isinstance(cfg, dict) else {}
    engine = inf.get("engine")
    if engine and engine != "off":
        return engine
    legacy = inf.get("inflection_mode") or cfg.get("agent", {}).get("inflection_mode")
    if legacy and legacy != "none":
        return legacy
    return engine or "off"
