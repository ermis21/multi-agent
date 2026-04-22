"""Counterfactual user-simulator primitives.

The dream simulator replays conversations under a candidate prompt. At each
turn the new agent response may diverge from the original. Replaying the
original user turn verbatim is wrong when that turn was written in reaction
to a now-different agent response. This module provides the machinery to:

  * compute dual similarity (lexical via rapidfuzz, semantic via chroma
    embeddings) between the old and new agent responses,
  * classify the divergence into a 5-level band,
  * build the briefing the `dream_user_simulator` role sees per turn,
  * sanitise that role's output (strip accidental tool calls, detect the
    `[[CF_ABORT:]]` sentinel, enforce single-message shape),
  * aggregate per-turn bands into a `fidelity` verdict consumed by
    `can_iterate` and the dreamer prompt.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from enum import Enum
from typing import Iterable

from rapidfuzz import fuzz


# ── Band + similarity types ─────────────────────────────────────────────────

class Band(str, Enum):
    IDENTICAL = "identical"
    MINOR = "minor_variation"
    SUBSTANTIAL = "substantial"
    DIVERGENT = "divergent"
    UNRELATED = "unrelated"


# Higher ordinal = more divergent. Used to pick the stricter (max) band when
# lex and sem disagree.
_BAND_ORDER = {
    Band.IDENTICAL: 0,
    Band.MINOR: 1,
    Band.SUBSTANTIAL: 2,
    Band.DIVERGENT: 3,
    Band.UNRELATED: 4,
}


@dataclass
class Similarity:
    lex: float        # 0..1 (rapidfuzz.token_set_ratio / 100)
    sem: float        # 0..1 (embedding cosine, clamped)
    lex_band: Band
    sem_band: Band
    band: Band        # stricter (max-ordinal) of lex_band and sem_band


# ── Defaults (mirrored in app/config_schema.DreamCounterfactualConfig) ──────

_DEFAULT_THRESHOLDS = {
    "identical_lex_min":   0.90,
    "identical_sem_min":   0.92,
    "minor_lex_min":       0.70,
    "minor_sem_min":       0.75,
    "substantial_lex_min": 0.40,
    "substantial_sem_min": 0.45,
    "divergent_lex_min":   0.15,
    "divergent_sem_min":   0.20,
}


def _cf_cfg(cfg: dict | None) -> dict:
    """Extract the counterfactual config subtree, with defaults for missing keys."""
    root = ((cfg or {}).get("dream") or {}).get("counterfactual") or {}
    return {**_DEFAULT_THRESHOLDS, **root}


# ── Embedding (semantic similarity) ─────────────────────────────────────────

_embed_fn = None  # lazy — chromadb import is heavy
_embed_cache: dict[str, list[float]] = {}
_EMBED_CACHE_MAX = 256


def _text_key(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="replace")).hexdigest()[:16]


def _get_embed_fn():
    global _embed_fn
    if _embed_fn is None:
        from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
        _embed_fn = DefaultEmbeddingFunction()
    return _embed_fn


def _embed(text: str) -> list[float]:
    key = _text_key(text)
    cached = _embed_cache.get(key)
    if cached is not None:
        return cached
    vecs = _get_embed_fn()([text])
    vec = list(vecs[0])
    if len(_embed_cache) >= _EMBED_CACHE_MAX:
        # drop an arbitrary entry; cheap FIFO approximation
        _embed_cache.pop(next(iter(_embed_cache)))
    _embed_cache[key] = vec
    return vec


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


# ── Similarity + band ───────────────────────────────────────────────────────

def _clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def _band_from(score: float, thresholds: dict, metric: str) -> Band:
    """Map a single score to a band using identical/minor/substantial/divergent
    thresholds for the given metric ("lex" or "sem"). Below divergent = unrelated."""
    if score >= thresholds[f"identical_{metric}_min"]:
        return Band.IDENTICAL
    if score >= thresholds[f"minor_{metric}_min"]:
        return Band.MINOR
    if score >= thresholds[f"substantial_{metric}_min"]:
        return Band.SUBSTANTIAL
    if score >= thresholds[f"divergent_{metric}_min"]:
        return Band.DIVERGENT
    return Band.UNRELATED


def classify_band(lex: float, sem: float, cfg: dict | None = None) -> tuple[Band, Band, Band]:
    """Return (lex_band, sem_band, band). `band` is the stricter (max-ordinal)."""
    th = _cf_cfg(cfg)
    lb = _band_from(lex, th, "lex")
    sb = _band_from(sem, th, "sem")
    band = lb if _BAND_ORDER[lb] >= _BAND_ORDER[sb] else sb
    return lb, sb, band


def compute_similarity(old: str, new: str, cfg: dict | None = None) -> Similarity:
    """Compute lex + sem similarity between two agent responses and classify.

    Fast-path: identical strings short-circuit to band=IDENTICAL without
    running the embedding model. Empty strings on either side yield
    band=UNRELATED (no signal).
    """
    if old == new:
        return Similarity(lex=1.0, sem=1.0, lex_band=Band.IDENTICAL,
                          sem_band=Band.IDENTICAL, band=Band.IDENTICAL)
    if not old or not new:
        return Similarity(lex=0.0, sem=0.0, lex_band=Band.UNRELATED,
                          sem_band=Band.UNRELATED, band=Band.UNRELATED)

    lex = _clamp01(fuzz.token_set_ratio(old, new) / 100.0)

    try:
        va = _embed(old)
        vb = _embed(new)
        sem = _clamp01(_cosine(va, vb))
    except Exception:
        # Embedding failure must not break the replay; fall back to lex-only
        # (semantic band matches lexical, so classification uses lex alone).
        sem = lex

    lb, sb, band = classify_band(lex, sem, cfg)
    return Similarity(lex=lex, sem=sem, lex_band=lb, sem_band=sb, band=band)


# ── Briefing builder ────────────────────────────────────────────────────────

_BAND_GUIDANCE = {
    Band.MINOR: (
        "minor_variation — keep the original turn almost verbatim; only touch "
        "wording that directly references something the new agent didn't say."
    ),
    Band.SUBSTANTIAL: (
        "substantial — react to what the NEW agent said. Preserve the user's "
        "underlying goal from the original turn, but do not echo content the "
        "new response didn't include."
    ),
    Band.DIVERGENT: (
        "divergent — the new response took a meaningfully different direction. "
        "Respond to it on its merits. Keep only the user's goal; everything "
        "else adapts."
    ),
}


def _format_replay_so_far(replay_so_far: list[dict], cap: int = 6) -> str:
    """Render the last `cap` interleaved messages as a plain transcript."""
    if not replay_so_far:
        return "(no prior simulated turns)"
    window = replay_so_far[-cap:]
    lines = []
    for m in window:
        role = m.get("role", "?")
        content = (m.get("content") or "").strip()
        if len(content) > 400:
            content = content[:400] + "…"
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def build_cf_briefing(
    *,
    original_user: str,
    old_agent: str,
    new_agent: str,
    goal: str,
    replay_so_far: list[dict],
    similarity: Similarity,
) -> str:
    """Return the single user-message string the dream_user_simulator sees.

    The briefing is a plain-text document with labeled sections. We do not
    wrap values in JSON — the user-sim produces a natural user reply, and a
    plain-text brief is less likely to bleed JSON scaffolding into the reply.
    """
    guidance = _BAND_GUIDANCE.get(
        similarity.band,
        "react to the new agent response while preserving the user's goal.",
    )
    return (
        "You are simulating a user whose real reply needs adjustment because the agent "
        "responded differently than in the original conversation.\n\n"
        f"# User's underlying goal\n{goal or '(unknown)'}\n\n"
        f"# Original user turn (what they actually said)\n{original_user}\n\n"
        f"# Old agent response (what the original user was reacting to)\n{old_agent}\n\n"
        f"# New agent response (what the new prompt produced)\n{new_agent}\n\n"
        f"# Similarity\n"
        f"lex={similarity.lex:.3f}  sem={similarity.sem:.3f}  band={similarity.band.value}\n"
        f"{guidance}\n\n"
        f"# Replay transcript so far\n{_format_replay_so_far(replay_so_far)}\n\n"
        "# Your output\n"
        "Write ONE user message as plain text. No preamble, no JSON, no tool calls, "
        "no meta-commentary. Match the original user's tone and length. If you "
        "cannot produce a plausible turn, output exactly:\n"
        "[[CF_ABORT: <one-line reason>]]"
    )


# ── Output sanitiser ────────────────────────────────────────────────────────

_TOOL_CALL_RE = re.compile(r"<\|tool_call\|>.*?<\|tool_call\|>", re.DOTALL)
_ABORT_RE = re.compile(r"\[\[CF_ABORT\s*:\s*([^\]]*)\]\]")
_LEADING_LABEL_RE = re.compile(r"^\s*(user|assistant)\s*:\s*", re.IGNORECASE)
_MAX_USER_TURN_CHARS = 4000


class SanitizeResult:
    __slots__ = ("text", "aborted", "abort_reason")

    def __init__(self, text: str | None, aborted: bool = False, abort_reason: str = ""):
        self.text = text
        self.aborted = aborted
        self.abort_reason = abort_reason


def sanitize_user_sim_output(raw: str) -> SanitizeResult:
    """Clean the user-sim's output.

    - Detects `[[CF_ABORT:<reason>]]` anywhere in the output and returns an
      aborted result (text=None).
    - Strips `<|tool_call|>...<|tool_call|>` spans.
    - Strips leading `user:` / `assistant:` role labels.
    - Caps length at _MAX_USER_TURN_CHARS.
    - Returns text=None with abort if the cleaned output is empty.
    """
    if raw is None:
        return SanitizeResult(None, aborted=True, abort_reason="empty_output")

    abort_m = _ABORT_RE.search(raw)
    if abort_m:
        reason = (abort_m.group(1) or "").strip() or "unspecified"
        return SanitizeResult(None, aborted=True, abort_reason=reason)

    cleaned = _TOOL_CALL_RE.sub("", raw)
    cleaned = _LEADING_LABEL_RE.sub("", cleaned).strip()

    if not cleaned:
        return SanitizeResult(None, aborted=True, abort_reason="empty_after_strip")

    if len(cleaned) > _MAX_USER_TURN_CHARS:
        cleaned = cleaned[:_MAX_USER_TURN_CHARS]

    return SanitizeResult(cleaned)


# ── Fidelity aggregation ────────────────────────────────────────────────────

def compute_fidelity(per_turn: Iterable[dict], cf_aborts: int) -> str:
    """Aggregate per-turn band records into a three-level fidelity verdict.

    Rules:
      - cf_aborts >= 2  → low (user-sim is failing to produce turns)
      - no divergent/unrelated turns AND no substantial turns → high
      - max band == substantial                              → moderate
      - max band in {divergent, unrelated}                   → low
    """
    if cf_aborts >= 2:
        return "low"
    turns = list(per_turn)
    if not turns:
        return "high"
    max_ord = -1
    for t in turns:
        try:
            b = Band(t["band"])
        except (KeyError, ValueError):
            continue
        max_ord = max(max_ord, _BAND_ORDER[b])
    if max_ord <= _BAND_ORDER[Band.MINOR]:
        return "high"
    if max_ord <= _BAND_ORDER[Band.SUBSTANTIAL]:
        return "moderate"
    return "low"


def summarize_metrics(per_turn: list[dict], cf_aborts: int, goal: str) -> dict:
    """Produce the `counterfactual` payload block for SimResult.to_payload."""
    if per_turn:
        lex_vals = [t.get("lex", 0.0) for t in per_turn]
        sem_vals = [t.get("sem", 0.0) for t in per_turn]
        avg_lex = sum(lex_vals) / len(lex_vals)
        avg_sem = sum(sem_vals) / len(sem_vals)
        turns_adjusted = sum(1 for t in per_turn if t.get("adjusted"))
        turns_verbatim = len(per_turn) - turns_adjusted
        max_band = max(
            (Band(t["band"]) for t in per_turn if t.get("band") in Band.__members__.values()
             or t.get("band") in {b.value for b in Band}),
            key=lambda b: _BAND_ORDER[b],
            default=Band.IDENTICAL,
        ).value
    else:
        avg_lex = avg_sem = 0.0
        turns_adjusted = turns_verbatim = 0
        max_band = Band.IDENTICAL.value

    return {
        "per_turn": per_turn,
        "avg_lex": avg_lex,
        "avg_sem": avg_sem,
        "turns_adjusted": turns_adjusted,
        "turns_verbatim": turns_verbatim,
        "max_band": max_band,
        "fidelity": compute_fidelity(per_turn, cf_aborts),
        "cf_aborts": cf_aborts,
        "goal": goal,
    }
