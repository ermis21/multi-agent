"""Detect edit loops across a phrase's history.

Two independent triggers:
  1. **Churn cap** — a phrase rewritten more than `max_history` times (default 8)
     is suspicious by sheer count alone, regardless of content.
  2. **Period detection** — the proposed new text echoes a version that lived
     `lag` revisions back (defaults: {2, 3}) at fuzzy similarity ≥ `threshold`
     (default 0.85) across a trailing window of `period_detection_window`
     samples (default 6). This catches the "A→B→A→B" oscillation pattern.

Only the **fuzzy** backend is implemented here (difflib.SequenceMatcher.ratio).
The plan reserves an `embedding` backend (ChromaDB collection
`dream_phrase_history`) but that lives in a future iteration — config honors
`similarity_backend: "fuzzy"` and rejects anything else for now.

Sibling phrases are other phrase_ids in the SAME pending batch whose history
trips period-detection at the SAME lag in phase with this phrase. The narrator
uses the sibling list to tell the dreamer "you're oscillating these ids
together."
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass, field


@dataclass
class LoopVerdict:
    loop_suspected: bool
    period_lag: int | None = None
    similarity: float = 0.0
    sibling_phrase_ids: list[str] = field(default_factory=list)
    reason: str = ""


def _ratio(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(a=a, b=b, autojunk=False).ratio()


def _detect_period(versions: list[str], lag: int, threshold: float, window: int) -> tuple[bool, float]:
    """Compare each v_i against v_{i-lag} across the trailing `window` samples.

    Returns `(suspected, avg_similarity)`. Suspected iff avg ≥ threshold AND
    at least one sample pair was comparable (avoids tripping on short history).
    """
    if lag <= 0 or len(versions) <= lag:
        return (False, 0.0)
    tail = versions[-window:] if window > 0 else versions
    if len(tail) <= lag:
        return (False, 0.0)
    ratios: list[float] = []
    for i in range(lag, len(tail)):
        ratios.append(_ratio(tail[i], tail[i - lag]))
    if not ratios:
        return (False, 0.0)
    avg = sum(ratios) / len(ratios)
    return (avg >= threshold, avg)


def _history_text_series(history: list[dict], candidate_new_text: str) -> list[str]:
    """Project the phrase's on-disk timeline: [v1=old_text_of_entry0, v2=new_text_of_entry0,
    v3=new_text_of_entry1, ..., v_{N+1}=new_text_of_entry_{N-1}, v_{N+2}=candidate].

    The first row's `old_text` is the ORIGINAL phrasing (before any dream edit).
    Each subsequent row's `new_text` is the state after that edit. We append the
    candidate so period detection can see whether the dreamer is re-proposing a
    prior shape.
    """
    if not history:
        return [candidate_new_text]
    series: list[str] = [history[0].get("old_text", "")]
    for row in history:
        series.append(row.get("new_text", ""))
    series.append(candidate_new_text)
    return [s for s in series if s]


def check_loop(
    phrase_id: str,
    new_text: str,
    history: list[dict],
    cfg: dict,
) -> LoopVerdict:
    """Single-phrase loop check. `cfg` is the full phoebe config dict.

    The caller (`dream_submit`) invokes `check_loop` per phrase, then enriches
    each flagged verdict with sibling ids via `find_siblings` below.
    """
    lg_cfg = (cfg or {}).get("dream", {}).get("loop_guard", {}) or {}
    backend = lg_cfg.get("similarity_backend", "fuzzy")
    if backend != "fuzzy":
        # Embedding backend not yet implemented — degrade to fuzzy rather than fail.
        backend = "fuzzy"
    max_history = int(lg_cfg.get("max_history", 8))
    threshold = float(lg_cfg.get("similarity_threshold", 0.85))
    window = int(lg_cfg.get("period_detection_window", 6))

    if not history:
        return LoopVerdict(loop_suspected=False)

    # Trigger 1: churn cap.
    if len(history) >= max_history:
        return LoopVerdict(
            loop_suspected=True,
            period_lag=None,
            similarity=0.0,
            reason=f"history length {len(history)} ≥ max_history={max_history}",
        )

    # Trigger 2: period detection.
    series = _history_text_series(history, new_text)
    best_lag: int | None = None
    best_sim = 0.0
    for lag in (2, 3):
        suspected, sim = _detect_period(series, lag, threshold, window)
        if suspected and sim > best_sim:
            best_lag, best_sim = lag, sim
    if best_lag is not None:
        return LoopVerdict(
            loop_suspected=True,
            period_lag=best_lag,
            similarity=round(best_sim, 4),
            reason=f"period lag={best_lag} avg similarity={best_sim:.3f}",
        )

    return LoopVerdict(loop_suspected=False)


def find_siblings(
    phrase_id: str,
    verdict: LoopVerdict,
    batch_histories: dict[str, list[dict]],
    batch_candidates: dict[str, str],
    cfg: dict,
) -> list[str]:
    """Return other phrase_ids in `batch_histories` whose history trips
    period-detection at the SAME lag as `verdict`. If `verdict.period_lag` is
    None (churn-cap trigger, not a period trigger), siblings are empty — the
    churn-cap case doesn't have a cycle partner by definition.
    """
    if not verdict.loop_suspected or verdict.period_lag is None:
        return []
    lg_cfg = (cfg or {}).get("dream", {}).get("loop_guard", {}) or {}
    threshold = float(lg_cfg.get("similarity_threshold", 0.85))
    window = int(lg_cfg.get("period_detection_window", 6))

    siblings: list[str] = []
    for other_id, other_hist in batch_histories.items():
        if other_id == phrase_id or not other_hist:
            continue
        other_candidate = batch_candidates.get(other_id, "")
        other_series = _history_text_series(other_hist, other_candidate)
        suspected, _sim = _detect_period(other_series, verdict.period_lag, threshold, window)
        if suspected:
            siblings.append(other_id)
    return siblings
