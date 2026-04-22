"""Local "checkpoint" LLM that composes narratives for flagged dream edits.

The dreamer never asks "is this a loop" or "does this conflict with history" —
the system flags those edits automatically and calls the narrator to render a
1–3-sentence prose explanation the dreamer can read inline with its staged
batch. Two templates:

  * **conflict narrative** — given the phrase's two newest prior versions
    (`history[-2:]`) + the proposed new text, describe how the proposal relates
    to recent drift.
  * **loop narrative** — given a wider excerpt + the sibling phrase_ids that
    loop-guard identified as co-oscillating + the detected period, state the
    period AND name each sibling. Sibling ids are pointers the dreamer uses to
    drill in via `phrase_history_recall`.

Cache: within a single `dream_submit` call, identical phrase_id → narrative
combinations are produced once and reused — avoids N LLM calls when the same
phrase shows up twice (e.g. dreamer re-submitted then `edit_revise`d).

`_llm_call` is module-level and easy to monkey-patch in tests. Production
routes it through `app.llm._llm_call` against `cfg.models.checkpoint`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Awaitable, Callable, Protocol


class _AsyncLLM(Protocol):
    async def __call__(self, prompt: str, cfg: dict) -> str: ...


async def _default_llm_call(prompt: str, cfg: dict) -> str:
    """Production LLM seam — pins the `checkpoint` model for low-latency narrative."""
    from app.llm import _content, _llm_call  # lazy to avoid import cycles
    resp = await _llm_call(
        messages=[
            {"role": "system", "content": "You explain prompt edits in 1-3 crisp sentences."},
            {"role": "user", "content": prompt},
        ],
        cfg=cfg,
        temperature=0.2,
        role_cfg={"model": "checkpoint"},
    )
    return _content(resp).strip()


# Test seam — monkeypatch this attribute to stub the LLM during unit tests.
_llm_call: _AsyncLLM = _default_llm_call


def _set_llm_call(fn: _AsyncLLM) -> None:
    """Hook for tests to replace the LLM seam (keeps the module importable)."""
    global _llm_call
    _llm_call = fn


@dataclass
class NarratorCache:
    """Per-submission cache — one instance lives for the duration of a
    `dream_submit` / `edit_revise` call. Keyed on (phrase_id, kind) so that a
    phrase's conflict narrative and loop narrative are stored independently
    if both ever apply (they shouldn't simultaneously, but the key shape is
    cheap insurance).
    """
    entries: dict[tuple[str, str], str] = field(default_factory=dict)

    def get(self, phrase_id: str, kind: str) -> str | None:
        return self.entries.get((phrase_id, kind))

    def put(self, phrase_id: str, kind: str, text: str) -> None:
        self.entries[(phrase_id, kind)] = text


# ── Prompt templates ─────────────────────────────────────────────────────────

def _format_conflict_prompt(
    phrase_id: str,
    section_path: str,
    history_excerpt: list[dict],
    new_text: str,
) -> str:
    lines = [
        f"Phrase id: {phrase_id}",
        f"Section: {section_path or '(preamble)'}",
        "",
        "This phrase has been edited before. The two NEWEST prior versions are:",
    ]
    for i, row in enumerate(history_excerpt, 1):
        lines.append(f"--- v{row.get('rev', '?')} (rationale: {row.get('rationale','')!s}) ---")
        lines.append(row.get("new_text", "").strip())
    lines += [
        "",
        "The dreamer now proposes replacing it with:",
        "--- proposed ---",
        new_text.strip(),
        "",
        "In 1-3 crisp sentences, describe how this proposal relates to the recent "
        "drift of this phrase. Do NOT recommend accept/reject — the dreamer decides. "
        "Just explain the relationship: is the proposal undoing, refining, "
        "diverging, or re-contesting prior edits?",
    ]
    return "\n".join(lines)


def _format_loop_prompt(
    phrase_id: str,
    section_path: str,
    history_excerpt: list[dict],
    sibling_phrase_ids: list[str],
    period_lag: int | None,
    new_text: str,
) -> str:
    lines = [
        f"Phrase id: {phrase_id}",
        f"Section: {section_path or '(preamble)'}",
    ]
    if period_lag:
        lines.append(f"Detected period: lag={period_lag} over recent versions.")
    else:
        lines.append("Detected trigger: churn cap — this phrase has been rewritten many times.")
    if sibling_phrase_ids:
        lines.append(
            "Co-oscillating sibling phrase_ids: "
            + ", ".join(sibling_phrase_ids)
            + "  (the dreamer can call phrase_history_recall on these to drill in)"
        )
    else:
        lines.append("No co-oscillating sibling phrases detected.")
    lines += ["", "History excerpt (first + last of the trailing window):"]
    for row in history_excerpt:
        lines.append(f"--- v{row.get('rev', '?')} ---")
        lines.append(row.get("new_text", "").strip())
    lines += [
        "",
        "Dreamer's current proposal:",
        new_text.strip(),
        "",
        "Write 1-3 crisp sentences that (a) state the detected pattern (include the period if known), "
        "and (b) explicitly name each sibling phrase_id above with a brief note on its role in the cycle. "
        "If there are no siblings, state that. Do NOT recommend accept/reject.",
    ]
    return "\n".join(lines)


# ── Public API ───────────────────────────────────────────────────────────────

async def narrate_conflict(
    *,
    phrase_id: str,
    section_path: str,
    history_excerpt: list[dict],
    new_text: str,
    cfg: dict,
    cache: NarratorCache | None = None,
) -> str:
    if cache and (hit := cache.get(phrase_id, "conflict")) is not None:
        return hit
    prompt = _format_conflict_prompt(phrase_id, section_path, history_excerpt, new_text)
    text = await _llm_call(prompt, cfg)
    if cache:
        cache.put(phrase_id, "conflict", text)
    return text


async def narrate_loop(
    *,
    phrase_id: str,
    section_path: str,
    history_excerpt: list[dict],
    sibling_phrase_ids: list[str],
    period_lag: int | None,
    new_text: str,
    cfg: dict,
    cache: NarratorCache | None = None,
) -> str:
    if cache and (hit := cache.get(phrase_id, "loop")) is not None:
        return hit
    prompt = _format_loop_prompt(
        phrase_id, section_path, history_excerpt, sibling_phrase_ids, period_lag, new_text,
    )
    text = await _llm_call(prompt, cfg)
    if cache:
        cache.put(phrase_id, "loop", text)
    return text
