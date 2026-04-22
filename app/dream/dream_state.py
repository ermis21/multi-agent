"""Per-conversation dream state: pending batch I/O + phase state machine.

One file per in-flight dreamer conversation at
`state/dream/runs/<date>/pending_<conv_sid>.json`. The file is the single source
of truth while a dreamer is revising a prompt; it persists the proposed new
prompt text, the classified edits, and the simulation counter.

Phase state machine:

  submit          — dreamer is still calling `dream_submit` / `edit_revise`.
                    Any other action (or a submit that completes with no flags)
                    triggers the auto-sim and transitions to `post_sim`.
  post_sim        — auto-sim has just run. If `model_match` and under the sim
                    cap, dreamer may revise further (stays in `post_sim`);
                    otherwise the only legal move is `dream_finalize`.
  finalize_only   — terminal non-commit state: model mismatch OR sim cap hit.
                    `dream_submit` / `edit_revise` both rejected with
                    `model_mismatch_no_further_edits`.

This module is deliberately I/O-only — it knows nothing about the LLM, the
dreamer's prompt, or the simulator. The runner owns auto-sim triggering and
calls `on_simulation_complete` to advance state.
"""

from __future__ import annotations

import json
import os
import secrets
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.dream.phrase_store import STATE_DIR  # single source of STATE_DIR

DREAM_RUNS_ROOT = STATE_DIR / "dream" / "runs"

# ── Phase constants ──────────────────────────────────────────────────────────

PHASE_SUBMIT = "submit"
PHASE_POST_SIM = "post_sim"
PHASE_FINALIZE_ONLY = "finalize_only"

# Error string the runner surfaces to the dreamer when a submit/revise is
# rejected under `can_iterate=false`. Mirrored in dream_tools.py.
ERR_MODEL_MISMATCH = "model_mismatch_no_further_edits"


class DreamStateError(RuntimeError):
    """Base error for state-machine violations."""


class NoPendingBatch(DreamStateError):
    """No pending batch exists for the conversation."""


# ── Path helpers ─────────────────────────────────────────────────────────────

def _run_date_dir(run_date: str | None = None) -> Path:
    """`state/dream/runs/YYYY-MM-DD/`. `run_date` defaults to UTC today."""
    d = run_date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return DREAM_RUNS_ROOT / d


def _pending_path(conv_sid: str, run_date: str | None = None) -> Path:
    return _run_date_dir(run_date) / f"pending_{conv_sid}.json"


def _find_pending_path(conv_sid: str) -> Path | None:
    """Scan run-date dirs for any `pending_{conv_sid}.json`. Returns newest or None.

    A run that spans midnight may end up under a different date than it started
    on; scanning avoids losing the file.
    """
    if not DREAM_RUNS_ROOT.exists():
        return None
    matches: list[Path] = []
    for day in DREAM_RUNS_ROOT.iterdir():
        if not day.is_dir():
            continue
        f = day / f"pending_{conv_sid}.json"
        if f.exists():
            matches.append(f)
    if not matches:
        return None
    return max(matches, key=lambda p: p.stat().st_mtime)


# ── File I/O ─────────────────────────────────────────────────────────────────

def _atomic_write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


# ── Batch shape helpers ──────────────────────────────────────────────────────

def _new_batch_id() -> str:
    return "pb-" + secrets.token_hex(4)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class PendingBatch:
    """Typed view over the on-disk pending batch file."""
    data: dict

    @property
    def pending_batch_id(self) -> str:
        return self.data["pending_batch_id"]

    @property
    def conversation_sid(self) -> str:
        return self.data["conversation_sid"]

    @property
    def target_prompt(self) -> str:
        return self.data["target_prompt"]

    @property
    def phase(self) -> str:
        return self.data.get("phase", PHASE_SUBMIT)

    @property
    def edits(self) -> list[dict]:
        return self.data.get("edits", [])

    @property
    def simulations_run(self) -> int:
        return int(self.data.get("simulations_run", 0))

    @property
    def last_sim_model_match(self) -> bool | None:
        return self.data.get("last_sim_model_match")

    def edit_by_phrase_id(self, phrase_id: str) -> dict | None:
        for e in self.edits:
            if e.get("phrase_id") == phrase_id:
                return e
        return None

    def summary_counts(self) -> dict[str, int]:
        counts = {"ok": 0, "possible_conflict": 0, "possible_loop": 0}
        for e in self.edits:
            s = e.get("status", "ok")
            counts[s] = counts.get(s, 0) + 1
        return counts

    def summary_line(self) -> str:
        c = self.summary_counts()
        return (
            f"{c.get('ok', 0)} ok, "
            f"{c.get('possible_conflict', 0)} possible_conflict, "
            f"{c.get('possible_loop', 0)} possible_loop"
        )

    def has_any_flag(self) -> bool:
        return any(e.get("status") != "ok" for e in self.edits)


# ── Public API — batch lifecycle ─────────────────────────────────────────────

def has_pending_batch(conv_sid: str) -> bool:
    return _find_pending_path(conv_sid) is not None


def load_pending(conv_sid: str) -> PendingBatch:
    p = _find_pending_path(conv_sid)
    if p is None:
        raise NoPendingBatch(f"no pending batch for conversation {conv_sid!r}")
    return PendingBatch(_read_json(p))


def save_pending(batch: PendingBatch, run_date: str | None = None) -> None:
    """Persist `batch` atomically. Caller is responsible for keeping
    `run_date` stable across writes to the same batch (or omitting it to use
    UTC today)."""
    path = _pending_path(batch.conversation_sid, run_date=run_date)
    _atomic_write_json(path, batch.data)


def delete_pending(conv_sid: str) -> bool:
    """Remove the pending batch file. Returns True if one existed."""
    p = _find_pending_path(conv_sid)
    if p is None:
        return False
    p.unlink()
    return True


def create_or_replace_pending(
    *,
    conversation_sid: str,
    target_prompt: str,
    new_prompt_text: str,
    rationale: str,
    edits: list[dict],
    run_date: str | None = None,
) -> PendingBatch:
    """Write a fresh pending batch, replacing any prior one wholesale.

    Fresh `dream_submit` semantics: start over; simulation counter resets.
    """
    data = {
        "pending_batch_id": _new_batch_id(),
        "conversation_sid": conversation_sid,
        "target_prompt": target_prompt,
        "new_prompt_text": new_prompt_text,
        "rationale": rationale,
        "submitted_at": _now_iso(),
        "phase": PHASE_SUBMIT,
        "edits": edits,
        "simulations_run": 0,
        "last_sim_model_match": None,
        "last_sim_at": None,
    }
    batch = PendingBatch(data)
    # Ensure we overwrite any prior-day file too.
    if (old := _find_pending_path(conversation_sid)) is not None:
        old.unlink()
    save_pending(batch, run_date=run_date)
    return batch


# ── State-machine transitions ────────────────────────────────────────────────

def can_accept_submit_or_revise(batch: PendingBatch) -> tuple[bool, str | None]:
    """Gate for `dream_submit` / `edit_revise`.

    Returns `(allowed, reason)`. `reason` is None when allowed; otherwise
    `ERR_MODEL_MISMATCH` when blocked by model-mismatch / cap exhaustion.
    """
    if batch.phase == PHASE_FINALIZE_ONLY:
        return False, ERR_MODEL_MISMATCH
    return True, None


def on_submit_resets_phase(batch: PendingBatch) -> None:
    """A new `dream_submit` (after a prior sim) puts the batch back in submit
    phase — the dreamer is revising based on the sim result."""
    batch.data["phase"] = PHASE_SUBMIT


def on_edit_revise_stays_in_phase(batch: PendingBatch) -> None:
    """`edit_revise` is a 'still revising' signal but does not change phase —
    it just keeps submit-phase live, or keeps post_sim revisable until the
    next auto-sim lands."""
    # No-op by design; kept as a named hook so the state-machine surface is
    # symmetric with on_submit / on_finalize.
    return None


def should_auto_sim(batch: PendingBatch, *, dreamer_just_called_submit_or_revise: bool) -> bool:
    """Auto-sim fires when:
      - we're in `submit` phase,
      - the dreamer's latest turn did NOT call `dream_submit` / `edit_revise`,
      - a pending batch exists (implicit — we were passed one).
    """
    if batch.phase != PHASE_SUBMIT:
        return False
    return not dreamer_just_called_submit_or_revise


def on_simulation_complete(
    batch: PendingBatch,
    *,
    model_match: bool,
    simulations_cap: int,
) -> None:
    """Advance the state machine after a simulation finishes.

    Rules:
      - Increment counter + stamp timestamps.
      - If `!model_match` OR `simulations_run >= cap` → `finalize_only`.
      - Else → `post_sim` (dreamer may revise again).
    """
    batch.data["simulations_run"] = batch.simulations_run + 1
    batch.data["last_sim_model_match"] = bool(model_match)
    batch.data["last_sim_at"] = _now_iso()
    if not model_match or batch.simulations_run >= simulations_cap:
        batch.data["phase"] = PHASE_FINALIZE_ONLY
    else:
        batch.data["phase"] = PHASE_POST_SIM


def simulations_remaining(batch: PendingBatch, simulations_cap: int) -> int:
    return max(0, simulations_cap - batch.simulations_run)


def can_iterate(batch: PendingBatch, simulations_cap: int) -> bool:
    """Convenience for the sim-result payload."""
    if batch.last_sim_model_match is False:
        return False
    return simulations_remaining(batch, simulations_cap) > 0


# ── Finalize coverage check ──────────────────────────────────────────────────

@dataclass
class FinalizeCoverage:
    ok: bool
    uncovered: list[str]
    unknown: list[str]
    reason: str = ""


def validate_finalize_coverage(
    batch: PendingBatch, keep: list[str], drop: list[str]
) -> FinalizeCoverage:
    """`keep ∪ drop` must equal the full phrase_id set of the batch.

    `uncovered` = staged phrase_ids not present in keep ∪ drop.
    `unknown`   = ids in keep or drop that aren't staged.
    """
    staged = {e["phrase_id"] for e in batch.edits}
    keep_set = set(keep)
    drop_set = set(drop)
    uncovered = sorted(staged - (keep_set | drop_set))
    unknown = sorted((keep_set | drop_set) - staged)
    overlap = sorted(keep_set & drop_set)
    if overlap:
        return FinalizeCoverage(
            ok=False, uncovered=uncovered, unknown=unknown,
            reason=f"phrase_ids present in both keep and drop: {overlap}",
        )
    if uncovered or unknown:
        reason_parts = []
        if uncovered:
            reason_parts.append(f"uncovered={uncovered}")
        if unknown:
            reason_parts.append(f"unknown={unknown}")
        return FinalizeCoverage(
            ok=False, uncovered=uncovered, unknown=unknown,
            reason="keep ∪ drop must equal staged phrase_ids (" + "; ".join(reason_parts) + ")",
        )
    return FinalizeCoverage(ok=True, uncovered=[], unknown=[])
