"""Orchestrate one nightly dream run.

`run_dream(date)`:
  1. Enumerate candidate sessions for `date` (via session_iter).
  2. For each, spawn `run_agent_role("dreamer", ...)` with a briefing that
     targets that conversation. The dreamer's auto-sim hook and FSM fire
     transparently inside that call; the runner only observes outcomes via
     `dream_state` after the agent call returns.
  3. After all per-session runs, spawn the meta-dreamer over the flagged
     phrases.
  4. Write `state/dream/runs/<date>/run.json` recording per-session outcomes
     + meta result for use by the email digest + diagnostics.

User-activity abort: we consult `app/dream/interrupt.py:UserActivityWatcher`
between sessions. On trigger, we short-circuit the loop, mark the run
`interrupted_at`, still attempt to write `run.json`, and let pending-batch
rollback happen via `run_agent_role`'s `finally` in `entrypoints.py`.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.dream import dream_state, meta_dreamer, phrase_store, session_iter

logger = logging.getLogger("dream.runner")

DREAM_RUNS_ROOT = phrase_store.STATE_DIR / "dream" / "runs"


def _run_dir(date_iso: str) -> Path:
    return DREAM_RUNS_ROOT / date_iso


def _atomic_write_json(path: Path, data: dict) -> None:
    import os
    import tempfile
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


def _briefing_for(candidate: session_iter.SessionCandidate, date_iso: str) -> str:
    """Message the dreamer receives per session.

    Stays minimal — the dreamer's prompt explains the full workflow; we just
    name the conversation and the target date so it can start reading.
    """
    return (
        f"Dream pass for conversation `{candidate.session_id}` on {date_iso} "
        f"(role={candidate.agent_role}, final turns={candidate.final_turn_count}). "
        "Read the turns, decide whether a prompt revision would help, and if "
        "so submit it via dream_submit. The system will auto-simulate after "
        "you stop revising. End with dream_finalize."
    )


def _collect_outcome(candidate: session_iter.SessionCandidate) -> dict[str, Any]:
    """Read per-conversation outcome from dream_state + phrase_store.

    Runs after `run_agent_role` returns for this conversation. The dreamer
    normally ends with `dream_finalize`, which deletes the pending batch and
    appends committed edits to phrase_history. We read those two sources.
    """
    conv_sid = candidate.session_id
    # If the pending batch still exists, the dreamer never called finalize —
    # entrypoints' finally has already rolled it back, so this is rare but
    # we still surface the fact for the report.
    if dream_state.has_pending_batch(conv_sid):
        return {
            "conversation_sid": conv_sid,
            "agent_role": candidate.agent_role,
            "status": "rolled_back_unfinalized",
            "committed": [],
            "dropped": [],
            "flagged": [],
        }
    # No pending file: either never submitted, or finalized.
    committed = _committed_today(conv_sid)
    flagged = _flagged_last_submit(conv_sid)
    return {
        "conversation_sid": conv_sid,
        "agent_role": candidate.agent_role,
        "status": "finalized" if committed or flagged else "no_submission",
        "committed": committed,
        "dropped": [],  # finalize drops are not retained on disk by design
        "flagged": flagged,
    }


def _committed_today(conv_sid: str) -> list[dict]:
    """Phrase-history entries where `session_id == conv_sid` and applied is True."""
    out: list[dict] = []
    hist_root = phrase_store.HISTORY_DIR
    if not hist_root.exists():
        return out
    for f in hist_root.glob("*.jsonl"):
        try:
            for line in f.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get("session_id") != conv_sid:
                    continue
                if not row.get("applied", True):
                    continue
                out.append({
                    "phrase_id": f.stem,
                    "prompt_name": row.get("role_template_name"),
                    "section_path": row.get("section_breadcrumb"),
                    "rev": row.get("rev") or row.get("version"),
                    "run_date": row.get("run_date"),
                })
        except OSError:
            continue
    return out


def _flagged_last_submit(conv_sid: str) -> list[dict]:
    """Surface the LAST known flagged list for this conversation.

    After finalize the pending file is gone, so we have no authoritative
    record of flagged edits — the runner treats the flagged list as a
    per-conversation signal only while the batch is live. We reconstruct it
    from the filesystem when possible; otherwise return [] so meta still
    runs off whatever data survived.

    A `finalized_batches.jsonl` append on finalize (future enhancement) would
    preserve this; for now we accept the lossy behavior.
    """
    return []


async def run_dream(
    date_iso: str,
    *,
    interrupt_event: asyncio.Event | None = None,
    meta_enabled: bool = True,
    top_k: int = 3,
) -> dict[str, Any]:
    """Sweep `date_iso`'s sessions and run the dreamer across each.

    `interrupt_event` (usually set by `UserActivityWatcher`) short-circuits
    the loop between conversations. We never cut a dreamer mid-call — the
    per-call `finally` rolls back pending edits cleanly.
    """
    started_at = datetime.now(timezone.utc).isoformat()
    run_dir = _run_dir(date_iso)
    run_dir.mkdir(parents=True, exist_ok=True)

    candidates = session_iter.iter_sessions_for_date(date_iso)
    record: dict[str, Any] = {
        "date": date_iso,
        "started_at": started_at,
        "ended_at": None,
        "interrupted_at": None,
        "session_ids_seen": [c.session_id for c in candidates],
        "session_ids_completed": [],
        "conversations": [],
        "meta": None,
    }

    # Lazy import — entrypoints pulls the whole agent stack.
    from app.entrypoints import run_agent_role

    try:
        for c in candidates:
            if interrupt_event is not None and interrupt_event.is_set():
                record["interrupted_at"] = datetime.now(timezone.utc).isoformat()
                break
            sid = f"dreamer_{date_iso}_{c.session_id}"
            body = {
                "messages": [{"role": "user", "content": _briefing_for(c, date_iso)}],
                "_source_trigger": {"type": "cron", "ref": "dreamer"},
                # The dreamer needs to know which conversation to dream about
                # inside the tool calls. We stamp it on the dreamer's own
                # session state so dream_submit can resolve conversation_sid.
                "_dream_conversation_sid": c.session_id,
            }
            try:
                await run_agent_role("dreamer", body, sid)
                record["session_ids_completed"].append(c.session_id)
            except Exception as e:
                logger.exception("dream pass for %s failed", c.session_id)
                record["conversations"].append({
                    "conversation_sid": c.session_id,
                    "agent_role": c.agent_role,
                    "status": "error",
                    "error": f"{type(e).__name__}: {e}",
                })
                continue
            record["conversations"].append(_collect_outcome(c))

        if meta_enabled and record["interrupted_at"] is None:
            try:
                record["meta"] = await meta_dreamer.run_meta_dreamer(
                    record, top_k=top_k,
                )
            except Exception as e:
                logger.exception("meta-dreamer failed")
                record["meta"] = {"status": "error", "error": f"{type(e).__name__}: {e}"}
    finally:
        record["ended_at"] = datetime.now(timezone.utc).isoformat()
        try:
            _atomic_write_json(run_dir / "run.json", record)
        except Exception:
            logger.exception("failed to write run.json")
    return record
