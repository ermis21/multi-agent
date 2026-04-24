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
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from app.dream import dream_state, meta_dreamer, phrase_store, session_iter

logger = logging.getLogger("dream.runner")

# Sids we never dream about: test-harness ephemera + one-shot slug-generation
# sessions (see discord/bot_worker.py:617 `rename_*`). These have ≥1 final turn
# but aren't "conversations" in any meaningful sense.
_EPHEMERAL_SID_PREFIXES: tuple[str, ...] = ("test_", "injtest_", "rename_", "e2e_")

# Only sids that trace back to a real Discord interaction count. API-direct
# one-shot /v1/chat/completions calls (sid = "YYYYMMDD_HHMMSS_hex") are
# filtered out — they're mostly test harness / smoke / curl calls, not
# conversations worth dreaming about. If you ever want to dream about
# programmatic API sessions, loosen this.
_REAL_CONVERSATION_SID_PREFIXES: tuple[str, ...] = ("discord_",)


def _is_dreamable_sid(sid: str) -> bool:
    if sid.startswith(_EPHEMERAL_SID_PREFIXES):
        return False
    return sid.startswith(_REAL_CONVERSATION_SID_PREFIXES)

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


def _roles_in_conversation(turns_path: Path) -> list[str]:
    """Return the distinct agent roles that appeared in this conversation.

    Reads turns.jsonl once and collects unique `role` values — we skip
    `"final"` because it's a replay marker, not an actual agent. Typical
    result is `["worker", "supervisor"]`; solo-worker sessions return just
    `["worker"]`.
    """
    roles: list[str] = []
    seen: set[str] = set()
    if not turns_path.exists():
        return roles
    try:
        with turns_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                r = row.get("role")
                if r and r != "final" and r not in seen:
                    seen.add(r)
                    roles.append(r)
    except OSError:
        pass
    return roles


def _candidate_prompt_files(
    candidate: session_iter.SessionCandidate,
) -> list[tuple[str, str]]:
    """List `(role, prompt_filename)` pairs for each role that appeared.

    Uses the 3-tier prompt resolver so the dreamer knows EXACTLY which
    files are in scope without inferring — e.g. `[("worker", "worker_full.md"),
    ("supervisor", "supervisor_full.md")]`. Roles that fail to resolve are
    surfaced with `"?"` so the dreamer can still investigate.
    """
    from app.prompt_generator import resolve_prompt_file_for_role
    mode = candidate.mode or "converse"
    pairs: list[tuple[str, str]] = []
    for role in _roles_in_conversation(candidate.turns_path):
        fname = resolve_prompt_file_for_role(role, mode) or "?"
        pairs.append((role, fname))
    return pairs


def _briefing_for(candidate: session_iter.SessionCandidate, date_iso: str) -> str:
    """Message the dreamer receives per session.

    The dreamer's prompt covers the full workflow; we name the conversation,
    target date, the turns.jsonl path, AND the prompt files driving each
    role that appeared — so a conversation with both worker and supervisor
    turns gets both prompts named, encouraging coordinated multi-target
    edits instead of a half-fix limited to the worker side.
    """
    turns_path = f"state/sessions/{candidate.session_id}/turns.jsonl"
    role_pairs = _candidate_prompt_files(candidate)
    if role_pairs:
        role_block = "; ".join(f"`{r}` (driven by `{f}`)" for r, f in role_pairs)
        roles_line = (
            f"Roles that appeared in this conversation: {role_block}. "
            f"Their prompts interact — if you fix one, consider whether the "
            f"others need coordinated edits. You get **one** simulation budget "
            f"for the whole batch, so stage all target prompts together in a "
            f"single `dream_submit(targets=[...], rationale=\"...\")` call."
        )
    else:
        roles_line = (
            f"This conversation was driven by role=`{candidate.agent_role}` "
            f"in mode=`{candidate.mode or '?'}` — target the prompt file "
            f"driving that role."
        )
    return (
        f"Dream pass for conversation `{candidate.session_id}` on {date_iso} "
        f"(final turns={candidate.final_turn_count}).\n\n"
        f"{roles_line}\n\n"
        f"READ THE TURNS FROM: `{turns_path}` via `file_read`. Each line is a "
        "JSON row with `role` (worker/supervisor/final), `messages`, and "
        "`response` — only `final` rows are authoritative user-visible turns.\n\n"
        "**Bias toward submitting a fix.** If you identify any concrete issue "
        "(a confusing instruction, a tool miscall, a hedging failure, a "
        "repeated supervisor critique), propose a targeted prompt revision via "
        "`dream_submit`. The user reviews every edit and can reject — so a "
        "proposed fix is cheap. Silent skips are expensive: the user has no "
        "signal and the run produced no value.\n\n"
        "End every conversation-turn with `dream_finalize(keep, drop)`. If the "
        "conversation is genuinely clean and nothing warrants a revision, call "
        "`dream_finalize(keep=[], drop=[], rationale=\"<≥20-char reason>\")` — "
        "the rationale is required and will be shown to the user so they see "
        "your reasoning for skipping."
    )


def _collect_outcome(
    candidate: session_iter.SessionCandidate,
    *,
    dreamer_sid: str | None = None,
) -> dict[str, Any]:
    """Read per-conversation outcome from dream_state + phrase_store.

    Runs after `run_agent_role` returns for this conversation. The dreamer
    normally ends with `dream_finalize`, which deletes the pending batch and
    appends committed edits to phrase_history.

    `dreamer_sid` lets us read the dreamer's session state to detect an
    explicit skip this pass — otherwise same-day commits from an earlier
    pass would masquerade as `finalized` and mislead the user.
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
    # Did the dreamer explicitly skip this pass? If so, report "skipped" and
    # surface the rationale — don't mistake prior-pass same-day commits for
    # "finalized this pass".
    skip_rationale: str | None = None
    if dreamer_sid:
        try:
            from app.sessions.state import SessionState as _SS
            _st = _SS.load_or_create(dreamer_sid)
            skip_rationale = _st.get("_dream_skip_rationale")
        except Exception:
            skip_rationale = None
    if skip_rationale:
        return {
            "conversation_sid": conv_sid,
            "agent_role": candidate.agent_role,
            "status": "skipped",
            "skip_rationale": skip_rationale,
            "committed": [],
            "dropped": [],
            "flagged": [],
        }
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
    date_iso: str | None = None,
    *,
    window_hours: float | None = None,
    end_ts: datetime | None = None,
    conversation_sids: list[str] | None = None,
    dreamer_model_override: str | None = None,
    interrupt_event: asyncio.Event | None = None,
    meta_enabled: bool = True,
    top_k: int = 3,
    trace_queue: asyncio.Queue | None = None,
    review_required: bool = False,
) -> dict[str, Any]:
    """Sweep sessions and run the dreamer across each.

    Two scoping modes:
      - **Calendar day** (back-compat): pass `date_iso="YYYY-MM-DD"`. Sessions
        whose last `final` turn falls on that UTC date are dreamed.
      - **Rolling window** (default): when `date_iso` is None, sweep sessions
        active in the last `window_hours` (defaults to 24) ending at `end_ts`
        (defaults to "now" UTC). This is what manual / cron invocations should
        use by default — "dream about the last 24 hours" matches user intent
        better than a UTC calendar slice that misses today's activity.

    `interrupt_event` (usually set by `UserActivityWatcher`) short-circuits
    the loop between conversations. We never cut a dreamer mid-call — the
    per-call `finally` rolls back pending edits cleanly.

    `trace_queue` (set by the manual SSE endpoint) receives runner events
    plus forwarded `_run_worker` events (worker_status / tool_trace / …).
    `review_required` arms `review_bus` for each conversation, gating
    `dream_finalize` on user decisions instead of auto-committing.
    """
    now_utc = datetime.now(timezone.utc)
    started_at = now_utc.isoformat()

    # Three scoping modes, in precedence order:
    #   1. Explicit sid list (user picked specific conversations via CLI/UI)
    #   2. Calendar day (back-compat for `date=YYYY-MM-DD`)
    #   3. Rolling window (default: last 24h)
    if conversation_sids:
        run_label = now_utc.strftime("%Y-%m-%d")
        window_meta = None
        raw_candidates = [
            c for c in (session_iter.load_candidate(s) for s in conversation_sids)
            if c is not None
        ]
    elif date_iso is None:
        hours = window_hours if window_hours is not None else 24.0
        window_end = end_ts or now_utc
        if window_end.tzinfo is None:
            window_end = window_end.replace(tzinfo=timezone.utc)
        window_start = window_end - timedelta(hours=hours)
        # Label the run by the window-end date so artifacts group naturally.
        run_label = window_end.strftime("%Y-%m-%d")
        window_meta = {
            "start": window_start.isoformat(),
            "end": window_end.isoformat(),
            "hours": hours,
        }
        raw_candidates = session_iter.iter_sessions_in_window(window_start, window_end)
    else:
        run_label = date_iso
        window_meta = None
        raw_candidates = session_iter.iter_sessions_for_date(date_iso)

    # Explicit sid selection trusts the user — no sid-prefix filter applied.
    # (They can always drop unwanted sids from the list client-side.) Window
    # and date modes apply the usual dreamable-sid gate.
    if conversation_sids:
        candidates = raw_candidates
        skipped_sids: list[str] = []
    else:
        candidates = [c for c in raw_candidates if _is_dreamable_sid(c.session_id)]
        skipped_sids = [c.session_id for c in raw_candidates if not _is_dreamable_sid(c.session_id)]

    run_dir = _run_dir(run_label)
    run_dir.mkdir(parents=True, exist_ok=True)

    record: dict[str, Any] = {
        "date": run_label,
        "window": window_meta,
        "started_at": started_at,
        "ended_at": None,
        "interrupted_at": None,
        "session_ids_seen": [c.session_id for c in candidates],
        "session_ids_skipped": skipped_sids,
        "session_ids_completed": [],
        "conversations": [],
        "meta": None,
    }

    def _emit(event: str, data: dict) -> None:
        if trace_queue is None:
            return
        try:
            trace_queue.put_nowait({"event": event, "data": data})
        except Exception:
            pass

    # Lazy import — entrypoints pulls the whole agent stack.
    from app.entrypoints import run_agent_role
    from app.config_loader import get_config
    from app.dream import review_bus

    _cfg = get_config()
    dreamer_model = (
        dreamer_model_override
        or ((_cfg.get("dream") or {}).get("model"))
        or (_cfg.get("llm") or {}).get("model")
        or "?"
    )

    _emit("dream_run_start", {
        "date": run_label,
        "window": window_meta,
        "candidates": len(candidates),
        "skipped": len(skipped_sids),
        "dreamer_model": dreamer_model,
        "review_required": bool(review_required),
    })

    try:
        for idx, c in enumerate(candidates):
            if interrupt_event is not None and interrupt_event.is_set():
                record["interrupted_at"] = datetime.now(timezone.utc).isoformat()
                break
            sid = f"dreamer_{run_label}_{c.session_id}"
            body: dict = {
                "messages": [{"role": "user", "content": _briefing_for(c, run_label)}],
                "_source_trigger": {"type": "cron", "ref": "dreamer"},
                # The dreamer needs to know which conversation to dream about
                # inside the tool calls. We stamp it on the dreamer's own
                # session state so dream_submit can resolve conversation_sid.
                "_dream_conversation_sid": c.session_id,
            }
            if dreamer_model_override:
                # entrypoints.run_agent_role copies role_cfg and overlays
                # body["model"] — this is how we pin the dreamer to the
                # user-picked model without mutating cfg.dream.model.
                body["model"] = dreamer_model_override
            # Parse the Discord channel id from the sid pattern
            # `discord_{channel_id}_{epoch}` so the Discord consumer can
            # resolve it to a channel name. CLI just renders the number.
            channel_id = None
            parts = c.session_id.split("_")
            if parts[:1] == ["discord"] and len(parts) >= 2 and parts[1].isdigit():
                channel_id = parts[1]
            # Enumerate roles + their prompt files so the CLI/Discord
            # consumer can show the user which prompts are in scope for
            # this conversation (matches the briefing the dreamer sees).
            prompt_files = [
                {"role": r, "file": f}
                for r, f in _candidate_prompt_files(c)
            ]
            _emit("dream_conversation_start", {
                "sid": c.session_id,
                "role": c.agent_role,
                "mode": c.mode,
                "worker_model": c.model,
                "channel_id": channel_id,
                "final_turn_count": c.final_turn_count,
                "last_final_ts": c.last_final_ts,
                "prompt_files": prompt_files,
                "idx": idx,
                "total": len(candidates),
                "dreamer_sid": sid,
                "dreamer_model": dreamer_model,
            })
            if review_required and trace_queue is not None:
                review_bus.register(sid, trace_queue)
            try:
                agent_result = await run_agent_role(
                    "dreamer", body, sid, trace_queue=trace_queue,
                )
                # run_agent_role swallows _run_worker exceptions and stuffs the
                # error string into the response; it now also sets result.error
                # when that happens. Promote it to a conversation_end error so
                # the stream shows the real failure instead of "no_submission".
                if isinstance(agent_result, dict) and agent_result.get("error"):
                    err_msg = str(agent_result.get("error"))
                    outcome = {
                        "conversation_sid": c.session_id,
                        "agent_role": c.agent_role,
                        "status": "error",
                        "error": err_msg,
                    }
                    record["conversations"].append(outcome)
                    _emit("dream_conversation_end", outcome | {"dreamer_sid": sid})
                    continue
                record["session_ids_completed"].append(c.session_id)
            except Exception as e:
                logger.exception("dream pass for %s failed", c.session_id)
                outcome = {
                    "conversation_sid": c.session_id,
                    "agent_role": c.agent_role,
                    "status": "error",
                    "error": f"{type(e).__name__}: {e}",
                }
                record["conversations"].append(outcome)
                _emit("dream_conversation_end", outcome | {"dreamer_sid": sid})
                continue
            finally:
                if review_required and trace_queue is not None:
                    review_bus.unregister(sid)
            outcome = _collect_outcome(c, dreamer_sid=sid)
            # If this pass was an explicit skip, emit a dream_skip event so
            # the CLI / Discord consumer can surface the rationale even when
            # review_bus isn't armed (review=off path).
            if outcome.get("status") == "skipped":
                _emit("dream_skip", {
                    "dreamer_sid": sid,
                    "conversation_sid": c.session_id,
                    "rationale": outcome.get("skip_rationale", ""),
                })
            record["conversations"].append(outcome)
            _emit("dream_conversation_end", outcome | {"dreamer_sid": sid})

        if meta_enabled and record["interrupted_at"] is None:
            flagged_total = sum(
                len((conv or {}).get("flagged") or [])
                for conv in record["conversations"]
            )
            _emit("dream_meta_start", {
                "flagged_total": flagged_total,
                "top_k": top_k,
            })
            try:
                record["meta"] = await meta_dreamer.run_meta_dreamer(
                    record,
                    top_k=top_k,
                    trace_queue=trace_queue,
                    review_required=review_required,
                    dreamer_model_override=dreamer_model_override,
                )
            except Exception as e:
                logger.exception("meta-dreamer failed")
                record["meta"] = {"status": "error", "error": f"{type(e).__name__}: {e}"}
            _emit("dream_meta_end", record["meta"] or {})
    finally:
        record["ended_at"] = datetime.now(timezone.utc).isoformat()
        try:
            _atomic_write_json(run_dir / "run.json", record)
        except Exception:
            logger.exception("failed to write run.json")
        _emit("dream_run_end", {
            "seen": len(record["session_ids_seen"]),
            "completed": len(record["session_ids_completed"]),
            "interrupted": bool(record["interrupted_at"]),
        })
    return record
