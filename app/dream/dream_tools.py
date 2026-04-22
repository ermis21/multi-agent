"""Four dreamer-facing tool implementations.

The dreamer role calls these via `LOCAL_TOOLS` dispatch in `app/mcp_client.py`.
Each function returns the JSON payload the dreamer sees as a tool result. The
per-conversation state machine lives in `app/dream/dream_state.py`; simulation
triggering lives in the runner wrapper; both read the pending batch this
module writes.

Public API:
  - dream_submit(path, new_full_text, rationale, *, conversation_sid, session_id, cfg)
  - edit_revise(phrase_id, new_text, rationale, *, conversation_sid, session_id, cfg)
  - dream_finalize(keep, drop, *, conversation_sid, session_id, cfg)
  - recal_historical_prompt(timestamp, prompt_name)

All four are async — `dream_submit` and `edit_revise` await the narrator LLM
for flagged edits; `dream_finalize` is sync internally but exposed async for
uniform dispatch.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.dream import diff as dream_diff
from app.dream import dream_state, loop_guard, narrator, phrase_store

# Excerpt sizes referenced by the plan: conflict uses history[-2:] (two newest);
# loop uses first K + last K of the trailing window.
_LOOP_EXCERPT_K = 3


# ── Helpers ──────────────────────────────────────────────────────────────────

def _loop_excerpt(history: list[dict], k: int = _LOOP_EXCERPT_K) -> list[dict]:
    """First k + last k entries, deduped by rev; returns at most 2k items."""
    if len(history) <= 2 * k:
        return list(history)
    head = history[:k]
    tail = history[-k:]
    seen_revs = {row.get("rev") for row in head}
    merged = list(head) + [row for row in tail if row.get("rev") not in seen_revs]
    return merged


def _last_history_timestamp(history: list[dict]) -> str | None:
    if not history:
        return None
    for row in reversed(history):
        ts = row.get("run_date")
        if ts:
            return ts
    return None


def _summary_counts_payload(batch: dream_state.PendingBatch) -> dict[str, Any]:
    """Top-level lists of flagged phrase_ids for the dreamer-facing payload."""
    conflicts: list[dict] = []
    loops: list[dict] = []
    for edit in batch.edits:
        status = edit.get("status")
        entry = {
            "phrase_id": edit["phrase_id"],
            "timestamp": edit.get("last_history_timestamp"),
        }
        if status == "possible_conflict":
            conflicts.append(entry)
        elif status == "possible_loop":
            loops.append(entry)
    return {"possible_conflicts": conflicts, "possible_loops": loops}


def _edit_record_for_payload(edit: dict) -> dict:
    """Shape a staged edit for the tool-result payload."""
    out = {
        "idx": edit["idx"],
        "phrase_id": edit["phrase_id"],
        "status": edit["status"],
        "section_path": edit.get("section_path", ""),
        "kind": edit.get("kind", "replace"),
    }
    if edit["status"] == "ok":
        out["old_text"] = edit.get("old_text", "")
        out["new_text"] = edit.get("new_text", "")
    else:
        out["new_text"] = edit.get("new_text", "")
        out["history_excerpt"] = edit.get("history_excerpt", [])
        out["narrative"] = edit.get("narrative", "")
        if edit["status"] == "possible_loop":
            out["loop_siblings"] = edit.get("loop_siblings", [])
            out["period_lag"] = edit.get("period_lag")
    return out


# ── Classification pipeline (shared by submit + revise) ──────────────────────

async def _classify_and_narrate(
    *,
    target_prompt: str,
    edit_records: list[dict],
    cfg: dict,
    narrator_cache: narrator.NarratorCache,
) -> None:
    """Mutate `edit_records` in place: run loop-guard + narrator for every
    record whose `new_text` has history; leave virgin edits at status=ok.

    Each record must already contain: `phrase_id, section_path, old_text,
    new_text, kind`.
    """
    # Load per-phrase history once (used for classification + sibling detection).
    histories: dict[str, list[dict]] = {}
    for rec in edit_records:
        histories[rec["phrase_id"]] = phrase_store.get_history(rec["phrase_id"])

    # First pass — classify each edit.
    verdicts: dict[str, loop_guard.LoopVerdict] = {}
    for rec in edit_records:
        pid = rec["phrase_id"]
        hist = histories[pid]
        rec["last_history_timestamp"] = _last_history_timestamp(hist)
        if not hist:
            rec["status"] = "ok"
            continue
        verdict = loop_guard.check_loop(pid, rec["new_text"], hist, cfg)
        verdicts[pid] = verdict
        if verdict.loop_suspected:
            rec["status"] = "possible_loop"
        else:
            rec["status"] = "possible_conflict"

    # Second pass — sibling detection for loop-flagged edits.
    batch_histories = {pid: histories[pid] for pid in verdicts}
    batch_candidates = {
        rec["phrase_id"]: rec["new_text"]
        for rec in edit_records
        if rec["phrase_id"] in verdicts
    }
    for rec in edit_records:
        if rec.get("status") != "possible_loop":
            continue
        v = verdicts[rec["phrase_id"]]
        siblings = loop_guard.find_siblings(
            rec["phrase_id"], v, batch_histories, batch_candidates, cfg,
        )
        rec["loop_siblings"] = siblings
        rec["period_lag"] = v.period_lag

    # Third pass — narrate flagged edits.
    for rec in edit_records:
        status = rec.get("status")
        if status not in ("possible_conflict", "possible_loop"):
            continue
        hist = histories[rec["phrase_id"]]
        if status == "possible_conflict":
            excerpt = hist[-2:]
            rec["history_excerpt"] = excerpt
            rec["narrative"] = await narrator.narrate_conflict(
                phrase_id=rec["phrase_id"],
                section_path=rec.get("section_path", ""),
                history_excerpt=excerpt,
                new_text=rec["new_text"],
                cfg=cfg,
                cache=narrator_cache,
            )
        else:  # possible_loop
            excerpt = _loop_excerpt(hist)
            rec["history_excerpt"] = excerpt
            rec["narrative"] = await narrator.narrate_loop(
                phrase_id=rec["phrase_id"],
                section_path=rec.get("section_path", ""),
                history_excerpt=excerpt,
                sibling_phrase_ids=rec.get("loop_siblings", []),
                period_lag=rec.get("period_lag"),
                new_text=rec["new_text"],
                cfg=cfg,
                cache=narrator_cache,
            )


# ── dream_submit ─────────────────────────────────────────────────────────────

async def dream_submit(
    path: str,
    new_full_text: str,
    rationale: str,
    *,
    conversation_sid: str,
    session_id: str,
    cfg: dict,
) -> dict:
    """Submit a full rewritten prompt. System diffs, flags, narrates."""
    # Respect model-mismatch gate if a batch already exists.
    if dream_state.has_pending_batch(conversation_sid):
        prior = dream_state.load_pending(conversation_sid)
        ok, reason = dream_state.can_accept_submit_or_revise(prior)
        if not ok:
            return {"error": reason}

    prompt_path = phrase_store._resolve_prompt_path(path)
    if not prompt_path.exists():
        return {"error": f"prompt file not found: {prompt_path}"}
    old_text = prompt_path.read_text(encoding="utf-8")
    role_template = phrase_store._role_template_name(prompt_path)

    edits = dream_diff.compute_edits(old_text, new_text=new_full_text,
                                     role_template_name=role_template)
    if not edits:
        return {
            "pending_batch_id": None,
            "edits": [],
            "possible_conflicts": [],
            "possible_loops": [],
            "summary": "0 ok, 0 possible_conflict, 0 possible_loop",
            "note": "submission identical to on-disk prompt — nothing to stage",
        }

    edit_records: list[dict] = []
    for idx, e in enumerate(edits):
        edit_records.append({
            "idx": idx,
            "phrase_id": e.phrase_id,
            "section_path": e.section_path,
            "old_text": e.old_text,
            "new_text": e.new_text,
            "kind": e.kind,
            "old_start": e.old_start,
            "new_start": e.new_start,
            "opcode_index": e.opcode_index,
            "status": "ok",
            "history_excerpt": [],
            "narrative": "",
            "loop_siblings": [],
            "period_lag": None,
            "last_history_timestamp": None,
        })

    cache = narrator.NarratorCache()
    await _classify_and_narrate(
        target_prompt=role_template,
        edit_records=edit_records,
        cfg=cfg,
        narrator_cache=cache,
    )

    batch = dream_state.create_or_replace_pending(
        conversation_sid=conversation_sid,
        target_prompt=role_template,
        new_prompt_text=new_full_text,
        rationale=rationale,
        edits=edit_records,
    )
    payload = _summary_counts_payload(batch)
    return {
        "pending_batch_id": batch.pending_batch_id,
        "edits": [_edit_record_for_payload(e) for e in batch.edits],
        "possible_conflicts": payload["possible_conflicts"],
        "possible_loops": payload["possible_loops"],
        "summary": batch.summary_line(),
    }


# ── edit_revise ──────────────────────────────────────────────────────────────

async def edit_revise(
    phrase_id: str,
    new_text: str,
    rationale: str,
    *,
    conversation_sid: str,
    session_id: str,
    cfg: dict,
) -> dict:
    """Patch a single staged edit's new_text; re-classify that edit in place."""
    try:
        batch = dream_state.load_pending(conversation_sid)
    except dream_state.NoPendingBatch:
        return {"error": f"no pending batch for conversation {conversation_sid!r}"}

    ok, reason = dream_state.can_accept_submit_or_revise(batch)
    if not ok:
        return {"error": reason}

    edit = batch.edit_by_phrase_id(phrase_id)
    if edit is None:
        return {"error": f"phrase_id {phrase_id!r} not present in pending batch"}

    edit["new_text"] = new_text
    edit["rationale"] = rationale
    # Reset flagged fields so re-classification lands cleanly.
    edit["status"] = "ok"
    edit["history_excerpt"] = []
    edit["narrative"] = ""
    edit["loop_siblings"] = []
    edit["period_lag"] = None

    cache = narrator.NarratorCache()
    await _classify_and_narrate(
        target_prompt=batch.target_prompt,
        edit_records=[edit],
        cfg=cfg,
        narrator_cache=cache,
    )

    dream_state.save_pending(batch)
    return {
        "pending_batch_id": batch.pending_batch_id,
        "edit": _edit_record_for_payload(edit),
        "summary": batch.summary_line(),
    }


# ── dream_finalize ───────────────────────────────────────────────────────────

async def dream_finalize(
    keep: list[str],
    drop: list[str],
    *,
    conversation_sid: str,
    session_id: str,
    cfg: dict,
) -> dict:
    """Commit kept edits to disk + history; drop the rest; delete pending batch."""
    try:
        batch = dream_state.load_pending(conversation_sid)
    except dream_state.NoPendingBatch:
        return {"error": f"no pending batch for conversation {conversation_sid!r}"}

    cov = dream_state.validate_finalize_coverage(batch, keep, drop)
    if not cov.ok:
        return {"error": cov.reason, "uncovered": cov.uncovered, "unknown": cov.unknown}

    decisions: dict[str, str] = {}
    for pid in keep:
        decisions[pid] = "keep"
    for pid in drop:
        decisions[pid] = "drop"

    # Resolve the prompt file from its basename (routes through PROMPTS_DIR).
    prompt_path = phrase_store._resolve_prompt_path(batch.target_prompt)
    if not prompt_path.exists():
        return {"error": f"prompt file missing: {prompt_path}"}

    old_text = prompt_path.read_text(encoding="utf-8")
    new_full_text = batch.data["new_prompt_text"]

    # Build the Edit list matching diff.py's shape so rebuild_with_decisions
    # can reconstruct the prompt.
    edits_for_rebuild = [
        dream_diff.Edit(
            kind=e["kind"],
            phrase_id=e["phrase_id"],
            section_path=e["section_path"],
            old_text=e["old_text"],
            new_text=e["new_text"],
            old_start=e.get("old_start", 0),
            new_start=e.get("new_start", 0),
            opcode_index=e["opcode_index"],
        )
        for e in batch.edits
    ]
    rebuilt = dream_diff.rebuild_with_decisions(
        old_text=old_text,
        new_text=new_full_text,
        edits=edits_for_rebuild,
        decisions=decisions,
    )
    phrase_store._atomic_write_text(prompt_path, rebuilt)

    run_date = phrase_store.datetime.now(phrase_store.timezone.utc).isoformat()
    committed: list[dict] = []
    dropped: list[dict] = []

    for edit in batch.edits:
        pid = edit["phrase_id"]
        if decisions.get(pid) == "drop":
            dropped.append({"phrase_id": pid, "kind": edit["kind"]})
            continue
        # keep
        if edit["kind"] == "insert":
            # Register the virgin phrase then stamp rev 1.
            phrase_store.tag_virgin_insert(
                batch.target_prompt,
                edit["section_path"],
                edit["new_text"],
                anchor_before="",
                anchor_after="",
            )
            rev = phrase_store.append_history_for_insert(
                pid, edit["new_text"],
                rationale=batch.data.get("rationale", ""),
                run_date=run_date,
                session_id=session_id,
            )
        else:  # replace or delete
            rev = phrase_store.record_committed_edit(
                pid,
                old_text=edit["old_text"],
                new_text=edit["new_text"] if edit["kind"] == "replace" else "",
                rationale=batch.data.get("rationale", ""),
                run_date=run_date,
                session_id=session_id,
                role_template=batch.target_prompt,
                section_path=edit["section_path"],
                path=batch.target_prompt,
            )
        committed.append({
            "phrase_id": pid,
            "kind": edit["kind"],
            "rev": rev,
        })

    dream_state.delete_pending(conversation_sid)
    return {
        "committed": committed,
        "dropped": dropped,
        "target_prompt": batch.target_prompt,
    }


# ── recal_historical_prompt ──────────────────────────────────────────────────

async def recal_historical_prompt(timestamp: str, prompt_name: str) -> dict:
    """Reconstruct a prompt file as it existed at `timestamp`.

    Thin async wrapper over phrase_store.reconstruct_prompt_at for uniform
    dispatch with the other three dreamer tools.
    """
    return phrase_store.reconstruct_prompt_at(timestamp, prompt_name)
