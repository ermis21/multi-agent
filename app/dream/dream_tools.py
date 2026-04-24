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


def _summarize_batch_for_review(
    batch: dream_state.PendingBatch,
    dreamer_sid: str,
) -> dict[str, Any]:
    """Shape the pending batch for the `dream_finalize_review` SSE event.

    The CLI / Discord consumer uses this to render per-edit diffs and collect
    user decisions. Full `old_text` / `new_text` are included — the reviewer
    needs them to make an informed call.
    """
    return {
        "dreamer_sid": dreamer_sid,
        "conversation_sid": batch.conversation_sid,
        "target_prompts": batch.target_prompts,
        "target_prompt": batch.target_prompt,  # back-compat for single-target UIs
        "rationale": batch.data.get("rationale", ""),
        "pending_batch_id": batch.data.get("pending_batch_id"),
        "edits": [
            {
                "phrase_id": e["phrase_id"],
                "kind": e["kind"],
                # target_prompt scopes the edit to its file — review UIs
                # group by this so the user sees one section per prompt.
                "target_prompt": e.get("target_prompt", batch.target_prompt),
                "section_path": e.get("section_path", ""),
                "old_text": e.get("old_text", ""),
                "new_text": e.get("new_text", ""),
                "status": e.get("status", "ok"),
                "narrative": e.get("narrative", ""),
            }
            for e in batch.edits
        ],
    }


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
        # `target_prompt` scopes the edit to its file — review UIs group by
        # this, and the dreamer uses it to plan edit_revise calls.
        "target_prompt": edit.get("target_prompt", ""),
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
    targets: list[dict] | None = None,
    rationale: str = "",
    *,
    conversation_sid: str,
    session_id: str,
    cfg: dict,
    path: str | None = None,             # legacy single-target shim
    new_full_text: str | None = None,    # legacy single-target shim
) -> dict:
    """Submit full rewritten prompts for one or more targets in ONE batch.

    New multi-target signature:
        targets=[{"path": "worker_full", "new_full_text": "..."},
                 {"path": "supervisor_full", "new_full_text": "..."}]

    Legacy single-target signature (still honored):
        path="worker_full", new_full_text="..."

    All edits across all targets land in one pending batch; one simulation
    validates them together; one finalize commits them together. This lets
    the dreamer edit interacting prompts (e.g. worker + supervisor) without
    burning separate simulation budgets.
    """
    # Shim: promote the legacy single-target call shape into the list form.
    if targets is None:
        if path is None or new_full_text is None:
            return {"error": "dream_submit requires `targets=[...]` (or legacy `path` + `new_full_text`)"}
        targets = [{"path": path, "new_full_text": new_full_text}]
    if not isinstance(targets, list) or not targets:
        return {"error": "`targets` must be a non-empty list of {path, new_full_text} objects"}

    # Respect model-mismatch gate if a batch already exists.
    if dream_state.has_pending_batch(conversation_sid):
        prior = dream_state.load_pending(conversation_sid)
        ok, reason = dream_state.can_accept_submit_or_revise(prior)
        if not ok:
            return {"error": reason}

    # Validate + resolve every target before touching state. Deduplicate by
    # role_template name — if the dreamer submits the same prompt twice, the
    # last one wins (treat it as a typo rather than erroring).
    resolved: list[tuple[str, Path, str, str]] = []  # (role_template, path, old_text, new_full_text)
    seen: set[str] = set()
    for i, tgt in enumerate(targets):
        if not isinstance(tgt, dict):
            return {"error": f"targets[{i}] must be an object with `path` and `new_full_text`"}
        tgt_path = str(tgt.get("path") or "").strip()
        tgt_text = tgt.get("new_full_text")
        if not tgt_path or not isinstance(tgt_text, str):
            return {"error": f"targets[{i}] missing `path` or `new_full_text` (str)"}
        prompt_path = phrase_store._resolve_prompt_path(tgt_path)
        if not prompt_path.exists():
            return {"error": f"prompt file not found: {prompt_path}"}
        role_template = phrase_store._role_template_name(prompt_path)
        old_text = prompt_path.read_text(encoding="utf-8")
        # Dedup: last write wins.
        resolved = [r for r in resolved if r[0] != role_template]
        seen.add(role_template)
        resolved.append((role_template, prompt_path, old_text, tgt_text))

    # Compute edits per target, stamping `target_prompt` on each record so
    # finalize can group + rewrite. phrase_ids are sha1-based and already
    # globally unique across files, so we can flatten into one list.
    edit_records: list[dict] = []
    flat_idx = 0
    for role_template, _prompt_path, old_text, new_full_text_i in resolved:
        edits_i = dream_diff.compute_edits(old_text, new_text=new_full_text_i,
                                           role_template_name=role_template)
        for e in edits_i:
            edit_records.append({
                "idx": flat_idx,
                "phrase_id": e.phrase_id,
                "target_prompt": role_template,
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
            flat_idx += 1

    if not edit_records:
        return {
            "pending_batch_id": None,
            "edits": [],
            "possible_conflicts": [],
            "possible_loops": [],
            "summary": "0 ok, 0 possible_conflict, 0 possible_loop",
            "note": "submission identical to on-disk prompts — nothing to stage",
        }

    # Narrate flags. `_classify_and_narrate` takes a single target_prompt
    # for history lookups; call it per-group so each file's flags resolve
    # against its own phrase history.
    cache = narrator.NarratorCache()
    for role_template, _p, _o, _n in resolved:
        group_records = [e for e in edit_records if e["target_prompt"] == role_template]
        if group_records:
            await _classify_and_narrate(
                target_prompt=role_template,
                edit_records=group_records,
                cfg=cfg,
                narrator_cache=cache,
            )

    target_prompts = [r[0] for r in resolved]
    new_prompt_texts = {r[0]: r[3] for r in resolved}

    batch = dream_state.create_or_replace_pending(
        conversation_sid=conversation_sid,
        target_prompts=target_prompts,
        new_prompt_texts=new_prompt_texts,
        rationale=rationale,
        edits=edit_records,
    )
    payload = _summary_counts_payload(batch)
    return {
        "pending_batch_id": batch.pending_batch_id,
        "target_prompts": target_prompts,
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

_SKIP_RATIONALE_MIN = 20


async def dream_finalize(
    keep: list[str],
    drop: list[str],
    *,
    conversation_sid: str,
    session_id: str,
    cfg: dict,
    rationale: str | None = None,
) -> dict:
    """Commit kept edits to disk + history; drop the rest; delete pending batch.

    Empty-empty finalize (`keep=[] + drop=[]` with no pending batch) is the
    "skip this conversation, no revision needed" path. It's legitimate, but
    the dreamer must explain WHY — otherwise it's too easy to mindlessly
    bail and leave the user with no signal. Require a `rationale` ≥20 chars
    to make skipping explicit + visible; the runner emits a `dream_skip`
    event carrying the rationale so the CLI/Discord surface shows the
    reasoning instead of "no_submission" going silent.
    """
    try:
        batch = dream_state.load_pending(conversation_sid)
    except dream_state.NoPendingBatch:
        if not keep and not drop:
            rationale = (rationale or "").strip()
            if len(rationale) < _SKIP_RATIONALE_MIN:
                return {
                    "error": (
                        f"empty-batch finalize requires a `rationale` string "
                        f"(≥{_SKIP_RATIONALE_MIN} chars) explaining WHY no "
                        f"revision was warranted. If you identified any "
                        f"conversational issue, prefer `dream_submit` with a "
                        f"targeted fix instead of skipping."
                    ),
                }
            # Stamp the skip onto the dreamer's session state so the runner
            # (a) emits a `dream_skip` event after run_agent_role returns and
            # (b) reports `status: "skipped"` in `_collect_outcome`. Keeping
            # the skip emission at the runner level — rather than here — means
            # it fires whether or not review_bus is armed.
            try:
                from app.sessions.state import SessionState as _SS
                _st = _SS.load_or_create(session_id)
                _st.set("_dream_skip_rationale", rationale)
                _st.save()
            except Exception:
                pass
            return {
                "committed": [],
                "dropped": [],
                "target_prompt": None,
                "noop": True,
                "skip_rationale": rationale,
            }
        return {"error": f"no pending batch for conversation {conversation_sid!r}"}

    cov = dream_state.validate_finalize_coverage(batch, keep, drop)
    if not cov.ok:
        return {"error": cov.reason, "uncovered": cov.uncovered, "unknown": cov.unknown}

    # Manual/verbose dream runs arm `review_bus` on this dreamer session; when
    # armed, the user's keep/drop verdict overrides the dreamer's. Missing
    # phrase_ids in the reply are treated as drop (safe default).
    from app.dream import review_bus
    if review_bus.is_active(session_id):
        all_pids = [e["phrase_id"] for e in batch.edits]
        user_decisions = await review_bus.request_decisions(
            session_id,
            _summarize_batch_for_review(batch, session_id),
        )
        keep = [pid for pid in all_pids if user_decisions.get(pid) == "keep"]
        drop = [pid for pid in all_pids if pid not in keep]

    decisions: dict[str, str] = {}
    for pid in keep:
        decisions[pid] = "keep"
    for pid in drop:
        decisions[pid] = "drop"

    # Group edits by target so we can rebuild each file once. Multi-target
    # batches may touch 2+ files (e.g. worker_full + supervisor_full); each
    # file is rewritten with only its own edits + decisions.
    edits_by_target: dict[str, list[dict]] = {}
    for e in batch.edits:
        edits_by_target.setdefault(e.get("target_prompt", batch.target_prompt), []).append(e)

    # Pre-resolve paths so we can fail fast before any disk write if one of
    # the target files has gone missing.
    target_paths: dict[str, Path] = {}
    for name in edits_by_target:
        p = phrase_store._resolve_prompt_path(name)
        if not p.exists():
            return {"error": f"prompt file missing: {p}"}
        target_paths[name] = p

    # Rewrite each target file with its own edits. Per-file atomicity via
    # `_atomic_write_text`; cross-file atomicity is best-effort (if the
    # second rename fails, the first file is already committed — accepted
    # trade-off; phrase-history per-edit still reflects what landed).
    new_prompt_texts = batch.new_prompt_texts
    for name, target_edits in edits_by_target.items():
        prompt_path = target_paths[name]
        old_text = prompt_path.read_text(encoding="utf-8")
        new_full_text_for_target = new_prompt_texts.get(name, old_text)
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
            for e in target_edits
        ]
        rebuilt = dream_diff.rebuild_with_decisions(
            old_text=old_text,
            new_text=new_full_text_for_target,
            edits=edits_for_rebuild,
            decisions=decisions,
        )
        phrase_store._atomic_write_text(prompt_path, rebuilt)

    run_date = phrase_store.datetime.now(phrase_store.timezone.utc).isoformat()
    committed: list[dict] = []
    dropped: list[dict] = []

    # Attribute history entries to the conversation being dreamed about, not
    # the dreamer's own sid. `_collect_outcome` (and phrase-history readers in
    # general) filter by `session_id == conv_sid` — stamping the dreamer sid
    # here broke that matching, causing finalized edits to show up as
    # "no_submission" in run.json.
    history_session_id = conversation_sid
    for edit in batch.edits:
        pid = edit["phrase_id"]
        target = edit.get("target_prompt", batch.target_prompt)
        if decisions.get(pid) == "drop":
            dropped.append({"phrase_id": pid, "kind": edit["kind"], "target_prompt": target})
            continue
        # keep
        if edit["kind"] == "insert":
            # Register the virgin phrase then stamp rev 1.
            phrase_store.tag_virgin_insert(
                target,
                edit["section_path"],
                edit["new_text"],
                anchor_before="",
                anchor_after="",
            )
            rev = phrase_store.append_history_for_insert(
                pid, edit["new_text"],
                rationale=batch.data.get("rationale", ""),
                run_date=run_date,
                session_id=history_session_id,
            )
        else:  # replace or delete
            rev = phrase_store.record_committed_edit(
                pid,
                old_text=edit["old_text"],
                new_text=edit["new_text"] if edit["kind"] == "replace" else "",
                rationale=batch.data.get("rationale", ""),
                run_date=run_date,
                session_id=history_session_id,
                role_template=target,
                section_path=edit["section_path"],
                path=target,
            )
        committed.append({
            "phrase_id": pid,
            "kind": edit["kind"],
            "target_prompt": target,
            "rev": rev,
        })

    dream_state.delete_pending(conversation_sid)
    return {
        "committed": committed,
        "dropped": dropped,
        "target_prompts": batch.target_prompts,
        "target_prompt": batch.target_prompt,  # back-compat for single-target callers
    }


# ── recal_historical_prompt ──────────────────────────────────────────────────

async def recal_historical_prompt(timestamp: str, prompt_name: str) -> dict:
    """Reconstruct a prompt file as it existed at `timestamp`.

    Thin async wrapper over phrase_store.reconstruct_prompt_at for uniform
    dispatch with the other three dreamer tools.
    """
    return phrase_store.reconstruct_prompt_at(timestamp, prompt_name)
