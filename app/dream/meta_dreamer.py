"""Meta-dreamer — edits `config/prompts/dreamer.md` after a nightly run.

After the per-role dream loop finishes for a date, the runner picks the
`top_k` phrases by conflict count (from `run.json`'s per-phrase conflict
tally) and spawns a meta-dreamer session targeting `dreamer.md`. Mechanically
this is just another `run_agent_role("dreamer", ...)` call — the meta pass
reuses the Phase 2 stack verbatim. The only thing meta-specific is the
briefing: we hand the dreamer a focused task ("these phrases are churning;
rewrite them") so it doesn't waste turns triaging.

Public API:
  - top_conflict_phrases(run_record, *, top_k=3) → list of `{phrase_id, count, prompt_name, last_text}`.
  - build_meta_briefing(top_phrases) → user-message body.
  - run_meta_dreamer(run_record, *, cfg=None, session_id=None) → the run_agent_role dict payload.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from app.dream import phrase_store

META_TARGET_PROMPT = "dreamer"  # corresponds to config/prompts/dreamer.md
META_SESSION_PREFIX = "meta_dreamer_"


def top_conflict_phrases(run_record: dict, *, top_k: int = 3) -> list[dict]:
    """Rank the phrases most often flagged `possible_conflict` / `possible_loop`
    across the run.

    `run_record` shape (from Phase 4's `app/dream/runner.py`):

        {
          "date": "YYYY-MM-DD",
          "conversations": [
            {
              "conversation_sid": "...",
              "committed": [{"phrase_id": "ph-aa", "prompt_name": "worker_full", ...}],
              "dropped":   [...],
              "flagged":   [{"phrase_id": "ph-bb", "status": "possible_conflict",
                             "prompt_name": "worker_full"}, ...]
            }, ...
          ]
        }

    We count each distinct `phrase_id` appearance in any `flagged` list across
    conversations (a phrase flagged in 4 different sessions gets count=4).
    Returns at most `top_k` entries, highest-count first, each enriched with
    the current phrase text from `phrase_store` so the meta briefing has
    grounded content.
    """
    counts: Counter[str] = Counter()
    meta_by_id: dict[str, dict] = {}
    for conv in run_record.get("conversations", []) or []:
        for edit in conv.get("flagged", []) or []:
            pid = edit.get("phrase_id")
            if not pid:
                continue
            counts[pid] += 1
            # Remember the last-seen prompt_name for each id (if multiple
            # conversations saw different targets, last one wins — fine for a
            # briefing hint, not load-bearing).
            if "prompt_name" in edit:
                meta_by_id.setdefault(pid, {})["prompt_name"] = edit["prompt_name"]
            if "status" in edit:
                meta_by_id.setdefault(pid, {})["last_status"] = edit["status"]

    ranked: list[dict] = []
    for pid, count in counts.most_common(top_k):
        entry = {
            "phrase_id": pid,
            "count": int(count),
            "prompt_name": meta_by_id.get(pid, {}).get("prompt_name"),
            "last_status": meta_by_id.get(pid, {}).get("last_status"),
        }
        # Pull current text from the phrase index if it's still live.
        try:
            idx = phrase_store._read_index(pid)
            entry["current_text"] = idx.get("current_text") or ""
            entry["section_path"] = idx.get("section_breadcrumb") or ""
            if not entry["prompt_name"]:
                entry["prompt_name"] = idx.get("role_template_name")
        except phrase_store.LocateFailure:
            entry["current_text"] = ""
            entry["section_path"] = ""
        ranked.append(entry)
    return ranked


def build_meta_briefing(top_phrases: list[dict]) -> str:
    """Render the user-message body handed to the meta-dreamer session.

    The meta-dreamer's *target* is `config/prompts/dreamer.md`; the hot
    phrases are evidence of oscillation in `dreamer.md` over the run. The
    briefing names each id + its current text so the dreamer can decide
    whether to rewrite, collapse, or leave alone.
    """
    if not top_phrases:
        return ("Meta-dreamer pass: no conflict-heavy phrases this run. "
                "No action required — end your turn with <|end|>.")
    lines = [
        "Meta-dreamer pass. These phrases in `config/prompts/dreamer.md` "
        "(or nearby role prompts it references) were flagged most often "
        "during tonight's run:",
        "",
    ]
    for i, p in enumerate(top_phrases, start=1):
        prompt = p.get("prompt_name") or "(unknown prompt)"
        status = p.get("last_status") or "flagged"
        section = p.get("section_path") or "(root)"
        txt = (p.get("current_text") or "").strip()
        if len(txt) > 400:
            txt = txt[:400].rstrip() + " …"
        lines.append(
            f"{i}. phrase_id={p['phrase_id']}  count={p['count']}  "
            f"prompt={prompt}  status={status}"
        )
        lines.append(f"   section: {section}")
        if txt:
            lines.append(f"   current text: {txt}")
        lines.append("")
    lines.append(
        "Decide whether to revise `config/prompts/dreamer.md`. If you revise, "
        "submit the whole rewritten file via dream_submit(path=\"config/prompts/dreamer.md\", "
        "new_full_text=..., rationale=...). You MUST finish with dream_finalize."
    )
    return "\n".join(lines)


def _meta_session_id() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{META_SESSION_PREFIX}{ts}_{uuid4().hex[:6]}"


async def run_meta_dreamer(
    run_record: dict,
    *,
    cfg: dict | None = None,
    session_id: str | None = None,
    top_k: int = 3,
) -> dict[str, Any]:
    """Spawn a meta-dreamer session targeting `dreamer.md`.

    Returns `{"status": "no_conflicts" | "ok" | "error", "session_id", ...}`.
    The actual agent call uses the standard `run_agent_role("dreamer", ...)`
    path, which means the auto-sim hook, pending-batch FSM, and unfinalized-
    rollback all apply to the meta run verbatim.
    """
    phrases = top_conflict_phrases(run_record, top_k=top_k)
    if not phrases:
        return {"status": "no_conflicts", "top_phrases": []}

    briefing = build_meta_briefing(phrases)
    sid = session_id or _meta_session_id()
    body = {
        "messages": [{"role": "user", "content": briefing}],
        "_source_trigger": {"type": "cron", "ref": "meta_dreamer"},
    }
    # Lazy import — entrypoints pulls agent/loop which is heavy.
    from app.entrypoints import run_agent_role
    try:
        result = await run_agent_role("dreamer", body, sid)
        return {
            "status": "ok",
            "session_id": sid,
            "top_phrases": phrases,
            "agent_result": result,
        }
    except Exception as e:
        return {
            "status": "error",
            "session_id": sid,
            "top_phrases": phrases,
            "error": f"{type(e).__name__}: {e}",
        }
