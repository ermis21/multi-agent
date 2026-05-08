"""
Rolling session compactor.

Long sessions outgrow the sliding-window attention budget. Rather than
truncate raw history and lose the early context, a background worker
role (`session_compactor`) folds completed turns into a dense summary
that becomes the canonical replay surface.

Design (plan §4):
  - Fire at most one compaction per session at a time (per-session asyncio.Lock).
  - Trigger after a turn when:
        turn_count - covers_up_to_turn >= K   (default 6)
      AND
        rebuilt-history token estimate >= budgets.history * 1.5
      AND
        now - last_compaction_ts > 60s        (churn guard)
  - Compactor emits the two-section contract defined in
    config/prompts/session_compactor.md. Output is wrapped as a single
    pseudo-turn in `state/sessions/{sid}/active.jsonl`, and
    `state.history.active` is repointed at it — subsequent replays read
    the compacted view instead of the raw turn log.

Failure modes are non-fatal: a crash during compaction leaves the raw
turn log and prior `active.jsonl` intact.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.config_loader import get_config
from app.sessions.state import SESSIONS_DIR, SessionState


# Module-global lock table — one asyncio.Lock per session_id, lazily created.
# asyncio.create_task-spawned compactors contend on this so we never run two
# at once for the same session.
_compactor_locks: dict[str, asyncio.Lock] = {}


def _lock_for(session_id: str) -> asyncio.Lock:
    lock = _compactor_locks.get(session_id)
    if lock is None:
        lock = asyncio.Lock()
        _compactor_locks[session_id] = lock
    return lock


def _parse_iso(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts)
    except (ValueError, TypeError):
        return None


def _seconds_since(ts: str | None) -> float:
    dt = _parse_iso(ts)
    if dt is None:
        return float("inf")
    now = datetime.now(timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return (now - dt).total_seconds()


def _history_budget(cfg: dict) -> int:
    return int(
        (cfg.get("context") or {})
        .get("budgets", {})
        .get("history", 6000)
    )


def _compaction_interval_turns(cfg: dict) -> int:
    # Plan default K=6 — keep as config knob for tuning without code changes.
    return int(
        (cfg.get("context") or {})
        .get("compaction_interval_turns", 6)
    )


def _churn_guard_seconds(cfg: dict) -> float:
    return float(
        (cfg.get("context") or {})
        .get("compaction_churn_seconds", 60.0)
    )


def _last_history_tokens(state: SessionState) -> int:
    """Token estimate for the last rebuilt history, written by prompt_generator."""
    section = state.get("context_stats.section_tokens") or {}
    # prompt_generator.generate emits curated sections + ALLOWED_TOOLS + SKILLS.
    # HISTORY is reserved for when loop.py starts tracking the rebuilt-context
    # size directly; until then we fall back to a conservative estimate based
    # on the full turn log length.
    if "HISTORY" in section:
        return int(section["HISTORY"])
    return 0


def _full_log_size_tokens(state: SessionState) -> int:
    """Fallback history size via the on-disk full turn log."""
    hist = state.get("history") or {}
    full_rel = hist.get("full")
    if not full_rel:
        return 0
    p = Path(full_rel if Path(full_rel).is_absolute() else f"/{full_rel}")
    if not p.exists():
        return 0
    try:
        # ~4 chars/token is the tiktoken-fallback ratio used elsewhere.
        return p.stat().st_size // 4
    except OSError:
        return 0


def should_trigger(state: SessionState, cfg: dict) -> bool:
    """Decide whether to spawn a compaction right now.

    All three conditions must hold:
      1. at least K turns have landed since the last compaction boundary
      2. estimated history tokens exceed 1.5× the configured budget
      3. the last compaction (if any) is older than the churn-guard window
    """
    if not (cfg.get("context") or {}).get("enabled", True):
        return False

    turn_count = int((state.get("stats") or {}).get("turn_count", 0))
    covers = state.get("history.compaction_covers_up_to_turn")
    covers = int(covers) if covers is not None else 0
    interval = _compaction_interval_turns(cfg)
    if turn_count - covers < interval:
        return False

    budget = _history_budget(cfg)
    est = _last_history_tokens(state) or _full_log_size_tokens(state)
    if budget and est < int(budget * 1.5):
        return False

    if _seconds_since(state.get("history.last_compaction_ts")) < _churn_guard_seconds(cfg):
        return False

    return True


def _format_scope_for_prompt(sid: str, turns: list[dict], from_turn: int, to_turn: int) -> str:
    """Produce the plan_context payload the compactor reads as its input.

    Only `role=final` turns are meaningful to the compactor — those carry the
    raw user message + clean assistant response. Worker/supervisor attempts
    are retry noise and would bloat the input.
    """
    finals = [t for t in turns if t.get("role") == "final"]
    scoped = finals[from_turn:to_turn]
    body_lines: list[str] = [
        f"Session: {sid}",
        f"Scope: turns {from_turn}..{to_turn} ({len(scoped)} finals)",
        "",
        "## Transcript",
        "",
    ]
    for i, t in enumerate(scoped, start=from_turn):
        for m in t.get("messages", []):
            body_lines.append(json.dumps({"turn": i, "role": m.get("role"), "content": m.get("content", "")}, ensure_ascii=False))
        body_lines.append(json.dumps({"turn": i, "role": "assistant", "content": t.get("response", "")}, ensure_ascii=False))
    return "\n".join(body_lines)


def _write_active_line(sid: str, body: str) -> Path:
    """Append a single `role=final` pseudo-turn wrapping the compactor output.

    Using role=final lets `_rebuild_session_context` consume the active.jsonl
    unchanged: it already filters on role=final, and the compacted body
    replaces what would otherwise be many-turn history.
    """
    sdir = SESSIONS_DIR / sid
    sdir.mkdir(parents=True, exist_ok=True)
    path = sdir / "active.jsonl"
    entry = {
        "role": "final",
        "messages": [{"role": "user", "content": "[compacted session history]"}],
        "response": body,
        "kind": "compacted",
        "ts": datetime.now(timezone.utc).isoformat(),
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return path


def _relative_active_path(sid: str) -> str:
    """Path stored in state.history.active — relative, so _rebuild_session_context
    works identically inside /app and on-host test environments."""
    return f"sessions/{sid}/active.jsonl"


async def run_compaction(session_id: str) -> dict[str, Any]:
    """Compact the uncompacted tail using whichever engine cfg selects.

    Returns a small telemetry dict for tests and diagnostics; never raises.
    Held under _lock_for(session_id) so concurrent triggers coalesce.

    Engines (cfg.context.compactor_engine):
      "agent"  → spawn `session_compactor` sub-agent (legacy, default).
      "lingua" → LongLLMLingua compresses the tail in-process (B2). Falls
                 back to "agent" if the package isn't installed so the
                 toggle never silently breaks compaction.
    """
    # Lazy imports — avoid circular (entrypoints -> compactor would loop).
    from app.sessions.logger import get_session

    lock = _lock_for(session_id)
    if lock.locked():
        # Another compaction is already in flight; silently drop.
        return {"session_id": session_id, "skipped": "already_locked"}

    async with lock:
        cfg = get_config()
        engine = (cfg.get("context") or {}).get("compactor_engine", "agent")

        state = SessionState.load_or_create(session_id)
        turns = get_session(session_id)
        finals = [t for t in turns if t.get("role") == "final"]
        turn_count = len(finals)
        covers = state.get("history.compaction_covers_up_to_turn") or 0
        covers = int(covers)
        if turn_count - covers < 1:
            return {"session_id": session_id, "skipped": "nothing_to_compact"}

        scope = _format_scope_for_prompt(session_id, turns, covers, turn_count)

        if engine == "lingua":
            body, lingua_meta = _compact_with_lingua(scope, cfg)
            if body is None:
                # Fallback: lingua not installed or failed → agent path.
                print(f"[compactor] lingua engine fell back to agent: {lingua_meta.get('error')}",
                      flush=True)
                body = await _compact_with_agent(scope, session_id)
                engine_used = "agent_fallback"
            else:
                engine_used = "lingua"
        else:
            body = await _compact_with_agent(scope, session_id)
            engine_used = "agent"

        if not body or "## RUNNING_SUMMARY" not in body:
            return {"session_id": session_id, "error": "malformed_output", "engine": engine_used}

        active = _write_active_line(session_id, body)
        state = SessionState.load_or_create(session_id)
        state.record_compaction(turn_count, _relative_active_path(session_id))
        state.save()
        return {
            "session_id":         session_id,
            "covers_up_to_turn":  turn_count,
            "active_path":        str(active),
            "body_chars":         len(body),
            "engine":             engine_used,
        }


async def _compact_with_agent(scope: str, session_id: str) -> str:
    """Legacy agent-based compaction. Spawns session_compactor sub-run."""
    from app.entrypoints import run_agent_role
    sub_sid = f"{session_id}_compactor"
    try:
        result = await run_agent_role(
            "session_compactor",
            {
                "messages": [{"role": "user", "content": "Compact the session transcript below into the two-section contract."}],
                "plan_context": scope,
                "_source_trigger": {"type": "sub_agent", "ref": session_id},
            },
            sub_sid,
        )
    except Exception as e:
        return ""
    return _extract_body(result)


def _compact_with_lingua(scope: str, cfg: dict) -> tuple[str | None, dict]:
    """LongLLMLingua compression of the uncompacted tail.

    Returns (compressed_body, meta). On any failure (package missing, model
    download blocked, runtime error), returns (None, {"error": ...}) and the
    caller falls back to the agent engine. The compressed body is wrapped
    with `## RUNNING_SUMMARY` so it satisfies the existing invariant that
    `app.loop._rebuild_session_context` reads.
    """
    try:
        # llmlingua's import is heavy (loads transformers); guard so a
        # missing/optional install doesn't break the api startup.
        from llmlingua import PromptCompressor  # type: ignore
    except ImportError as e:
        return None, {"error": f"llmlingua not installed: {e}"}

    ctx = cfg.get("context") or {}
    target_token = int(ctx.get("lingua_target_token", 1500))
    rate = float(ctx.get("lingua_rate", 0.5))

    try:
        # Cache the compressor on the module so we don't reload weights per
        # compaction (model is ~350MB).
        global _lingua_compressor
        if "_lingua_compressor" not in globals() or _lingua_compressor is None:
            _lingua_compressor = PromptCompressor(
                model_name="microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
                use_llmlingua2=True,
                device_map="cpu",
            )
        compressed = _lingua_compressor.compress_prompt(
            scope,
            target_token=target_token,
            rate=rate,
        )
        # PromptCompressor returns dict with `compressed_prompt` key
        text = compressed.get("compressed_prompt", "") if isinstance(compressed, dict) else str(compressed)
    except Exception as e:
        return None, {"error": f"lingua compression failed: {e}"}

    if not text.strip():
        return None, {"error": "empty compression output"}

    # Wrap to satisfy the `## RUNNING_SUMMARY` invariant that the agent
    # path's prompt forces. _rebuild_session_context only requires the
    # marker is present somewhere in the body.
    body = (
        "## RUNNING_SUMMARY\n"
        "_(LongLLMLingua compression of the uncompacted tail — token-level reduction, "
        "no LLM call. Originals preserved in turns.jsonl.)_\n\n"
        f"{text}\n"
    )
    return body, {"engine": "lingua", "input_chars": len(scope), "output_chars": len(body)}


_lingua_compressor = None  # cached PromptCompressor instance


def _extract_body(result: dict | None) -> str:
    if not isinstance(result, dict):
        return ""
    choices = result.get("choices") or []
    if not choices:
        return ""
    msg = choices[0].get("message") or {}
    return str(msg.get("content") or "")


def maybe_spawn(session_id: str, cfg: dict) -> asyncio.Task | None:
    """Fire-and-forget spawn wrapper called by run_agent_loop's finally block.

    Returns the Task for tests to await on; production callers drop it.
    """
    try:
        state = SessionState.load_or_create(session_id)
    except Exception:
        return None
    if not should_trigger(state, cfg):
        return None
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # No running loop (e.g. sync test context) — skip silently.
        return None
    return loop.create_task(run_compaction(session_id))
