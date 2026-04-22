"""Iterate the previous day's conversation sessions for the dreamer.

Source-of-truth: `state/sessions/{sid}/turns.jsonl`. A session is a candidate
for dreaming if it has at least one `role=="final"` turn whose timestamp falls
on the target UTC date, and it is not a sub-agent session (no
`parent_session_id`) or a dreamer-spawned session (so we don't dream about
dream runs).

This module is pure I/O — no LLM, no state mutation — so the rest of the
dream pipeline can iterate candidates without hitting the runner.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from app.dream.phrase_store import STATE_DIR

SESSIONS_ROOT = STATE_DIR / "sessions"

# Roles that are internal pipeline infrastructure, not user conversations. Dreaming
# over them would recurse or waste cycles.
_EXCLUDED_ROLES: frozenset[str] = frozenset({
    "dreamer",
    "meta_dreamer",
    "soul_updater",
    "session_compactor",
    "prompt_suggester",
    "webfetch_summarizer",
    "supervisor",
})


@dataclass
class SessionCandidate:
    """Minimal metadata the dreamer needs to pick a target prompt + replay."""
    session_id: str
    agent_role: str
    mode: str | None
    model: str | None
    final_turn_count: int
    last_final_ts: str
    turns_path: Path


def _iter_date_from_iso(ts: str) -> str:
    """Normalize an ISO-8601 timestamp to a `YYYY-MM-DD` UTC date string.

    Accepts the Python isoformat output (`+00:00` offset); returns `""` on
    unparseable input so filtering ignores malformed rows.
    """
    try:
        dt = datetime.fromisoformat(ts)
    except (TypeError, ValueError):
        return ""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%d")


def _read_state(sid_dir: Path) -> dict:
    """Return the session's state.json as a dict, or `{}` when absent/unreadable.

    state.json is advisory — turns.jsonl is authoritative for turn data, so a
    missing state file shouldn't mask a valid session. But when it exists we
    use it to filter by `agent_role`, `parent_session_id`, and the
    `source_trigger.ref` (dream-spawned sessions).
    """
    p = sid_dir / "state.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _is_excluded(state: dict) -> bool:
    """Role/parent-based filtering of infrastructure sessions.

    - Sub-agent sessions: `parent_session_id` is set.
    - Dream-triggered sessions: `source_trigger.ref` names a dream role.
    - Any role in `_EXCLUDED_ROLES` (dreamer, supervisor, compactor…).
    """
    if state.get("parent_session_id"):
        return True
    role = (state.get("agent_role") or "").strip()
    if role in _EXCLUDED_ROLES:
        return True
    trig = state.get("source_trigger") or {}
    ref = str(trig.get("ref") or "").lower()
    if "dream" in ref:
        return True
    return False


def _scan_turns_for_date(turns_path: Path, date_iso: str) -> tuple[int, str]:
    """Scan turns.jsonl for `final` turns on the target date.

    Returns `(count, latest_ts)`. `latest_ts` is the newest ISO timestamp of
    any final turn on that date, or `""` when no match.
    """
    if not turns_path.exists():
        return 0, ""
    count = 0
    latest = ""
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
                if row.get("role") != "final":
                    continue
                ts = row.get("timestamp") or ""
                if _iter_date_from_iso(ts) != date_iso:
                    continue
                count += 1
                if ts > latest:
                    latest = ts
    except OSError:
        return 0, ""
    return count, latest


def iter_sessions_for_date(date_iso: str) -> list[SessionCandidate]:
    """Return candidates with `≥1 final turn on `date_iso`, sorted by recency.

    `date_iso` is `YYYY-MM-DD` in UTC. Sessions are excluded if they are
    sub-agents, the dreamer's own, or any infrastructure role (compactor,
    supervisor, soul_updater, …).
    """
    if not SESSIONS_ROOT.exists():
        return []
    out: list[SessionCandidate] = []
    for sid_dir in sorted(SESSIONS_ROOT.iterdir()):
        if not sid_dir.is_dir():
            continue
        state = _read_state(sid_dir)
        if _is_excluded(state):
            continue
        turns_path = sid_dir / "turns.jsonl"
        count, latest = _scan_turns_for_date(turns_path, date_iso)
        if count == 0:
            continue
        role = (state.get("agent_role") or "worker").strip() or "worker"
        out.append(SessionCandidate(
            session_id=sid_dir.name,
            agent_role=role,
            mode=state.get("mode"),
            model=state.get("model"),
            final_turn_count=count,
            last_final_ts=latest,
            turns_path=turns_path,
        ))
    out.sort(key=lambda c: c.last_final_ts, reverse=True)
    return out


def iter_yesterday_sessions() -> list[SessionCandidate]:
    """Convenience wrapper: today-minus-one in UTC. Dreamer cron entry point."""
    today = datetime.now(timezone.utc).date()
    # date subtraction; avoid dateutil
    yesterday = today.fromordinal(today.toordinal() - 1)
    return iter_sessions_for_date(yesterday.strftime("%Y-%m-%d"))
