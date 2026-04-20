"""
JSONL session logger.

Each session lives in its own directory: `{sessions_dir}/{session_id}/turns.jsonl`.
Each line in `turns.jsonl` is one turn (JSON object). Sidecars (state.json,
approvals.jsonl, tool_errors.jsonl, active.jsonl) live beside it.
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

SESSIONS_DIR = Path(os.environ.get("SESSIONS_DIR", "/state/sessions"))


def _session_dir(session_id: str) -> Path:
    return SESSIONS_DIR / session_id


def _turns_path(session_id: str) -> Path:
    return _session_dir(session_id) / "turns.jsonl"


class SessionLogger:
    def __init__(self, session_id: str):
        self.session_id = session_id
        _session_dir(session_id).mkdir(parents=True, exist_ok=True)
        self.path = _turns_path(session_id)

    def log_turn(
        self,
        attempt:    int,
        role:       str,
        messages:   list[dict],
        response:   str,
        supervisor: dict | None = None,
        usage:      dict | None = None,
    ) -> None:
        entry = {
            "timestamp":  datetime.now(timezone.utc).isoformat(),
            "session_id": self.session_id,
            "attempt":    attempt,
            "role":       role,
            "messages":   messages,
            "response":   response,
            "supervisor": supervisor,
            "usage":      usage or {},
        }
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def list_sessions() -> list[dict]:
    """Return summary of all sessions (id + turn count + last timestamp)."""
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    result = []
    for p in sorted(SESSIONS_DIR.glob("*/turns.jsonl"), key=lambda x: x.stat().st_mtime, reverse=True):
        lines = p.read_text(encoding="utf-8").strip().splitlines()
        last_ts = ""
        if lines:
            try:
                last_ts = json.loads(lines[-1]).get("timestamp", "")
            except json.JSONDecodeError:
                pass
        result.append({
            "session_id": p.parent.name,
            "turns":      len(lines),
            "last_turn":  last_ts,
            "size_bytes": p.stat().st_size,
        })
    return result


def get_session(session_id: str) -> list[dict]:
    """Return all turns for a session."""
    path = _turns_path(session_id)
    if not path.exists():
        return []
    turns = []
    for line in path.read_text(encoding="utf-8").strip().splitlines():
        try:
            turns.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return turns
