"""
JSONL session logger.

Each session lives in its own directory: `{sessions_dir}/{session_id}/turns.jsonl`.
Each line in `turns.jsonl` is one turn (JSON object). Sidecars (state.json,
approvals.jsonl, tool_errors.jsonl, active.jsonl) live beside it.
"""

import json
import os
import tempfile
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

    def truncate_to_final(self, target_turn_index: int) -> tuple[list[dict], list[dict]]:
        """Drop the `target_turn_index`-th final turn and everything after it.

        Worker/supervisor attempts that led into the dropped final are part of
        its "attempt chain" and are dropped with it — they sit between the
        previous final and the target final. Atomic rewrite via tmp + rename.

        Returns `(surviving_turns, dropped_turns)` as parsed dicts so callers
        can recompute aggregated state (stats, message_index, …).

        Raises ValueError if target_turn_index is negative or beyond the
        number of final turns in the log.
        """
        if target_turn_index < 0:
            raise ValueError(f"target_turn_index must be >= 0 (got {target_turn_index})")
        if not self.path.exists():
            raise ValueError("turns.jsonl does not exist")

        raw_lines = self.path.read_text(encoding="utf-8").splitlines()
        parsed: list[tuple[str, dict]] = []
        for line in raw_lines:
            if not line.strip():
                continue
            try:
                parsed.append((line, json.loads(line)))
            except json.JSONDecodeError:
                # Preserve corrupt lines on their original side of the cut:
                # attribute them to the surviving slice only if they precede
                # the target final's attempt chain.
                parsed.append((line, {}))

        final_count = 0
        cut_idx: int | None = None  # index of first line to DROP
        last_final_idx = -1
        for idx, (_, obj) in enumerate(parsed):
            if obj.get("role") != "final":
                continue
            if final_count == target_turn_index:
                # Drop from the line after the previous final (start of this
                # final's attempt chain).
                cut_idx = last_final_idx + 1
                break
            final_count += 1
            last_final_idx = idx

        if cut_idx is None:
            raise ValueError(
                f"target_turn_index={target_turn_index} out of range "
                f"(log has {final_count} final turn(s))"
            )

        surviving_lines = [ln for ln, _ in parsed[:cut_idx]]
        surviving_turns = [obj for _, obj in parsed[:cut_idx] if obj]
        dropped_turns = [obj for _, obj in parsed[cut_idx:] if obj]

        # Atomic write: tmp file in the same dir, then os.replace().
        sdir = _session_dir(self.session_id)
        fd, tmp_name = tempfile.mkstemp(prefix=".turns.", suffix=".tmp", dir=str(sdir))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                for line in surviving_lines:
                    f.write(line + "\n")
            os.replace(tmp_name, self.path)
        except Exception:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
            raise

        return surviving_turns, dropped_turns


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
