"""
One-shot migration from flat `sessions/{sid}.*` files to folder-per-session
`sessions/{sid}/{file}` layout.

Invoked once from `app.main.lifespan` at startup, before the scheduler starts
(so no in-flight writers exist). Idempotent: re-running on an already-migrated
directory is a no-op.

File moves:
    sessions/{sid}.jsonl              -> sessions/{sid}/turns.jsonl
    sessions/{sid}.state.json         -> sessions/{sid}/state.json
    sessions/{sid}.approvals.jsonl    -> sessions/{sid}/approvals.jsonl
    sessions/{sid}.tool_errors.jsonl  -> sessions/{sid}/tool_errors.jsonl
    sessions/{sid}.active.jsonl       -> sessions/{sid}/active.jsonl
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

_SIDECARS = {
    ".approvals.jsonl":    "approvals.jsonl",
    ".tool_errors.jsonl":  "tool_errors.jsonl",
    ".active.jsonl":       "active.jsonl",
    ".state.json":         "state.json",
}


def _sid_for_flat_file(name: str) -> str | None:
    """Return the session_id a flat filename belongs to, or None if it's not a flat sidecar/turnlog.

    Matches:
        {sid}.jsonl               -> sid
        {sid}.state.json          -> sid
        {sid}.approvals.jsonl     -> sid
        {sid}.tool_errors.jsonl   -> sid
        {sid}.active.jsonl        -> sid
    Rejects hidden/temp files (dotfiles) and anything already under a subdir.
    """
    if name.startswith("."):
        return None
    # Check longer suffixes first so `.approvals.jsonl` wins over `.jsonl`.
    for suffix in (".approvals.jsonl", ".tool_errors.jsonl", ".active.jsonl", ".state.json"):
        if name.endswith(suffix):
            return name[: -len(suffix)] or None
    if name.endswith(".jsonl"):
        return name[: -len(".jsonl")] or None
    return None


def _target_name(flat_name: str) -> str:
    """Given a flat filename, return the target basename inside the session folder."""
    for suffix, target in _SIDECARS.items():
        if flat_name.endswith(suffix):
            return target
    # Plain turn log
    return "turns.jsonl"


def _rewrite_history_pointer(state_path: Path, session_id: str) -> None:
    """Update history.full in state.json to the new layout. Best-effort."""
    try:
        data = json.loads(state_path.read_text(encoding="utf-8"))
    except Exception:
        return
    hist = data.get("history") or {}
    expected = f"sessions/{session_id}/turns.jsonl"
    if hist.get("full") == expected:
        return
    hist["full"] = expected
    data["history"] = hist
    state_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def migrate_flat_to_folders(sessions_dir: Path) -> dict:
    """Move flat `sessions/{sid}.*` files into `sessions/{sid}/` folders.

    Returns a summary dict: {"moved": N, "sessions": M, "skipped_existing": K}.
    """
    sessions_dir = Path(sessions_dir)
    if not sessions_dir.exists():
        return {"moved": 0, "sessions": 0, "skipped_existing": 0}

    moved = 0
    skipped = 0
    touched_sids: set[str] = set()

    for p in sorted(sessions_dir.iterdir()):
        if not p.is_file():
            continue
        sid = _sid_for_flat_file(p.name)
        if sid is None:
            continue
        target_dir = sessions_dir / sid
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / _target_name(p.name)
        if target.exists():
            # Already migrated or manually resolved — leave the flat file alone.
            skipped += 1
            continue
        shutil.move(str(p), str(target))
        moved += 1
        touched_sids.add(sid)

    # Rewrite history.full pointer on every state.json we just placed.
    for sid in touched_sids:
        state_path = sessions_dir / sid / "state.json"
        if state_path.exists():
            _rewrite_history_pointer(state_path, sid)

    return {"moved": moved, "sessions": len(touched_sids), "skipped_existing": skipped}
