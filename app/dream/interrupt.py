"""Watch for real user activity during a dream run.

`UserActivityWatcher` polls `app.main._active_sessions` on a tick. Any active
session whose `source_trigger.type == "user"` and whose start timestamp is
*after* the dream run started is a real user typing in — we abort.

The watcher doesn't cancel in-flight agent calls; it only sets an
`asyncio.Event` that the runner checks between sessions. That gives the
per-conversation `run_agent_role` finally-block time to roll back any
pending batch cleanly.

Public API:
  - `class UserActivityWatcher(poll_interval_s=2.0)`
  - `.event` — asyncio.Event; set when user activity is detected
  - `async def start()` — spawns the background task
  - `async def stop()`  — cancels the background task
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Iterable


def _session_started_after(session_dict: dict, cutoff_iso: str) -> bool:
    """True when the session's start ISO timestamp is strictly after `cutoff_iso`.

    The `_active_sessions` entry shape isn't formally specified — we accept
    either a `created_at` string, a `start_iso` string, or fall back to True
    (session exists and we have no timestamp → treat as recent).
    """
    ts = (session_dict.get("created_at")
          or session_dict.get("start_iso")
          or session_dict.get("started_at"))
    if not ts:
        return True
    try:
        return ts > cutoff_iso
    except TypeError:
        return True


def detect_user_activity(active_sessions: dict | None, run_started_iso: str) -> bool:
    """Return True if any active session looks like a real user message
    that started after `run_started_iso`.

    `active_sessions` is the shape of `app.main._active_sessions` — a mapping
    `sid → state_dict`. `state_dict` usually carries a `source_trigger`
    (`{"type": "user" | "cron" | "api" | "sub_agent", ...}`) in its
    persistent state; at runtime the dict itself may or may not carry it
    depending on the code path. We tolerate both.
    """
    if not active_sessions:
        return False
    for sid, rec in active_sessions.items():
        if not isinstance(rec, dict):
            continue
        # Skip any session spawned by the dream pipeline.
        if sid.startswith(("dreamer_", "meta_dreamer_")):
            continue
        trig = rec.get("source_trigger")
        if isinstance(trig, dict) and trig.get("type") == "user":
            if _session_started_after(rec, run_started_iso):
                return True
        # Some paths don't stamp source_trigger on the runtime dict — fall
        # back to the persistent state file if a session_state loader is
        # available. We keep this simple: the runtime dict is authoritative,
        # and if an "unknown origin" session exists we err on the safe side.
        if "cancel" in rec and trig is None and _session_started_after(rec, run_started_iso):
            # Runtime session with unknown trigger → treat as user.
            return True
    return False


class UserActivityWatcher:
    """Background asyncio task that flips `.event` when a user message arrives."""

    def __init__(self, poll_interval_s: float = 2.0,
                 active_sessions_source=None):
        self.poll_interval_s = float(poll_interval_s)
        self.event: asyncio.Event = asyncio.Event()
        self.started_iso: str = datetime.now(timezone.utc).isoformat()
        self._task: asyncio.Task | None = None
        self._active_sessions_source = active_sessions_source

    def _current_active(self) -> dict:
        if self._active_sessions_source is not None:
            src = self._active_sessions_source
            if callable(src):
                try:
                    return src() or {}
                except Exception:
                    return {}
            return src or {}
        # Default: import lazily so tests can avoid pulling app.main.
        try:
            from app.main import _active_sessions
            return _active_sessions or {}
        except Exception:
            return {}

    async def _poll(self) -> None:
        try:
            while not self.event.is_set():
                if detect_user_activity(self._current_active(), self.started_iso):
                    self.event.set()
                    return
                await asyncio.sleep(self.poll_interval_s)
        except asyncio.CancelledError:
            return

    async def start(self) -> None:
        if self._task is not None and not self._task.done():
            return
        self.started_iso = datetime.now(timezone.utc).isoformat()
        self.event = asyncio.Event()
        self._task = asyncio.create_task(self._poll())

    async def stop(self) -> None:
        if self._task is not None and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
        self._task = None
