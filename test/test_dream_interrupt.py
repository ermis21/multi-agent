"""Tests for app.dream.interrupt — UserActivityWatcher + detect_user_activity.

Covers:
  - detect_user_activity returns False when no active sessions.
  - detect_user_activity returns True for a user-triggered session started
    after the dream cutoff.
  - detect_user_activity ignores dreamer-spawned sessions (sid prefix).
  - detect_user_activity ignores sub-agent and cron sessions.
  - _session_started_after defaults True when timestamp is missing (be safe).
  - UserActivityWatcher.start/stop lifecycle + .event fires on detection.
  - UserActivityWatcher uses a callable source when provided.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from app.dream import interrupt


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat()


# ── detect_user_activity ─────────────────────────────────────────────────────

def test_detect_returns_false_when_empty():
    cutoff = _iso(datetime.now(timezone.utc))
    assert interrupt.detect_user_activity({}, cutoff) is False
    assert interrupt.detect_user_activity(None, cutoff) is False


def test_detect_returns_true_for_user_session_started_after_cutoff():
    cutoff = _iso(datetime.now(timezone.utc) - timedelta(minutes=10))
    active = {
        "user_sid_123": {
            "source_trigger": {"type": "user", "ref": "discord:999"},
            "created_at": _iso(datetime.now(timezone.utc)),
        }
    }
    assert interrupt.detect_user_activity(active, cutoff) is True


def test_detect_ignores_dreamer_sessions():
    cutoff = _iso(datetime.now(timezone.utc) - timedelta(minutes=10))
    active = {
        "dreamer_2026-04-21_foo": {
            "source_trigger": {"type": "user"},  # even if mis-stamped
            "created_at": _iso(datetime.now(timezone.utc)),
        },
        "meta_dreamer_xyz": {
            "source_trigger": {"type": "user"},
            "created_at": _iso(datetime.now(timezone.utc)),
        },
    }
    assert interrupt.detect_user_activity(active, cutoff) is False


def test_detect_ignores_non_user_triggers():
    cutoff = _iso(datetime.now(timezone.utc) - timedelta(minutes=10))
    now_iso = _iso(datetime.now(timezone.utc))
    active = {
        "cron_soul": {"source_trigger": {"type": "cron"}, "created_at": now_iso},
        "api_svc": {"source_trigger": {"type": "api"}, "created_at": now_iso},
        "child_session": {"source_trigger": {"type": "sub_agent"}, "created_at": now_iso},
    }
    assert interrupt.detect_user_activity(active, cutoff) is False


def test_detect_ignores_user_session_that_predates_cutoff():
    cutoff = _iso(datetime.now(timezone.utc))
    old = _iso(datetime.now(timezone.utc) - timedelta(hours=2))
    active = {
        "old_user": {"source_trigger": {"type": "user"}, "created_at": old},
    }
    assert interrupt.detect_user_activity(active, cutoff) is False


def test_detect_unknown_trigger_with_cancel_event_is_flagged():
    """A runtime session with no source_trigger but a cancel Event — err on
    the safe side and treat as user activity."""
    cutoff = _iso(datetime.now(timezone.utc) - timedelta(minutes=10))
    active = {
        "unknown_sid": {
            "cancel": asyncio.Event(),
            "created_at": _iso(datetime.now(timezone.utc)),
        }
    }
    assert interrupt.detect_user_activity(active, cutoff) is True


def test_detect_session_started_after_missing_timestamp_defaults_true():
    """Missing timestamp means we can't rule it out — be safe."""
    rec = {"source_trigger": {"type": "user"}}  # no created_at
    assert interrupt._session_started_after(rec, "2026-04-21T04:00:00+00:00") is True


# ── UserActivityWatcher ──────────────────────────────────────────────────────

def test_watcher_start_stop_no_activity():
    async def _run():
        watcher = interrupt.UserActivityWatcher(
            poll_interval_s=0.02,
            active_sessions_source=lambda: {},
        )
        await watcher.start()
        await asyncio.sleep(0.05)
        flag = watcher.event.is_set()
        await watcher.stop()
        return flag
    assert asyncio.run(_run()) is False


def test_watcher_flips_event_on_user_session():
    sessions: dict = {}

    def src():
        return sessions

    async def _run():
        watcher = interrupt.UserActivityWatcher(
            poll_interval_s=0.02, active_sessions_source=src,
        )
        await watcher.start()
        await asyncio.sleep(0.05)
        sessions["new_user"] = {
            "source_trigger": {"type": "user"},
            "created_at": _iso(datetime.now(timezone.utc)),
        }
        for _ in range(25):
            if watcher.event.is_set():
                break
            await asyncio.sleep(0.02)
        flag = watcher.event.is_set()
        await watcher.stop()
        return flag
    assert asyncio.run(_run()) is True


def test_watcher_start_is_idempotent():
    async def _run():
        watcher = interrupt.UserActivityWatcher(
            poll_interval_s=0.02, active_sessions_source=lambda: {},
        )
        await watcher.start()
        first_task = watcher._task
        await watcher.start()  # no-op while first still running
        same = watcher._task is first_task
        await watcher.stop()
        return same
    assert asyncio.run(_run()) is True


def test_watcher_stop_is_idempotent():
    async def _run():
        watcher = interrupt.UserActivityWatcher(
            poll_interval_s=0.02, active_sessions_source=lambda: {},
        )
        await watcher.start()
        await watcher.stop()
        await watcher.stop()  # no crash
    asyncio.run(_run())
