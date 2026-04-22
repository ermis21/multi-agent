"""Tests for dream cron wiring + /internal/dream-* endpoints.

Covers:
  - _run_dream_cron computes yesterday's date and passes it to runner.run_dream.
  - _run_dream_cron passes the _active_sessions-backed watcher's event.
  - _run_dream_cron catches runner exceptions so the scheduler loop stays healthy.
  - _run_dream_digest_cron awaits mailer.send_digest with yesterday's date + cfg.
  - /internal/dream-run accepts an explicit date override.
  - /internal/dream-run defaults to yesterday when no body.
  - /internal/dream-digest accepts an explicit date override.
"""

from __future__ import annotations

import asyncio
import json
from datetime import date, datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient


# ── _run_dream_cron ──────────────────────────────────────────────────────────

def test_run_dream_cron_invokes_runner_with_yesterday(monkeypatch):
    from app import main

    captured: dict = {}

    async def fake_run_dream(date_iso, *, interrupt_event=None, meta_enabled=True, top_k=3):
        captured["date"] = date_iso
        captured["has_event"] = interrupt_event is not None
        return {"session_ids_seen": [], "session_ids_completed": [],
                "interrupted_at": None}

    from app.dream import runner
    monkeypatch.setattr(runner, "run_dream", fake_run_dream)

    asyncio.run(main._run_dream_cron())
    expected = (datetime.now(timezone.utc).date() - timedelta(days=1)).isoformat()
    assert captured["date"] == expected
    assert captured["has_event"] is True


def test_run_dream_cron_swallows_runner_errors(monkeypatch, capsys):
    """Cron loop must not die on a bad dream — just log + move on."""
    from app import main

    async def boom(*_a, **_kw):
        raise RuntimeError("runner kaboom")

    from app.dream import runner
    monkeypatch.setattr(runner, "run_dream", boom)

    asyncio.run(main._run_dream_cron())
    out = capsys.readouterr().out
    assert "dream run" in out
    assert "failed" in out
    assert "runner kaboom" in out


def test_run_dream_digest_cron_invokes_mailer(monkeypatch):
    from app import main

    captured: dict = {}

    async def fake_send(date_iso, cfg, *, smtp_lib=None, now=None):
        captured["date"] = date_iso
        captured["cfg_has_dream"] = "dream" in (cfg or {})
        return {"transport": "gmail", "ok": True}

    from app.dream import mailer
    monkeypatch.setattr(mailer, "send_digest", fake_send)

    asyncio.run(main._run_dream_digest_cron())
    expected = (datetime.now(timezone.utc).date() - timedelta(days=1)).isoformat()
    assert captured["date"] == expected


# ── /internal/dream-run ──────────────────────────────────────────────────────

def test_internal_dream_run_accepts_explicit_date(monkeypatch):
    from app import main

    captured: dict = {}

    async def fake_run_dream(date_iso, *, interrupt_event=None, meta_enabled=True, top_k=3):
        captured["date"] = date_iso
        captured["meta_enabled"] = meta_enabled
        return {"ok": True, "session_ids_seen": []}

    from app.dream import runner
    monkeypatch.setattr(runner, "run_dream", fake_run_dream)

    client = TestClient(main.app)
    r = client.post("/internal/dream-run", json={"date": "2026-04-19", "meta_enabled": False})
    assert r.status_code == 200
    payload = r.json()
    assert payload["status"] == "ok"
    assert payload["date"] == "2026-04-19"
    assert captured["date"] == "2026-04-19"
    assert captured["meta_enabled"] is False


def test_internal_dream_run_defaults_to_yesterday(monkeypatch):
    from app import main

    captured: dict = {}

    async def fake_run_dream(date_iso, *, interrupt_event=None, meta_enabled=True, top_k=3):
        captured["date"] = date_iso
        return {"ok": True, "session_ids_seen": []}

    from app.dream import runner
    monkeypatch.setattr(runner, "run_dream", fake_run_dream)

    client = TestClient(main.app)
    r = client.post("/internal/dream-run")
    assert r.status_code == 200
    expected = (datetime.now(timezone.utc).date() - timedelta(days=1)).isoformat()
    assert captured["date"] == expected


def test_internal_dream_digest_accepts_explicit_date(monkeypatch):
    from app import main

    captured: dict = {}

    async def fake_send(date_iso, cfg, *, smtp_lib=None, now=None):
        captured["date"] = date_iso
        return {"transport": "gmail", "ok": True}

    from app.dream import mailer
    monkeypatch.setattr(mailer, "send_digest", fake_send)

    client = TestClient(main.app)
    r = client.post("/internal/dream-digest", json={"date": "2026-04-19"})
    assert r.status_code == 200
    assert r.json()["date"] == "2026-04-19"
    assert captured["date"] == "2026-04-19"
