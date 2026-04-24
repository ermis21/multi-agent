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

def test_run_dream_cron_invokes_runner_with_rolling_window(monkeypatch):
    """Cron uses a rolling 24h window, not a UTC calendar day.

    "Dream about what just happened" — dreaming about yesterday's UTC slice
    misses late-night activity of the most recent local day. Rolling window
    ending at cron fire time is what we want.
    """
    from app import main

    captured: dict = {}

    async def fake_run_dream(date_iso=None, *, window_hours=None, end_ts=None,
                             interrupt_event=None, meta_enabled=True, top_k=3,
                             **_kwargs):
        captured["date_iso"] = date_iso
        captured["window_hours"] = window_hours
        captured["has_event"] = interrupt_event is not None
        return {"date": "2026-04-23", "session_ids_seen": [],
                "session_ids_completed": [], "interrupted_at": None}

    from app.dream import runner
    monkeypatch.setattr(runner, "run_dream", fake_run_dream)

    asyncio.run(main._run_dream_cron())
    assert captured["date_iso"] is None
    assert captured["window_hours"] == 24
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

    async def fake_run_dream(date_iso=None, *, window_hours=None, end_ts=None,
                             interrupt_event=None, meta_enabled=True, top_k=3,
                             **_kwargs):
        captured["date_iso"] = date_iso
        captured["window_hours"] = window_hours
        captured["meta_enabled"] = meta_enabled
        return {"ok": True, "date": date_iso, "session_ids_seen": []}

    from app.dream import runner
    monkeypatch.setattr(runner, "run_dream", fake_run_dream)

    client = TestClient(main.app)
    r = client.post(
        "/internal/dream-run",
        json={"date": "2026-04-19", "meta_enabled": False, "verbose": False},
    )
    assert r.status_code == 200
    payload = r.json()
    assert payload["status"] == "ok"
    assert payload["date"] == "2026-04-19"
    assert captured["date_iso"] == "2026-04-19"
    # Explicit date → calendar-day mode; no rolling window applied.
    assert captured["window_hours"] is None
    assert captured["meta_enabled"] is False


def test_internal_dream_run_defaults_to_rolling_window(monkeypatch):
    """No body → rolling 24h window, not UTC calendar yesterday."""
    from app import main

    captured: dict = {}

    async def fake_run_dream(date_iso=None, *, window_hours=None, end_ts=None,
                             interrupt_event=None, meta_enabled=True, top_k=3,
                             **_kwargs):
        captured["date_iso"] = date_iso
        captured["window_hours"] = window_hours
        return {"ok": True, "date": "2026-04-23", "session_ids_seen": []}

    from app.dream import runner
    monkeypatch.setattr(runner, "run_dream", fake_run_dream)

    client = TestClient(main.app)
    r = client.post("/internal/dream-run", json={"verbose": False})
    assert r.status_code == 200
    assert captured["date_iso"] is None
    assert captured["window_hours"] == 24.0


def test_internal_dream_run_accepts_conversation_sids(monkeypatch):
    """Body-level sid list overrides window/date and is forwarded verbatim."""
    from app import main

    captured: dict = {}

    async def fake_run_dream(date_iso=None, *, window_hours=None, end_ts=None,
                             conversation_sids=None, dreamer_model_override=None,
                             interrupt_event=None, meta_enabled=True, top_k=3,
                             **_kwargs):
        captured["conversation_sids"] = conversation_sids
        captured["dreamer_model_override"] = dreamer_model_override
        captured["window_hours"] = window_hours
        captured["date_iso"] = date_iso
        return {"ok": True, "date": "2026-04-23", "session_ids_seen": []}

    from app.dream import runner
    monkeypatch.setattr(runner, "run_dream", fake_run_dream)

    client = TestClient(main.app)
    r = client.post("/internal/dream-run", json={
        "verbose": False,
        "conversation_sids": ["discord_1_a", "discord_2_b"],
        "dreamer_model": "claude_opus_4_7",
    })
    assert r.status_code == 200
    assert captured["conversation_sids"] == ["discord_1_a", "discord_2_b"]
    assert captured["dreamer_model_override"] == "claude_opus_4_7"
    # Explicit sids → window/date must not also be passed.
    assert captured["window_hours"] is None
    assert captured["date_iso"] is None


def test_internal_dream_candidates_window(monkeypatch, tmp_path):
    """Preview endpoint returns candidate metadata + channel_id parse."""
    from app import main
    from app.dream import phrase_store, session_iter

    root = tmp_path / "sessions"
    root.mkdir()
    monkeypatch.setattr(phrase_store, "STATE_DIR", tmp_path)
    monkeypatch.setattr(session_iter, "SESSIONS_ROOT", root)

    # Seed one dreamable + one skipped session, both inside the default 24h
    # window (use "now" ISO).
    now_iso = datetime.now(timezone.utc).isoformat()
    for sid, role in [("discord_999_abc", "worker"), ("test_harness_x", "worker")]:
        d = root / sid
        d.mkdir()
        (d / "state.json").write_text(json.dumps({
            "agent_role": role, "mode": "converse",
            "source_trigger": {"type": "user", "ref": None},
        }))
        (d / "turns.jsonl").write_text(json.dumps({
            "role": "final", "timestamp": now_iso,
            "messages": [], "response": "ok",
        }) + "\n")

    client = TestClient(main.app)
    r = client.post("/internal/dream-candidates", json={"window_hours": 1})
    assert r.status_code == 200
    payload = r.json()
    assert payload["scope"]["mode"] == "window"
    sids = [c["session_id"] for c in payload["candidates"]]
    assert "discord_999_abc" in sids
    assert "test_harness_x" not in sids
    skipped_sids = [c["session_id"] for c in payload["skipped"]]
    assert "test_harness_x" in skipped_sids
    discord_row = next(c for c in payload["candidates"] if c["session_id"] == "discord_999_abc")
    assert discord_row["channel_id"] == "999"


def test_internal_dream_models_lists_defaults_and_overrides(monkeypatch):
    """Model endpoint returns current default + named overrides."""
    from app import main
    from app.config_loader import get_config

    fake_cfg = {
        "llm":   {"model": "base-model"},
        "dream": {"model": "dream-pinned"},
        "models": {
            "alias_a": {"model": "real-a"},
            "alias_b": {"model": "real-b"},
        },
    }
    monkeypatch.setattr(main, "get_config", lambda: fake_cfg)

    client = TestClient(main.app)
    r = client.get("/internal/dream-models")
    assert r.status_code == 200
    data = r.json()
    assert data["default"] == "dream-pinned"
    names = [o["name"] for o in data["options"]]
    assert "dream-pinned" in names
    assert "alias_a" in names
    assert "alias_b" in names
    assert "base-model" in names
    # default must be first so the picker can highlight it.
    assert names[0] == "dream-pinned"


def test_internal_dream_run_accepts_custom_window_hours(monkeypatch):
    from app import main

    captured: dict = {}

    async def fake_run_dream(date_iso=None, *, window_hours=None, end_ts=None,
                             interrupt_event=None, meta_enabled=True, top_k=3,
                             **_kwargs):
        captured["window_hours"] = window_hours
        return {"ok": True, "date": "2026-04-23", "session_ids_seen": []}

    from app.dream import runner
    monkeypatch.setattr(runner, "run_dream", fake_run_dream)

    client = TestClient(main.app)
    r = client.post("/internal/dream-run", json={"verbose": False, "window_hours": 6})
    assert r.status_code == 200
    assert captured["window_hours"] == 6.0


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
