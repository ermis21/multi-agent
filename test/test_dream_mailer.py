"""Tests for app.dream.mailer — Gmail/SMTP/Discord/disk cascade.

Covers:
  - render_digest_body prefers curated report markdown over run.json.
  - render_digest_body falls back to run.json synthesis.
  - render_digest_body falls back to a placeholder when neither exists.
  - render_digest_diff walks phrase_history for `run_date` matches.
  - send_digest chooses Gmail when DREAM_GMAIL_USER + DREAM_GMAIL_APP_PASSWORD set.
  - send_digest falls back to generic SMTP when Gmail env absent.
  - send_digest falls back to Discord when no SMTP env.
  - send_digest writes email_failed.txt when all transports fail.
  - Gmail SMTP class receives login() and send_message() calls.
"""

from __future__ import annotations

import asyncio
import json
from email.message import EmailMessage
from pathlib import Path

import pytest

from app.dream import mailer, phrase_store


@pytest.fixture
def rootfs(tmp_path, monkeypatch):
    state = tmp_path / "state"
    runs = state / "dream" / "runs"
    reports = state / "dream" / "reports"
    history = state / "dream" / "phrase_history"
    for p in (runs, reports, history):
        p.mkdir(parents=True)

    monkeypatch.setattr(phrase_store, "STATE_DIR", state)
    monkeypatch.setattr(phrase_store, "HISTORY_DIR", history)
    monkeypatch.setattr(mailer, "RUNS_ROOT", runs)
    monkeypatch.setattr(mailer, "REPORTS_ROOT", reports)

    # Clear env so tests pick explicit transport per scenario.
    for k in ("DREAM_GMAIL_USER", "DREAM_GMAIL_APP_PASSWORD",
              "DREAM_SMTP_HOST", "DREAM_SMTP_PORT",
              "DREAM_SMTP_USER", "DREAM_SMTP_PASS", "DREAM_SMTP_FROM"):
        monkeypatch.delenv(k, raising=False)

    return {"state": state, "runs": runs, "reports": reports, "history": history}


# ── render_digest_body ───────────────────────────────────────────────────────

def test_render_body_prefers_curated_report(rootfs):
    (rootfs["reports"] / "2026-04-20.md").write_text(
        "# Curated report body", encoding="utf-8",
    )
    out = mailer.render_digest_body("2026-04-20")
    assert "Curated report body" in out


def test_render_body_falls_back_to_run_json(rootfs):
    run_dir = rootfs["runs"] / "2026-04-20"
    run_dir.mkdir()
    (run_dir / "run.json").write_text(json.dumps({
        "date": "2026-04-20",
        "session_ids_seen": ["a", "b"],
        "session_ids_completed": ["a"],
        "conversations": [{
            "conversation_sid": "a",
            "status": "finalized",
            "committed": [{"phrase_id": "ph-1"}],
            "flagged": [],
        }],
        "meta": {"status": "no_conflicts"},
    }), encoding="utf-8")
    out = mailer.render_digest_body("2026-04-20")
    assert "Phoebe Dream Digest — 2026-04-20" in out
    assert "Sessions seen: 2" in out
    assert "Sessions completed: 1" in out
    assert "Committed phrase edits: 1" in out
    assert "Meta-dreamer: no_conflicts" in out


def test_render_body_no_data_placeholder(rootfs):
    out = mailer.render_digest_body("2026-04-20")
    assert "no data found" in out


# ── render_digest_diff ───────────────────────────────────────────────────────

def test_render_diff_collects_matching_run_date(rootfs):
    hf = rootfs["history"] / "ph-aaaaaaaaaa.jsonl"
    hf.write_text(json.dumps({
        "rev": 1,
        "role_template_name": "worker_full",
        "section_breadcrumb": "Rules",
        "old_text": "alpha\nbeta", "new_text": "alpha\ngamma",
        "run_date": "2026-04-20T04:03:12+00:00",
    }) + "\n" + json.dumps({
        "rev": 2,
        "role_template_name": "worker_full",
        "section_breadcrumb": "Rules",
        "old_text": "x", "new_text": "y",
        "run_date": "2025-01-01T00:00:00+00:00",  # different date — ignored
    }) + "\n", encoding="utf-8")
    out = mailer.render_digest_diff("2026-04-20")
    assert "Phrase-history diff — 2026-04-20" in out
    assert "ph-aaaaaaaaaa" in out
    # difflib unified_diff emits -beta / +gamma somewhere.
    assert "-beta" in out
    assert "+gamma" in out


def test_render_diff_empty_date(rootfs):
    out = mailer.render_digest_diff("2026-04-20")
    assert "no phrase-history entries on this date" in out


# ── send_digest transport cascade ────────────────────────────────────────────

class _FakeSMTP:
    """Context-manager stub capturing login() + send_message() calls."""
    instances: list["_FakeSMTP"] = []

    def __init__(self, host, port, context=None):
        self.host = host
        self.port = port
        self.context = context
        self.logged_in: tuple | None = None
        self.sent: list[EmailMessage] = []
        _FakeSMTP.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def login(self, user, password):
        self.logged_in = (user, password)

    def send_message(self, msg: EmailMessage):
        self.sent.append(msg)

    def starttls(self):
        pass


def test_send_digest_uses_gmail_when_env_set(rootfs, monkeypatch):
    _FakeSMTP.instances.clear()
    monkeypatch.setenv("DREAM_GMAIL_USER", "phoebe.dream@gmail.com")
    monkeypatch.setenv("DREAM_GMAIL_APP_PASSWORD", "appppppppp")

    cfg = {"dream": {"email": {"to": "ekatsaounis@uth.gr",
                               "fallback_channel_id": "999"}}}
    result = asyncio.run(mailer.send_digest("2026-04-20", cfg, smtp_lib=_FakeSMTP))

    assert result["transport"] == "gmail"
    assert result["ok"] is True
    assert result["to"] == "ekatsaounis@uth.gr"
    assert len(_FakeSMTP.instances) == 1
    inst = _FakeSMTP.instances[0]
    assert inst.host == "smtp.gmail.com"
    assert inst.port == 465
    assert inst.logged_in == ("phoebe.dream@gmail.com", "appppppppp")
    assert len(inst.sent) == 1
    msg = inst.sent[0]
    assert msg["To"] == "ekatsaounis@uth.gr"
    assert msg["From"] == "phoebe.dream@gmail.com"
    assert "2026-04-20" in msg["Subject"]


def test_send_digest_uses_smtp_when_gmail_absent(rootfs, monkeypatch):
    _FakeSMTP.instances.clear()
    monkeypatch.setenv("DREAM_SMTP_HOST", "smtp.example.com")
    monkeypatch.setenv("DREAM_SMTP_PORT", "465")
    monkeypatch.setenv("DREAM_SMTP_USER", "dream@example.com")
    monkeypatch.setenv("DREAM_SMTP_PASS", "sekret")
    monkeypatch.setenv("DREAM_SMTP_FROM", "phoebe@example.com")

    cfg = {"dream": {"email": {"to": "user@uth.gr"}}}
    result = asyncio.run(mailer.send_digest("2026-04-20", cfg, smtp_lib=_FakeSMTP))

    assert result["transport"] == "smtp"
    assert result["ok"] is True
    inst = _FakeSMTP.instances[0]
    assert inst.host == "smtp.example.com"
    assert inst.port == 465
    assert inst.logged_in == ("dream@example.com", "sekret")
    assert inst.sent[0]["From"] == "phoebe@example.com"


def test_send_digest_falls_back_to_discord(rootfs, monkeypatch):
    """No SMTP env → Discord fallback via call_tool."""
    calls: list[dict] = []

    async def fake_call_tool(method, params, allowed, mode, approved):
        calls.append({"method": method, "params": params})
        return {"ok": True}

    import app.mcp_client as mc
    monkeypatch.setattr(mc, "call_tool", fake_call_tool)

    cfg = {"dream": {"email": {"to": "user@uth.gr",
                               "fallback_channel_id": "chan-123"}}}
    result = asyncio.run(mailer.send_digest("2026-04-20", cfg))

    assert result["transport"] == "discord"
    assert result["ok"] is True
    assert result["channel_id"] == "chan-123"
    assert len(calls) == 1
    assert calls[0]["method"] == "discord_send"
    assert calls[0]["params"]["channel_id"] == "chan-123"
    assert "Phoebe Dream Digest" in calls[0]["params"]["content"]


def test_send_digest_writes_email_failed_when_all_fail(rootfs, monkeypatch):
    """SMTP raises, Discord raises, no fallback_channel_id even — disk note written."""
    class ExplodingSMTP(_FakeSMTP):
        def __enter__(self):
            raise RuntimeError("smtp down")

    monkeypatch.setenv("DREAM_SMTP_HOST", "smtp.example.com")
    monkeypatch.setenv("DREAM_SMTP_PORT", "465")

    async def boom_call_tool(*_a, **_kw):
        raise RuntimeError("discord api down")

    import app.mcp_client as mc
    monkeypatch.setattr(mc, "call_tool", boom_call_tool)

    cfg = {"dream": {"email": {"to": "user@uth.gr",
                               "fallback_channel_id": "chan-1"}}}
    result = asyncio.run(
        mailer.send_digest("2026-04-20", cfg, smtp_lib=ExplodingSMTP),
    )

    assert result["transport"] == "none"
    assert result["ok"] is False
    fail_file = rootfs["runs"] / "2026-04-20" / "email_failed.txt"
    assert fail_file.exists()
    contents = fail_file.read_text(encoding="utf-8")
    assert "2026-04-20" in contents


def test_send_digest_no_transport_configured(rootfs, monkeypatch):
    """No env, no fallback_channel_id → disk note with 'no transport' error."""
    cfg = {"dream": {"email": {"to": "user@uth.gr"}}}
    result = asyncio.run(mailer.send_digest("2026-04-20", cfg))
    assert result["transport"] == "none"
    assert result["ok"] is False
    assert "no transport configured" in result["error"]


# ── SMTP STARTTLS path ───────────────────────────────────────────────────────

def test_send_smtp_starttls_path(rootfs, monkeypatch):
    _FakeSMTP.instances.clear()
    monkeypatch.setenv("DREAM_SMTP_HOST", "smtp.example.com")
    monkeypatch.setenv("DREAM_SMTP_PORT", "587")
    monkeypatch.setenv("DREAM_SMTP_USER", "u")
    monkeypatch.setenv("DREAM_SMTP_PASS", "p")

    cfg = {"dream": {"email": {"to": "user@uth.gr"}}}
    result = asyncio.run(mailer.send_digest("2026-04-20", cfg, smtp_lib=_FakeSMTP))

    assert result["transport"] == "smtp"
    inst = _FakeSMTP.instances[0]
    assert inst.port == 587
    assert inst.logged_in == ("u", "p")
