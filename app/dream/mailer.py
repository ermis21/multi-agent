"""Daily dream-digest email with Discord fallback.

Cascade (env-driven):
  1. Gmail via SMTP_SSL 465 — if `DREAM_GMAIL_USER` + `DREAM_GMAIL_APP_PASSWORD`.
  2. Generic SMTP — if `DREAM_SMTP_HOST/PORT/USER/PASS/FROM`.
  3. Discord fallback — `discord_send` to `cfg.dream.email.fallback_channel_id`.
  4. Last-ditch: write `state/dream/runs/<date>/email_failed.txt` with traceback.

Subject: `Phoebe Dream Digest — YYYY-MM-DD`.
Body: `state/dream/reports/<date>.md` when present, else a generated summary
from `run.json`.
Attachment: unified diff over phrase histories for that run (difflib-based,
no git dependency).

Credentials are **env-only**. Config carries addresses + toggles only.
"""

from __future__ import annotations

import json
import logging
import os
import ssl
import traceback
from datetime import datetime, timezone
from email.message import EmailMessage
from pathlib import Path

from app.dream import phrase_store

logger = logging.getLogger("dream.mailer")

REPORTS_ROOT = phrase_store.STATE_DIR / "dream" / "reports"
RUNS_ROOT = phrase_store.STATE_DIR / "dream" / "runs"


# ── Body + attachment rendering ──────────────────────────────────────────────

def _render_body_from_run(record: dict) -> str:
    """Synthesize a plain-text body from a `run.json` record when no curated
    report markdown is available."""
    date = record.get("date") or "?"
    seen = record.get("session_ids_seen") or []
    done = record.get("session_ids_completed") or []
    convs = record.get("conversations") or []
    committed_total = sum(len(c.get("committed") or []) for c in convs)
    flagged_total = sum(len(c.get("flagged") or []) for c in convs)
    lines = [
        f"Phoebe Dream Digest — {date}",
        "",
        f"Sessions seen: {len(seen)}",
        f"Sessions completed: {len(done)}",
        f"Conversations with dream pass: {len(convs)}",
        f"Committed phrase edits: {committed_total}",
        f"Phrases flagged during run: {flagged_total}",
    ]
    if record.get("interrupted_at"):
        lines.append(f"Interrupted at: {record['interrupted_at']} (user activity detected)")
    meta = record.get("meta") or {}
    if meta.get("status"):
        lines.append(f"Meta-dreamer: {meta['status']}")
    lines.append("")
    lines.append("Per-conversation status:")
    for c in convs:
        lines.append(
            f"  {c.get('conversation_sid')}: {c.get('status')}  "
            f"committed={len(c.get('committed') or [])} flagged={len(c.get('flagged') or [])}"
        )
    return "\n".join(lines)


def render_digest_body(date_iso: str) -> str:
    """Digest body. Prefer a curated report markdown; fall back to run.json."""
    report = REPORTS_ROOT / f"{date_iso}.md"
    if report.exists():
        try:
            return report.read_text(encoding="utf-8")
        except OSError:
            pass
    run_json = RUNS_ROOT / date_iso / "run.json"
    if run_json.exists():
        try:
            record = json.loads(run_json.read_text(encoding="utf-8"))
            return _render_body_from_run(record)
        except (OSError, json.JSONDecodeError):
            pass
    return f"Phoebe Dream Digest — {date_iso}\n\n(no data found)"


def render_digest_diff(date_iso: str) -> str:
    """Unified diff of phrase_history edits landed on `date_iso`.

    We walk `state/dream/phrase_history/*.jsonl` and collect rows whose
    `run_date` starts with `date_iso`. For each, emit a simple unified-diff
    style block using difflib.unified_diff over `old_text` → `new_text`.
    """
    import difflib
    out_lines: list[str] = [f"Phrase-history diff — {date_iso}", ""]
    hist_root = phrase_store.HISTORY_DIR
    if not hist_root.exists():
        out_lines.append("(no phrase history directory)")
        return "\n".join(out_lines)
    any_entries = False
    for f in sorted(hist_root.glob("*.jsonl")):
        try:
            rows = [json.loads(line) for line in f.read_text(encoding="utf-8").splitlines() if line.strip()]
        except (OSError, json.JSONDecodeError):
            continue
        for row in rows:
            rd = str(row.get("run_date") or "")
            if not rd.startswith(date_iso):
                continue
            any_entries = True
            header = (f"--- {f.stem}  (rev={row.get('rev') or row.get('version')})\n"
                      f"+++ {row.get('role_template_name') or '?'}"
                      f"  section={row.get('section_breadcrumb') or ''}")
            out_lines.append(header)
            diff = difflib.unified_diff(
                (row.get("old_text") or "").splitlines(keepends=False),
                (row.get("new_text") or "").splitlines(keepends=False),
                lineterm="",
                n=2,
            )
            out_lines.extend(diff)
            out_lines.append("")
    if not any_entries:
        out_lines.append("(no phrase-history entries on this date)")
    return "\n".join(out_lines)


# ── Transports ───────────────────────────────────────────────────────────────

class MailerError(RuntimeError):
    pass


def _build_message(to_addr: str, from_addr: str, subject: str,
                   body: str, attachment_name: str,
                   attachment_text: str) -> EmailMessage:
    msg = EmailMessage()
    msg["From"] = from_addr
    msg["To"] = to_addr
    msg["Subject"] = subject
    msg.set_content(body)
    msg.add_attachment(
        attachment_text.encode("utf-8"),
        maintype="text", subtype="plain", filename=attachment_name,
    )
    return msg


def _send_gmail(msg: EmailMessage, *, user: str, password: str,
                smtp_lib=None) -> None:
    """Gmail via SMTP_SSL. `smtp_lib` is an injection point for tests."""
    import smtplib
    cls = smtp_lib or smtplib.SMTP_SSL
    ctx = ssl.create_default_context()
    with cls("smtp.gmail.com", 465, context=ctx) as s:
        s.login(user, password)
        s.send_message(msg)


def _send_smtp(msg: EmailMessage, *, host: str, port: int, user: str | None,
               password: str | None, smtp_lib=None) -> None:
    """Generic SMTP. When port==465 we use SMTP_SSL; else STARTTLS on 587."""
    import smtplib
    if port == 465:
        cls = smtp_lib or smtplib.SMTP_SSL
        ctx = ssl.create_default_context()
        with cls(host, port, context=ctx) as s:
            if user and password:
                s.login(user, password)
            s.send_message(msg)
    else:
        cls = smtp_lib or smtplib.SMTP
        with cls(host, port) as s:
            try:
                s.starttls()
            except Exception:
                pass
            if user and password:
                s.login(user, password)
            s.send_message(msg)


async def _send_discord_fallback(channel_id: str, body: str,
                                 attachment_name: str,
                                 attachment_text: str) -> None:
    """Fallback transport: post body to Discord channel via MCP discord_send."""
    from app.mcp_client import call_tool
    # Attach diff as a code block below the body — the discord_send tool
    # doesn't support file attachments uniformly across channel types.
    composed = f"{body}\n\n```\n{attachment_text[:1800]}\n```"
    result = await call_tool(
        "discord_send",
        {"channel_id": str(channel_id), "content": composed[:3800]},
        ["discord_send"], "build", [],
    )
    if "error" in (result or {}):
        raise MailerError(f"discord fallback failed: {result.get('error')}")


# ── Public API ───────────────────────────────────────────────────────────────

async def send_digest(
    date_iso: str,
    cfg: dict,
    *,
    smtp_lib=None,
    now: datetime | None = None,
) -> dict:
    """Send the digest for `date_iso`. Returns `{transport, ok, detail}`.

    `smtp_lib` is an injection point so tests can pass a mock SMTP class.
    """
    email_cfg = ((cfg or {}).get("dream") or {}).get("email") or {}
    to_addr = email_cfg.get("to") or "ekatsaounis@uth.gr"
    fallback_channel_id = email_cfg.get("fallback_channel_id")

    subject = f"Phoebe Dream Digest — {date_iso}"
    body = render_digest_body(date_iso)
    diff = render_digest_diff(date_iso)
    attach_name = f"dream_diff_{date_iso}.txt"

    # Credentials env-only.
    gmail_user = os.environ.get("DREAM_GMAIL_USER")
    gmail_pw   = os.environ.get("DREAM_GMAIL_APP_PASSWORD")
    smtp_host  = os.environ.get("DREAM_SMTP_HOST")
    smtp_port  = os.environ.get("DREAM_SMTP_PORT")
    smtp_user  = os.environ.get("DREAM_SMTP_USER")
    smtp_pass  = os.environ.get("DREAM_SMTP_PASS")
    smtp_from  = os.environ.get("DREAM_SMTP_FROM")

    last_error: Exception | None = None

    # 1. Gmail
    if gmail_user and gmail_pw:
        try:
            msg = _build_message(to_addr, gmail_user, subject, body, attach_name, diff)
            _send_gmail(msg, user=gmail_user, password=gmail_pw, smtp_lib=smtp_lib)
            return {"transport": "gmail", "ok": True, "to": to_addr}
        except Exception as e:
            last_error = e
            logger.warning("gmail transport failed: %s", e)

    # 2. Generic SMTP
    if smtp_host and smtp_port:
        try:
            msg = _build_message(to_addr, smtp_from or (smtp_user or "dream@localhost"),
                                 subject, body, attach_name, diff)
            _send_smtp(msg, host=smtp_host, port=int(smtp_port),
                       user=smtp_user, password=smtp_pass, smtp_lib=smtp_lib)
            return {"transport": "smtp", "ok": True, "to": to_addr}
        except Exception as e:
            last_error = e
            logger.warning("smtp transport failed: %s", e)

    # 3. Discord fallback
    if fallback_channel_id:
        try:
            await _send_discord_fallback(str(fallback_channel_id), body, attach_name, diff)
            return {"transport": "discord", "ok": True, "channel_id": str(fallback_channel_id)}
        except Exception as e:
            last_error = e
            logger.warning("discord fallback failed: %s", e)

    # 4. Last resort — disk note.
    try:
        fail_dir = RUNS_ROOT / date_iso
        fail_dir.mkdir(parents=True, exist_ok=True)
        trace = ""
        if last_error is not None:
            trace = "".join(traceback.format_exception(type(last_error), last_error,
                                                       last_error.__traceback__))
        (fail_dir / "email_failed.txt").write_text(
            f"failed to send digest for {date_iso} at {(now or datetime.now(timezone.utc)).isoformat()}\n\n"
            f"{trace}\n",
            encoding="utf-8",
        )
    except Exception:
        pass
    return {"transport": "none", "ok": False,
            "error": f"{type(last_error).__name__}: {last_error}" if last_error
                     else "no transport configured"}
