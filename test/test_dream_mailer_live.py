"""Live integration tests for the dream-digest emailing system.

Exercises `POST /internal/dream-digest` against the running stack. The endpoint
runs the full mailer.send_digest cascade (Gmail → SMTP → Discord → disk note)
and we assert it returns a well-formed result dict regardless of which
transport happened to fire.

We deliberately target an ancient date with no on-disk data so:
  - the body synthesises to "no data found" (harmless if a real transport is
    configured and actually delivers),
  - no real phrase-history diff content leaks into a test email,
  - the endpoint stays fast (no run.json parsing, no large diff).

Part of `make test-integration` / `make test-full` — skipped by default
`make test` / `make test-fast`.
"""

from __future__ import annotations

import uuid

import pytest

pytestmark = pytest.mark.live


_PROBE_DATE = "1999-12-31"  # ancient; guaranteed no dream artefacts
_EXPECTED_TRANSPORTS = {"gmail", "smtp", "discord", "none"}


def test_dream_digest_endpoint_reachable(client):
    """Endpoint exists, returns 200, and produces the documented shape."""
    r = client.post("/internal/dream-digest", json={"date": _PROBE_DATE})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body.get("status") == "ok"
    assert body.get("date") == _PROBE_DATE
    result = body.get("result") or {}
    assert "transport" in result, f"missing transport in result: {result}"
    assert result["transport"] in _EXPECTED_TRANSPORTS, \
        f"unexpected transport: {result['transport']}"
    assert "ok" in result


def test_dream_digest_defaults_to_yesterday(client):
    """Omitting `date` defaults to yesterday's UTC date."""
    import datetime as _dt
    expected = (_dt.datetime.now(_dt.timezone.utc).date()
                - _dt.timedelta(days=1)).isoformat()
    r = client.post("/internal/dream-digest", json={})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body.get("status") == "ok"
    assert body.get("date") == expected


def test_dream_digest_unknown_transport_writes_disk_note(client):
    """When no transport is configured, the cascade lands on the disk note.

    We PATCH config to clear `dream.email.fallback_channel_id` so the Discord
    rung can't succeed for this probe. Gmail/SMTP creds come from env and are
    outside our reach — if they're set on this stack the test falls through
    to the transport=="gmail"/"smtp" branches and we skip the disk-note check.
    """
    pre = client.get("/config").json()
    pre_fallback = (pre.get("dream") or {}).get("email", {}).get("fallback_channel_id")
    patched = client.patch("/config", json={
        "dream": {"email": {"fallback_channel_id": None}},
    })
    patched.raise_for_status()
    try:
        r = client.post("/internal/dream-digest",
                        json={"date": f"1999-12-{uuid.uuid4().int % 28 + 1:02d}"})
        r.raise_for_status()
        result = r.json()["result"]
        if result["transport"] != "none":
            pytest.skip(f"real transport configured ({result['transport']}); "
                        "disk-note path not exercised")
        assert result["ok"] is False
        assert "error" in result
    finally:
        client.patch("/config", json={
            "dream": {"email": {"fallback_channel_id": pre_fallback}},
        })


def test_dream_digest_body_survives_missing_date(client):
    """Render-body path handles a date with no run.json and no report.md —
    the endpoint must NOT raise; it synthesises a minimal placeholder body."""
    r = client.post("/internal/dream-digest", json={"date": _PROBE_DATE})
    assert r.status_code == 200, r.text
    # We can't read the body text directly from this endpoint (the mailer
    # returns only the transport result). The assertion is that the endpoint
    # completes — a 500 or timeout here means render_digest_body crashed.
    assert r.json().get("status") == "ok"
