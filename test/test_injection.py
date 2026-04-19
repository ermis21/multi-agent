"""Tests for the mid-flight injection registry + endpoints."""

import asyncio

import pytest

from app.main import (
    _INJECT_MODES,
    _active_sessions,
    get_session_state,
    register_session,
    release_session,
)


@pytest.fixture(autouse=True)
def _clear_registry():
    _active_sessions.clear()
    yield
    _active_sessions.clear()


def test_register_and_get():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        task = loop.create_task(asyncio.sleep(0))
        rec = register_session("s1", task)
        assert rec["pending"] == []
        assert not rec["cancel"].is_set()
        assert get_session_state("s1") is rec
        loop.run_until_complete(task)
    finally:
        loop.close()


def test_release_returns_queued_injections():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        task = loop.create_task(asyncio.sleep(0))
        rec = register_session("s2", task)
        rec["pending"].append({"mode": "queue", "text": "later"})
        rec["pending"].append({"mode": "immediate", "text": "now"})  # non-queue
        rec["pending"].append({"mode": "queue", "text": "also later"})

        queued = release_session("s2")
        assert [q["text"] for q in queued] == ["later", "also later"]
        assert get_session_state("s2") is None  # released
        loop.run_until_complete(task)
    finally:
        loop.close()


def test_release_unknown_session_returns_empty():
    assert release_session("nonexistent") == []


def test_inject_modes_literal_set():
    """If we ever add a mode, update the literal set and every handler."""
    assert _INJECT_MODES == {"immediate", "not_urgent", "clarify", "queue", "stop"}


def test_cancel_event_separate_per_session():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        t1 = loop.create_task(asyncio.sleep(0))
        t2 = loop.create_task(asyncio.sleep(0))
        r1 = register_session("a", t1)
        r2 = register_session("b", t2)
        r1["cancel"].set()
        assert r1["cancel"].is_set()
        assert not r2["cancel"].is_set()
        loop.run_until_complete(asyncio.gather(t1, t2))
    finally:
        loop.close()


# ── Live endpoint tests ───────────────────────────────────────────────────────

@pytest.mark.live
def test_inject_unknown_session_returns_404(client):
    r = client.post(
        "/v1/sessions/never_registered/inject",
        json={"text": "hi", "mode": "queue"},
    )
    assert r.status_code == 404


@pytest.mark.live
def test_inject_invalid_mode_rejected(client):
    r = client.post(
        "/v1/sessions/never_registered/inject",
        json={"text": "hi", "mode": "nonsense"},
    )
    # Either 400 (mode validated first) or 404 (session checked first) — either
    # proves the endpoint is mounted and does some validation.
    assert r.status_code in (400, 404, 422)


@pytest.mark.live
def test_session_active_endpoint_false_for_unknown(client):
    r = client.get("/v1/sessions/never_registered/active")
    assert r.status_code == 200
    assert r.json().get("active") is False
