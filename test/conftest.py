"""Shared fixtures for the phoebe test suite.

Covers two tiers:
- pure-unit tests: no stack running, no LLM, no network. Import modules directly.
- live tests: gated behind the ``live`` pytest marker. Set PHOEBE_LIVE=1 or run
  ``pytest -m live`` to include them. The default ``pytest`` run skips them.
"""

import json
import os
import sys
import time
from contextlib import contextmanager
from typing import Iterator

import httpx
import pytest

if "/app" not in sys.path:
    sys.path.insert(0, "/app")

BASE_URL = os.environ.get("PHOEBE_URL", "http://localhost:8090")


# ── Default: skip live tests unless -m live or PHOEBE_LIVE=1 ────────────────────

def pytest_collection_modifyitems(config, items):
    """Skip tests marked 'live' unless explicitly requested.

    Triggers that include live tests:
      - ``pytest -m live`` or ``-m "live or ..."`` via the -m option
      - ``PHOEBE_LIVE=1`` env var (useful in Makefile targets and CI)
    """
    m_option = config.getoption("-m", default="") or ""
    if "live" in m_option or os.environ.get("PHOEBE_LIVE"):
        return
    skip = pytest.mark.skip(reason="live test (run with PHOEBE_LIVE=1 or pytest -m live)")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip)


# ── Shared fixtures (live) ────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def base_url():
    return BASE_URL


@pytest.fixture(scope="session")
def client():
    with httpx.Client(base_url=BASE_URL, timeout=180) as c:
        yield c


# ── Live helpers (importable by test modules) ────────────────────────────────

def chat(client: httpx.Client, message: str, session_id: str | None = None, mode: str | None = None) -> dict:
    """Send a non-streaming chat message and return the parsed response."""
    body: dict = {"messages": [{"role": "user", "content": message}]}
    if session_id:
        body["session_id"] = session_id
    if mode:
        body["mode"] = mode
    r = client.post("/v1/chat/completions", json=body)
    r.raise_for_status()
    return r.json()


def answer(response: dict) -> str:
    return response["choices"][0]["message"]["content"]


def stream_chat(
    client: httpx.Client,
    message: str,
    session_id: str | None = None,
    mode: str | None = None,
    timeout: float = 120.0,
) -> dict:
    """Consume an SSE stream and return {events, final, session_id, usage}.

    Each event dict: {"event": str, "data": dict}. The terminal "done" event
    carries the final assistant text in ``data["response"]`` (when it exists).
    """
    body: dict = {"messages": [{"role": "user", "content": message}], "stream": True}
    if session_id:
        body["session_id"] = session_id
    if mode:
        body["mode"] = mode

    events: list[dict] = []
    final = ""
    resolved_sid = session_id or ""
    current_event = ""
    with client.stream("POST", "/v1/chat/completions", json=body, timeout=timeout) as r:
        for raw in r.iter_lines():
            if not raw:
                current_event = ""
                continue
            if raw.startswith("event: "):
                current_event = raw[len("event: "):].strip()
            elif raw.startswith("data: "):
                payload = raw[len("data: "):]
                try:
                    data = json.loads(payload)
                except json.JSONDecodeError:
                    data = {"_raw": payload}
                events.append({"event": current_event or "message", "data": data})
                if current_event == "done":
                    final = data.get("response", final) if isinstance(data, dict) else final
                    resolved_sid = (data.get("session_id") or resolved_sid) if isinstance(data, dict) else resolved_sid
    return {"events": events, "final": final, "session_id": resolved_sid}


def wait_for_active(client: httpx.Client, session_id: str, timeout: float = 10.0) -> bool:
    """Poll ``/v1/sessions/{sid}/active`` until it returns True or timeout."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = client.get(f"/v1/sessions/{session_id}/active", timeout=2.0)
            if r.status_code == 200 and r.json().get("active"):
                return True
        except Exception:
            pass
        time.sleep(0.1)
    return False


@contextmanager
def with_config_patch(client: httpx.Client, patch: dict) -> Iterator[dict]:
    """PATCH /config with ``patch``, yield the updated config, restore on exit.

    Restore is a best-effort second PATCH using the pre-patch values for the
    same top-level keys. Patches that fail schema validation raise HTTPStatusError.
    """
    pre = client.get("/config").json()
    client.patch("/config", json=patch).raise_for_status()
    try:
        updated = client.get("/config").json()
        yield updated
    finally:
        restore: dict = {}
        for key in patch:
            if key in pre:
                restore[key] = pre[key]
        if restore:
            try:
                client.patch("/config", json=restore)
            except Exception:
                pass
