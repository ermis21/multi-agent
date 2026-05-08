"""Phase 2 — sandbox/ keeps a paste-copy of app/safety.strip_injection_tokens
because it can't import from app/. This test asserts the key regex tokens are
present in both copies, so a fix in one is forced into the other.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import app.safety as safety


_KEY_TOKENS = (
    r"<\|tool_call",
    r"user_(?:note|interjection|clarification)",
    "[redacted-injection-token]",
)


def _read_sandbox_source() -> str | None:
    # /Phebe is the read-only project mount inside phoebe-api; the host repo
    # path is the natural fallback for local pytest runs outside docker.
    for cand in (Path("/Phebe/sandbox/mcp_server.py"),
                 Path(__file__).resolve().parent.parent / "sandbox/mcp_server.py",
                 Path("/host/sandbox/mcp_server.py")):
        if cand.exists():
            return cand.read_text()
    return None


def test_app_safety_has_all_tokens():
    src = Path(safety.__file__).read_text()
    for token in _KEY_TOKENS:
        assert token in src, f"app/safety.py missing token: {token!r}"


def test_sandbox_paste_copy_has_same_tokens():
    src = _read_sandbox_source()
    if src is None:
        pytest.skip("sandbox source not reachable from this test runner")
    for token in _KEY_TOKENS:
        assert token in src, f"sandbox/mcp_server.py paste-copy missing token: {token!r}"
