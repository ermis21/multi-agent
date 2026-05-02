"""Tests for Pi.dev Remote Control bridge (app/remote/bridge.py).

Unit tests — no live stack required. Tests event mapping, connection registry,
and command formatting.
"""

import asyncio
import json
import pytest

from app.remote.bridge import (
    EventMapper,
    PiConnection,
    get_connection,
    get_connection_for_channel,
    list_connections,
    _format_tool_args,
    _extract_text,
    _format_assistant_message,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

class FakeWS:
    """Minimal WebSocket mock for testing."""

    def __init__(self):
        self.sent = []
        self.closed = False

    async def send_text(self, text):
        self.sent.append(text)

    async def close(self, code=1000, reason=""):
        self.closed = True

    async def iter_text(self):
        return iter([])


class FakeConn:
    """Minimal PiConnection mock for EventMapper tests."""

    def __init__(self, channel_id=123456):
        self.channel_id = channel_id
        self.metadata = {}
        self.pending_ui = {}


# ── Event mapper tests ──────────────────────────────────────────────────────

class TestEventMapper:
    def test_format_tool_args_empty(self):
        assert _format_tool_args({}) == ""

    def test_format_tool_args_single(self):
        result = _format_tool_args({"file": "test.py"})
        assert "file=test.py" in result

    def test_format_tool_args_truncates(self):
        long_val = "x" * 100
        result = _format_tool_args({"data": long_val})
        assert "…" in result

    def test_extract_text_string(self):
        assert _extract_text("hello") == "hello"

    def test_extract_text_list(self):
        blocks = [{"type": "text", "text": "hello"}, {"type": "text", "text": "world"}]
        assert _extract_text(blocks) == "hello\nworld"

    def test_extract_text_mixed(self):
        blocks = [{"type": "text", "text": "hello"}, {"type": "image"}]
        assert _extract_text(blocks) == "hello"

    def test_format_assistant_message_simple(self):
        msg = {"content": "Hello world"}
        result = _format_assistant_message(msg)
        assert "Hello world" in result

    def test_format_assistant_message_with_usage(self):
        msg = {
            "content": "Done",
            "usage": {"input": 100, "output": 50},
        }
        result = _format_assistant_message(msg)
        assert "100in/50out" in result

    def test_format_assistant_message_with_cost(self):
        msg = {
            "content": "Done",
            "usage": {"input": 100, "output": 50, "cost": {"total": 0.005}},
        }
        result = _format_assistant_message(msg)
        assert "$" in result


# ── Connection registry tests ───────────────────────────────────────────────

class TestConnectionRegistry:
    @pytest.fixture(autouse=True)
    def cleanup(self):
        """Clear connections after each test."""
        from app.remote.bridge import _connections
        yield
        _connections.clear()

    def test_get_connection_empty(self):
        assert get_connection("nonexistent") is None

    def test_get_connection_for_channel_empty(self):
        assert get_connection_for_channel(999) is None

    def test_list_connections_empty(self):
        assert list_connections() == []

    def test_connection_lifecycle(self):
        ws = FakeWS()
        conn = PiConnection(ws, channel_id=123)
        assert conn.conn_id.startswith("pi_")
        assert conn.channel_id == 123
        assert conn.is_active

    def test_connection_status(self):
        ws = FakeWS()
        conn = PiConnection(ws, channel_id=456)
        status = conn.get_status()
        assert status["channel_id"] == 456
        assert status["is_streaming"] is False
        assert status["is_compacting"] is False
        assert "conn_id" in status
        assert "created_at" in status


# ── Command formatting tests ────────────────────────────────────────────────

class TestCommandFormatting:
    def test_prompt_command(self):
        cmd = {"type": "prompt", "message": "hello"}
        assert cmd["type"] == "prompt"
        assert cmd["message"] == "hello"

    def test_steer_command(self):
        cmd = {"type": "steer", "message": "try a different approach"}
        assert cmd["type"] == "steer"

    def test_bash_command(self):
        cmd = {"type": "bash", "command": "ls -la"}
        assert cmd["type"] == "bash"
        assert cmd["command"] == "ls -la"

    def test_set_model_command(self):
        cmd = {"type": "set_model", "provider": "anthropic", "modelId": "claude-sonnet-4-20250514"}
        assert cmd["type"] == "set_model"
        assert cmd["provider"] == "anthropic"


# ── Async tests (run via asyncio.run) ────────────────────────────────────────

def test_pi_connection_send_command():
    async def _run():
        ws = FakeWS()
        conn = PiConnection(ws, channel_id=789)
        await conn.send_command({"type": "prompt", "message": "test"})
        assert len(ws.sent) == 1
        sent = json.loads(ws.sent[0])
        assert sent["type"] == "prompt"
        assert sent["message"] == "test"
        assert "id" in sent  # auto-generated
    asyncio.run(_run())


def test_pi_connection_send_ui_response():
    async def _run():
        ws = FakeWS()
        conn = PiConnection(ws, channel_id=789)
        await conn.send_ui_response("req123", {"value": "selected"})
        assert len(ws.sent) == 1
        sent = json.loads(ws.sent[0])
        assert sent["type"] == "extension_ui_response"
        assert sent["id"] == "req123"
        assert sent["value"] == "selected"
    asyncio.run(_run())


def test_pi_connection_stop():
    async def _run():
        ws = FakeWS()
        conn = PiConnection(ws, channel_id=789)
        await conn.stop()
        assert not conn.is_active
        assert ws.closed
    asyncio.run(_run())
