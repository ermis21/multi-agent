"""
Pi.dev Remote Bridge — WebSocket server that accepts Pi RPC connections,
multiplexes them, and relays events/commands between Pi and Discord.

Architecture:
  Pi (any machine) ──WS──► Phoebe bridge (:9300) ──HTTP──► phoebe-discord

Each Pi connection is bound to a Discord channel. Events from Pi are rendered
as Discord messages. User messages in that channel are sent back to Pi as
prompt/steer commands.
"""

import asyncio
import json
import logging
import time
import uuid
from typing import Any

import httpx

logger = logging.getLogger("phoebe.remote.bridge")

# ── Connection registry ───────────────────────────────────────────────────────

# connection_id → PiConnection
_connections: dict[str, "PiConnection"] = {}
_next_conn_id = 0


def _make_conn_id() -> str:
    global _next_conn_id
    _next_conn_id += 1
    return f"pi_{_next_conn_id}_{uuid.uuid4().hex[:6]}"


# ── Discord gateway helpers ───────────────────────────────────────────────────

import os as _os
import socket as _socket

# Try multiple Discord hostnames (phoebe-discord-pycord is the active service)
_DISCORD_CANDIDATES = [
    _os.environ.get("DISCORD_URL", "http://phoebe-discord:4000"),
    "http://phoebe-discord-pycord:4000",
]

# Cache the resolved URL with a simple module-level variable
_cached_discord_url: str | None = None


def _resolve_discord_url() -> str:
    """Resolve Discord URL, trying multiple candidates."""
    global _cached_discord_url
    if _cached_discord_url is not None:
        return _cached_discord_url
    for url in _DISCORD_CANDIDATES:
        host = url.replace("http://", "").replace("https://", "").split(":")[0]
        try:
            _socket.getaddrinfo(host, 4000)
            _cached_discord_url = url
            return url
        except _socket.gaierror:
            continue
    # Fallback to first candidate (will retry on next call)
    return _DISCORD_CANDIDATES[0]


async def _discord_send(channel_id: int, content: str) -> bool:
    """Send a message to a Discord channel via the phoebe-discord HTTP gateway."""
    discord_url = _resolve_discord_url()
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(
                f"{discord_url}/discord/send",
                json={"channel_id": str(channel_id), "content": content},
            )
            return resp.status_code == 200
    except Exception as e:
        # If DNS fails, clear cache to force re-resolution next time
        global _cached_discord_url
        if "name resolution" in str(e).lower():
            _cached_discord_url = None
        logger.warning(f"Discord send failed for channel {channel_id}: {e}")
        return False


async def _discord_send_embed(channel_id: int, embed: dict) -> bool:
    """Send an embed message to a Discord channel."""
    discord_url = _resolve_discord_url()
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(
                f"{discord_url}/discord/send",
                json={"channel_id": str(channel_id), "embed": embed},
            )
            return resp.status_code == 200
    except Exception as e:
        global _cached_discord_url
        if "name resolution" in str(e).lower():
            _cached_discord_url = None
        logger.warning(f"Discord embed send failed for channel {channel_id}: {e}")
        return False


async def _rename_channel(channel_id: int, name: str) -> bool:
    """Rename a Discord channel."""
    discord_url = _resolve_discord_url()
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(
                f"{discord_url}/discord/edit_channel",
                json={"channel_id": str(channel_id), "name": name[:100]},
            )
            return resp.status_code == 200
    except Exception as e:
        global _cached_discord_url
        if "name resolution" in str(e).lower():
            _cached_discord_url = None
        logger.warning(f"Discord channel rename failed for channel {channel_id}: {e}")
        return False


# ── Event mapper: Pi RPC events → Discord messages ───────────────────────────

def _format_tool_args(args: dict) -> str:
    """Format tool arguments as a short preview."""
    if not args:
        return ""
    parts = []
    for k, v in list(args.items())[:3]:
        vs = str(v)
        if len(vs) > 60:
            vs = vs[:60] + "…"
        parts.append(f"{k}={vs}")
    return " ".join(parts)


def _extract_text(content) -> str:
    """Extract text from Pi content blocks."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                texts.append(block.get("text", ""))
        return "\n".join(texts)
    return str(content)


def _format_assistant_message(msg: dict) -> str:
    """Format a Pi assistant message for Discord."""
    content = msg.get("content", "")
    text = _extract_text(content)
    model = msg.get("model") or msg.get("provider", "?")
    if isinstance(model, dict):
        model = model.get("id", model.get("name", "?"))
    stop = msg.get("stopReason", "")
    usage = msg.get("usage", {})

    lines = []
    if text:
        lines.append(text)

    # Usage footer
    usage_parts = []
    if usage:
        inp = usage.get("input", 0)
        out = usage.get("output", 0)
        cost = usage.get("cost", {})
        if isinstance(cost, dict):
            total_cost = cost.get("total", 0)
            if total_cost:
                usage_parts.append(f"${total_cost:.4f}")
        if inp or out:
            usage_parts.append(f"{inp}in/{out}out")

    if usage_parts:
        lines.append(f"\n-# 📊 {' · '.join(usage_parts)}")

    return "\n".join(lines) if lines else "(empty response)"


class EventMapper:
    """Maps Pi RPC events to Discord message content."""

    def __init__(self, conn: "PiConnection"):
        self.conn = conn
        self._last_msg_text: str = ""
        self._last_msg_id: str | None = None
        self._current_tool_name: str = ""
        self._current_tool_call_id: str = ""
        self._thinking_text: str = ""
        self._turn_depth: int = 0

    async def handle_event(self, event: dict) -> None:
        """Process a single Pi RPC event and render to Discord."""
        event_type = event.get("type", "")
        channel_id = self.conn.channel_id

        try:
            if event_type == "agent_start":
                self.conn.is_streaming = True
                await _discord_send(channel_id, "-# 🚀 Agent started")

            elif event_type == "agent_end":
                self.conn.is_streaming = False
                await _discord_send(channel_id, "-# ✈️ Agent idle")

            elif event_type == "turn_start":
                self._turn_depth += 1

            elif event_type == "turn_end":
                msg = event.get("message", {})
                text = _format_assistant_message(msg)
                if text:
                    await _discord_send(channel_id, text)
                self._turn_depth = max(0, self._turn_depth - 1)

            elif event_type == "message_update":
                await self._handle_message_update(event)

            elif event_type == "tool_execution_start":
                tool_name = event.get("toolName", "?")
                args = event.get("args", {})
                self._current_tool_name = tool_name
                self._current_tool_call_id = event.get("toolCallId", "")
                args_preview = _format_tool_args(args)
                preview = f" {args_preview}" if args_preview else ""
                await _discord_send(channel_id, f"-# ⏳ `{tool_name}`{preview}…")

            elif event_type == "tool_execution_update":
                # Partial tool output — update in place if possible
                tool_name = event.get("toolName", self._current_tool_name)
                partial = event.get("partialResult", {})
                content = partial.get("content", [])
                text = _extract_text(content)
                if text:
                    preview = text.replace("\n", " ")[:200]
                    if len(text) > 200:
                        preview += "…"
                    await _discord_send(channel_id, f"-# 🔧 `{tool_name}`: {preview}")

            elif event_type == "tool_execution_end":
                tool_name = event.get("toolName", self._current_tool_name)
                is_error = event.get("isError", False)
                result = event.get("result", {})
                content = result.get("content", [])
                text = _extract_text(content)
                lines = text.count("\n") + 1 if text else 0
                icon = "❌" if is_error else "✅"
                preview = ""
                if text:
                    t = text.replace("\n", " ")[:120]
                    if len(text) > 120:
                        t += "…"
                    preview = f" {t}"
                await _discord_send(channel_id, f"-# {icon} `{tool_name}` ({lines} lines){preview}")
                self._current_tool_name = ""
                self._current_tool_call_id = ""

            elif event_type == "compaction_start":
                reason = event.get("reason", "manual")
                await _discord_send(channel_id, f"-# 📦 Compacting ({reason})…")

            elif event_type == "compaction_end":
                result = event.get("result", {})
                tokens_before = result.get("tokensBefore", 0) if result else 0
                aborted = event.get("aborted", False)
                icon = "❌" if aborted else "✅"
                await _discord_send(channel_id, f"-# {icon} Compacted (was {tokens_before:,} tok)")

            elif event_type == "queue_update":
                steering = event.get("steering", [])
                follow_up = event.get("followUp", [])
                parts = []
                if steering:
                    parts.append(f"{len(steering)} steering")
                if follow_up:
                    parts.append(f"{len(follow_up)} follow-up")
                if parts:
                    await _discord_send(channel_id, f"-# 📨 {', '.join(parts)} queued")

            elif event_type == "auto_retry_start":
                attempt = event.get("attempt", 1)
                max_att = event.get("maxAttempts", 3)
                error = event.get("errorMessage", "?")
                await _discord_send(channel_id, f"-# 🔄 Retry {attempt}/{max_att}: {error[:100]}")

            elif event_type == "auto_retry_end":
                success = event.get("success", False)
                attempt = event.get("attempt", 0)
                icon = "✅" if success else "❌"
                await _discord_send(channel_id, f"-# {icon} Retry {attempt} {'succeeded' if success else 'failed'}")

            elif event_type == "extension_ui_request":
                await self._handle_extension_ui_request(event)

            elif event_type == "extension_error":
                ext_path = event.get("extensionPath", "?")
                error = event.get("error", "?")
                await _discord_send(channel_id, f"-# ⚠️ Extension error ({ext_path}): {error[:200]}")

            elif event_type == "message_start":
                msg = event.get("message", {})
                # Capture model info from first assistant message
                if msg.get("role") == "assistant" and not self.conn._model:
                    self.conn._model = msg.get("model", "")
                    provider = msg.get("provider", "")
                    model_display = f"{provider}/{self.conn._model}" if provider else self.conn._model
                    await _discord_send(channel_id, f"-# 🤖 Model: {model_display}")

                # Rename channel based on first user prompt
                if msg.get("role") == "user" and not self.conn._session_name:
                    content = msg.get("content", [])
                    if content:
                        prompt = content[0].get("text", "") if isinstance(content[0], dict) else str(content[0])
                        # Truncate and sanitize for channel name
                        title = prompt.strip()[:50]
                        if len(title) > 100:
                            title = title[:97] + "…"
                        if title:
                            self.conn._session_name = title
                            await _rename_channel(channel_id, title)

            # message_end — informational, skip for Discord
        except Exception as e:
            logger.error(f"Event mapping failed for {event_type}: {e}", exc_info=True)

    async def _handle_message_update(self, event: dict) -> None:
        """Handle streaming message updates (text_delta, thinking_delta, etc.)."""
        delta_event = event.get("assistantMessageEvent", {})
        delta_type = delta_event.get("type", "")
        channel_id = self.conn.channel_id

        if delta_type == "text_delta":
            delta_text = delta_event.get("delta", "")
            if delta_text:
                self._last_msg_text += delta_text
                # Send text in chunks to avoid spamming Discord
                # For now, send the full accumulated text as a subtext line
                # In a full implementation, we'd edit the last message in-place
                preview = self._last_msg_text.replace("\n", " ")
                if len(preview) > 500:
                    preview = preview[:500] + "…"
                # Only send periodically to avoid spam
                if len(self._last_msg_text) % 200 < len(delta_text):
                    await _discord_send(channel_id, f"-# 💬 {preview}")

        elif delta_type == "thinking_delta":
            delta_text = delta_event.get("delta", "")
            if delta_text:
                self._thinking_text += delta_text

        elif delta_type == "thinking_end":
            if self._thinking_text:
                preview = self._thinking_text.replace("\n", " ")[:300]
                if len(self._thinking_text) > 300:
                    preview += "…"
                await _discord_send(channel_id, f"-# 🧠 thinking: {preview}")
                self._thinking_text = ""

        elif delta_type == "toolcall_start":
            tool_name = delta_event.get("partial", {}).get("name", "?")
            await _discord_send(channel_id, f"-# ⏳ `{tool_name}` starting…")

        elif delta_type == "toolcall_end":
            tool_call = delta_event.get("toolCall", {})
            tool_name = tool_call.get("name", "?")
            await _discord_send(channel_id, f"-# 🔧 `{tool_name}` args received")

        elif delta_type == "done":
            reason = delta_event.get("reason", "stop")
            if reason == "stop":
                await _discord_send(channel_id, "-# ✈️ Agent idle")
            elif reason == "error":
                await _discord_send(channel_id, "-# ❌ Agent error")
            elif reason == "aborted":
                await _discord_send(channel_id, "-# 🛑 Agent aborted")
            self._last_msg_text = ""
            self._thinking_text = ""

        elif delta_type == "error":
            reason = delta_event.get("reason", "?")
            await _discord_send(channel_id, f"-# ❌ Stream error: {reason}")

    async def _handle_extension_ui_request(self, event: dict) -> None:
        """Handle extension UI requests (select, confirm, input, notify, etc.)."""
        channel_id = self.conn.channel_id
        method = event.get("method", "")
        req_id = event.get("id", "")

        if method == "notify":
            msg = event.get("message", "")
            notify_type = event.get("notifyType", "info")
            icon = {"info": "ℹ️", "warning": "⚠️", "error": "❌"}.get(notify_type, "ℹ️")
            await _discord_send(channel_id, f"-# {icon} {msg}")

        elif method == "setStatus":
            status_text = event.get("statusText", "")
            status_key = event.get("statusKey", "")
            if status_text:
                await _discord_send(channel_id, f"-# 📌 [{status_key}] {status_text}")

        elif method == "setTitle":
            title = event.get("title", "")
            # Store for /remote status, don't spam Discord
            self.conn.metadata["title"] = title

        elif method in ("select", "confirm", "input", "editor"):
            # These are dialog methods — we need user interaction.
            # For now, log them. Full implementation would use Discord buttons.
            title = event.get("title", "")
            options = event.get("options", [])
            timeout = event.get("timeout", 30000)

            if method == "select" and options:
                options_str = "\n".join(f"  • {opt}" for opt in options)
                await _discord_send(
                    channel_id,
                    f"-# 🔘 **{title}**\n{options_str}\n"
                    f"_Use `/remote ui-select {req_id} <option>` to respond_"
                )
                # Store the request for later response
                self.conn.pending_ui[req_id] = {"method": method, "options": options, "timeout": timeout}

            elif method == "confirm":
                message = event.get("message", "")
                await _discord_send(
                    channel_id,
                    f"-# 🔘 **{title}**\n{message}\n"
                    f"_Use `/remote ui-confirm {req_id} <yes|no>` to respond_"
                )
                self.conn.pending_ui[req_id] = {"method": method, "timeout": timeout}

            elif method == "input":
                placeholder = event.get("placeholder", "")
                await _discord_send(
                    channel_id,
                    f"-# 📝 **{title}**\n_placeholder: {placeholder}_\n"
                    f"_Use `/remote ui-input {req_id} <value>` to respond_"
                )
                self.conn.pending_ui[req_id] = {"method": method, "timeout": timeout}

            elif method == "editor":
                prefill = event.get("prefill", "")[:500]
                await _discord_send(
                    channel_id,
                    f"-# 📝 **{title}**\n"
                    f"_Use `/remote ui-editor {req_id} <text>` to respond_"
                )
                self.conn.pending_ui[req_id] = {"method": method, "timeout": timeout}


# ── Pi Connection ─────────────────────────────────────────────────────────────

class PiConnection:
    """Represents a single Pi.dev RPC connection."""

    def __init__(
        self,
        ws,
        channel_id: int,
        metadata: dict | None = None,
    ):
        self.ws = ws
        self.channel_id = channel_id
        self.conn_id = _make_conn_id()
        self.metadata = metadata or {}
        self.event_mapper = EventMapper(self)
        self.pending_ui: dict[str, dict] = {}  # req_id → ui request
        self._is_streaming = False
        self._is_compacting = False
        self._session_id: str | None = None
        self._session_name: str | None = None
        self._model: str | None = None
        self._created_at = time.time()
        self._last_activity = time.time()
        self._running = True
        self._read_task: asyncio.Task | None = None
        self._write_task: asyncio.Task | None = None

    @property
    def is_active(self) -> bool:
        return self._running and self.ws is not None

    async def start(self) -> None:
        """Start the read/write loop for this connection."""
        self._read_task = asyncio.create_task(self._read_loop(), name=f"pi_read_{self.conn_id}")
        self._write_task = asyncio.create_task(self._write_loop(), name=f"pi_write_{self.conn_id}")

    async def stop(self) -> None:
        """Stop the connection gracefully."""
        self._running = False
        if self._read_task:
            self._read_task.cancel()
        if self._write_task:
            self._write_task.cancel()
        try:
            if self.ws:
                await self.ws.close(code=1000, reason="Phoebe disconnecting")
        except Exception:
            pass
        if self.conn_id in _connections:
            del _connections[self.conn_id]

    async def send_command(self, command: dict) -> None:
        """Send an RPC command to Pi."""
        command["id"] = f"phoebe_{uuid.uuid4().hex[:8]}"
        line = json.dumps(command, ensure_ascii=False) + "\n"
        try:
            if self.ws:
                await self.ws.send_text(line)
            self._last_activity = time.time()
        except Exception as e:
            logger.error(f"Failed to send command to Pi {self.conn_id}: {e}")
            await self.stop()

    async def send_ui_response(self, req_id: str, response: dict) -> None:
        """Send an extension UI response to Pi."""
        resp = {"type": "extension_ui_response", "id": req_id, **response}
        line = json.dumps(resp, ensure_ascii=False) + "\n"
        try:
            if self.ws:
                await self.ws.send_text(line)
        except Exception as e:
            logger.error(f"Failed to send UI response to Pi {self.conn_id}: {e}")

    async def _read_loop(self) -> None:
        """Read messages from Pi and dispatch events."""
        try:
            async for message in self.ws.iter_text():
                self._last_activity = time.time()
                if not message.strip():
                    continue
                try:
                    event = json.loads(message)
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON from Pi {self.conn_id}: {message[:200]}")
                    continue

                # Track state from events
                if event.get("type") == "agent_start":
                    self._is_streaming = True
                elif event.get("type") == "agent_end":
                    self._is_streaming = False
                elif event.get("type") == "compaction_start":
                    self._is_compacting = True
                elif event.get("type") == "compaction_end":
                    self._is_compacting = False

                # Track state from responses
                if event.get("type") == "response" and event.get("command") == "get_state":
                    data = event.get("data", {})
                    self._is_streaming = data.get("isStreaming", False)
                    self._is_compacting = data.get("isCompacting", False)
                    self._session_id = data.get("sessionId")
                    self._session_name = data.get("sessionName")
                    model = data.get("model")
                    if model:
                        if isinstance(model, dict):
                            self._model = model.get("id", model.get("name", "?"))
                        else:
                            self._model = str(model)

                # Dispatch to event mapper
                await self.event_mapper.handle_event(event)

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"Read loop error for Pi {self.conn_id}: {e}")
        finally:
            await self.stop()

    async def _write_loop(self) -> None:
        """Keep the connection alive. Commands are sent via send_command()."""
        try:
            while self._running:
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            raise

    def get_status(self) -> dict:
        """Get current connection status."""
        return {
            "conn_id": self.conn_id,
            "channel_id": self.channel_id,
            "is_streaming": self._is_streaming,
            "is_compacting": self._is_compacting,
            "session_id": self._session_id,
            "session_name": self._session_name,
            "model": self._model,
            "created_at": self._created_at,
            "last_activity": self._last_activity,
            "pending_ui": list(self.pending_ui.keys()),
            "metadata": self.metadata,
        }


# ── WebSocket handler ─────────────────────────────────────────────────────────

async def handle_pi_connection(
    ws,
    channel_id: int,
    metadata: dict | None = None,
) -> PiConnection:
    """Handle a new Pi.dev WebSocket connection.

    Returns the PiConnection object. Callers should keep a reference
    to manage the connection lifecycle.
    """
    conn = PiConnection(ws, channel_id, metadata)
    _connections[conn.conn_id] = conn
    await conn.start()
    logger.info(f"Pi connection {conn.conn_id} established for channel {channel_id}")

    # Send initial context to Discord
    await _discord_send(channel_id, f"-# 🔗 Pi connected  id=`{conn.conn_id}`")
    await _discord_send(channel_id, "-# 💡 Send messages here to control Pi. Pi will respond in this channel.")
    await _discord_send(channel_id, "-# 📖 Use `/remote` slash commands for status, abort, bash, model, etc.")

    return conn


# ── Bridge API ────────────────────────────────────────────────────────────────

def get_connection(conn_id: str) -> PiConnection | None:
    """Get a connection by ID."""
    return _connections.get(conn_id)


def get_connection_for_channel(channel_id: int) -> PiConnection | None:
    """Get the active connection for a Discord channel."""
    for conn in _connections.values():
        if conn.channel_id == channel_id and conn.is_active:
            return conn
    return None


def list_connections() -> list[dict]:
    """List all active connections."""
    return [c.get_status() for c in _connections.values() if c.is_active]


def get_all_connections() -> list[PiConnection]:
    """Get all active connection objects."""
    return [c for c in _connections.values() if c.is_active]
