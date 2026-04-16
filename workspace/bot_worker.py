"""
Worker Discord bot — bidirectional: listens in the mab-worker channel and
routes messages to mab-api, then posts the response back.

Each bot-created channel gets its own session_id for conversation continuity.
Legacy (non-bot-created) channels use a per-user+channel session_id.

Slash commands: /new, /mode, /model, /status, /help
"""

import asyncio
import json
import re
import time
import os
from datetime import datetime, timezone
from pathlib import Path
from pathlib import Path

import discord
from discord import app_commands
import httpx

from utils import is_allowed, split_message

MAB_API_URL      = os.environ.get("MAB_API_URL",             "http://mab-api:8090")
WORKER_TOKEN     = os.environ.get("DISCORD_TOKEN_WORKER",   "")
GUILD_ID         = int(os.environ.get("DISCORD_GUILD_ID",   "0"))
WORKER_NICKNAME  = os.environ.get("DISCORD_WORKER_NICKNAME", "Gemma")

# Channel IDs the worker bot listens in — populated dynamically on_ready
# (env var is an optional static fallback)
WORKER_CHANNEL_IDS: set[int] = {
    int(c) for c in os.environ.get("DISCORD_WORKER_CHANNELS", "").split(",") if c.strip()
}

intents = discord.Intents.default()
intents.message_content = True
intents.guilds = True         # needed for on_guild_channel_create
intents.voice_states = True   # needed to check if user is in a voice channel

client = discord.Client(intents=intents)
tree   = app_commands.CommandTree(client)
_http  = httpx.AsyncClient(timeout=180)

# ── Per-user+channel state ────────────────────────────────────────────────────
_session_ids: dict[str, str] = {}
_user_modes:  dict[str, str] = {}

# ── Per-channel session state (bot-created and auto-connected channels) ───────
_channel_sessions: dict[int, str] = {}        # channel_id → session_id
_channel_message_counts: dict[int, int] = {}  # channel_id → user msg count (ephemeral)
_renamed_channels: set[int] = set()           # channels already auto-renamed (persisted)

_CONVERSATIONS_CATEGORY = "Conversations"

# Per-session tools the user has said "Always allow" — bypasses the approval prompt
_session_always_allow: dict[str, set[str]] = {}

# ── Thinking indicator ────────────────────────────────────────────────────────
# Set by main.py after both bots are initialised; avoids circular imports.
_thinking_client: discord.Client | None = None
_thinking_tasks:  dict[str, asyncio.Task] = {}   # session_id → running task


def set_thinking_client(client_ref: discord.Client) -> None:
    global _thinking_client
    _thinking_client = client_ref


async def _run_thinking_indicator(channel: discord.TextChannel) -> None:
    """Rename mod bot and keep it typing until cancelled."""
    guild_id   = channel.guild.id
    channel_id = channel.id
    nick_changed = False
    try:
        if _thinking_client:
            try:
                mod_guild = _thinking_client.get_guild(guild_id) or \
                            await _thinking_client.fetch_guild(guild_id)
                await mod_guild.me.edit(nick="🧠 thinking")
                nick_changed = True
            except Exception:
                pass  # nickname change is best-effort

            try:
                ch = _thinking_client.get_channel(channel_id) or \
                     await _thinking_client.fetch_channel(channel_id)
                # typing() context manager re-pings Discord every 5 s automatically
                async with ch.typing():
                    while True:
                        await asyncio.sleep(1)
            except Exception as e:
                print(f"[thinking] typing failed: {e!r}", flush=True)
    except asyncio.CancelledError:
        pass
    finally:
        if nick_changed and _thinking_client:
            try:
                mod_guild = _thinking_client.get_guild(guild_id) or \
                            await _thinking_client.fetch_guild(guild_id)
                await mod_guild.me.edit(nick=None)
            except Exception:
                pass


def _start_thinking(session_id: str, channel: discord.TextChannel) -> None:
    if session_id in _thinking_tasks:
        return
    task = asyncio.create_task(
        _run_thinking_indicator(channel),
        name=f"thinking_{session_id}",
    )
    _thinking_tasks[session_id] = task


def _stop_thinking(session_id: str) -> None:
    task = _thinking_tasks.pop(session_id, None)
    if task and not task.done():
        task.cancel()


def _format_tool_trace(traces: list[dict]) -> str:
    """Format tool call telemetry as Discord subtext (small, muted lines).

    Each line is prefixed with -# so Discord renders it in small grey text,
    keeping tool reports visually subordinate to the actual response.
    Errors get a ⚠ prefix to stay visible despite the small size.
    """
    lines = []
    for t in traces:
        name     = t.get("tool", "?")
        duration = t.get("duration_s", 0)
        error    = t.get("error")
        if error:
            lines.append(f"-# ⚠ {name} ({duration:.2f}s) {error}")
        else:
            n = t.get("lines", 0)
            lines.append(f"-# {name} ({duration:.2f}s, {n} lines)")
    return "\n".join(lines)


# ── State persistence — survives container restarts ───────────────────────────
_STATE_FILE = Path(os.environ.get("STATE_DIR", "/app/state")) / "bot_worker_state.json"


def _save_state() -> None:
    """Write session IDs, user modes, and channel sessions to disk."""
    try:
        _STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        _STATE_FILE.write_text(
            json.dumps({
                "session_ids":     _session_ids,
                "user_modes":      _user_modes,
                "channel_sessions": {str(k): v for k, v in _channel_sessions.items()},
                "renamed_channels": list(_renamed_channels),
            }, ensure_ascii=False),
            encoding="utf-8",
        )
    except Exception as e:
        print(f"[worker-bot] state save failed: {e}", flush=True)


def _load_state() -> None:
    """Restore state from disk on startup."""
    if not _STATE_FILE.exists():
        return
    try:
        data = json.loads(_STATE_FILE.read_text(encoding="utf-8"))
        _session_ids.update(data.get("session_ids", {}))
        _user_modes.update(data.get("user_modes", {}))
        for k, v in data.get("channel_sessions", {}).items():
            _channel_sessions[int(k)] = v
            WORKER_CHANNEL_IDS.add(int(k))
        _renamed_channels.update(int(x) for x in data.get("renamed_channels", []))
        print(
            f"[worker-bot] restored {len(_session_ids)} sessions, "
            f"{len(_user_modes)} modes, "
            f"{len(_channel_sessions)} channel-sessions from state file",
            flush=True,
        )
    except Exception as e:
        print(f"[worker-bot] state load failed: {e}", flush=True)


def _session_key(channel_id: int, user_id: int) -> str:
    return f"{channel_id}_{user_id}"


def _get_session_id(channel_id: int, user_id: int) -> str:
    k = _session_key(channel_id, user_id)
    if k not in _session_ids:
        _session_ids[k] = f"discord_{channel_id}_{user_id}"
        _save_state()
    return _session_ids[k]


def _get_mode(channel_id: int, user_id: int) -> str:
    return _user_modes.get(_session_key(channel_id, user_id), "converse")


# ── Auto-rename ───────────────────────────────────────────────────────────────

async def _auto_rename_channel(channel: discord.TextChannel) -> None:
    """Background task: read first user messages, ask LLM for a slug name, rename.

    Uses a one-shot ephemeral session so rename requests never pollute the
    channel's real conversation history or each other.
    After renaming, ensures the worker bot's guild nickname is set.
    """
    try:
        msgs = []
        async for m in channel.history(limit=5):
            if not m.author.bot and m.content.strip():
                msgs.append(m.content.strip())
        if not msgs:
            return

        context = "\n".join(reversed(msgs))
        # Unique session per call — never reused, so no stale context leaks in.
        ephemeral_session = f"rename_{channel.id}_{int(time.time())}"
        resp = await _http.post(f"{MAB_API_URL}/v1/chat/completions", json={
            "messages": [{"role": "user", "content": (
                f"Given these Discord messages:\n\n{context}\n\n"
                "Reply with ONLY a channel name slug: lowercase letters and hyphens only, "
                "3-5 words, max 40 chars. No explanation, no punctuation."
            )}],
            "session_id": ephemeral_session,
            "mode": "converse",
        })
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"].strip().lower()
        # Take the last non-empty line — the slug is always the final output line,
        # never mixed with any preamble the model may have emitted.
        lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
        candidate = lines[-1] if lines else raw
        slug = re.sub(r"[^a-z0-9-]", "-", candidate)
        slug = re.sub(r"-{2,}", "-", slug).strip("-")[:50]
        if slug:
            await channel.edit(name=slug)
            _save_state()
            print(f"[worker-bot] renamed channel {channel.id} to #{slug}", flush=True)

        # Ensure the worker bot keeps its configured nickname after the rename.
        try:
            await channel.guild.me.edit(nick=WORKER_NICKNAME)
        except Exception:
            pass
    except Exception as e:
        print(f"[worker-bot] auto-rename failed for channel {channel.id}: {e}", flush=True)


# ── Approval UI (buttons + embed) ────────────────────────────────────────────

def _make_approval_embed(approval: dict, mode: str) -> discord.Embed:
    """Build the orange approval-request embed shown when a tool needs confirmation."""
    tool   = approval["tool"]
    params = approval.get("params", {})
    embed  = discord.Embed(
        title=f"🔐  Approval Required: `{tool}`",
        color=discord.Color.orange(),
    )
    embed.add_field(name="Mode", value=f"`{mode}`", inline=True)
    if params:
        params_str = json.dumps(params, indent=2, ensure_ascii=False)
        if len(params_str) > 900:
            params_str = params_str[:900] + "\n…"
        embed.add_field(name="Parameters", value=f"```json\n{params_str}\n```", inline=False)
    embed.set_footer(text="✅ Yes — ❌ No — 🔒 Always allow this tool in the current session")
    return embed


class SpeakView(discord.ui.View):
    """Single 🔊 button attached to agent text responses for on-demand TTS playback."""

    def __init__(self, text: str, channel_id: int):
        super().__init__(timeout=900)  # 15 minutes
        self.text       = text
        self.channel_id = channel_id

    @discord.ui.button(emoji="🔊", label="Listen", style=discord.ButtonStyle.secondary)
    async def speak_btn(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        button.disabled = True
        await interaction.response.edit_message(view=self)

        # Prefer voice channel playback; fall back to WAV file in text channel
        voice_state = interaction.user.voice if interaction.guild else None
        voice_ch    = voice_state.channel if voice_state else None

        try:
            if voice_ch:
                tts_resp = await _http.post("http://localhost:4000/discord/speak_voice", json={
                    "voice_channel_id": voice_ch.id,
                    "text":             self.text,
                })
            else:
                tts_resp = await _http.post("http://localhost:4000/discord/speak", json={
                    "channel_id": self.channel_id,
                    "text":       self.text,
                })
            tts_resp.raise_for_status()
            result = tts_resp.json()
            if not result.get("ok"):
                err = result.get("error", "unknown error")
                await interaction.followup.send(f"Voice generation failed: {err}", ephemeral=True)
        except Exception as e:
            await interaction.followup.send(f"Voice generation failed: {e}", ephemeral=True)


class ApprovalView(discord.ui.View):
    """Three-button confirmation UI: Yes / No / Always allow."""

    def __init__(self, approval: dict, session_id: str, mode: str, original_message: str = ""):
        super().__init__(timeout=300)
        self.approval         = approval
        self.session_id       = session_id
        self.mode             = mode
        self.original_message = original_message
        self.message: discord.Message | None = None

    async def _resume(self, interaction: discord.Interaction, *, approved: bool, always: bool = False) -> None:
        # Acknowledge the button click immediately so Discord doesn't show "failed"
        await interaction.response.defer()

        # Disable all buttons on the approval embed
        for item in self.children:
            item.disabled = True  # type: ignore[attr-defined]
        if self.message:
            try:
                await self.message.edit(view=self)
            except Exception:
                pass

        tool = self.approval["tool"]

        if always:
            _session_always_allow.setdefault(self.session_id, set()).add(tool)

        always_allow_list = list(_session_always_allow.get(self.session_id, set()))

        if approved:
            # Send a minimal resume message; the API will restore mid-execution state
            # from disk (saved when PendingApprovalError was raised) and continue
            # exactly where it paused rather than replaying the whole request.
            payload = {
                "messages":       [{"role": "user", "content": self.original_message or "[approved] Yes, proceed."}],
                "session_id":     self.session_id,
                "mode":           self.mode,
                "approved_tools": always_allow_list + [tool],
            }
        else:
            payload = {
                "messages": [{"role": "user", "content": (
                    f"The user declined to run `{tool}`. Do not execute this tool. "
                    "Explain what you were trying to do and ask how they'd like to proceed."
                )}],
                "session_id": self.session_id,
                "mode":       self.mode,
            }

        _start_thinking(self.session_id, interaction.channel)
        data = None
        try:
            resp = await _http.post(f"{MAB_API_URL}/v1/chat/completions", json=payload)
            resp.raise_for_status()
            data = resp.json()
            if data.get("pending_approval"):
                embed = _make_approval_embed(data["pending_approval"], self.mode)
                view  = ApprovalView(data["pending_approval"], self.session_id, self.mode,
                                     original_message=self.original_message)
                sent  = await interaction.followup.send(embed=embed, view=view)
                view.message = sent
                return
            answer = data["choices"][0]["message"]["content"]
        except Exception as e:
            err_str = str(e) or f"{type(e).__name__} (no message)"
            answer = f"[error: {err_str}]"
        finally:
            _stop_thinking(self.session_id)

        if data is not None and data.get("tool_trace"):
            await interaction.followup.send(_format_tool_trace(data["tool_trace"]))

        chunks = split_message(answer)
        for i, chunk in enumerate(chunks):
            is_last = i == len(chunks) - 1
            if is_last:
                await interaction.followup.send(chunk, view=SpeakView(answer, interaction.channel_id))
            else:
                await interaction.followup.send(chunk)

    async def on_timeout(self) -> None:
        for item in self.children:
            item.disabled = True  # type: ignore[attr-defined]
        if self.message:
            try:
                await self.message.edit(content="⏱️ Approval timed out — action cancelled.", view=self)
            except Exception:
                pass

    @discord.ui.button(label="Yes, proceed", style=discord.ButtonStyle.green, emoji="✅")
    async def yes(self, interaction: discord.Interaction, _button: discord.ui.Button) -> None:
        await self._resume(interaction, approved=True)

    @discord.ui.button(label="No, cancel", style=discord.ButtonStyle.red, emoji="❌")
    async def no(self, interaction: discord.Interaction, _button: discord.ui.Button) -> None:
        await self._resume(interaction, approved=False)

    @discord.ui.button(label="Always allow", style=discord.ButtonStyle.blurple, emoji="🔒")
    async def always_allow(self, interaction: discord.Interaction, _button: discord.ui.Button) -> None:
        await self._resume(interaction, approved=True, always=True)


# ── Channel auto-create ───────────────────────────────────────────────────────

async def _ensure_channel(guild: discord.Guild, name: str, topic: str = "") -> discord.TextChannel:
    ch = discord.utils.get(guild.text_channels, name=name)
    if ch is None:
        ch = await guild.create_text_channel(name, topic=topic)
        print(f"[worker-bot] created #{name} ({ch.id})", flush=True)
    return ch


# ── Slash commands ────────────────────────────────────────────────────────────

@tree.command(name="new", description="Start a fresh conversation in a new channel")
async def cmd_new(interaction: discord.Interaction):
    if not is_allowed(interaction.user.id):
        await interaction.response.send_message("Not authorized.", ephemeral=True)
        return

    await interaction.response.defer(ephemeral=True)   # channel creation takes > 3s

    guild = interaction.guild
    if guild is None:
        await interaction.followup.send("This command must be used in a server.", ephemeral=True)
        return

    category = discord.utils.get(guild.categories, name=_CONVERSATIONS_CATEGORY)
    if category is None:
        category = await guild.create_category(_CONVERSATIONS_CATEGORY)
        print(f"[worker-bot] created category '{_CONVERSATIONS_CATEGORY}'", flush=True)

    ts = datetime.now(timezone.utc)
    channel_name = ts.strftime("chat-%m%d-%H%M")
    new_channel = await guild.create_text_channel(channel_name, category=category)
    print(f"[worker-bot] created channel #{channel_name} ({new_channel.id})", flush=True)

    session_id = f"discord_{new_channel.id}_{int(ts.timestamp())}"
    _channel_sessions[new_channel.id] = session_id
    WORKER_CHANNEL_IDS.add(new_channel.id)
    _save_state()

    await new_channel.send(
        f"Hey {interaction.user.mention}! New conversation started. What's on your mind?"
    )
    await interaction.followup.send(
        f"New conversation created: {new_channel.mention}", ephemeral=True
    )


@tree.command(name="clear", description="Reset conversation and delete all messages in this channel")
async def cmd_clear(interaction: discord.Interaction):
    if not is_allowed(interaction.user.id):
        await interaction.response.send_message("Not authorized.", ephemeral=True)
        return

    await interaction.response.defer(ephemeral=True)   # purge may take a moment

    channel = interaction.channel

    # Reset session for this channel
    if channel.id in _channel_sessions:
        old_sid = _channel_sessions[channel.id]
        new_sid = f"discord_{channel.id}_{int(time.time())}"
        _channel_sessions[channel.id] = new_sid
        _renamed_channels.discard(channel.id)        # allow re-rename after new messages come in
        _channel_message_counts.pop(channel.id, None)
        _session_always_allow.pop(old_sid, None)
    else:
        k = _session_key(channel.id, interaction.user.id)
        old_sid = _session_ids.get(k)
        _session_ids[k] = f"discord_{channel.id}_{interaction.user.id}_{int(time.time())}"
        if old_sid:
            _session_always_allow.pop(old_sid, None)

    _save_state()

    # Delete all messages in THIS channel only
    try:
        deleted = await channel.purge(limit=None)
        count   = len(deleted)
    except discord.Forbidden:
        await interaction.followup.send(
            "I need the **Manage Messages** permission to delete messages here.", ephemeral=True
        )
        return
    except Exception as e:
        await interaction.followup.send(f"Could not delete messages: {e}", ephemeral=True)
        return

    await channel.send("🔄 Conversation cleared. Fresh start!")
    await interaction.followup.send(f"Deleted {count} messages and reset the session.", ephemeral=True)


@tree.command(name="mode", description="Switch agent mode")
@app_commands.describe(mode="plan (read-only analysis), build (full tools), converse (casual chat)")
@app_commands.choices(mode=[
    app_commands.Choice(name="plan",     value="plan"),
    app_commands.Choice(name="build",    value="build"),
    app_commands.Choice(name="converse", value="converse"),
])
async def cmd_mode(interaction: discord.Interaction, mode: app_commands.Choice[str]):
    if not is_allowed(interaction.user.id):
        await interaction.response.send_message("Not authorized.", ephemeral=True)
        return
    _user_modes[_session_key(interaction.channel_id, interaction.user.id)] = mode.value
    _save_state()
    descriptions = {
        "plan":     "Read-only analysis — write tools disabled, lower temperature.",
        "build":    "Full tool access — can write files, run commands, make changes.",
        "converse": "Casual chat — write/exec tools disabled, slightly higher temperature.",
    }
    await interaction.response.send_message(
        f"Switched to **{mode.value}** mode. {descriptions[mode.value]}"
    )


_MODEL_PRESETS: dict[str, dict] = {
    "local":     {"provider": "local",     "model": "local-model",                "url": None, "enable_thinking": True},
    "vpn_local": {"provider": "local",     "model": "vpn-model",                  "url": "http://10.64.82.60:8000", "enable_thinking": False},
    "haiku":     {"provider": "anthropic", "model": "claude-haiku-4-5-20251001",  "url": None, "enable_thinking": False},
    "sonnet":    {"provider": "anthropic", "model": "claude-sonnet-4-6",          "url": None, "enable_thinking": False},
    "opus":      {"provider": "anthropic", "model": "claude-opus-4-6",            "url": None, "enable_thinking": False},
}


@tree.command(name="model", description="Show or switch the current LLM model")
@app_commands.describe(preset="Switch to a model (leave blank to show current)")
@app_commands.choices(preset=[
    app_commands.Choice(name="local     — local GPU",                         value="local"),
    app_commands.Choice(name="vpn_local — UTh intranet GPU",                  value="vpn_local"),
    app_commands.Choice(name="haiku     — Claude Haiku 4.5 (fast, cheap)",   value="haiku"),
    app_commands.Choice(name="sonnet    — Claude Sonnet 4.6 (balanced)",     value="sonnet"),
    app_commands.Choice(name="opus      — Claude Opus 4.6 (powerful)",       value="opus"),
])
async def cmd_model(
    interaction: discord.Interaction,
    preset: app_commands.Choice[str] | None = None,
):
    if not is_allowed(interaction.user.id):
        await interaction.response.send_message("Not authorized.", ephemeral=True)
        return

    if preset is not None:
        try:
            patch = dict(_MODEL_PRESETS[preset.value])
            resp = await _http.patch(
                f"{MAB_API_URL}/config",
                json={"llm": patch},
            )
            resp.raise_for_status()
            # PATCH returns {"updated": true, "config": {...}}
            llm = resp.json().get("config", {}).get("llm", {})
            # Query through mab-api (which is on the VPN) to discover the real model name
            if patch.get("provider") == "local":
                try:
                    r = await _http.get(f"{MAB_API_URL}/models", timeout=8)
                    r.raise_for_status()
                    models_data = r.json().get("data", [])
                    if models_data:
                        real_model = models_data[0]["id"]
                        await _http.patch(f"{MAB_API_URL}/config", json={"llm": {"model": real_model}})
                        llm["model"] = real_model
                except Exception:
                    pass  # keep preset default if probe fails
            msg = (
                f"Switched to **{preset.name}**.\n"
                f"**Provider**: `{llm.get('provider', 'local')}`\n"
                f"**Model**: `{llm.get('model', '?')}`"
            )
        except Exception as e:
            msg = f"Failed to switch model: {e}"
    else:
        try:
            resp = await _http.get(f"{MAB_API_URL}/config")
            resp.raise_for_status()
            llm = resp.json().get("llm", {})
            msg = (
                f"**Provider**: `{llm.get('provider', 'local')}`\n"
                f"**Model**: `{llm.get('model', '?')}`\n"
                f"**Temperature**: `{llm.get('temperature', '?')}`\n"
                f"**Max tokens**: `{llm.get('max_tokens', '?')}`\n"
                f"**Thinking**: `{llm.get('enable_thinking', False)}`"
            )
        except Exception as e:
            msg = f"Could not fetch config: {e}"

    await interaction.response.send_message(msg)


@tree.command(name="status", description="Show current session mode, model, and system status")
async def cmd_status(interaction: discord.Interaction):
    if not is_allowed(interaction.user.id):
        await interaction.response.send_message("Not authorized.", ephemeral=True)
        return
    mode = _get_mode(interaction.channel_id, interaction.user.id)
    # Show channel session if applicable, otherwise user+channel session
    if interaction.channel_id in _channel_sessions:
        session_id = _channel_sessions[interaction.channel_id]
    else:
        session_id = _get_session_id(interaction.channel_id, interaction.user.id)
    try:
        resp = await _http.get(f"{MAB_API_URL}/config")
        resp.raise_for_status()
        cfg = resp.json()
        llm   = cfg.get("llm", {})
        agent = cfg.get("agent", {})
        soul  = cfg.get("soul", {})
        msg = (
            f"**Mode**: {mode}\n"
            f"**Session**: `{session_id}`\n"
            f"**Model**: `{llm.get('model', '?')}`\n"
            f"**Supervisor**: {'enabled' if agent.get('supervisor_enabled') else 'disabled'}\n"
            f"**Soul schedule**: `{soul.get('schedule', '?')}`"
        )
    except Exception as e:
        msg = f"**Mode**: {mode}\n**Session**: `{session_id}`\n(Config unavailable: {e})"
    await interaction.response.send_message(msg)


@tree.command(name="help", description="List available slash commands")
async def cmd_help(interaction: discord.Interaction):
    msg = (
        "**Available commands:**\n"
        "• `/new` — Start a fresh conversation in a new channel\n"
        "• `/clear` — Reset conversation and delete all messages in this channel\n"
        "• `/mode [plan|build|converse]` — Switch agent mode\n"
        "• `/model` — Show current LLM model and settings\n"
        "• `/speak <prompt>` — Ask the agent a question and get a voice response\n"
        "• `/status` — Show your current mode, session, and system status\n"
        "• `/help` — This message\n\n"
        "**Modes:**\n"
        "• **plan** — Read-only analysis, lower temperature\n"
        "• **build** — Full tool access, normal temperature\n"
        "• **converse** — Casual chat, write tools off, slightly higher temperature\n\n"
        "**Tip:** Create a channel in the `Conversations` category and I'll auto-connect to it."
    )
    await interaction.response.send_message(msg)


@tree.command(name="speak", description="Ask the agent a question and get a voice response")
@app_commands.describe(prompt="What to ask the agent")
async def cmd_speak(interaction: discord.Interaction, prompt: str):
    if not is_allowed(interaction.user.id):
        await interaction.response.send_message("Not authorized.", ephemeral=True)
        return

    await interaction.response.defer()

    if interaction.channel_id in _channel_sessions:
        session_id = _channel_sessions[interaction.channel_id]
    else:
        session_id = _get_session_id(interaction.channel_id, interaction.user.id)
    mode = _get_mode(interaction.channel_id, interaction.user.id)

    _start_thinking(session_id, interaction.channel)
    data = None
    try:
        resp = await _http.post(f"{MAB_API_URL}/v1/chat/completions", json={
            "messages":   [{"role": "user", "content": prompt}],
            "session_id": session_id,
            "mode":       mode,
        })
        resp.raise_for_status()
        data = resp.json()
        answer = data["choices"][0]["message"]["content"]
    except Exception as e:
        err_str = str(e) or f"{type(e).__name__} (no message)"
        await interaction.followup.send(f"[error: {err_str}]")
        return
    finally:
        _stop_thinking(session_id)

    if data is not None and data.get("tool_trace"):
        await interaction.followup.send(_format_tool_trace(data["tool_trace"]))

    # Play TTS — prefer voice channel, fall back to WAV file in text channel
    voice_state = interaction.user.voice if interaction.guild else None
    voice_ch    = voice_state.channel if voice_state else None
    tts_ok = False
    try:
        if voice_ch:
            tts_resp = await _http.post("http://localhost:4000/discord/speak_voice", json={
                "voice_channel_id": voice_ch.id,
                "text":             answer,
            })
        else:
            tts_resp = await _http.post("http://localhost:4000/discord/speak", json={
                "channel_id": interaction.channel_id,
                "text":       answer,
            })
        tts_resp.raise_for_status()
        tts_ok = tts_resp.json().get("ok", False)
    except Exception as e:
        print(f"[worker-bot] TTS failed: {e}", flush=True)

    # Always send the text too so the response is readable
    for chunk in split_message(answer):
        await interaction.followup.send(chunk)

    if not tts_ok:
        await interaction.followup.send("_(voice generation failed — text response above)_", ephemeral=True)


# ── Background model-name poller ─────────────────────────────────────────────

async def _poll_model_name() -> None:
    """
    Every 60 s: if the current llm provider is local, query /models through
    mab-api and patch the model name back into config if it changed.
    This keeps the displayed model name current even when the server swaps models.
    """
    await asyncio.sleep(15)  # let the API finish starting before first poll
    while True:
        try:
            cfg_resp = await _http.get(f"{MAB_API_URL}/config", timeout=5)
            cfg_resp.raise_for_status()
            llm = cfg_resp.json().get("llm", {})
            if llm.get("provider", "local") == "local":
                r = await _http.get(f"{MAB_API_URL}/models", timeout=8)
                r.raise_for_status()
                models_data = r.json().get("data", [])
                if models_data:
                    real_model = models_data[0]["id"]
                    if real_model != llm.get("model"):
                        await _http.patch(f"{MAB_API_URL}/config", json={"llm": {"model": real_model}})
                        print(f"[worker-bot] model name updated: {llm.get('model')!r} → {real_model!r}", flush=True)
        except Exception:
            pass
        await asyncio.sleep(60)


# ── Bot events ────────────────────────────────────────────────────────────────

@client.event
async def on_ready():
    _load_state()
    # Re-register all channel-session channels as listened channels (after state load)
    for ch_id in _channel_sessions:
        WORKER_CHANNEL_IDS.add(ch_id)
    print(f"[worker-bot] logged in as {client.user}", flush=True)

    # Set bot avatar from the bundled Gemma logo (best-effort; Discord rate-limits
    # avatar changes to roughly once per 10 minutes, so failures are non-fatal).
    try:
        logo = Path(__file__).parent / "gemma_logo.jpg"
        if logo.exists():
            await client.user.edit(avatar=logo.read_bytes())
            print("[worker-bot] avatar set from gemma_logo.jpg", flush=True)
    except Exception as e:
        print(f"[worker-bot] avatar set skipped: {e}", flush=True)

    if GUILD_ID:
        try:
            guild = client.get_guild(GUILD_ID) or await client.fetch_guild(GUILD_ID)
            ch = await _ensure_channel(guild, "mab-worker", "Multi-agent worker chat")
            WORKER_CHANNEL_IDS.add(ch.id)
            print(f"[worker-bot] listening on #{ch.name} ({ch.id})", flush=True)
        except Exception as e:
            print(f"[worker-bot] channel setup failed: {e}", flush=True)
        try:
            guild_obj = discord.Object(id=GUILD_ID)
            tree.copy_global_to(guild=guild_obj)
            cmds = await tree.sync(guild=guild_obj)
            print(f"[worker-bot] synced {len(cmds)} commands to guild {GUILD_ID}", flush=True)
        except Exception as e:
            print(f"[worker-bot] slash command sync failed: {e}", flush=True)
        # Set the worker bot's persistent display name in this guild.
        try:
            guild = client.get_guild(GUILD_ID) or await client.fetch_guild(GUILD_ID)
            await guild.me.edit(nick=WORKER_NICKNAME)
            print(f"[worker-bot] guild nickname set to {WORKER_NICKNAME!r}", flush=True)
        except Exception as e:
            print(f"[worker-bot] nickname set skipped: {e}", flush=True)
    asyncio.create_task(_poll_model_name(), name="model_name_poller")


@client.event
async def on_guild_channel_create(channel: discord.abc.GuildChannel) -> None:
    """Auto-connect when a user manually creates a text channel in the Conversations category."""
    if not isinstance(channel, discord.TextChannel):
        return
    if channel.category is None or channel.category.name != _CONVERSATIONS_CATEGORY:
        return
    # Guard: /new already registered it before Discord fires this event
    if channel.id in _channel_sessions:
        return

    session_id = f"discord_{channel.id}_{int(time.time())}"
    _channel_sessions[channel.id] = session_id
    WORKER_CHANNEL_IDS.add(channel.id)
    _save_state()

    try:
        await channel.send("👋 I'm connected to this channel. What's on your mind?")
    except Exception as e:
        print(f"[worker-bot] failed to send welcome to #{channel.name}: {e}", flush=True)

    print(f"[worker-bot] auto-connected to #{channel.name} ({channel.id})", flush=True)


@client.event
async def on_message(msg: discord.Message):
    if msg.author.bot:
        return
    if not is_allowed(msg.author.id):
        return
    if WORKER_CHANNEL_IDS and msg.channel.id not in WORKER_CHANNEL_IDS:
        return

    # Channel-session routing: bot-created channels share one session for the whole channel
    if msg.channel.id in _channel_sessions:
        session_id = _channel_sessions[msg.channel.id]
        # Track user message count for auto-rename trigger
        _channel_message_counts[msg.channel.id] = (
            _channel_message_counts.get(msg.channel.id, 0) + 1
        )
        if (_channel_message_counts[msg.channel.id] == 2
                and msg.channel.id not in _renamed_channels):
            _renamed_channels.add(msg.channel.id)
            asyncio.create_task(
                _auto_rename_channel(msg.channel),
                name=f"rename_{msg.channel.id}",
            )
    else:
        session_id = _get_session_id(msg.channel.id, msg.author.id)

    mode         = _get_mode(msg.channel.id, msg.author.id)
    always_allow = list(_session_always_allow.get(session_id, set()))

    _start_thinking(session_id, msg.channel)
    data = None
    try:
        resp = await _http.post(
            f"{MAB_API_URL}/v1/chat/completions",
            json={
                "messages":       [{"role": "user", "content": msg.content}],
                "session_id":     session_id,
                "mode":           mode,
                "approved_tools": always_allow,
            },
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        err_str = str(e) or f"{type(e).__name__} (no message)"
        await msg.channel.send(f"[error: {err_str}]")
        return
    finally:
        _stop_thinking(session_id)

    if data.get("pending_approval"):
        embed = _make_approval_embed(data["pending_approval"], mode)
        view  = ApprovalView(data["pending_approval"], session_id, mode,
                             original_message=msg.content)
        sent  = await msg.channel.send(embed=embed, view=view)
        view.message = sent
        return

    if data.get("tool_trace"):
        await msg.channel.send(_format_tool_trace(data["tool_trace"]))

    answer = data["choices"][0]["message"]["content"]
    chunks = split_message(answer)
    for i, chunk in enumerate(chunks):
        is_last = i == len(chunks) - 1
        if is_last:
            await msg.channel.send(chunk, view=SpeakView(answer, msg.channel.id))
        else:
            await msg.channel.send(chunk)


async def run():
    if not WORKER_TOKEN:
        print("[worker-bot] DISCORD_TOKEN_WORKER not set — skipping", flush=True)
        return
    await client.start(WORKER_TOKEN)
