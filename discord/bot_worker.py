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

import discord
from discord import app_commands
import httpx

from utils import is_allowed, split_message

MAB_API_URL  = os.environ.get("MAB_API_URL",    "http://mab-api:8090")
WORKER_TOKEN = os.environ.get("DISCORD_TOKEN_WORKER", "")
GUILD_ID     = int(os.environ.get("DISCORD_GUILD_ID", "0"))

# Channel IDs the worker bot listens in — populated dynamically on_ready
# (env var is an optional static fallback)
WORKER_CHANNEL_IDS: set[int] = {
    int(c) for c in os.environ.get("DISCORD_WORKER_CHANNELS", "").split(",") if c.strip()
}

intents = discord.Intents.default()
intents.message_content = True
intents.guilds = True   # needed for on_guild_channel_create

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
    """Background task: read first user messages, ask LLM for a slug name, rename."""
    try:
        msgs = []
        async for m in channel.history(limit=5):
            if not m.author.bot and m.content.strip():
                msgs.append(m.content.strip())
        if not msgs:
            return

        context = "\n".join(reversed(msgs))
        resp = await _http.post(f"{MAB_API_URL}/v1/chat/completions", json={
            "messages": [{"role": "user", "content": (
                f"Given these Discord messages:\n\n{context}\n\n"
                "Reply with ONLY a channel name slug: lowercase letters and hyphens only, "
                "3-5 words, max 40 chars. No explanation, no punctuation."
            )}],
            "session_id": f"rename_{channel.id}",
            "mode": "converse",
        })
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"].strip().lower()
        slug = re.sub(r"[^a-z0-9-]", "-", raw)
        slug = re.sub(r"-{2,}", "-", slug).strip("-")[:50]
        if slug:
            await channel.edit(name=slug)
            _save_state()
            print(f"[worker-bot] renamed channel {channel.id} to #{slug}", flush=True)
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
        await interaction.response.defer()
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

        if approved:
            always_allow_list = list(_session_always_allow.get(self.session_id, set()))
            messages = [{"role": "user", "content": self.original_message}] if self.original_message else \
                       [{"role": "user", "content": "[approved] Yes, proceed."}]
            payload = {
                "messages":       messages,
                "session_id":     self.session_id,
                "mode":           self.mode,
                "approved_tools": always_allow_list + [tool],
            }
        else:
            if self.original_message:
                messages = [
                    {"role": "user", "content": self.original_message},
                    {"role": "user", "content": f"[context] The user declined to run `{tool}`. Do not execute this tool."},
                ]
            else:
                messages = [{"role": "user", "content": f"[denied] No, do not run `{tool}`."}]
            payload = {
                "messages":   messages,
                "session_id": self.session_id,
                "mode":       self.mode,
            }

        _start_thinking(self.session_id, interaction.channel)
        try:
            resp = await _http.post(f"{MAB_API_URL}/v1/chat/completions", json=payload)
            resp.raise_for_status()
            data = resp.json()
            if data.get("pending_approval"):
                embed = _make_approval_embed(data["pending_approval"], self.mode)
                view  = ApprovalView(data["pending_approval"], self.session_id, self.mode,
                                     original_message=self.original_message)
                sent  = await interaction.channel.send(embed=embed, view=view)
                view.message = sent
                return
            answer = data["choices"][0]["message"]["content"]
        except Exception as e:
            answer = f"[error: {e}]"
        finally:
            _stop_thinking(self.session_id)

        for chunk in split_message(answer):
            await interaction.channel.send(chunk)

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
    "local":  {"provider": "local",     "model": "local-model"},
    "haiku":  {"provider": "anthropic", "model": "claude-haiku-4-5-20251001"},
    "sonnet": {"provider": "anthropic", "model": "claude-sonnet-4-6"},
    "opus":   {"provider": "anthropic", "model": "claude-opus-4-6"},
}


@tree.command(name="model", description="Show or switch the current LLM model")
@app_commands.describe(preset="Switch to a model (leave blank to show current)")
@app_commands.choices(preset=[
    app_commands.Choice(name="local  — Gemma 4 (local GPU)",            value="local"),
    app_commands.Choice(name="haiku  — Claude Haiku 4.5 (fast, cheap)", value="haiku"),
    app_commands.Choice(name="sonnet — Claude Sonnet 4.6 (balanced)",   value="sonnet"),
    app_commands.Choice(name="opus   — Claude Opus 4.6 (powerful)",     value="opus"),
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
            resp = await _http.patch(
                f"{MAB_API_URL}/config",
                json={"llm": _MODEL_PRESETS[preset.value]},
            )
            resp.raise_for_status()
            llm = resp.json().get("llm", {})
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
        "• `/status` — Show your current mode, session, and system status\n"
        "• `/help` — This message\n\n"
        "**Modes:**\n"
        "• **plan** — Read-only analysis, lower temperature\n"
        "• **build** — Full tool access, normal temperature\n"
        "• **converse** — Casual chat, write tools off, slightly higher temperature\n\n"
        "**Tip:** Create a channel in the `Conversations` category and I'll auto-connect to it."
    )
    await interaction.response.send_message(msg)


# ── Bot events ────────────────────────────────────────────────────────────────

@client.event
async def on_ready():
    _load_state()
    # Re-register all channel-session channels as listened channels (after state load)
    for ch_id in _channel_sessions:
        WORKER_CHANNEL_IDS.add(ch_id)
    print(f"[worker-bot] logged in as {client.user}", flush=True)
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
        await msg.channel.send(f"[error: {e}]")
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

    answer = data["choices"][0]["message"]["content"]
    for chunk in split_message(answer):
        await msg.channel.send(chunk)


async def run():
    if not WORKER_TOKEN:
        print("[worker-bot] DISCORD_TOKEN_WORKER not set — skipping", flush=True)
        return
    await client.start(WORKER_TOKEN)
