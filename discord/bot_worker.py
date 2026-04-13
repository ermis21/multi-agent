"""
Worker Discord bot — bidirectional: listens in the mab-worker channel and
routes messages to mab-api, then posts the response back.

Each user+channel pair gets its own session_id for continuity.
Slash commands: /new, /mode, /model, /status, /help
"""

import time
import os

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

client = discord.Client(intents=intents)
tree   = app_commands.CommandTree(client)
_http  = httpx.AsyncClient(timeout=180)

# ── Per-user+channel state ────────────────────────────────────────────────────
_session_ids: dict[str, str] = {}
_user_modes:  dict[str, str] = {}


def _session_key(channel_id: int, user_id: int) -> str:
    return f"{channel_id}_{user_id}"


def _get_session_id(channel_id: int, user_id: int) -> str:
    k = _session_key(channel_id, user_id)
    if k not in _session_ids:
        _session_ids[k] = f"discord_{channel_id}_{user_id}"
    return _session_ids[k]


def _get_mode(channel_id: int, user_id: int) -> str:
    return _user_modes.get(_session_key(channel_id, user_id), "converse")


# ── Channel auto-create ───────────────────────────────────────────────────────

async def _ensure_channel(guild: discord.Guild, name: str, topic: str = "") -> discord.TextChannel:
    ch = discord.utils.get(guild.text_channels, name=name)
    if ch is None:
        ch = await guild.create_text_channel(name, topic=topic)
        print(f"[worker-bot] created #{name} ({ch.id})", flush=True)
    return ch


# ── Slash commands ────────────────────────────────────────────────────────────

@tree.command(name="new", description="Start a fresh conversation (resets session and mode)")
async def cmd_new(interaction: discord.Interaction):
    if not is_allowed(interaction.user.id):
        await interaction.response.send_message("Not authorized.", ephemeral=True)
        return
    k = _session_key(interaction.channel_id, interaction.user.id)
    _session_ids[k] = f"discord_{interaction.channel_id}_{interaction.user.id}_{int(time.time())}"
    _user_modes.pop(k, None)
    await interaction.response.send_message("New session started. Mode reset to **converse**.")


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
    descriptions = {
        "plan":     "Read-only analysis — write tools disabled, lower temperature.",
        "build":    "Full tool access — can write files, run commands, make changes.",
        "converse": "Casual chat — write/exec tools disabled, slightly higher temperature.",
    }
    await interaction.response.send_message(
        f"Switched to **{mode.value}** mode. {descriptions[mode.value]}"
    )


@tree.command(name="model", description="Show the current LLM model and configuration")
async def cmd_model(interaction: discord.Interaction):
    if not is_allowed(interaction.user.id):
        await interaction.response.send_message("Not authorized.", ephemeral=True)
        return
    try:
        resp = await _http.get(f"{MAB_API_URL}/config")
        resp.raise_for_status()
        cfg = resp.json()
        llm = cfg.get("llm", {})
        msg = (
            f"**Model**: `{llm.get('model', '?')}`\n"
            f"**Base URL**: `{llm.get('base_url', '?')}`\n"
            f"**Temperature**: `{llm.get('temperature', '?')}`\n"
            f"**Max tokens**: `{llm.get('max_tokens', '?')}`"
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
        "• `/new` — Start a fresh conversation (resets session and mode)\n"
        "• `/mode [plan|build|converse]` — Switch agent mode\n"
        "• `/model` — Show current LLM model and settings\n"
        "• `/status` — Show your current mode, session, and system status\n"
        "• `/help` — This message\n\n"
        "**Modes:**\n"
        "• **plan** — Read-only analysis, lower temperature\n"
        "• **build** — Full tool access, normal temperature\n"
        "• **converse** — Casual chat, write tools off, slightly higher temperature"
    )
    await interaction.response.send_message(msg)


# ── Bot events ────────────────────────────────────────────────────────────────

@client.event
async def on_ready():
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
async def on_message(msg: discord.Message):
    if msg.author.bot:
        return
    if not is_allowed(msg.author.id):
        return
    if WORKER_CHANNEL_IDS and msg.channel.id not in WORKER_CHANNEL_IDS:
        return

    mode       = _get_mode(msg.channel.id, msg.author.id)
    session_id = _get_session_id(msg.channel.id, msg.author.id)

    async with msg.channel.typing():
        try:
            resp = await _http.post(
                f"{MAB_API_URL}/v1/chat/completions",
                json={
                    "messages":   [{"role": "user", "content": msg.content}],
                    "session_id": session_id,
                    "mode":       mode,
                },
            )
            resp.raise_for_status()
            data   = resp.json()
            answer = data["choices"][0]["message"]["content"]
        except Exception as e:
            answer = f"[error: {e}]"

    for chunk in split_message(answer):
        await msg.channel.send(chunk)


async def run():
    if not WORKER_TOKEN:
        print("[worker-bot] DISCORD_TOKEN_WORKER not set — skipping", flush=True)
        return
    await client.start(WORKER_TOKEN)
