"""
Config Discord bot — routes messages to the mab-api config agent.
Used for guided configuration via Discord channels.

Slash commands: /help, /status
Auto-creates #mab-config channel on startup if it doesn't exist.
"""

import os

import discord
from discord import app_commands
import httpx

from utils import is_allowed, split_message

MAB_API_URL  = os.environ.get("MAB_API_URL", "http://mab-api:8090")
CONFIG_TOKEN = os.environ.get("DISCORD_TOKEN_CONFIG", "")
GUILD_ID     = int(os.environ.get("DISCORD_GUILD_ID", "0"))

# Channel IDs the config bot listens in — populated dynamically on_ready
CONFIG_CHANNEL_IDS: set[int] = {
    int(c) for c in os.environ.get("DISCORD_CONFIG_CHANNELS", "").split(",") if c.strip()
}

intents = discord.Intents.default()
intents.message_content = True

client = discord.Client(intents=intents)
tree   = app_commands.CommandTree(client)
_http  = httpx.AsyncClient(timeout=120)


# ── Channel auto-create ───────────────────────────────────────────────────────

async def _ensure_channel(guild: discord.Guild, name: str, topic: str = "") -> discord.TextChannel:
    ch = discord.utils.get(guild.text_channels, name=name)
    if ch is None:
        ch = await guild.create_text_channel(name, topic=topic)
        print(f"[config-bot] created #{name} ({ch.id})", flush=True)
    return ch


# ── Slash commands ────────────────────────────────────────────────────────────

@tree.command(name="status", description="Show system configuration status")
async def cmd_status(interaction: discord.Interaction):
    if not is_allowed(interaction.user.id):
        await interaction.response.send_message("Not authorized.", ephemeral=True)
        return
    try:
        resp = await _http.get(f"{MAB_API_URL}/config")
        resp.raise_for_status()
        cfg   = resp.json()
        llm   = cfg.get("llm", {})
        agent = cfg.get("agent", {})
        soul  = cfg.get("soul", {})
        msg = (
            f"**Model**: `{llm.get('model', '?')}`\n"
            f"**Base URL**: `{llm.get('base_url', '?')}`\n"
            f"**Temperature**: `{llm.get('temperature', '?')}`\n"
            f"**Supervisor**: {'enabled' if agent.get('supervisor_enabled') else 'disabled'}\n"
            f"**Max retries**: `{agent.get('max_retries', '?')}`\n"
            f"**Prompt mode**: `{cfg.get('prompts', {}).get('mode', '?')}`\n"
            f"**Soul**: {'enabled' if soul.get('enabled') else 'disabled'} — `{soul.get('schedule', '?')}`"
        )
    except Exception as e:
        msg = f"Could not fetch config: {e}"
    await interaction.response.send_message(msg)


@tree.command(name="help", description="List available slash commands for the config bot")
async def cmd_help(interaction: discord.Interaction):
    msg = (
        "**Config bot commands:**\n"
        "• `/status` — Show current system configuration\n"
        "• `/help` — This message\n\n"
        "Send any message in this channel to chat with the config agent.\n"
        "The config agent can read and update your system settings interactively."
    )
    await interaction.response.send_message(msg)


# ── Bot events ────────────────────────────────────────────────────────────────

@client.event
async def on_ready():
    print(f"[config-bot] logged in as {client.user}", flush=True)
    if GUILD_ID:
        try:
            guild = client.get_guild(GUILD_ID) or await client.fetch_guild(GUILD_ID)
            ch = await _ensure_channel(guild, "mab-config", "Multi-agent config assistant")
            CONFIG_CHANNEL_IDS.add(ch.id)
            print(f"[config-bot] listening on #{ch.name} ({ch.id})", flush=True)
        except Exception as e:
            print(f"[config-bot] channel setup failed: {e}", flush=True)
        try:
            guild_obj = discord.Object(id=GUILD_ID)
            tree.copy_global_to(guild=guild_obj)
            cmds = await tree.sync(guild=guild_obj)
            print(f"[config-bot] synced {len(cmds)} commands to guild {GUILD_ID}", flush=True)
        except Exception as e:
            print(f"[config-bot] slash command sync failed: {e}", flush=True)


@client.event
async def on_message(msg: discord.Message):
    if msg.author.bot:
        return
    if not is_allowed(msg.author.id):
        return
    if CONFIG_CHANNEL_IDS and msg.channel.id not in CONFIG_CHANNEL_IDS:
        return

    session_id = f"discord_config_{msg.channel.id}_{msg.author.id}"

    async with msg.channel.typing():
        try:
            resp = await _http.post(
                f"{MAB_API_URL}/config/agent",
                json={
                    "messages":   [{"role": "user", "content": msg.content}],
                    "session_id": session_id,
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
    if not CONFIG_TOKEN:
        print("[config-bot] DISCORD_TOKEN_CONFIG not set — skipping", flush=True)
        return
    await client.start(CONFIG_TOKEN)
