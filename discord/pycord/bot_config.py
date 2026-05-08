"""Pycord port of bot_config.py (A2 PR2 Phase 2a).

Migration delta vs ../bot_config.py:
  * `discord.Client` + `app_commands.CommandTree(client)` → `discord.Bot(...)`.
    Bot inherits from Client and exposes `.slash_command` directly; no tree.
  * `@tree.command(...)` → `@bot.slash_command(...)`.
  * `interaction: discord.Interaction` → `ctx: discord.ApplicationContext`.
  * `interaction.response.send_message(...)` → `ctx.respond(...)`.
  * `interaction.user.id` → `ctx.author.id`.
  * Slash command sync: pycord registers automatically when `Bot(guild_ids=…)`
    is passed; no manual `tree.copy_global_to / sync` needed in on_ready.
  * Everything else (intents, on_message, _http, run()) stays identical.

Same module-level surface as the legacy file so main.py can import either.
"""

import os
from pathlib import Path

import discord
import httpx

from utils import is_allowed, split_message

PHOEBE_API_URL  = os.environ.get("PHOEBE_API_URL", "http://phoebe-api:8090")
CONFIG_TOKEN = os.environ.get("DISCORD_TOKEN_CONFIG", "")
GUILD_ID     = int(os.environ.get("DISCORD_GUILD_ID", "0"))
CONFIG_NICKNAME = os.environ.get("DISCORD_CONFIG_NICKNAME", "Phoebe-config")

CONFIG_CHANNEL_IDS: set[int] = {
    int(c) for c in os.environ.get("DISCORD_CONFIG_CHANNELS", "").split(",") if c.strip()
}

intents = discord.Intents.default()
intents.message_content = True

# Pycord: pass guild_ids so slash commands register to that guild automatically
# (instant availability vs the global ~1h propagation). Falls through to global
# registration when GUILD_ID isn't set.
_bot_kwargs = {"intents": intents}
if GUILD_ID:
    _bot_kwargs["debug_guilds"] = [GUILD_ID]

client = discord.Bot(**_bot_kwargs)
# Alias preserves the legacy import shape so main.py can still
# `from bot_config import client`. Bot is a subclass of Client.
_http = httpx.AsyncClient(timeout=120)


# ── Slash commands ────────────────────────────────────────────────────────────

@client.slash_command(name="status", description="Show system configuration status")
async def cmd_status(ctx: discord.ApplicationContext):
    if not is_allowed(ctx.author.id):
        await ctx.respond("Not authorized.", ephemeral=True)
        return
    try:
        resp = await _http.get(f"{PHOEBE_API_URL}/config")
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
    await ctx.respond(msg)


@client.slash_command(name="help", description="List available slash commands for the config bot")
async def cmd_help(ctx: discord.ApplicationContext):
    msg = (
        "**Config bot commands:**\n"
        "• `/status` — Show current system configuration\n"
        "• `/help` — This message\n\n"
        "Send any message in this channel to chat with the config agent.\n"
        "The config agent can read and update your system settings interactively."
    )
    await ctx.respond(msg)


# ── Bot events ────────────────────────────────────────────────────────────────

@client.event
async def on_ready():
    print(f"[config-bot] logged in as {client.user}", flush=True)

    # Set bot avatar from the bundled Phoebe-config logo (best-effort; Discord
    # rate-limits avatar changes to roughly once per 10 minutes).
    try:
        logo = Path(__file__).parent / "Phoebe_config.png"
        if logo.exists():
            await client.user.edit(avatar=logo.read_bytes())
            print("[config-bot] avatar set from Phoebe_config.png", flush=True)
    except Exception as e:
        print(f"[config-bot] avatar set skipped: {e}", flush=True)

    if GUILD_ID:
        try:
            guild = client.get_guild(GUILD_ID) or await client.fetch_guild(GUILD_ID)
            ch = discord.utils.get(guild.text_channels, name="phoebe-config")
            if ch:
                CONFIG_CHANNEL_IDS.add(ch.id)
                print(f"[config-bot] listening on #{ch.name} ({ch.id})", flush=True)
            else:
                print("[config-bot] #phoebe-config not found yet — mod-bot will create it", flush=True)
        except Exception as e:
            print(f"[config-bot] channel lookup failed: {e}", flush=True)
        try:
            guild = client.get_guild(GUILD_ID) or await client.fetch_guild(GUILD_ID)
            await guild.me.edit(nick=CONFIG_NICKNAME)
            print(f"[config-bot] guild nickname set to {CONFIG_NICKNAME!r}", flush=True)
        except Exception as e:
            print(f"[config-bot] nickname set skipped: {e}", flush=True)
    # Pycord's default on_ready does the slash-command sync. A user-defined
    # @client.event on_ready REPLACES it (not additive), so we must call
    # sync_commands ourselves or the slashes silently disappear.
    try:
        synced = await client.sync_commands()
        n = len(synced) if synced else len(client.pending_application_commands)
        print(f"[config-bot] synced {n} slash commands", flush=True)
    except Exception as e:
        print(f"[config-bot] slash sync failed: {e}", flush=True)


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
                f"{PHOEBE_API_URL}/config/agent",
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
