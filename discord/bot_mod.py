"""
Server management bot — handles all channel/category operations that require
the Manage Channels permission, and acts as the thinking indicator bot
(nickname changes + typing pings while the worker bot waits for the LLM).

Required Discord permissions for this bot:
  - Manage Channels  (create / delete / edit channels and categories)
  - Manage Nicknames (rename itself for the thinking indicator)
  - View Channels    (needed to find existing channels)
  - Send Messages    (needed for trigger_typing)

No slash commands, no message routing — purely infrastructure.
"""

import os
from pathlib import Path

import discord

MOD_TOKEN = os.environ.get("DISCORD_TOKEN_MOD", "")
GUILD_ID  = int(os.environ.get("DISCORD_GUILD_ID", "0"))
MOD_NICKNAME = os.environ.get("DISCORD_MOD_NICKNAME", "Phoebe_mod")

intents         = discord.Intents.default()
intents.guilds  = True  # needed to receive on_guild_channel_create and manage channels

client = discord.Client(intents=intents)


@client.event
async def on_ready():
    print(f"[mod-bot] logged in as {client.user}", flush=True)

    # Set bot avatar from the bundled Phoebe-mod logo (best-effort; Discord
    # rate-limits avatar changes to roughly once per 10 minutes).
    try:
        logo = Path(__file__).parent / "Phoebe_mod.png"
        if logo.exists():
            await client.user.edit(avatar=logo.read_bytes())
            print("[mod-bot] avatar set from Phoebe_mod.png", flush=True)
    except Exception as e:
        print(f"[mod-bot] avatar set skipped: {e}", flush=True)

    if not GUILD_ID:
        return
    try:
        guild = client.get_guild(GUILD_ID) or await client.fetch_guild(GUILD_ID)
        # Ensure the config bot's channel exists — previously the config bot tried
        # to create this itself but lacked Manage Channels.
        ch = discord.utils.get(guild.text_channels, name="phoebe-config")
        if ch is None:
            ch = await guild.create_text_channel(
                "phoebe-config", topic="Multi-agent config assistant"
            )
            print(f"[mod-bot] created #phoebe-config ({ch.id})", flush=True)
        else:
            print(f"[mod-bot] found #phoebe-config ({ch.id})", flush=True)
    except Exception as e:
        print(f"[mod-bot] channel setup failed: {e}", flush=True)
    try:
        guild = client.get_guild(GUILD_ID) or await client.fetch_guild(GUILD_ID)
        await guild.me.edit(nick=MOD_NICKNAME)
        print(f"[mod-bot] guild nickname set to {MOD_NICKNAME!r}", flush=True)
    except Exception as e:
        print(f"[mod-bot] nickname set skipped: {e}", flush=True)


async def run():
    if not MOD_TOKEN:
        print("[mod-bot] DISCORD_TOKEN_MOD not set — skipping", flush=True)
        return
    await client.start(MOD_TOKEN)
