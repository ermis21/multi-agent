"""
Discord service — FastAPI HTTP API (port 4000) + both bots.

Endpoints (called by sandbox discord_* tools):
  POST /discord/send              — send a message via named bot
  GET  /discord/read              — fetch recent messages from a channel
  POST /discord/set_nickname      — rename a guild member
  POST /discord/edit_channel      — update channel name/topic/category
  POST /discord/create_channel    — create a new text channel
  POST /discord/delete_channel    — delete a channel
  GET  /discord/list_channels     — list all guild channels with metadata
  POST /discord/create_category   — create a new category channel
  GET  /health
"""

import asyncio
import os
from contextlib import asynccontextmanager

import discord
from fastapi import FastAPI
from pydantic import BaseModel

import bot_config
import bot_mod
import bot_worker


def _log_task_error(task: asyncio.Task) -> None:
    """Surface bot task exceptions to the container log instead of swallowing them."""
    if not task.cancelled() and task.exception():
        print(f"[discord] bot task {task.get_name()} failed: {task.exception()}", flush=True)


async def _wire_thinking_client() -> None:
    """Wait for the mod bot to be ready, then expose its client to bot_worker."""
    await bot_mod.client.wait_until_ready()
    bot_worker.set_thinking_client(bot_mod.client)
    print("[discord] thinking client wired to mod bot", flush=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    tasks = [
        asyncio.create_task(bot_worker.run(), name="worker-bot"),
        asyncio.create_task(bot_config.run(), name="config-bot"),
        asyncio.create_task(bot_mod.run(), name="mod-bot"),
        asyncio.create_task(_wire_thinking_client(), name="wire-thinking"),
    ]
    for t in tasks:
        t.add_done_callback(_log_task_error)
    yield
    for t in tasks:
        t.cancel()


def _get_bot_client(bot: str) -> discord.Client:
    if bot == "worker":
        return bot_worker.client
    if bot == "mod":
        return bot_mod.client
    return bot_config.client


app = FastAPI(title="mab-discord", lifespan=lifespan, docs_url=None, redoc_url=None)


# ── Request models ──────────────────────────────────────────────────────────

class SendRequest(BaseModel):
    channel_id: int
    content:    str
    bot:        str = "worker"

class NicknameRequest(BaseModel):
    guild_id:  int
    user_id:   int
    nickname:  str
    bot:       str = "worker"

class EditChannelRequest(BaseModel):
    channel_id:  int
    name:        str | None = None
    topic:       str | None = None
    category_id: int | None = None
    bot:         str = "mod"

class CreateChannelRequest(BaseModel):
    name:        str
    topic:       str = ""
    category_id: int | None = None
    guild_id:    int | None = None
    bot:         str = "mod"

class DeleteChannelRequest(BaseModel):
    channel_id: int
    bot:        str = "mod"

class CreateCategoryRequest(BaseModel):
    name:     str
    guild_id: int | None = None
    bot:      str = "mod"


# ── Routes ──────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"ok": True}


@app.post("/discord/send")
async def send_message(req: SendRequest):
    bot_client = _get_bot_client(req.bot)
    try:
        channel = bot_client.get_channel(req.channel_id)
        if channel is None:
            channel = await bot_client.fetch_channel(req.channel_id)
        msg = await channel.send(req.content)
        return {"message_id": str(msg.id), "ok": True}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.get("/discord/read")
async def read_messages(channel_id: int, limit: int = 20, bot: str = "worker"):
    bot_client = _get_bot_client(bot)
    try:
        channel = bot_client.get_channel(channel_id)
        if channel is None:
            channel = await bot_client.fetch_channel(channel_id)
        messages = []
        async for m in channel.history(limit=limit):
            messages.append({
                "id":      str(m.id),
                "author":  str(m.author),
                "content": m.content,
                "ts":      m.created_at.isoformat(),
            })
        return {"messages": messages}
    except Exception as e:
        return {"messages": [], "error": str(e)}


@app.post("/discord/set_nickname")
async def set_nickname(req: NicknameRequest):
    bot_client = _get_bot_client(req.bot)
    try:
        guild  = bot_client.get_guild(req.guild_id) or await bot_client.fetch_guild(req.guild_id)
        member = guild.get_member(req.user_id) or await guild.fetch_member(req.user_id)
        await member.edit(nick=req.nickname)
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.post("/discord/edit_channel")
async def edit_channel(req: EditChannelRequest):
    bot_client = _get_bot_client(req.bot)
    try:
        channel = bot_client.get_channel(req.channel_id)
        if channel is None:
            channel = await bot_client.fetch_channel(req.channel_id)
        kwargs = {}
        if req.name  is not None: kwargs["name"]  = req.name
        if req.topic is not None: kwargs["topic"] = req.topic
        if req.category_id is not None:
            cat = channel.guild.get_channel(req.category_id) or \
                  await bot_client.fetch_channel(req.category_id)
            kwargs["category"] = cat
        await channel.edit(**kwargs)
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.post("/discord/create_channel")
async def create_channel(req: CreateChannelRequest):
    bot_client = _get_bot_client(req.bot)
    gid = req.guild_id or bot_worker.GUILD_ID
    try:
        guild = bot_client.get_guild(gid) or await bot_client.fetch_guild(gid)
        category = None
        if req.category_id is not None:
            category = guild.get_channel(req.category_id) or \
                       await bot_client.fetch_channel(req.category_id)
        ch = await guild.create_text_channel(req.name, topic=req.topic, category=category)
        return {"ok": True, "channel_id": str(ch.id), "name": ch.name}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.post("/discord/delete_channel")
async def delete_channel(req: DeleteChannelRequest):
    bot_client = _get_bot_client(req.bot)
    try:
        ch = bot_client.get_channel(req.channel_id) or \
             await bot_client.fetch_channel(req.channel_id)
        await ch.delete()
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.get("/discord/list_channels")
async def list_channels(guild_id: int | None = None, bot: str = "mod"):
    bot_client = _get_bot_client(bot)
    gid = guild_id or bot_worker.GUILD_ID
    try:
        guild = bot_client.get_guild(gid) or await bot_client.fetch_guild(gid)

        async def _last_ts(ch) -> str | None:
            try:
                msgs = [m async for m in ch.history(limit=1)]
                return msgs[0].created_at.isoformat() if msgs else None
            except Exception:
                return None

        text_channels = [c for c in guild.channels if c.type == discord.ChannelType.text]
        timestamps    = await asyncio.gather(*[_last_ts(c) for c in text_channels])
        ts_map        = {c.id: ts for c, ts in zip(text_channels, timestamps)}

        channels = []
        for ch in guild.channels:
            channels.append({
                "id":              str(ch.id),
                "name":            ch.name,
                "type":            str(ch.type),
                "category_id":     str(ch.category_id) if ch.category_id else None,
                "category_name":   ch.category.name if ch.category else None,
                "topic":           getattr(ch, "topic", None),
                "position":        ch.position,
                "last_message_ts": ts_map.get(ch.id),
            })
        return {"channels": channels}
    except Exception as e:
        return {"channels": [], "error": str(e)}


@app.post("/discord/create_category")
async def create_category(req: CreateCategoryRequest):
    bot_client = _get_bot_client(req.bot)
    gid = req.guild_id or bot_worker.GUILD_ID
    try:
        guild = bot_client.get_guild(gid) or await bot_client.fetch_guild(gid)
        cat   = await guild.create_category(req.name)
        return {"ok": True, "category_id": str(cat.id), "name": cat.name}
    except Exception as e:
        return {"ok": False, "error": str(e)}
