"""
Discord service — FastAPI HTTP API (port 4000) + both bots.

Endpoints (called by sandbox discord_* tools):
  POST /discord/send           — send a message via named bot
  GET  /discord/read           — fetch recent messages from a channel
  POST /discord/set_nickname   — rename a guild member
  POST /discord/edit_channel   — update channel name/topic
  GET  /health
"""

import asyncio
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel

import bot_config
import bot_worker


def _log_task_error(task: asyncio.Task) -> None:
    """Surface bot task exceptions to the container log instead of swallowing them."""
    if not task.cancelled() and task.exception():
        print(f"[discord] bot task {task.get_name()} failed: {task.exception()}", flush=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    tasks = [
        asyncio.create_task(bot_worker.run(), name="worker-bot"),
        asyncio.create_task(bot_config.run(), name="config-bot"),
    ]
    for t in tasks:
        t.add_done_callback(_log_task_error)
    yield
    for t in tasks:
        t.cancel()


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
    channel_id: int
    name:       str | None = None
    topic:      str | None = None
    bot:        str = "worker"


# ── Routes ──────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"ok": True}


@app.post("/discord/send")
async def send_message(req: SendRequest):
    bot_client = bot_worker.client if req.bot == "worker" else bot_config.client
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
    bot_client = bot_worker.client if bot == "worker" else bot_config.client
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
    bot_client = bot_worker.client if req.bot == "worker" else bot_config.client
    try:
        guild  = bot_client.get_guild(req.guild_id) or await bot_client.fetch_guild(req.guild_id)
        member = guild.get_member(req.user_id) or await guild.fetch_member(req.user_id)
        await member.edit(nick=req.nickname)
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.post("/discord/edit_channel")
async def edit_channel(req: EditChannelRequest):
    bot_client = bot_worker.client if req.bot == "worker" else bot_config.client
    try:
        channel = bot_client.get_channel(req.channel_id)
        if channel is None:
            channel = await bot_client.fetch_channel(req.channel_id)
        kwargs = {}
        if req.name  is not None: kwargs["name"]  = req.name
        if req.topic is not None: kwargs["topic"] = req.topic
        await channel.edit(**kwargs)
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "error": str(e)}
