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
import io
import os
import wave
from contextlib import asynccontextmanager

import discord
from fastapi import FastAPI
from piper.voice import PiperVoice
from pydantic import BaseModel

import bot_config
import bot_mod
import bot_worker

# ── Piper TTS ────────────────────────────────────────────────────────────────

_PIPER_MODEL  = os.environ.get("PIPER_MODEL", "/models/en_US-ryan-low.onnx")
_piper_voice: PiperVoice | None = None


def _get_voice() -> PiperVoice:
    global _piper_voice
    if _piper_voice is None:
        _piper_voice = PiperVoice.load(_PIPER_MODEL)
    return _piper_voice


def _synthesize(text: str) -> bytes:
    """Synthesize text → WAV bytes (s16le, mono, native sample rate)."""
    import numpy as np
    voice  = _get_voice()
    chunks = list(voice.synthesize(text))
    if not chunks:
        raise ValueError("Piper returned no audio chunks")
    # Concatenate all sentence chunks, convert float32 → int16
    audio = np.concatenate([c.audio_float_array for c in chunks])
    audio = np.clip(audio, -1.0, 1.0)
    pcm16 = (audio * 32767).astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(chunks[0].sample_rate)
        wf.writeframes(pcm16.tobytes())
    buf.seek(0)
    return buf.read()


def _wav_to_discord_pcm(wav_bytes: bytes) -> bytes:
    """Convert WAV bytes → raw s16le PCM at 48 kHz stereo for discord.PCMAudio."""
    import audioop
    buf = io.BytesIO(wav_bytes)
    with wave.open(buf, "rb") as wf:
        framerate = wf.getframerate()
        nchannels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        frames    = wf.readframes(wf.getnframes())
    if framerate != 48000:
        frames, _ = audioop.ratecv(frames, sampwidth, nchannels, framerate, 48000, None)
    if nchannels == 1:
        frames = audioop.tostereo(frames, sampwidth, 1, 1)
    return frames


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

class ApprovalRequest(BaseModel):
    tool:        str
    params:      dict
    approval_id: str
    session_id:  str

class SpeakRequest(BaseModel):
    channel_id: int
    text:       str
    bot:        str = "worker"

class SpeakVoiceRequest(BaseModel):
    voice_channel_id: int
    text:             str
    bot:              str = "worker"


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


@app.post("/discord/request_approval")
async def request_approval(req: ApprovalRequest):
    """Called by mab-api when a tool needs user approval. Shows buttons in Discord."""
    channel_id = bot_worker.get_channel_for_session(req.session_id)
    if channel_id is None:
        return {"ok": False, "error": "No channel for session"}
    channel = bot_worker.client.get_channel(channel_id)
    if channel is None:
        try:
            channel = await bot_worker.client.fetch_channel(channel_id)
        except Exception:
            return {"ok": False, "error": f"Cannot fetch channel {channel_id}"}
    mode = bot_worker.get_mode_for_channel(channel_id)
    embed = bot_worker._make_approval_embed({"tool": req.tool, "params": req.params}, mode)
    view = bot_worker.CallbackApprovalView(req.approval_id, req.tool, req.params, req.session_id)
    sent = await channel.send(embed=embed, view=view)
    view.message = sent
    bot_worker._stop_thinking(req.session_id)
    return {"ok": True}


@app.post("/discord/speak")
async def speak_message(req: SpeakRequest):
    """Generate TTS audio from text and send it as a WAV file to a Discord channel."""
    bot_client = _get_bot_client(req.bot)
    try:
        channel = bot_client.get_channel(req.channel_id)
        if channel is None:
            channel = await bot_client.fetch_channel(req.channel_id)
        loop      = asyncio.get_event_loop()
        wav_bytes = await loop.run_in_executor(None, _synthesize, req.text)
        audio     = discord.File(io.BytesIO(wav_bytes), filename="response.wav")
        msg       = await channel.send(file=audio)
        return {"ok": True, "message_id": str(msg.id)}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.post("/discord/speak_voice")
async def speak_voice(req: SpeakVoiceRequest):
    """Synthesize TTS and play it directly in a Discord voice channel."""
    bot_client = _get_bot_client(req.bot)
    vc = None
    try:
        voice_channel = bot_client.get_channel(req.voice_channel_id)
        if voice_channel is None:
            voice_channel = await bot_client.fetch_channel(req.voice_channel_id)

        loop      = asyncio.get_event_loop()
        wav_bytes = await loop.run_in_executor(None, _synthesize, req.text)
        pcm_bytes = await loop.run_in_executor(None, _wav_to_discord_pcm, wav_bytes)

        vc     = await voice_channel.connect(timeout=10.0, reconnect=False)
        source = discord.PCMAudio(io.BytesIO(pcm_bytes))

        done = asyncio.Event()

        def _after(err):
            if err:
                print(f"[voice] playback error: {err}", flush=True)
            loop.call_soon_threadsafe(done.set)

        vc.play(source, after=_after)
        await asyncio.wait_for(done.wait(), timeout=120.0)
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "error": str(e)}
    finally:
        if vc and vc.is_connected():
            await vc.disconnect(force=True)
