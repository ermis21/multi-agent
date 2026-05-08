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
import time
import wave
from contextlib import asynccontextmanager

import audioop

import discord
import httpx
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

import bot_config
import bot_mod
import bot_worker
from tts_backend import get_backend

# ── TTS ──────────────────────────────────────────────────────────────────────
# Backend chosen by `cfg.tts.backend` from /config (live-mtime-cached on the
# api side). Defaults to "piper" until the user opts in via config_agent or
# direct PATCH. See discord/tts_backend.py for the per-backend implementations.

PHOEBE_API_URL_LOCAL = os.environ.get("PHOEBE_API_URL", "http://phoebe-vpn:8090")


async def _resolve_backend_name() -> str:
    """Read `cfg.tts.backend` over HTTP. Falls back to 'piper' on any error."""
    try:
        async with httpx.AsyncClient(timeout=3) as c:
            r = await c.get(f"{PHOEBE_API_URL_LOCAL}/config")
            r.raise_for_status()
            return (r.json().get("tts") or {}).get("backend") or "piper"
    except Exception:
        return "piper"


def _synthesize(text: str, backend_name: str) -> bytes:
    """Synthesize text → WAV bytes (s16le, mono, native sample rate).
    Concatenates the streaming chunks at the call site for the file path."""
    backend = get_backend(backend_name)
    chunks = list(backend.synthesize(text))
    if not chunks:
        raise ValueError(f"{backend_name} returned no audio chunks")
    sample_rate = chunks[0][1]
    audio = np.concatenate([c[0] for c in chunks])

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio.tobytes())
    buf.seek(0)
    return buf.read()


def _wav_to_discord_pcm(wav_bytes: bytes) -> bytes:
    """Convert WAV bytes → raw s16le PCM at 48 kHz stereo for discord.PCMAudio."""
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


app = FastAPI(title="phoebe-discord", lifespan=lifespan, docs_url=None, redoc_url=None)


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
    block:            bool = True  # legacy default; listen button passes False


# ── Routes ──────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"ok": True}


@app.get("/discord/in_flight")
async def channel_in_flight(channel_id: int):
    """Authoritative bot-side readiness check for a channel.

    Returns {"in_flight": bool, "session_id": str|None}. The bot clears
    `_channel_in_flight` only after the SSE stream finishes rendering, which
    is strictly after the api clears `_active_sessions` — so this endpoint is
    the correct signal for e2e scenarios to gate on before starting a new run.
    """
    sid = bot_worker._channel_in_flight.get(channel_id)
    return {"in_flight": sid is not None, "session_id": sid}


@app.post("/discord/channel_reset")
async def channel_reset(channel_id: int):
    """Escape hatch: forcibly clear a channel's in-flight flag.

    Used by the e2e runner when `wait_for_idle` times out. Also cancels any
    thinking indicator tied to the stale session."""
    sid = bot_worker._channel_in_flight.pop(channel_id, None)
    if sid:
        bot_worker._stop_thinking(sid)
    return {"cleared": sid is not None, "session_id": sid}


@app.post("/discord/session_reset")
async def session_reset(channel_id: int, user_id: int | None = None):
    """Rotate the session id for a channel — no prior conversation history will
    be replayed into the next message. Used by the e2e runner before each
    scenario to prevent context bleed (stale write_config tool-calls from prior
    runs were the load-bearing cause)."""
    return bot_worker.reset_channel_session(channel_id, user_id)


class PurgeRequest(BaseModel):
    channel_id: int
    bot: str = "mod"


@app.post("/discord/purge_channel")
async def purge_channel(req: PurgeRequest):
    """Delete every message in the channel that the requesting bot can reach.

    Used by the e2e runner at startup to clear leftover state from prior runs.
    Uses the mod bot by default (has Manage Messages). Caps at 1000 messages
    to avoid pathological loops."""
    bot_client = _get_bot_client(req.bot)
    try:
        channel = bot_client.get_channel(req.channel_id) or \
                  await bot_client.fetch_channel(req.channel_id)
        deleted = await channel.purge(limit=1000)
        return {"ok": True, "deleted": len(deleted)}
    except Exception as e:
        return {"ok": False, "error": str(e)}


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
        # Register channel with worker bot so on_message routes messages here
        ch_id = ch.id
        if ch_id not in bot_worker.WORKER_CHANNEL_IDS:
            bot_worker.WORKER_CHANNEL_IDS.add(ch_id)
            bot_worker._channel_sessions[ch_id] = f"discord_{ch_id}_{int(time.time())}"
            bot_worker._save_state()
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
async def list_channels(
    guild_id: int | None = None,
    bot: str = "mod",
    names_only: bool = False,
):
    """List guild channels. `names_only=true` skips the per-channel
    history(limit=1) probe used to compute `last_message_ts` — cuts latency
    from ~N round-trips to Discord down to zero. Cheap-to-call consumers like
    the dream-candidates preview use this fast path."""
    bot_client = _get_bot_client(bot)
    gid = guild_id or bot_worker.GUILD_ID
    try:
        guild = bot_client.get_guild(gid) or await bot_client.fetch_guild(gid)

        if names_only:
            ts_map: dict[int, str | None] = {}
        else:
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


class AskQuestionRequest(BaseModel):
    question:    str
    options:     list[str]
    question_id: str
    session_id:  str


@app.post("/discord/ask_question")
async def ask_question(req: AskQuestionRequest):
    """Post a multiple-choice question to the user's Discord channel."""
    channel_id = bot_worker.get_channel_for_session(req.session_id)
    if channel_id is None:
        return {"ok": False, "error": "No channel for session"}
    channel = bot_worker.client.get_channel(channel_id)
    if channel is None:
        try:
            channel = await bot_worker.client.fetch_channel(channel_id)
        except Exception:
            return {"ok": False, "error": f"Cannot fetch channel {channel_id}"}

    view = bot_worker.QuestionView(req.question_id, req.options, req.session_id)
    embed = discord.Embed(
        title="Clarification needed",
        description=req.question,
        color=0x5865F2,
    )
    letters = "ABCDE"
    for i, opt in enumerate(req.options):
        embed.add_field(name=f"{letters[i]}", value=opt, inline=False)

    view.message = await channel.send(embed=embed, view=view)
    bot_worker._stop_thinking(req.session_id)
    return {"ok": True}


@app.post("/discord/request_approval")
async def request_approval(req: ApprovalRequest):
    """Called by phoebe-api when a tool needs user approval. Shows buttons in Discord."""
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
        backend_name = await _resolve_backend_name()
        loop      = asyncio.get_event_loop()
        wav_bytes = await loop.run_in_executor(None, _synthesize, req.text, backend_name)
        audio     = discord.File(io.BytesIO(wav_bytes), filename="response.wav")
        msg       = await channel.send(file=audio)
        return {"ok": True, "message_id": str(msg.id)}
    except Exception as e:
        return {"ok": False, "error": str(e)}


# ── Voice playback registry (A3 listen button + auto-dictate) ───────────────
# Tracks live VoiceClient handles per voice-channel so the SpeakView pause /
# resume buttons (in views.py) can act on a specific playback. Cleared by the
# `after` callback when audio ends naturally OR by /discord/playback/stop.

_active_playback: dict[int, dict] = {}  # voice_channel_id → {vc, finished_event, paused}


async def _start_voice_playback(
    voice_channel_id: int,
    text: str,
    bot_name: str,
    block: bool,
) -> dict:
    """Synthesize + start playing in a voice channel. If `block=True` waits
    for the audio to finish (legacy /speak shape); else returns as soon as
    playback starts so the View can attach pause/resume controls."""
    bot_client = _get_bot_client(bot_name)
    voice_channel = bot_client.get_channel(voice_channel_id)
    if voice_channel is None:
        voice_channel = await bot_client.fetch_channel(voice_channel_id)

    backend_name = await _resolve_backend_name()
    loop = asyncio.get_event_loop()
    wav_bytes = await loop.run_in_executor(None, _synthesize, text, backend_name)
    pcm_bytes = await loop.run_in_executor(None, _wav_to_discord_pcm, wav_bytes)

    # If something is still playing in this VC, stop + disconnect it first.
    prior = _active_playback.pop(voice_channel_id, None)
    if prior:
        try:
            if prior["vc"].is_playing() or prior["vc"].is_paused():
                prior["vc"].stop()
            if prior["vc"].is_connected():
                await prior["vc"].disconnect(force=True)
        except Exception as e:
            print(f"[voice] prior cleanup warn: {e}", flush=True)

    vc = await voice_channel.connect(timeout=10.0, reconnect=False)
    source = discord.PCMAudio(io.BytesIO(pcm_bytes))
    done = asyncio.Event()
    state = {"vc": vc, "finished_event": done, "paused": False}
    _active_playback[voice_channel_id] = state

    def _after(err):
        if err:
            print(f"[voice] playback error: {err}", flush=True)
        loop.call_soon_threadsafe(done.set)

    async def _disconnect_when_done() -> None:
        try:
            await asyncio.wait_for(done.wait(), timeout=600.0)
        except asyncio.TimeoutError:
            print(f"[voice] playback timeout (10min) on {voice_channel_id}", flush=True)
        finally:
            _active_playback.pop(voice_channel_id, None)
            try:
                if vc.is_connected():
                    await vc.disconnect(force=True)
            except Exception:
                pass

    vc.play(source, after=_after)
    if block:
        # Legacy / auto-dictate path: hold the request open until done.
        await asyncio.wait_for(done.wait(), timeout=120.0)
        _active_playback.pop(voice_channel_id, None)
        try:
            if vc.is_connected():
                await vc.disconnect(force=True)
        except Exception:
            pass
        return {"ok": True}
    # Listen-button path: return immediately, run cleanup as a bg task.
    asyncio.create_task(_disconnect_when_done(), name=f"voice_cleanup_{voice_channel_id}")
    return {"ok": True, "voice_channel_id": voice_channel_id}


@app.post("/discord/speak_voice")
async def speak_voice(req: SpeakVoiceRequest):
    """Synthesize TTS and play it in a Discord voice channel.

    Default `block=True` matches the legacy shape (used by auto-dictate).
    The listen button uses `block=False` to hand control back so pause /
    resume buttons can act mid-playback."""
    block = getattr(req, "block", True)
    try:
        return await _start_voice_playback(req.voice_channel_id, req.text, req.bot, block)
    except Exception as e:
        return {"ok": False, "error": str(e)}


class PlaybackControlRequest(BaseModel):
    voice_channel_id: int


@app.post("/discord/playback/pause")
async def playback_pause(req: PlaybackControlRequest):
    state = _active_playback.get(req.voice_channel_id)
    if not state or not state["vc"].is_connected():
        return {"ok": False, "error": "no active playback"}
    try:
        state["vc"].pause()
        state["paused"] = True
        return {"ok": True, "paused": True}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.post("/discord/playback/resume")
async def playback_resume(req: PlaybackControlRequest):
    state = _active_playback.get(req.voice_channel_id)
    if not state or not state["vc"].is_connected():
        return {"ok": False, "error": "no active playback"}
    try:
        state["vc"].resume()
        state["paused"] = False
        return {"ok": True, "paused": False}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.get("/discord/playback/status")
async def playback_status(voice_channel_id: int):
    state = _active_playback.get(voice_channel_id)
    if not state:
        return {"active": False, "playing": False, "paused": False}
    vc = state["vc"]
    return {
        "active":  True,
        "playing": vc.is_playing(),
        "paused":  state["paused"] or vc.is_paused(),
    }
