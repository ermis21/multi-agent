"""
Worker Discord bot — bidirectional: listens in the phoebe-worker channel and
routes messages to phoebe-api, then posts the response back.

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
from types import SimpleNamespace

import discord
from discord import app_commands
import httpx

from utils import is_allowed, split_message

PHOEBE_API_URL      = os.environ.get("PHOEBE_API_URL",             "http://phoebe-api:8090")
WORKER_TOKEN     = os.environ.get("DISCORD_TOKEN_WORKER",   "")
GUILD_ID         = int(os.environ.get("DISCORD_GUILD_ID",   "0"))
WORKER_NICKNAME  = os.environ.get("DISCORD_WORKER_NICKNAME", "Gemma")

# Test-driver bot user id: when set, this specific bot author is NOT ignored by
# the author.bot early-return in on_message. Used by discord/e2e_scenarios.py
# so the config bot can drive the worker from another bot account.
TEST_DRIVER_USER_ID = int(os.environ.get("DISCORD_TEST_DRIVER_USER_ID", "0") or "0")

# When set to "1", enables !-prefixed text-command fallbacks (!mode, !model, !btw,
# !stop) equivalent to the slash commands. One bot cannot invoke another bot's
# slash commands, so E2E scenarios use these text fallbacks.
ENABLE_TEXT_COMMANDS = os.environ.get("PHOEBE_ENABLE_TEXT_COMMANDS", "").strip() == "1"

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
_http  = httpx.AsyncClient(timeout=600)

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

# Per-session plan state — populated when user accepts a plan via PlanReviewView
_session_plans: dict[str, str] = {}                  # session_id → plan markdown text
_session_privileged_paths: dict[str, list[str]] = {}  # session_id → scope paths (for Accept+Privileged)

# channel_id → session_id currently running a worker loop in that channel.
# Drives mid-flight injection dispatcher + /btw + /stop slash commands.
_channel_in_flight: dict[int, str] = {}


# ── Session-state sync (sessions/{sid}.state.json via api) ────────────────────
# The in-memory dicts above (_session_plans, _session_privileged_paths,
# _session_always_allow) remain the fast path read during on_message; every
# mutation also fires a best-effort PATCH to the api so the persistent state
# file stays in sync and other subsystems (diagnostics, future UIs) can read it.

def _sync_session_state(session_id: str) -> None:
    """Fire-and-forget PATCH that mirrors the three per-session bot dicts into
    `sessions/{sid}.state.json`. Never blocks the caller or raises."""
    if not session_id:
        return
    patch = {
        "plan": _session_plans.get(session_id, "") or None,
        "permissions": {
            "privileged_paths": list(_session_privileged_paths.get(session_id, []) or []),
            "approved_tools": sorted(_session_always_allow.get(session_id, set()) or set()),
        },
    }
    async def _do() -> None:
        try:
            await _http.patch(
                f"{PHOEBE_API_URL}/v1/sessions/{session_id}/state",
                json=patch, timeout=5,
            )
        except Exception:
            pass
    try:
        asyncio.create_task(_do())
    except RuntimeError:
        # No running loop yet (startup edge case) — skip; next mutation will sync.
        pass

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


def _format_single_trace(t: dict) -> str:
    """Format one tool trace as Discord subtext."""
    name     = t.get("tool", "?")
    duration = t.get("duration_s", 0)
    error    = t.get("error")
    preview  = (t.get("params_preview") or "").strip()
    suffix   = f" {preview}" if preview else ""
    if error:
        return f"-# ⚠ {name}{suffix} ({duration:.2f}s) {error}"
    n = t.get("lines", 0)
    return f"-# {name}{suffix} ({duration:.2f}s, {n} lines)"


def _format_tool_started(data: dict) -> str:
    """Format the in-flight placeholder shown while a tool is running."""
    name    = data.get("tool", "?")
    preview = (data.get("params_preview") or "").strip()
    suffix  = f" {preview}" if preview else ""
    return f"-# ⏳ {name}{suffix}…"


def _format_tool_trace(traces: list[dict]) -> str:
    """Format tool call telemetry as Discord subtext (small, muted lines).

    Each line is prefixed with -# so Discord renders it in small grey text,
    keeping tool reports visually subordinate to the actual response.
    Errors get a ⚠ prefix to stay visible despite the small size.
    """
    return "\n".join(_format_single_trace(t) for t in traces)


def _truncate(text: str, limit: int = 900) -> str:
    if len(text) <= limit:
        return text
    return text[:limit - 1] + "…"


def _verdict_color(score: float, threshold: float) -> int:
    """Red for very low scores, orange for mid, yellow near threshold."""
    if score < 0.4:
        return 0xE74C3C  # red
    if score < 0.6:
        return 0xE67E22  # orange
    return 0xF1C40F      # yellow


async def _send_via_mod(channel_id: int, embed: discord.Embed) -> None:
    """Send an embed via the mod bot into the same channel the worker bot is in."""
    import bot_mod  # lazy to avoid any import-order surprises
    mod = bot_mod.client
    if mod is None or not mod.is_ready():
        return  # mod bot not up yet — skip silently
    mod_channel = mod.get_channel(channel_id)
    if mod_channel is None:
        try:
            mod_channel = await mod.fetch_channel(channel_id)
        except Exception:
            return
    try:
        await mod_channel.send(embed=embed)
    except Exception:
        pass


async def _render_supervisor_verdict(channel_id: int, data: dict) -> None:
    attempt   = data.get("attempt", 0)
    score     = float(data.get("score", 0.0))
    threshold = float(data.get("pass_threshold", 0.7))

    embed = discord.Embed(
        title=f"⚠ Supervisor flagged retry (attempt {attempt})",
        description=_truncate(data.get("feedback", "(no summary)"), 1500),
        color=_verdict_color(score, threshold),
    )

    for label, key in [
        ("Tool Usage",         "tool_issues"),
        ("Source Verification", "source_gaps"),
        ("Research",           "research_gaps"),
        ("Accuracy",           "accuracy_issues"),
        ("Completeness",       "completeness_issues"),
    ]:
        items = data.get(key) or []
        if not items:
            continue
        bullets = "\n".join(f"• {item}" for item in items)
        embed.add_field(name=label, value=_truncate(bullets), inline=False)

    suggestions = []
    if data.get("suggest_spawn"):
        suggestions.append(f"spawn: `{data['suggest_spawn']}`")
    if data.get("suggest_debate"):
        suggestions.append(f"debate: {data['suggest_debate']}")
    if suggestions:
        embed.add_field(name="Suggestion", value=_truncate("\n".join(suggestions)), inline=False)

    embed.set_footer(text=f"Score: {score:.2f} / threshold {threshold:.2f}")
    await _send_via_mod(channel_id, embed)


async def _render_worker_review(channel_id: int, data: dict) -> None:
    review = (data.get("review") or "").strip()
    if not review:
        return
    # Amber if the worker pushed back on anything; green if it fully accepted.
    has_rejected = any(
        line.strip().startswith("REJECTED:") for line in review.split("\n")
    )
    color = 0xE67E22 if has_rejected else 0x2ECC71
    embed = discord.Embed(
        title="Worker response to supervisor",
        description=_truncate(review, 1800),
        color=color,
    )
    await _send_via_mod(channel_id, embed)


async def _consume_sse_stream(stream, channel: discord.TextChannel) -> dict | None:
    """Read SSE events from an httpx streaming response, dispatching traces in real-time.

    Returns the full API response dict from the ``done`` event, or None on error.
    """
    current_event: str | None = None
    current_data:  str | None = None
    # call_id → discord.Message for in-place edit when tool_trace lands
    pending_tool_msgs: dict[str, discord.Message] = {}
    async for raw_line in stream.aiter_lines():
        line = raw_line.strip()
        if not line:
            # Blank line marks end of an SSE event
            if current_event and current_data is not None:
                if current_event == "tool_started":
                    try:
                        info = json.loads(current_data)
                        msg  = await channel.send(_format_tool_started(info))
                        call_id = info.get("call_id")
                        if call_id:
                            pending_tool_msgs[call_id] = msg
                    except Exception:
                        pass
                elif current_event == "tool_trace":
                    trace = json.loads(current_data)
                    call_id = trace.get("call_id")
                    msg = pending_tool_msgs.pop(call_id, None) if call_id else None
                    if msg is not None:
                        try:
                            await msg.edit(content=_format_single_trace(trace))
                        except Exception:
                            await channel.send(_format_single_trace(trace))
                    else:
                        await channel.send(_format_single_trace(trace))
                elif current_event == "retry":
                    info = json.loads(current_data)
                    await channel.send(f"-# \u21bb supervisor retry (attempt {info['attempt']})")
                elif current_event == "supervisor_verdict":
                    try:
                        await _render_supervisor_verdict(channel.id, json.loads(current_data))
                    except Exception:
                        pass
                elif current_event == "worker_review":
                    try:
                        await _render_worker_review(channel.id, json.loads(current_data))
                    except Exception:
                        pass
                elif current_event == "worker_status":
                    try:
                        info = json.loads(current_data)
                        text = (info.get("text") or "").strip()
                        if text:
                            snippet = text.replace("\n", " ")
                            if len(snippet) > 500:
                                snippet = snippet[:500] + "…"
                            await channel.send(f"-# \U0001f4ad {snippet}")
                    except Exception:
                        pass
                elif current_event == "done":
                    return json.loads(current_data)
                elif current_event == "error":
                    err = json.loads(current_data)
                    await channel.send(f"[stream error: {err.get('error', 'unknown')}]")
                    return None
            current_event = None
            current_data = None
            continue
        if line.startswith("event:"):
            current_event = line[len("event:"):].strip()
        elif line.startswith("data:"):
            current_data = line[len("data:"):].strip()
    return None  # stream ended without a done event


def get_channel_for_session(session_id: str) -> int | None:
    """Look up the Discord channel ID associated with a session_id."""
    for ch_id, sid in _channel_sessions.items():
        if sid == session_id:
            return ch_id
    for key, sid in _session_ids.items():
        if sid == session_id:
            return int(key.split("_")[0])
    return None


def reset_channel_session(channel_id: int, user_id: int | None = None) -> dict:
    """Rotate the session id for a channel (and optionally a specific user).

    Returns {"old_session_id": ..., "new_session_id": ...} (either may be None)."""
    old_sids: list[str] = []
    new_sid: str | None = None

    if channel_id in _channel_sessions:
        old_sids.append(_channel_sessions[channel_id])
        new_sid = f"discord_{channel_id}_{int(time.time())}"
        _channel_sessions[channel_id] = new_sid
        _renamed_channels.discard(channel_id)
        _channel_message_counts.pop(channel_id, None)
    elif user_id is not None:
        k = _session_key(channel_id, user_id)
        if k in _session_ids:
            old_sids.append(_session_ids[k])
        new_sid = f"discord_{channel_id}_{user_id}_{int(time.time())}"
        _session_ids[k] = new_sid
    else:
        for key in list(_session_ids.keys()):
            if key.startswith(f"{channel_id}_"):
                old_sids.append(_session_ids.pop(key))

    for sid in old_sids:
        _session_always_allow.pop(sid, None)
        _stop_thinking(sid)
    _channel_in_flight.pop(channel_id, None)
    _save_state()
    return {"old_session_id": old_sids[-1] if old_sids else None, "new_session_id": new_sid}


def get_mode_for_channel(channel_id: int) -> str:
    """Return the agent mode for a channel (first user found), defaulting to converse."""
    for key, mode in _user_modes.items():
        if key.startswith(f"{channel_id}_"):
            return mode
    return "converse"


def _parse_plan_scope(plan_text: str) -> list[str]:
    """Extract backtick-wrapped paths from the ## Scope section of a plan."""
    lines = plan_text.split("\n")
    in_scope = False
    paths: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("## Scope"):
            in_scope = True
            continue
        if in_scope:
            if stripped.startswith("## "):
                break  # next section
            match = re.search(r"`([^`]+)`", stripped)
            if match:
                paths.append(match.group(1))
    return paths


def _set_mode_for_session(session_id: str, mode: str) -> None:
    """Set the agent mode for all users associated with a session's channel."""
    channel_id = get_channel_for_session(session_id)
    if channel_id is None:
        return
    for key in list(_user_modes.keys()):
        if key.startswith(f"{channel_id}_"):
            _user_modes[key] = mode
    # If no user keys exist yet (channel-session only), create a wildcard entry
    if not any(k.startswith(f"{channel_id}_") for k in _user_modes):
        _user_modes[f"{channel_id}_0"] = mode
    _save_state()


# ── State persistence — survives container restarts ───────────────────────────
_STATE_FILE = Path(os.environ.get("STATE_DIR", "/app/state")) / "bot_worker_state.json"


def _save_state() -> None:
    """Write session IDs, user modes, channel sessions, and plan state to disk."""
    try:
        _STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        _STATE_FILE.write_text(
            json.dumps({
                "session_ids":     _session_ids,
                "user_modes":      _user_modes,
                "channel_sessions": {str(k): v for k, v in _channel_sessions.items()},
                "renamed_channels": list(_renamed_channels),
                "session_plans":           _session_plans,
                "session_privileged_paths": _session_privileged_paths,
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
        _session_plans.update(data.get("session_plans", {}))
        _session_privileged_paths.update(data.get("session_privileged_paths", {}))
        print(
            f"[worker-bot] restored {len(_session_ids)} sessions, "
            f"{len(_user_modes)} modes, "
            f"{len(_channel_sessions)} channel-sessions, "
            f"{len(_session_plans)} plans from state file",
            flush=True,
        )
    except Exception as e:
        print(f"[worker-bot] state load failed: {e}", flush=True)

    # Fallback: hydrate from authoritative per-session state files written by the api.
    # Runs after the JSON restore so the bot-local file wins on conflicts, but picks up
    # sessions the api created while the bot was down (or if bot_worker_state.json was lost).
    try:
        import glob as _glob
        hydrated = 0
        for p in _glob.glob("/sessions/*/state.json"):
            try:
                data = json.loads(Path(p).read_text(encoding="utf-8"))
            except Exception:
                continue
            ch = data.get("channel_id")
            sid = data.get("session_id")
            if not (ch and sid):
                continue
            try:
                ch = int(ch)
            except (TypeError, ValueError):
                continue
            if ch not in _channel_sessions:
                _channel_sessions[ch] = sid
                WORKER_CHANNEL_IDS.add(ch)
                hydrated += 1
            plan = data.get("plan")
            if isinstance(plan, dict):
                md = plan.get("markdown") or plan.get("context")
                if md and sid not in _session_plans:
                    _session_plans[sid] = md
            mode = data.get("mode")
            if mode:
                _user_modes.setdefault(f"{ch}_0", mode)
            perms = data.get("permissions") or {}
            pp = perms.get("privileged_paths")
            if pp and sid not in _session_privileged_paths:
                _session_privileged_paths[sid] = list(pp)
            at = perms.get("approved_tools")
            if at and sid not in _session_always_allow:
                _session_always_allow[sid] = set(at)
        if hydrated:
            print(f"[worker-bot] hydrated {hydrated} channel-sessions from /sessions/*/state.json", flush=True)
    except Exception as e:
        print(f"[worker-bot] state hydration from /sessions failed: {e}", flush=True)


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
        resp = await _http.post(f"{PHOEBE_API_URL}/v1/chat/completions", json={
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


class CallbackApprovalView(discord.ui.View):
    """Three-button confirmation UI that POSTs back to phoebe-api to unblock call_tool()."""

    def __init__(self, approval_id: str, tool: str, params: dict, session_id: str):
        super().__init__(timeout=600)
        self.approval_id = approval_id
        self.tool        = tool
        self.params      = params
        self.session_id  = session_id
        self.message: discord.Message | None = None

    async def _respond(self, interaction: discord.Interaction, approved: bool, always: bool = False) -> None:
        await interaction.response.defer()

        # Capture channel before deleting — needed to restart thinking
        channel = self.message.channel if self.message else None

        # Delete the approval embed entirely — keeps the channel clean
        if self.message:
            try:
                await self.message.delete()
            except Exception:
                pass

        if always:
            _session_always_allow.setdefault(self.session_id, set()).add(self.tool)
            _sync_session_state(self.session_id)

        try:
            resp = await _http.post(f"{PHOEBE_API_URL}/v1/approval_response", json={
                "approval_id": self.approval_id,
                "approved": approved,
                "always": always,
            })
            resp.raise_for_status()
        except Exception:
            pass

        # Restart thinking — it was stopped when the approval embed was shown
        if approved and channel:
            _start_thinking(self.session_id, channel)

    async def on_timeout(self) -> None:
        for item in self.children:
            item.disabled = True  # type: ignore[attr-defined]
        if self.message:
            try:
                await self.message.edit(content="⏱️ Approval timed out — action cancelled.", view=self)
            except Exception:
                pass
        # Deny so call_tool() can return instead of hanging
        try:
            await _http.post(f"{PHOEBE_API_URL}/v1/approval_response", json={
                "approval_id": self.approval_id,
                "approved": False,
            })
        except Exception:
            pass

    @discord.ui.button(label="Yes, proceed", style=discord.ButtonStyle.green, emoji="✅")
    async def yes(self, interaction: discord.Interaction, _button: discord.ui.Button) -> None:
        await self._respond(interaction, approved=True)

    @discord.ui.button(label="No, cancel", style=discord.ButtonStyle.red, emoji="❌")
    async def no(self, interaction: discord.Interaction, _button: discord.ui.Button) -> None:
        await self._respond(interaction, approved=False)

    @discord.ui.button(label="Always allow", style=discord.ButtonStyle.blurple, emoji="🔒")
    async def always_allow(self, interaction: discord.Interaction, _button: discord.ui.Button) -> None:
        await self._respond(interaction, approved=True, always=True)


class QuestionView(discord.ui.View):
    """Multiple-choice question buttons. Posts answer back to phoebe-api."""

    def __init__(self, question_id: str, options: list[str], session_id: str):
        super().__init__(timeout=300)
        self.question_id = question_id
        self.options = options
        self.session_id = session_id

        letters = "ABCDE"
        styles = [
            discord.ButtonStyle.primary,
            discord.ButtonStyle.secondary,
            discord.ButtonStyle.success,
            discord.ButtonStyle.primary,
            discord.ButtonStyle.secondary,
        ]

        for i, opt in enumerate(options):
            label = f"{letters[i]}: {opt[:75]}"
            btn = discord.ui.Button(
                label=label,
                style=styles[i % len(styles)],
                custom_id=f"q_{question_id}_{letters[i]}",
            )
            btn.callback = self._make_callback(letters[i], opt)
            self.add_item(btn)

    def _make_callback(self, letter: str, option_text: str):
        async def callback(interaction: discord.Interaction):
            for item in self.children:
                item.disabled = True  # type: ignore[attr-defined]
            await interaction.response.edit_message(
                content=f"Selected: **{letter}** — {option_text}",
                view=self,
            )
            try:
                await _http.post(f"{PHOEBE_API_URL}/v1/question_response", json={
                    "question_id": self.question_id,
                    "answer": letter,
                    "answer_text": option_text,
                })
            except Exception as e:
                print(f"[discord] Failed to send question response: {e}", flush=True)
            self.stop()
        return callback

    async def on_timeout(self) -> None:
        for item in self.children:
            item.disabled = True  # type: ignore[attr-defined]


async def _execute_plan(session_id: str, channel: discord.TextChannel) -> None:
    """Kick off the agent loop to execute an accepted plan.

    Mirrors the on_message flow: streams the API call via SSE and posts
    tool traces + final answer to the channel.
    """
    always_allow     = list(_session_always_allow.get(session_id, set()))
    plan_context     = _session_plans.get(session_id, "")
    privileged_paths = _session_privileged_paths.get(session_id, [])

    _start_thinking(session_id, channel)
    data = None
    try:
        async with _http.stream(
            "POST",
            f"{PHOEBE_API_URL}/v1/chat/completions",
            json={
                "messages":         [{"role": "user", "content": "Execute the plan."}],
                "session_id":       session_id,
                "mode":             "build",
                "approved_tools":   always_allow,
                "stream":           True,
                "plan_context":     plan_context,
                "privileged_paths": privileged_paths,
            },
        ) as stream:
            stream.raise_for_status()
            data = await _consume_sse_stream(stream, channel)
    except Exception as e:
        err_str = str(e) or f"{type(e).__name__} (no message)"
        await channel.send(f"[error: {err_str}]")
        return
    finally:
        _stop_thinking(session_id)

    if data is None:
        await channel.send("[error: stream ended unexpectedly]")
        return

    answer = data.get("choices", [{}])[0].get("message", {}).get("content", "[no response]")
    for chunk in split_message(answer):
        await channel.send(chunk)


class PlanReviewView(discord.ui.View):
    """Three-button view shown when the agent produces a plan in plan mode.

    Accept           → store plan, switch to build mode
    Accept+Privileged → store plan + scope paths as auto-allow, switch to build
    Keep Planning    → user sends feedback, agent revises (session continuity handles this)
    """

    def __init__(self, plan_text: str, session_id: str):
        super().__init__(timeout=None)  # no timeout — plans can be reviewed indefinitely
        self.plan_text  = plan_text
        self.session_id = session_id
        self.message: discord.Message | None = None

    async def _accept(self, interaction: discord.Interaction, privileged: bool = False) -> None:
        await interaction.response.defer()

        # Store the plan
        _session_plans[self.session_id] = self.plan_text

        # Extract and store scope paths if privileged
        if privileged:
            scope_paths = _parse_plan_scope(self.plan_text)
            _session_privileged_paths[self.session_id] = scope_paths
            scope_msg = "\n".join(f"- `{p}`" for p in scope_paths) if scope_paths else "_(none detected)_"
        else:
            _session_privileged_paths.pop(self.session_id, None)

        # Switch to build mode
        _set_mode_for_session(self.session_id, "build")
        _save_state()
        _sync_session_state(self.session_id)

        # Disable buttons
        for item in self.children:
            item.disabled = True  # type: ignore[attr-defined]
        if self.message:
            await self.message.edit(view=self)

        if privileged:
            await interaction.followup.send(
                f"**Plan accepted with privileged access.** Executing plan...\n"
                f"Auto-approved scope:\n{scope_msg}"
            )
        else:
            await interaction.followup.send("**Plan accepted.** Executing plan...")

        # Auto-execute: kick off the agent loop with the plan
        channel = interaction.channel
        if channel is not None:
            asyncio.create_task(
                _execute_plan(self.session_id, channel),
                name=f"plan_exec_{self.session_id}",
            )

    @discord.ui.button(label="Accept", style=discord.ButtonStyle.green, emoji="\u2705")
    async def accept(self, interaction: discord.Interaction, _button: discord.ui.Button) -> None:
        await self._accept(interaction, privileged=False)

    @discord.ui.button(label="Accept + Privileged", style=discord.ButtonStyle.blurple, emoji="\u26a1")
    async def accept_privileged(self, interaction: discord.Interaction, _button: discord.ui.Button) -> None:
        await self._accept(interaction, privileged=True)

    @discord.ui.button(label="Keep Planning", style=discord.ButtonStyle.secondary, emoji="\ud83d\udcdd")
    async def keep_planning(self, interaction: discord.Interaction, _button: discord.ui.Button) -> None:
        await interaction.response.defer()
        for item in self.children:
            item.disabled = True  # type: ignore[attr-defined]
        if self.message:
            await self.message.edit(view=self)
        await interaction.followup.send(
            "Staying in **plan** mode. Type your feedback and the agent will revise the plan."
        )

    async def on_timeout(self) -> None:
        for item in self.children:
            item.disabled = True  # type: ignore[attr-defined]
        if self.message:
            try:
                await self.message.edit(content="Plan review timed out.", view=self)
            except Exception:
                pass


# ── Mid-flight injection dispatcher ──────────────────────────────────────────

_VALID_MODES = {"plan", "build", "converse"}


async def _handle_text_command(msg: discord.Message) -> bool:
    """Parse !-prefixed text commands (mode/model/btw/stop). Returns True if the
    message was a command (whether or not it succeeded) so on_message skips
    normal routing."""
    parts = msg.content[1:].split(maxsplit=1)
    if not parts:
        return False
    cmd = parts[0].lower()
    rest = parts[1].strip() if len(parts) > 1 else ""

    if cmd == "mode":
        if rest not in _VALID_MODES:
            await msg.channel.send(f"Unknown mode. Use: {', '.join(sorted(_VALID_MODES))}.")
            return True
        # For channels with a persistent session, set on that session; else
        # use the legacy per-user+channel key.
        sid = _channel_sessions.get(msg.channel.id)
        if sid:
            _set_mode_for_session(sid, rest)
        else:
            _user_modes[_session_key(msg.channel.id, msg.author.id)] = rest
            _save_state()
        await msg.channel.send(f"Switched to **{rest}** mode.")
        return True

    if cmd == "model":
        if not rest:
            await msg.channel.send(f"Known presets: {', '.join(_MODEL_PRESETS)}")
            return True
        preset = rest.split()[0]
        if preset not in _MODEL_PRESETS:
            await msg.channel.send(f"Unknown preset '{preset}'. Known: {', '.join(_MODEL_PRESETS)}.")
            return True
        try:
            resp = await _http.patch(f"{PHOEBE_API_URL}/config", json={"llm": dict(_MODEL_PRESETS[preset])})
            resp.raise_for_status()
            await msg.channel.send(f"Switched to **{preset}**.")
        except Exception as e:
            await msg.channel.send(f"Model switch failed: {e}")
        return True

    if cmd == "btw":
        sid = _channel_in_flight.get(msg.channel.id)
        if not sid:
            await msg.channel.send("No active run in this channel.")
            return True
        await _post_injection(sid, rest, mode="not_urgent")
        snippet = rest if len(rest) <= 80 else rest[:80] + "…"
        await msg.channel.send(f"-# \U0001f4dd injected: {snippet}")
        return True

    if cmd == "stop":
        sid = _channel_in_flight.get(msg.channel.id)
        if not sid:
            await msg.channel.send("No active run in this channel.")
            return True
        await _post_injection(sid, "", mode="stop")
        await msg.channel.send("\U0001f6d1 stop requested — worker will exit at the next iteration.")
        return True

    # Not a recognised command — fall through to normal message handling.
    return False


async def _post_injection(session_id: str, text: str, mode: str) -> dict | None:
    """POST a mid-flight injection into an in-flight session. Returns response JSON or None."""
    try:
        resp = await _http.post(
            f"{PHOEBE_API_URL}/v1/sessions/{session_id}/inject",
            json={"text": text, "mode": mode},
            timeout=8,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"[worker-bot] inject failed ({mode}): {e}", flush=True)
        return None


class InjectionView(discord.ui.View):
    """Four-button popup when a user messages a channel that has a live worker run.

    Immediate    → appended as a user turn before the next LLM call.
    Not urgent   → stapled onto the next tool_result as a [user_note] block.
    Clarify      → like Not urgent, with "this is clarification, not a new task" suffix.
    Queue        → held; delivered as a new turn after the current run finishes.
    """

    def __init__(self, session_id: str, text: str, author_id: int):
        super().__init__(timeout=120)
        self.session_id = session_id
        self.text = text
        self.author_id = author_id
        self.message: discord.Message | None = None

    async def _handle(self, interaction: discord.Interaction, mode: str, label: str) -> None:
        if interaction.user.id != self.author_id:
            await interaction.response.send_message("This choice belongs to whoever sent the message.", ephemeral=True)
            return
        await interaction.response.defer()
        result = await _post_injection(self.session_id, self.text, mode)
        for item in self.children:
            item.disabled = True  # type: ignore[attr-defined]
        if self.message is not None:
            try:
                if result is None:
                    await self.message.edit(content=f"❌ Injection failed ({label}).", view=self)
                else:
                    await self.message.edit(content=f"✅ Injected as **{label}**.", view=self)
            except Exception:
                pass

    @discord.ui.button(label="Immediate", style=discord.ButtonStyle.danger, emoji="\u26a1")
    async def btn_immediate(self, interaction: discord.Interaction, _button: discord.ui.Button) -> None:
        await self._handle(interaction, "immediate", "immediate")

    @discord.ui.button(label="Not urgent", style=discord.ButtonStyle.primary, emoji="\U0001f4dd")
    async def btn_not_urgent(self, interaction: discord.Interaction, _button: discord.ui.Button) -> None:
        await self._handle(interaction, "not_urgent", "not urgent")

    @discord.ui.button(label="Clarify", style=discord.ButtonStyle.primary, emoji="\U0001f4ac")
    async def btn_clarify(self, interaction: discord.Interaction, _button: discord.ui.Button) -> None:
        await self._handle(interaction, "clarify", "clarification")

    @discord.ui.button(label="Queue", style=discord.ButtonStyle.secondary, emoji="\U0001f5c3\ufe0f")
    async def btn_queue(self, interaction: discord.Interaction, _button: discord.ui.Button) -> None:
        await self._handle(interaction, "queue", "queued")

    async def on_timeout(self) -> None:
        if self.message is not None:
            try:
                for item in self.children:
                    item.disabled = True  # type: ignore[attr-defined]
                await self.message.edit(content="Injection prompt timed out.", view=self)
            except Exception:
                pass


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

    reset_channel_session(channel.id, interaction.user.id)

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
@app_commands.describe(mode="plan (investigate + create plan), build (execute plan), converse (casual chat)")
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
        "plan":     "Investigation mode — investigates, then produces a structured plan for review.",
        "build":    "Execution mode — follows the active plan step by step with full tool access.",
        "converse": "Conversational — answers questions, quick lookups. Suggests plan mode for complex tasks.",
    }
    await interaction.response.send_message(
        f"Switched to **{mode.value}** mode. {descriptions[mode.value]}"
    )


async def _fetch_model_presets() -> dict[str, dict]:
    """Pull the named-model list from phoebe-api's live config. Filters out
    internal-use entries (debate/checkpoint helpers with tiny max_tokens)."""
    try:
        resp = await _http.get(f"{PHOEBE_API_URL}/config", timeout=5)
        resp.raise_for_status()
        models = resp.json().get("models", {}) or {}
    except Exception:
        return {}
    # Drop internal helpers that aren't meaningful as full-session models
    return {
        name: spec for name, spec in models.items()
        if isinstance(spec, dict) and spec.get("max_tokens", 0) >= 1024
    }


@tree.command(name="model", description="Show or switch the current LLM model")
@app_commands.describe(preset="Switch to a named model from config.models (leave blank to show current)")
async def cmd_model(
    interaction: discord.Interaction,
    preset: str | None = None,
):
    if not is_allowed(interaction.user.id):
        await interaction.response.send_message("Not authorized.", ephemeral=True)
        return

    if preset is not None:
        try:
            presets = await _fetch_model_presets()
            if preset not in presets:
                available = ", ".join(sorted(presets)) or "<none>"
                await interaction.response.send_message(
                    f"Unknown model `{preset}`. Available: {available}", ephemeral=True
                )
                return
            patch = dict(presets[preset])
            resp = await _http.patch(
                f"{PHOEBE_API_URL}/config",
                json={"llm": patch},
            )
            resp.raise_for_status()
            llm = resp.json().get("config", {}).get("llm", {})
            # For local providers, probe /models to replace any placeholder model id
            if patch.get("provider") == "local":
                try:
                    r = await _http.get(f"{PHOEBE_API_URL}/models", timeout=8)
                    r.raise_for_status()
                    models_data = r.json().get("data", [])
                    if models_data:
                        real_model = models_data[0]["id"]
                        await _http.patch(f"{PHOEBE_API_URL}/config", json={"llm": {"model": real_model}})
                        llm["model"] = real_model
                except Exception:
                    pass
            msg = (
                f"Switched to **{preset}**.\n"
                f"**Provider**: `{llm.get('provider', 'local')}`\n"
                f"**Model**: `{llm.get('model', '?')}`"
            )
        except Exception as e:
            msg = f"Failed to switch model: {e}"
    else:
        try:
            resp = await _http.get(f"{PHOEBE_API_URL}/config")
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


@cmd_model.autocomplete("preset")
async def _model_autocomplete(interaction: discord.Interaction, current: str):
    presets = await _fetch_model_presets()
    cur = (current or "").lower()
    items: list[app_commands.Choice[str]] = []
    for name, spec in presets.items():
        if cur and cur not in name.lower() and cur not in str(spec.get("model", "")).lower():
            continue
        label = f"{name} — {spec.get('provider', '?')} / {spec.get('model', '?')}"
        items.append(app_commands.Choice(name=label[:100], value=name))
        if len(items) >= 25:  # Discord caps autocomplete at 25 entries
            break
    return items


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
        resp = await _http.get(f"{PHOEBE_API_URL}/config")
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


@tree.command(name="btw", description="Add context to the current run without stopping it")
@app_commands.describe(text="The note to pass to the agent (stapled onto the next tool result)")
async def cmd_btw(interaction: discord.Interaction, text: str):
    if not is_allowed(interaction.user.id):
        await interaction.response.send_message("Not authorized.", ephemeral=True)
        return
    sid = _channel_in_flight.get(interaction.channel_id)
    if not sid:
        await interaction.response.send_message("No active run in this channel.", ephemeral=True)
        return
    result = await _post_injection(sid, text, mode="not_urgent")
    if result is None:
        await interaction.response.send_message("Injection failed.", ephemeral=True)
        return
    snippet = text if len(text) <= 80 else text[:80] + "…"
    await interaction.response.send_message(f"-# \U0001f4dd injected: {snippet}")


@tree.command(name="stop", description="Stop the current worker run in this channel")
async def cmd_stop(interaction: discord.Interaction):
    if not is_allowed(interaction.user.id):
        await interaction.response.send_message("Not authorized.", ephemeral=True)
        return
    sid = _channel_in_flight.get(interaction.channel_id)
    if not sid:
        await interaction.response.send_message("No active run in this channel.", ephemeral=True)
        return
    result = await _post_injection(sid, "", mode="stop")
    if result is None:
        await interaction.response.send_message("Stop request failed.", ephemeral=True)
        return
    await interaction.response.send_message("\U0001f6d1 stop requested — worker will exit at the next iteration.")


@tree.command(name="help", description="List available slash commands")
async def cmd_help(interaction: discord.Interaction):
    msg = (
        "**Available commands:**\n"
        "• `/new` — Start a fresh conversation in a new channel\n"
        "• `/clear` — Reset conversation and delete all messages in this channel\n"
        "• `/mode [plan|build|converse]` — Switch agent mode\n"
        "• `/model` — Show current LLM model and settings\n"
        "• `/speak <prompt>` — Ask the agent a question and get a voice response\n"
        "• `/btw <text>` — Add context mid-flight without stopping the agent\n"
        "• `/stop` — Cancel the currently running worker loop in this channel\n"
        "• `/status` — Show your current mode, session, and system status\n"
        "• `/help` — This message\n\n"
        "**Modes:**\n"
        "• **plan** — Investigate and produce a structured plan for review\n"
        "• **build** — Execute the active plan step by step with full tools\n"
        "• **converse** — Quick questions and casual chat; suggests plan mode for complex tasks\n\n"
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
        resp = await _http.post(f"{PHOEBE_API_URL}/v1/chat/completions", json={
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
    phoebe-api and patch the model name back into config if it changed.
    This keeps the displayed model name current even when the server swaps models.
    """
    await asyncio.sleep(15)  # let the API finish starting before first poll
    while True:
        try:
            cfg_resp = await _http.get(f"{PHOEBE_API_URL}/config", timeout=5)
            cfg_resp.raise_for_status()
            llm = cfg_resp.json().get("llm", {})
            if llm.get("provider", "local") == "local":
                r = await _http.get(f"{PHOEBE_API_URL}/models", timeout=8)
                r.raise_for_status()
                models_data = r.json().get("data", [])
                if models_data:
                    real_model = models_data[0]["id"]
                    if real_model != llm.get("model"):
                        await _http.patch(f"{PHOEBE_API_URL}/config", json={"llm": {"model": real_model}})
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
            ch = await _ensure_channel(guild, "phoebe-worker", "Multi-agent worker chat")
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
    # Allow one designated driver bot (test/e2e) through the bot-ignore gate.
    is_test_driver = TEST_DRIVER_USER_ID and msg.author.id == TEST_DRIVER_USER_ID
    if msg.author.bot and not is_test_driver:
        return
    if not is_test_driver and not is_allowed(msg.author.id):
        return
    if WORKER_CHANNEL_IDS and msg.channel.id not in WORKER_CHANNEL_IDS:
        return

    # Text-command fallbacks (only when explicitly enabled). Bots can't invoke
    # each other's slash commands, so E2E scenarios need this.
    if ENABLE_TEXT_COMMANDS and msg.content.startswith("!"):
        handled = await _handle_text_command(msg)
        if handled:
            return

    # Mid-flight message: the channel already has a worker loop running.
    # Pop the 4-option dispatcher instead of starting a second session.
    inflight_sid = _channel_in_flight.get(msg.channel.id)
    if inflight_sid:
        view = InjectionView(session_id=inflight_sid, text=msg.content, author_id=msg.author.id)
        try:
            sent = await msg.channel.send(
                "⏳ A run is already in progress. How should I handle this message?",
                view=view,
            )
            view.message = sent
        except Exception as e:
            print(f"[worker-bot] injection dispatcher failed: {e}", flush=True)
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

    # Include plan context and privileged paths if a plan is active
    plan_context     = _session_plans.get(session_id, "")
    privileged_paths = _session_privileged_paths.get(session_id, [])

    _start_thinking(session_id, msg.channel)
    _channel_in_flight[msg.channel.id] = session_id
    data = None
    try:
        async with _http.stream(
            "POST",
            f"{PHOEBE_API_URL}/v1/chat/completions",
            json={
                "messages":         [{"role": "user", "content": msg.content}],
                "session_id":       session_id,
                "mode":             mode,
                "approved_tools":   always_allow,
                "stream":           True,
                "plan_context":     plan_context,
                "privileged_paths": privileged_paths,
            },
        ) as stream:
            stream.raise_for_status()
            data = await _consume_sse_stream(stream, msg.channel)
    except Exception as e:
        err_str = str(e) or f"{type(e).__name__} (no message)"
        await msg.channel.send(f"[error: {err_str}]")
        _channel_in_flight.pop(msg.channel.id, None)
        return
    finally:
        _stop_thinking(session_id)
        _channel_in_flight.pop(msg.channel.id, None)

    if data is None:
        await msg.channel.send("[error: stream ended unexpectedly]")
        return

    answer = data.get("choices", [{}])[0].get("message", {}).get("content", "[no response]")

    # Detect MODE_SWITCH sentinel from converse mode → auto-switch to plan
    if "[MODE_SWITCH: plan]" in answer:
        answer = answer.replace("[MODE_SWITCH: plan]", "").strip()
        _set_mode_for_session(session_id, "plan")

    is_plan_output = (
        mode == "plan"
        and re.search(r"^## Scope\b", answer, re.MULTILINE) is not None
        and re.search(r"^## Steps\b", answer, re.MULTILINE) is not None
    )

    chunks = split_message(answer)
    for i, chunk in enumerate(chunks):
        is_last = i == len(chunks) - 1
        if is_last and is_plan_output:
            view = PlanReviewView(answer, session_id)
            sent = await msg.channel.send(chunk, view=view)
            view.message = sent
        elif is_last:
            await msg.channel.send(chunk, view=SpeakView(answer, msg.channel.id))
        else:
            await msg.channel.send(chunk)

    # Replay any "queue"-mode injections captured during the run as fresh turns.
    queued = data.get("queued_injections") or []
    for q_text in queued:
        if not q_text:
            continue
        try:
            await msg.channel.send(f"-# \U0001f5c3\ufe0f replaying queued: {q_text[:120]}")
            fake = SimpleNamespace(
                author=msg.author, channel=msg.channel, content=q_text,
                guild=msg.guild,
            )
            await on_message(fake)  # type: ignore[arg-type]
        except Exception as e:
            print(f"[worker-bot] queued-injection replay failed: {e}", flush=True)


async def run():
    if not WORKER_TOKEN:
        print("[worker-bot] DISCORD_TOKEN_WORKER not set — skipping", flush=True)
        return
    await client.start(WORKER_TOKEN)
