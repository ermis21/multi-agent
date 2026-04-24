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
from views import (
    CallbackApprovalView,
    DreamEditReviewView,
    InjectionView,
    PlanReviewView,
    QuestionView,
    SpeakView,
)

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
_thinking_started_at: dict[str, float] = {}     # session_id → epoch seconds
_thinking_watchdog_task: asyncio.Task | None = None
_THINKING_MAX_AGE_S = 15 * 60  # force-cancel after 15 min — catches mod-bot crashes


def set_thinking_client(client_ref: discord.Client) -> None:
    global _thinking_client
    _thinking_client = client_ref


async def _thinking_watchdog() -> None:
    """Sweep stale thinking indicators every 60s.

    Protects against mod-bot crashes + orphan tasks — without this, a missed
    `_stop_thinking` call leaves the nickname stranded as "🧠 thinking" and
    the typing loop running indefinitely.
    """
    while True:
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            return
        now = time.time()
        for sid, started in list(_thinking_started_at.items()):
            if now - started > _THINKING_MAX_AGE_S:
                print(f"[thinking] watchdog force-cancel sid={sid} age={now - started:.0f}s",
                      flush=True)
                _stop_thinking(sid)


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
    global _thinking_watchdog_task
    if session_id in _thinking_tasks:
        return
    task = asyncio.create_task(
        _run_thinking_indicator(channel),
        name=f"thinking_{session_id}",
    )
    _thinking_tasks[session_id] = task
    _thinking_started_at[session_id] = time.time()
    # Launch the sweeper lazily on first indicator start.
    if _thinking_watchdog_task is None or _thinking_watchdog_task.done():
        _thinking_watchdog_task = asyncio.create_task(
            _thinking_watchdog(), name="thinking_watchdog"
        )


def _stop_thinking(session_id: str) -> None:
    task = _thinking_tasks.pop(session_id, None)
    _thinking_started_at.pop(session_id, None)
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
    max_att   = data.get("max_attempts") or 0
    score     = float(data.get("score", 0.0))
    threshold = float(data.get("pass_threshold", 0.7))

    # "attempt 2/3" reads as progress; bare "attempt 2" reads as a dupe.
    title_suffix = f"attempt {attempt}/{max_att}" if max_att else f"attempt {attempt}"
    embed = discord.Embed(
        title=f"⚠ Supervisor flagged retry ({title_suffix})",
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


async def _consume_sse_stream(
    stream,
    channel: discord.TextChannel,
    bot_msg_ids: list[str] | None = None,
) -> dict | None:
    """Read SSE events from an httpx streaming response, dispatching traces in real-time.

    Returns the full API response dict from the ``done`` event, or None on error.

    When `bot_msg_ids` is provided, every discord.Message authored by this bot
    (tool-start stubs, tool traces, retry/status chirps, error notices) is
    recorded into the list so the caller can index them against the turn for
    later message-edit rewind. Mod-bot embeds (supervisor verdict, worker
    review) are NOT tracked — different author, not deletable by this bot.
    """

    def _track(m: discord.Message | None) -> None:
        if m is not None and bot_msg_ids is not None:
            bot_msg_ids.append(str(m.id))

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
                        _track(msg)
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
                            _track(await channel.send(_format_single_trace(trace)))
                    else:
                        _track(await channel.send(_format_single_trace(trace)))
                elif current_event == "retry":
                    info = json.loads(current_data)
                    _track(await channel.send(f"-# \u21bb supervisor retry (attempt {info['attempt']})"))
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
                            _track(await channel.send(f"-# \U0001f4ad {snippet}"))
                    except Exception:
                        pass
                elif current_event == "done":
                    return json.loads(current_data)
                elif current_event == "error":
                    err = json.loads(current_data)
                    _track(await channel.send(f"[stream error: {err.get('error', 'unknown')}]"))
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

def _approval_intent_summary(tool: str, params: dict) -> str:
    """One-sentence plain-English intent for an approval prompt.

    Raw JSON params + bare tool name invited misclicks on "Always allow" —
    this reduces the cognitive load to a single skimmable line. Falls back
    to tool + short params for tools we don't have a template for.
    """
    def _short(v, n=80):
        s = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
        return s if len(s) <= n else s[:n - 1] + "…"

    if tool in ("file_write", "file_edit"):
        return f"Write to `{params.get('path', '?')}`"
    if tool == "file_move":
        return f"Move `{params.get('src', '?')}` → `{params.get('dst', '?')}`"
    if tool in ("shell_exec", "execute_command"):
        return f"Run shell: `{_short(params.get('command', ''))}`"
    if tool == "git_commit":
        return f"Git commit: {_short(params.get('message', 'agent: automated update'), 100)}"
    if tool == "git_rollback":
        return "Git rollback: revert HEAD"
    if tool == "write_config":
        patch = params.get("patch") or params
        keys = list(patch.keys()) if isinstance(patch, dict) else []
        return f"Change config: {', '.join(keys) or 'unspecified'}"
    if tool == "web_fetch":
        return f"Fetch web page: {_short(params.get('url', '?'), 100)}"
    if tool in ("discord_send", "discord_edit_channel", "discord_create_channel",
                "discord_delete_channel"):
        return f"Discord: `{tool}` on channel `{params.get('channel_id', '?')}`"
    if tool == "docker_test_up":
        return "Start the Docker test stack"
    if tool == "run_agent":
        return f"Spawn sub-agent: `{params.get('role', '?')}`"
    if tool == "deliberate":
        return "Start a multi-advocate deliberation"
    # Fallback: tool name + short params preview.
    short_params = _short(params, 100)
    return f"Call `{tool}` with {short_params}"


def _make_approval_embed(approval: dict, mode: str) -> discord.Embed:
    """Build the orange approval-request embed shown when a tool needs confirmation."""
    tool   = approval["tool"]
    params = approval.get("params", {})
    intent = _approval_intent_summary(tool, params)
    embed  = discord.Embed(
        title=f"🔐  Approval Required: `{tool}`",
        description=intent,
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


async def _execute_retry(session_id: str, channel: discord.TextChannel, text: str, mode: str) -> None:
    """Re-send a prior user message through the full worker pipeline."""
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
                "messages":         [{"role": "user", "content": text}],
                "session_id":       session_id,
                "mode":             mode,
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
            spec = dict(_MODEL_PRESETS[preset])
            resp = await _http.patch(f"{PHOEBE_API_URL}/config", json={"llm": spec})
            resp.raise_for_status()
            await msg.channel.send(f"Switched to **{preset}**.{_format_pricing_line(spec)}")
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


async def _register_bot_msgs(
    session_id: str, turn_index: int, msg_ids: list[str], channel_id: str,
) -> None:
    """Index bot reply msg_ids against a turn so a future edit can delete them."""
    try:
        await _http.post(
            f"{PHOEBE_API_URL}/v1/sessions/{session_id}/message_index/bot_msgs",
            json={
                "turn_index": int(turn_index),
                "discord_msg_ids": list(msg_ids),
                "channel_id": channel_id,
            },
            timeout=5,
        )
    except Exception as e:
        print(f"[worker-bot] register bot_msgs failed for {session_id}: {e}", flush=True)


async def _handle_user_edit(after: discord.Message) -> None:
    """Native Discord edit → session rewind → re-dispatch as fresh turn.

    Flow: look up the turn index by `after.id`, kill any in-flight run,
    truncate turns + state via POST /rewind, delete the bot's Discord replies
    that came after, then re-enter on_message() with the edited message.
    """
    channel = after.channel
    session_id = _channel_sessions.get(channel.id)
    if not session_id:
        return

    # Resolve turn_index by looking up the edited message id in state.
    try:
        resp = await _http.get(
            f"{PHOEBE_API_URL}/v1/sessions/{session_id}/state", timeout=5,
        )
        resp.raise_for_status()
        state = resp.json()
    except Exception as e:
        print(f"[worker-bot] edit: state fetch failed for {session_id}: {e}", flush=True)
        return

    user_msgs = (state.get("message_index") or {}).get("user_msgs") or []
    mid = str(after.id)
    entry = next((m for m in user_msgs if str(m.get("discord_msg_id")) == mid), None)
    if entry is None:
        try:
            await channel.send(
                "-# edit ignored: this message isn't in my session history "
                "(pre-dates the session, or was never linked)."
            )
        except Exception:
            pass
        return
    target_turn = int(entry.get("turn_index", -1))
    if target_turn < 0:
        return

    # Stop any in-flight run on this channel so rewind is safe. Poll briefly
    # for _channel_in_flight to clear — the bot's finally block clears it
    # when the SSE stream ends / cancels.
    if _channel_in_flight.get(channel.id):
        try:
            await _http.post(
                f"{PHOEBE_API_URL}/v1/sessions/{session_id}/kill", timeout=5,
            )
        except Exception:
            pass
        for _ in range(40):  # up to ~8s
            if not _channel_in_flight.get(channel.id):
                break
            await asyncio.sleep(0.2)
        else:
            try:
                await channel.send(
                    "-# edit ignored: still processing the previous run — try again."
                )
            except Exception:
                pass
            return

    # Rewind server-side.
    try:
        resp = await _http.post(
            f"{PHOEBE_API_URL}/v1/sessions/{session_id}/rewind",
            json={"target_turn_index": target_turn, "reason": "user_edit"},
            timeout=15,
        )
        resp.raise_for_status()
        result = resp.json()
    except Exception as e:
        print(f"[worker-bot] edit: rewind failed for {session_id}: {e}", flush=True)
        try:
            await channel.send(f"-# edit failed: {e}")
        except Exception:
            pass
        return

    # Delete bot replies + user follow-ups posted after the edited message.
    # The edited message itself (mid) is preserved — it's the new seed.
    to_delete: list[str] = []
    to_delete.extend(result.get("dropped_bot_msg_ids") or [])
    for uid in (result.get("dropped_user_msg_ids") or []):
        if str(uid) != mid:
            to_delete.append(str(uid))
    for did in to_delete:
        try:
            dmsg = await channel.fetch_message(int(did))
            await dmsg.delete()
        except (discord.NotFound, discord.Forbidden):
            continue
        except Exception as e:
            print(f"[worker-bot] delete msg {did} failed: {e}", flush=True)

    # Re-dispatch: the edited message routes through on_message as a fresh turn.
    try:
        await on_message(after)  # type: ignore[arg-type]
    except Exception as e:
        print(f"[worker-bot] edit: re-dispatch failed: {e}", flush=True)


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

    # Bot-created channels share one session across every user in the channel;
    # legacy channels scope sessions per user. The confirmation copy has to
    # reflect that so user A doesn't accidentally nuke user B's context
    # without realising.
    is_shared = channel.id in _channel_sessions
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

    if is_shared:
        await channel.send(
            "🔄 **Shared session reset.** This channel's context is wiped for everyone — "
            "fresh start for all users."
        )
        await interaction.followup.send(
            f"Deleted {count} messages and reset the shared channel session. "
            f"Any other users here are also starting from scratch.",
            ephemeral=True,
        )
    else:
        await channel.send("🔄 Conversation cleared. Fresh start!")
        await interaction.followup.send(
            f"Deleted {count} messages and reset your session.",
            ephemeral=True,
        )


_MODE_DESCRIPTIONS = {
    "plan":     "Investigation mode — investigates, then produces a structured plan for review.",
    "build":    "Execution mode — follows the active plan step by step with full tool access.",
    "converse": "Conversational — answers questions, quick lookups. Suggests plan mode for complex tasks.",
}


def _apply_mode(channel_id: int, user_id: int, mode: str) -> None:
    """Persist a mode switch. Routes through the channel session if one exists,
    so every user in that channel sees the same mode."""
    sid = _channel_sessions.get(channel_id)
    if sid:
        _set_mode_for_session(sid, mode)
    else:
        _user_modes[_session_key(channel_id, user_id)] = mode
        _save_state()


@tree.command(name="mode", description="Show or switch agent mode")
@app_commands.describe(mode="Leave blank to show current; otherwise: plan, build, or converse")
@app_commands.choices(mode=[
    app_commands.Choice(name="plan",     value="plan"),
    app_commands.Choice(name="build",    value="build"),
    app_commands.Choice(name="converse", value="converse"),
])
async def cmd_mode(
    interaction: discord.Interaction,
    mode: app_commands.Choice[str] | None = None,
):
    if not is_allowed(interaction.user.id):
        await interaction.response.send_message("Not authorized.", ephemeral=True)
        return
    if mode is None:
        current = _get_mode(interaction.channel_id, interaction.user.id)
        await interaction.response.send_message(
            f"**Current mode**: `{current}`. {_MODE_DESCRIPTIONS.get(current, '')}"
        )
        return
    _apply_mode(interaction.channel_id, interaction.user.id, mode.value)
    await interaction.response.send_message(
        f"Switched to **{mode.value}** mode. {_MODE_DESCRIPTIONS[mode.value]}"
    )


@tree.command(name="plan", description="Switch to plan mode (investigate + create plan)")
async def cmd_plan(interaction: discord.Interaction):
    if not is_allowed(interaction.user.id):
        await interaction.response.send_message("Not authorized.", ephemeral=True)
        return
    _apply_mode(interaction.channel_id, interaction.user.id, "plan")
    await interaction.response.send_message(
        f"Switched to **plan** mode. {_MODE_DESCRIPTIONS['plan']}"
    )


@tree.command(name="build", description="Switch to build mode (execute active plan)")
async def cmd_build(interaction: discord.Interaction):
    if not is_allowed(interaction.user.id):
        await interaction.response.send_message("Not authorized.", ephemeral=True)
        return
    _apply_mode(interaction.channel_id, interaction.user.id, "build")
    await interaction.response.send_message(
        f"Switched to **build** mode. {_MODE_DESCRIPTIONS['build']}"
    )


@tree.command(name="converse", description="Switch to converse mode (quick chat)")
async def cmd_converse(interaction: discord.Interaction):
    if not is_allowed(interaction.user.id):
        await interaction.response.send_message("Not authorized.", ephemeral=True)
        return
    _apply_mode(interaction.channel_id, interaction.user.id, "converse")
    await interaction.response.send_message(
        f"Switched to **converse** mode. {_MODE_DESCRIPTIONS['converse']}"
    )


# /model is hit repeatedly for "what's current?" checks; each call was a
# 5–8s cold GET to phoebe-api. 60s TTL is long enough to coalesce bursts and
# short enough that a config edit shows up on the next tick.
_model_presets_cache: dict[str, object] = {"data": None, "fetched_at": 0.0}
_MODEL_PRESETS_TTL_S = 60.0


def _format_pricing_line(spec: dict) -> str:
    """Returns `\\n**Price**: $X in / $Y out per 1M tokens` or `""` if no price."""
    p_in = spec.get("price_input")
    p_out = spec.get("price_output")
    if p_in is None and p_out is None:
        return ""
    in_s = f"${p_in:.2f}" if isinstance(p_in, (int, float)) else "?"
    out_s = f"${p_out:.2f}" if isinstance(p_out, (int, float)) else "?"
    return f"\n**Price**: `{in_s} in / {out_s} out per 1M tokens`"


async def _fetch_model_presets(force: bool = False) -> dict[str, dict]:
    """Pull the named-model list from phoebe-api's live config. Filters out
    internal-use entries (debate/checkpoint helpers with tiny max_tokens).

    Cached for _MODEL_PRESETS_TTL_S seconds; pass force=True to bypass.
    """
    now = time.time()
    cached = _model_presets_cache.get("data")
    fetched_at = float(_model_presets_cache.get("fetched_at") or 0.0)
    if not force and cached is not None and now - fetched_at < _MODEL_PRESETS_TTL_S:
        return cached  # type: ignore[return-value]
    try:
        resp = await _http.get(f"{PHOEBE_API_URL}/config", timeout=5)
        resp.raise_for_status()
        models = resp.json().get("models", {}) or {}
    except Exception:
        return cached or {}  # type: ignore[return-value]
    filtered = {
        name: spec for name, spec in models.items()
        if isinstance(spec, dict) and spec.get("max_tokens", 0) >= 1024
    }
    _model_presets_cache["data"] = filtered
    _model_presets_cache["fetched_at"] = now
    return filtered


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
            # Config just changed — invalidate caches so next /model shows truth.
            _model_presets_cache["data"] = None
            _model_presets_cache["fetched_at"] = 0.0
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
                f"{_format_pricing_line(llm)}"
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
                f"{_format_pricing_line(llm)}"
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
        p_in = spec.get("price_input")
        p_out = spec.get("price_output")
        if isinstance(p_in, (int, float)) and isinstance(p_out, (int, float)):
            label += f"  (${p_in:.2f}/${p_out:.2f} per 1M)"
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


@tree.command(name="context", description="Show context-window stats for this channel's session")
async def cmd_context(interaction: discord.Interaction):
    if not is_allowed(interaction.user.id):
        await interaction.response.send_message("Not authorized.", ephemeral=True)
        return
    if interaction.channel_id in _channel_sessions:
        session_id = _channel_sessions[interaction.channel_id]
    else:
        session_id = _get_session_id(interaction.channel_id, interaction.user.id)

    try:
        state_resp, cfg_resp = await asyncio.gather(
            _http.get(f"{PHOEBE_API_URL}/v1/sessions/{session_id}/state", timeout=5),
            _http.get(f"{PHOEBE_API_URL}/config", timeout=5),
        )
        state_resp.raise_for_status()
        cfg_resp.raise_for_status()
    except Exception as e:
        await interaction.response.send_message(
            f"Could not fetch session state: {e}", ephemeral=True
        )
        return

    state = state_resp.json()
    cfg = cfg_resp.json()
    ctx = cfg.get("context", {}) or {}
    budgets = ctx.get("budgets", {}) or {}
    soft_cap = ctx.get("total_soft_cap")

    cs = state.get("context_stats", {}) or {}
    last_tokens = cs.get("last_prompt_tokens") or 0
    pre_comp    = cs.get("last_prompt_tokens_pre_compression") or 0
    ratio       = cs.get("last_compression_ratio")
    sections    = cs.get("section_tokens") or {}
    handles     = cs.get("handles_created") or 0
    triggers    = cs.get("compression_triggers") or {}
    soft_over   = bool(cs.get("soft_cap_exceeded"))
    prefix_hash = cs.get("last_kv_prefix_hash") or "—"

    stats = state.get("stats", {}) or {}
    turn_count = stats.get("turn_count") or 0
    usage = stats.get("token_usage") or {}

    tool_results = state.get("tool_results") or {}
    handle_bytes = sum(int(meta.get("bytes") or 0) for meta in tool_results.values())

    def fmt_tok(n: int | float | None) -> str:
        return f"{int(n):,}" if n else "0"

    # Build section-token breakdown vs budgets, stable ordering
    section_order = ["soul", "user", "memory", "identity", "tool_docs",
                     "skills", "history", "tool_result"]
    section_lines: list[str] = []
    for name in section_order:
        used = sections.get(name)
        budget = budgets.get(name if name != "tool_result" else "tool_result_inline")
        if used is None and budget is None:
            continue
        used_s = fmt_tok(used) if used is not None else "—"
        budget_s = fmt_tok(budget) if budget is not None else "—"
        marker = " ⚠" if (used and budget and used > budget) else ""
        section_lines.append(f"  • `{name}`: {used_s} / {budget_s}{marker}")

    # Compression triggers (only show non-zero)
    trig_active = [f"{k}×{v}" for k, v in triggers.items() if v]
    triggers_s = ", ".join(trig_active) if trig_active else "none"

    cap_marker = " ⚠ over soft cap" if soft_over else ""
    ratio_s = f"{ratio:.2f}" if isinstance(ratio, (int, float)) and ratio else "—"

    msg = (
        f"**Session**: `{session_id}` (turn {turn_count})\n"
        f"**Last prompt**: {fmt_tok(last_tokens)} tok"
        f" / soft cap {fmt_tok(soft_cap)}{cap_marker}\n"
        f"**Pre-compression**: {fmt_tok(pre_comp)} tok · ratio {ratio_s}\n"
        f"**Prefix hash**: `{prefix_hash}`\n"
        f"**Section tokens** (used / budget):\n"
        + ("\n".join(section_lines) if section_lines else "  _(no telemetry yet — send a message first)_")
        + "\n"
        f"**Compression triggers**: {triggers_s}\n"
        f"**Tool-result handles**: {handles} created, "
        f"{len(tool_results)} live ({fmt_tok(handle_bytes)} B on disk)\n"
        f"**Cumulative usage**: in={fmt_tok(usage.get('input'))} "
        f"out={fmt_tok(usage.get('output'))} "
        f"thinking={fmt_tok(usage.get('thinking'))}"
    )
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


# channel_id → epoch seconds of the last /stop request. Used to upgrade a
# second /stop within the cooperative-grace window into an immediate hard
# cancel via /kill, per UX audit item 9.
_stop_request_ts: dict[int, float] = {}
_STOP_HARD_CANCEL_WINDOW_S = 15.0  # matches app/main.py _hard_cancel_after


@tree.command(name="stop", description="Stop the current worker run in this channel")
async def cmd_stop(interaction: discord.Interaction):
    if not is_allowed(interaction.user.id):
        await interaction.response.send_message("Not authorized.", ephemeral=True)
        return
    sid = _channel_in_flight.get(interaction.channel_id)
    if not sid:
        await interaction.response.send_message("No active run in this channel.", ephemeral=True)
        return

    now = time.time()
    last_stop = _stop_request_ts.get(interaction.channel_id)
    if last_stop is not None and now - last_stop < _STOP_HARD_CANCEL_WINDOW_S:
        # Second /stop within the grace window → escalate to hard cancel.
        try:
            resp = await _http.post(f"{PHOEBE_API_URL}/v1/sessions/{sid}/kill", timeout=5)
            if resp.status_code == 200:
                await interaction.response.send_message(
                    "\U0001f6d1 **Hard cancel** — worker terminated immediately."
                )
            else:
                await interaction.response.send_message(
                    f"Hard cancel returned HTTP {resp.status_code}.", ephemeral=True
                )
        except Exception as e:
            await interaction.response.send_message(f"Hard cancel failed: {e}", ephemeral=True)
        finally:
            _stop_request_ts.pop(interaction.channel_id, None)
        return

    result = await _post_injection(sid, "", mode="stop")
    if result is None:
        await interaction.response.send_message("Stop request failed.", ephemeral=True)
        return
    _stop_request_ts[interaction.channel_id] = now
    await interaction.response.send_message(
        "\U0001f6d1 Stopping at the next safe point (up to 15s). "
        "Send `/stop` again within that window to force-kill immediately."
    )


@tree.command(name="help", description="List available slash commands")
async def cmd_help(interaction: discord.Interaction):
    msg = (
        "**Always available**\n"
        "• `/new` — Start a fresh conversation in a new channel\n"
        "• `/clear` — Reset conversation and delete all messages in this channel\n"
        "• `/mode [plan|build|converse]` — Show current mode (blank) or switch\n"
        "• `/plan`, `/build`, `/converse` — One-shot mode switches\n"
        "• `/model` — Show current LLM model and settings\n"
        "• `/speak <prompt>` — Ask the agent a question and get a voice response\n"
        "• `/status` — Show your current mode, session, and system status\n"
        "• `/help` — This message\n\n"
        "**Diagnostics**\n"
        "• `/diag` — System health check (sandbox + LLM)\n"
        "• `/context` — Context-window usage and compression stats\n"
        "• `/todos` — Todos and checkpoints for this session\n"
        "• `/memory <query>` — Search stored memories (ChromaDB)\n"
        "• `/soul` — Dump current SOUL.md\n"
        "• `/plan-show` — Dump this session's active plan\n\n"
        "**Session control**\n"
        "• `/retry` — Re-run the last user message\n"
        "• `/revoke <tool>` — Revoke a session-approved tool\n"
        "• `/resume <sid>` — Bind this channel to a different session id\n"
        "• `/compact` — Force-run the rolling compactor\n"
        "• `/kill` — Hard-cancel the in-flight worker immediately\n"
        "• `/dream-run [date] [meta] [review]` — Trigger a verbose dream run with per-edit review\n\n"
        "**Only while I'm working on something** (worker run in flight)\n"
        "• `/btw <text>` — Add context mid-flight without stopping the agent\n"
        "• `/stop` — Cooperative stop (then escalates to hard cancel if repeated)\n"
        "*Tip:* just sending a normal message during a run pops up a 4-button\n"
        "popup (Immediate / Not urgent / Clarify / Queue) — usually easier than `/btw`.\n\n"
        "**Modes**\n"
        "• **plan** — Investigate and produce a structured plan for review\n"
        "• **build** — Execute the active plan step by step with full tools\n"
        "• **converse** — Quick questions and casual chat; suggests plan mode for complex tasks\n\n"
        "**Tip:** Create a channel in the `Conversations` category and I'll auto-connect to it."
    )
    await interaction.response.send_message(msg)


# ── Diagnostics / introspection ──────────────────────────────────────────────

def _session_for_channel(interaction: discord.Interaction) -> str:
    """Return the channel's session id, creating a per-user one as fallback."""
    if interaction.channel_id in _channel_sessions:
        return _channel_sessions[interaction.channel_id]
    return _get_session_id(interaction.channel_id, interaction.user.id)


@tree.command(name="diag", description="Run system diagnostic checks and show overall health")
async def cmd_diag(interaction: discord.Interaction):
    if not is_allowed(interaction.user.id):
        await interaction.response.send_message("Not authorized.", ephemeral=True)
        return
    await interaction.response.defer()
    try:
        resp = await _http.get(f"{PHOEBE_API_URL}/internal/diagnostics", timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        await interaction.followup.send(f"Diagnostics unavailable: {e}", ephemeral=True)
        return

    overall = data.get("overall", "?")
    icon = {"pass": "✅", "warn": "⚠️", "fail": "❌"}.get(overall, "❔")
    summary = data.get("summary", "")

    fails: list[str] = []
    sandbox = data.get("sandbox_checks", {}) or {}
    for check in sandbox.get("checks", []) or []:
        if check.get("status") in ("fail", "warn"):
            tag = "❌" if check["status"] == "fail" else "⚠️"
            fails.append(f"{tag} `{check.get('name','?')}` — {check.get('detail','')[:100]}")
    api_llm = data.get("api_checks", {}).get("llm_api_from_api", {}) or {}
    if api_llm.get("status") in ("fail", "warn"):
        tag = "❌" if api_llm["status"] == "fail" else "⚠️"
        fails.append(f"{tag} `llm_api_from_api` — {api_llm.get('detail','')[:100]}")

    lines = [f"{icon} **Overall**: `{overall}` — {summary}"]
    if fails:
        lines.append("**Issues:**")
        lines.extend(f"• {f}" for f in fails[:8])
    await interaction.followup.send("\n".join(lines))


@tree.command(name="memory", description="Search stored memories (ChromaDB)")
@app_commands.describe(query="What to search for", n="How many results (default 5)")
async def cmd_memory(interaction: discord.Interaction, query: str, n: int = 5):
    if not is_allowed(interaction.user.id):
        await interaction.response.send_message("Not authorized.", ephemeral=True)
        return
    await interaction.response.defer()
    try:
        resp = await _http.get(
            f"{PHOEBE_API_URL}/internal/memory_search",
            params={"q": query, "n": max(1, min(10, n))},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        await interaction.followup.send(f"Memory search failed: {e}", ephemeral=True)
        return

    results = data.get("results") or []
    if not results:
        await interaction.followup.send(f"No memories matched `{query}`.")
        return

    lines = [f"**Memory search** for `{query}` — {len(results)} hit(s):"]
    for r in results[:5]:
        score = r.get("score")
        score_s = f"{score:.2f}" if isinstance(score, (int, float)) else "—"
        content = str(r.get("content", "")).replace("\n", " ").strip()
        if len(content) > 200:
            content = content[:200] + "…"
        lines.append(f"-# `{score_s}`  {content}")
    await interaction.followup.send("\n".join(lines))


@tree.command(name="todos", description="Show todos and checkpoints for this session")
async def cmd_todos(interaction: discord.Interaction):
    if not is_allowed(interaction.user.id):
        await interaction.response.send_message("Not authorized.", ephemeral=True)
        return
    sid = _session_for_channel(interaction)
    try:
        resp = await _http.get(f"{PHOEBE_API_URL}/v1/sessions/{sid}/state", timeout=5)
        resp.raise_for_status()
        state = resp.json()
    except Exception as e:
        await interaction.response.send_message(f"Could not read state: {e}", ephemeral=True)
        return

    todos = state.get("todos") or []
    checkpoints = state.get("checkpoints") or []
    if not todos and not checkpoints:
        await interaction.response.send_message(
            f"No todos or checkpoints recorded for `{sid}`."
        )
        return

    def _fmt(items: list, header: str) -> list[str]:
        if not items:
            return []
        out = [f"**{header}**"]
        for item in items[:15]:
            if isinstance(item, dict):
                txt = item.get("text") or item.get("title") or item.get("description") or str(item)
                status = item.get("status")
                prefix = f"[{status}] " if status else ""
                out.append(f"• {prefix}{txt}")
            else:
                out.append(f"• {item}")
        if len(items) > 15:
            out.append(f"…and {len(items) - 15} more")
        return out

    lines = [f"**Session** `{sid}`"]
    lines.extend(_fmt(todos, "Todos"))
    lines.extend(_fmt(checkpoints, "Checkpoints"))
    await interaction.response.send_message("\n".join(lines))


@tree.command(name="kill", description="Hard-cancel the in-flight worker run immediately")
async def cmd_kill(interaction: discord.Interaction):
    if not is_allowed(interaction.user.id):
        await interaction.response.send_message("Not authorized.", ephemeral=True)
        return
    sid = _channel_in_flight.get(interaction.channel_id)
    if not sid:
        await interaction.response.send_message("No active run in this channel.", ephemeral=True)
        return
    try:
        resp = await _http.post(f"{PHOEBE_API_URL}/v1/sessions/{sid}/kill", timeout=5)
    except Exception as e:
        await interaction.response.send_message(f"Hard cancel failed: {e}", ephemeral=True)
        return

    if resp.status_code == 404:
        await interaction.response.send_message(
            "Run already finished — nothing to kill.", ephemeral=True
        )
        return
    if resp.status_code != 200:
        await interaction.response.send_message(
            f"Hard cancel returned HTTP {resp.status_code}.", ephemeral=True
        )
        return
    _stop_request_ts.pop(interaction.channel_id, None)
    await interaction.response.send_message("🛑 **Hard cancel** — worker terminated.")


@tree.command(name="retry", description="Re-run the last user message in this session")
async def cmd_retry(interaction: discord.Interaction):
    if not is_allowed(interaction.user.id):
        await interaction.response.send_message("Not authorized.", ephemeral=True)
        return
    if _channel_in_flight.get(interaction.channel_id):
        await interaction.response.send_message(
            "A run is already in flight — `/stop` or `/kill` first.", ephemeral=True
        )
        return

    sid = _session_for_channel(interaction)
    try:
        resp = await _http.get(f"{PHOEBE_API_URL}/sessions/{sid}", timeout=5)
    except Exception as e:
        await interaction.response.send_message(f"Could not read session: {e}", ephemeral=True)
        return
    if resp.status_code == 404:
        await interaction.response.send_message("No history in this session yet.", ephemeral=True)
        return
    if resp.status_code != 200:
        await interaction.response.send_message(
            f"Session fetch returned HTTP {resp.status_code}.", ephemeral=True
        )
        return

    turns = resp.json().get("turns") or []
    last_user = next(
        (str(t.get("content") or "") for t in reversed(turns) if t.get("role") == "user"),
        "",
    )
    if not last_user.strip():
        await interaction.response.send_message("Nothing to retry.", ephemeral=True)
        return

    snippet = last_user if len(last_user) <= 80 else last_user[:80] + "…"
    await interaction.response.send_message(f"🔁 Retrying: {snippet}")
    mode = _get_mode(interaction.channel_id, interaction.user.id)
    asyncio.create_task(_execute_retry(sid, interaction.channel, last_user, mode))


@tree.command(name="revoke", description="Revoke a session-approved tool (next use re-prompts)")
@app_commands.describe(tool="The tool name to revoke from this session")
async def cmd_revoke(interaction: discord.Interaction, tool: str):
    if not is_allowed(interaction.user.id):
        await interaction.response.send_message("Not authorized.", ephemeral=True)
        return
    sid = _session_for_channel(interaction)
    approved = _session_always_allow.get(sid)
    if not approved or tool not in approved:
        await interaction.response.send_message(
            f"`{tool}` is not in this session's always-allow list.", ephemeral=True
        )
        return
    approved.discard(tool)
    _sync_session_state(sid)
    _save_state()
    await interaction.response.send_message(
        f"Revoked `{tool}`. Next invocation will prompt for approval again."
    )


@cmd_revoke.autocomplete("tool")
async def _revoke_autocomplete(interaction: discord.Interaction, current: str):
    sid = _session_for_channel(interaction)
    approved = sorted(_session_always_allow.get(sid, set()))
    cur = (current or "").lower()
    return [
        app_commands.Choice(name=t, value=t)
        for t in approved if not cur or cur in t.lower()
    ][:25]


@tree.command(name="resume", description="Bind this channel to a different session id")
@app_commands.describe(session_id="The session id to resume (see /sessions via api for a list)")
async def cmd_resume(interaction: discord.Interaction, session_id: str):
    if not is_allowed(interaction.user.id):
        await interaction.response.send_message("Not authorized.", ephemeral=True)
        return
    try:
        resp = await _http.get(f"{PHOEBE_API_URL}/v1/sessions/{session_id}/state", timeout=5)
        resp.raise_for_status()
        state = resp.json()
    except Exception as e:
        await interaction.response.send_message(f"Could not fetch session: {e}", ephemeral=True)
        return

    turn_count = (state.get("stats") or {}).get("turn_count") or 0
    if turn_count <= 0:
        await interaction.response.send_message(
            f"Session `{session_id}` has no recorded turns — refusing to bind (likely a typo).",
            ephemeral=True,
        )
        return

    _channel_sessions[interaction.channel_id] = session_id
    WORKER_CHANNEL_IDS.add(interaction.channel_id)
    _save_state()
    await interaction.response.send_message(
        f"Bound this channel to `{session_id}` ({turn_count} turns)."
    )


@tree.command(name="compact", description="Force-run the rolling compactor for this session")
async def cmd_compact(interaction: discord.Interaction):
    if not is_allowed(interaction.user.id):
        await interaction.response.send_message("Not authorized.", ephemeral=True)
        return
    sid = _session_for_channel(interaction)
    await interaction.response.defer()
    try:
        resp = await _http.post(f"{PHOEBE_API_URL}/internal/compact/{sid}", timeout=600)
        resp.raise_for_status()
        result = resp.json()
    except Exception as e:
        await interaction.followup.send(f"Compaction failed: {e}", ephemeral=True)
        return

    if "skipped" in result:
        await interaction.followup.send(f"Skipped: `{result['skipped']}`.")
        return
    if "error" in result:
        await interaction.followup.send(f"❌ Compaction error: `{result['error']}`.")
        return

    covers = result.get("covers_up_to_turn", "?")
    size = result.get("body_chars", 0)
    await interaction.followup.send(
        f"✅ Compacted through turn **{covers}** — summary size {size:,} chars."
    )


async def _render_dream_event(event: str, data: dict, channel: discord.TextChannel) -> None:
    """Dispatch a single dream SSE event to the channel.

    Delegates worker_status / tool_started / tool_trace / injection to the
    worker formatters via a shared pending_tool_msgs dict; dream-level events
    (conversation_start/end, meta, finalize_review) render as embeds or
    subtext lines.
    """
    if event == "dream_run_start":
        window = data.get("window") or {}
        scope = (
            f"last {window.get('hours', '?')}h" if window
            else f"date {data.get('date', '?')}"
        )
        await channel.send(
            f"-# 🌙 dream run starting ({scope})  "
            f"candidates={data.get('candidates', 0)}  "
            f"skipped={data.get('skipped', 0)}  "
            f"dreamer=`{data.get('dreamer_model', '?')}`  "
            f"review={'on' if data.get('review_required') else 'off'}"
        )
    elif event == "dream_conversation_start":
        ch_id = data.get("channel_id")
        ch_display = ""
        if ch_id:
            try:
                resolved = client.get_channel(int(ch_id))
                if resolved is not None:
                    ch_display = f"  #{resolved.name}"
                else:
                    ch_display = f"  channel={ch_id}"
            except Exception:
                ch_display = f"  channel={ch_id}"
        worker_model = data.get("worker_model") or "?"
        mode = data.get("mode") or "?"
        # `prompt_files` (new, from multi-target plumbing) lists the role →
        # prompt-file pairs the dreamer can see in scope for this conv.
        pf_line = ""
        pf = data.get("prompt_files") or []
        if pf:
            pf_line = "  scope: " + ", ".join(
                f"`{p.get('file', '?')}`" for p in pf if p.get("file")
            )
        await channel.send(
            f"-# ▸ conv {data.get('idx', '?')}/{data.get('total', '?')}  "
            f"role=`{data.get('role', '?')}` mode=`{mode}` "
            f"worker_model=`{worker_model}`{ch_display}{pf_line}  "
            f"sid=`{data.get('sid', '?')}`"
        )
    elif event == "dream_conversation_end":
        status = data.get("status", "?")
        committed = len(data.get("committed") or [])
        flagged = len(data.get("flagged") or [])
        err = data.get("error")
        marker = "✗" if err else "✓"
        line = f"-# {marker} {status}  committed={committed}  flagged={flagged}"
        if err:
            line += f"  err=`{err[:120]}`"
        await channel.send(line)
    elif event == "dream_meta_start":
        await channel.send(
            f"-# 🌀 meta-dreamer  flagged_total={data.get('flagged_total', 0)}  "
            f"top_k={data.get('top_k', 0)}"
        )
    elif event == "dream_meta_briefing":
        head = (data.get("head") or "")[:500].replace("\n", " ")
        if head:
            await channel.send(f"-# briefing: {head}")
    elif event == "dream_meta_end":
        await channel.send(f"-# meta: `{data.get('status', '?')}`")
    elif event == "dream_run_end":
        await channel.send(
            f"-# 🌙 done  seen={data.get('seen', 0)}  "
            f"completed={data.get('completed', 0)}  "
            f"interrupted={data.get('interrupted', False)}"
        )
    elif event == "dream_finalize_review_timeout":
        await channel.send(
            f"⚠️ Review for `{data.get('dreamer_sid', '?')}` timed out — all edits dropped."
        )
    elif event == "dream_skip":
        rationale = (data.get("rationale") or "(no reason given)").replace("`", "'")
        await channel.send(
            f"-# ⊘ dreamer skipped  rationale: _{rationale[:400]}_"
        )


async def _consume_dream_sse(stream, channel: discord.TextChannel) -> None:
    """SSE consumer for `/dream-run`.

    Renders dream-specific events; on `dream_finalize_review`, posts a
    `DreamEditReviewView` that POSTs user decisions to
    `/v1/dream/review_response`. Worker events (worker_status/tool_trace)
    flow through the same channel.
    """
    current_event: str | None = None
    data_lines: list[str] = []
    pending_tool_msgs: dict[str, discord.Message] = {}

    async for raw_line in stream.aiter_lines():
        line = raw_line.rstrip("\r\n")
        if line == "":
            if current_event is None:
                continue
            blob = "\n".join(data_lines) if data_lines else "{}"
            try:
                data = json.loads(blob)
            except Exception:
                data = {"_raw": blob}
            try:
                if current_event == "tool_started":
                    msg = await channel.send(_format_tool_started(data))
                    call_id = data.get("call_id")
                    if call_id:
                        pending_tool_msgs[call_id] = msg
                elif current_event == "tool_trace":
                    call_id = data.get("call_id")
                    msg = pending_tool_msgs.pop(call_id, None) if call_id else None
                    formatted = _format_single_trace(data)
                    if msg is not None:
                        try:
                            await msg.edit(content=formatted)
                        except Exception:
                            await channel.send(formatted)
                    else:
                        await channel.send(formatted)
                elif current_event == "worker_status":
                    text = (data.get("text") or "").strip().replace("\n", " ")
                    if text:
                        await channel.send(f"-# 💭 {text[:500]}")
                elif current_event == "dream_finalize_review":
                    raw_edits = data.get("edits") or []
                    # Sort by target_prompt so all edits for one file cluster
                    # together. Multi-target batches get a "## target.md"
                    # divider field before each group.
                    def _grp_key(e):
                        return (e.get("target_prompt") or data.get("target_prompt") or "", e.get("phrase_id") or "")
                    edits = sorted(raw_edits, key=_grp_key)
                    targets = data.get("target_prompts") or (
                        [data.get("target_prompt")] if data.get("target_prompt") else []
                    )
                    sid = data.get("dreamer_sid", "?")
                    if len(targets) > 1:
                        header_target = f"{len(targets)} prompts: {', '.join(targets)}"
                    else:
                        header_target = f"`{targets[0] if targets else '?'}`"
                    embed = discord.Embed(
                        title=f"🌙 Review {len(edits)} dream edit(s)",
                        description=(
                            f"**target:** {header_target}\n"
                            f"**dreamer session:** `{sid}`\n"
                            f"_rationale:_ {(data.get('rationale') or '')[:400]}"
                        ),
                        color=0x9b5de5,
                    )
                    last_target: str | None = None
                    for i, e in enumerate(edits[:10], start=1):
                        et = e.get("target_prompt") or (targets[0] if targets else "?")
                        if et != last_target:
                            # Group divider — a field without a diff body.
                            embed.add_field(
                                name=f"── {et} ──",
                                value="​",  # zero-width space placeholder
                                inline=False,
                            )
                            last_target = et
                        old = (e.get("old_text") or "").replace("```", "`​``")
                        new = (e.get("new_text") or "").replace("```", "`​``")
                        body = (
                            f"**kind:** `{e.get('kind', '?')}`  "
                            f"**status:** `{e.get('status', 'ok')}`\n"
                            f"```diff\n- {old[:300]}\n+ {new[:300]}\n```"
                        )
                        embed.add_field(
                            name=f"{i}. {(e.get('phrase_id') or '')[:60]}",
                            value=body[:1000],
                            inline=False,
                        )
                    if len(edits) > 10:
                        embed.set_footer(text=f"showing 10 of {len(edits)} edits — use Select… for the full list")
                    view = DreamEditReviewView(dreamer_sid=sid, edits=edits)
                    msg = await channel.send(embed=embed, view=view)
                    view.message = msg
                elif current_event == "done":
                    return
                elif current_event == "error":
                    await channel.send(f"❌ dream stream error: `{data.get('error', 'unknown')}`")
                    return
                else:
                    await _render_dream_event(current_event, data, channel)
            except Exception as e:
                print(f"[dream-stream] render {current_event} failed: {e}", flush=True)
            current_event = None
            data_lines = []
            continue
        if line.startswith("event:"):
            current_event = line[len("event:"):].strip()
        elif line.startswith("data:"):
            data_lines.append(line[len("data:"):].lstrip())


@tree.command(name="dream-run", description="Manually trigger a verbose dream run with per-edit review")
@app_commands.describe(
    date="UTC calendar day YYYY-MM-DD (default: rolling last 24h)",
    window_hours="Rolling-window size in hours (default 24; ignored when date is set)",
    meta="Run the meta-dreamer after per-conversation passes (default: true)",
    review="Gate finalize on your accept/drop decisions (default: true)",
)
async def cmd_dream_run(
    interaction: discord.Interaction,
    date: str | None = None,
    window_hours: float | None = None,
    meta: bool = True,
    review: bool = True,
):
    if not is_allowed(interaction.user.id):
        await interaction.response.send_message("Not authorized.", ephemeral=True)
        return
    channel = interaction.channel
    if not isinstance(channel, discord.TextChannel):
        await interaction.response.send_message(
            "Use this in a text channel so the stream has somewhere to render.", ephemeral=True
        )
        return
    scope = (
        f"`{date}`" if date
        else f"last {window_hours}h" if window_hours is not None
        else "last 24h"
    )
    await interaction.response.send_message(
        f"🌙 starting dream run ({scope})  "
        f"meta={'on' if meta else 'off'}  review={'on' if review else 'off'}"
    )
    payload: dict = {"verbose": True, "meta_enabled": bool(meta), "review": bool(review)}
    if date:
        payload["date"] = date
    elif window_hours is not None:
        payload["window_hours"] = window_hours
    try:
        async with _http.stream(
            "POST",
            f"{PHOEBE_API_URL}/internal/dream-run",
            json=payload,
            headers={"Accept": "text/event-stream"},
            timeout=None,
        ) as stream:
            if stream.status_code != 200:
                body = await stream.aread()
                await channel.send(f"❌ dream-run HTTP {stream.status_code}: {body.decode('utf-8', 'replace')[:500]}")
                return
            await _consume_dream_sse(stream, channel)
    except Exception as e:
        await channel.send(f"❌ dream-run failed: `{e}`")


@tree.command(name="soul", description="Show the current SOUL.md contents")
async def cmd_soul(interaction: discord.Interaction):
    if not is_allowed(interaction.user.id):
        await interaction.response.send_message("Not authorized.", ephemeral=True)
        return
    await interaction.response.defer()
    try:
        resp = await _http.get(f"{PHOEBE_API_URL}/internal/soul", timeout=5)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        await interaction.followup.send(f"Could not read SOUL: {e}", ephemeral=True)
        return

    content = (data.get("content") or "").strip()
    if not content:
        await interaction.followup.send("SOUL.md is empty or missing.")
        return

    chunks = split_message(content)
    await interaction.followup.send(f"**SOUL.md** ({len(content):,} chars)")
    for chunk in chunks:
        await interaction.channel.send(chunk)


@tree.command(name="plan-show", description="Show this session's active plan")
async def cmd_plan_show(interaction: discord.Interaction):
    if not is_allowed(interaction.user.id):
        await interaction.response.send_message("Not authorized.", ephemeral=True)
        return
    sid = _session_for_channel(interaction)
    try:
        resp = await _http.get(f"{PHOEBE_API_URL}/v1/sessions/{sid}/state", timeout=5)
        resp.raise_for_status()
        state = resp.json()
    except Exception as e:
        await interaction.response.send_message(f"Could not read state: {e}", ephemeral=True)
        return

    plan = state.get("plan") or _session_plans.get(sid, "")
    plan = (plan or "").strip()
    if not plan:
        await interaction.response.send_message("No active plan for this session.")
        return

    chunks = split_message(plan)
    await interaction.response.send_message(f"**Active plan** — {len(plan):,} chars")
    for chunk in chunks:
        await interaction.channel.send(chunk)


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
async def on_message_edit(before: discord.Message, after: discord.Message) -> None:
    """Native Discord edit hook → rewind the session and re-run the edited message.

    Fires when the edited message is in discord.py's cache (which holds
    recent messages). Older edits fall through to on_raw_message_edit below.
    """
    is_test_driver = TEST_DRIVER_USER_ID and after.author.id == TEST_DRIVER_USER_ID
    if after.author.bot and not is_test_driver:
        return
    if not is_test_driver and not is_allowed(after.author.id):
        return
    if WORKER_CHANNEL_IDS and after.channel.id not in WORKER_CHANNEL_IDS:
        return
    if (before.content or "") == (after.content or ""):
        # Discord fires edit events on embed unfurl, pin, etc. — ignore no-op edits.
        return
    await _handle_user_edit(after)


@client.event
async def on_raw_message_edit(payload: discord.RawMessageUpdateEvent) -> None:
    """Fallback edit hook for messages that fell out of discord.py's cache."""
    if payload.cached_message is not None:
        # on_message_edit handled it with full before/after context.
        return
    ch = client.get_channel(payload.channel_id)
    if ch is None:
        try:
            ch = await client.fetch_channel(payload.channel_id)
        except Exception:
            return
    try:
        after = await ch.fetch_message(payload.message_id)
    except Exception:
        return
    is_test_driver = TEST_DRIVER_USER_ID and after.author.id == TEST_DRIVER_USER_ID
    if after.author.bot and not is_test_driver:
        return
    if not is_test_driver and not is_allowed(after.author.id):
        return
    if WORKER_CHANNEL_IDS and after.channel.id not in WORKER_CHANNEL_IDS:
        return
    await _handle_user_edit(after)


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
    bot_msg_ids: list[str] = []
    # _channel_in_flight must stay set through chunk rendering so a rapid
    # follow-up message routes to the InjectionView dispatcher, not a second
    # worker. It's cleared in the outer finally, after the last chunk lands.
    try:
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
                    "discord_msg_id":   str(msg.id),
                    "channel_id":       str(msg.channel.id),
                },
            ) as stream:
                stream.raise_for_status()
                data = await _consume_sse_stream(stream, msg.channel, bot_msg_ids)
        except Exception as e:
            err_str = str(e) or f"{type(e).__name__} (no message)"
            sent = await msg.channel.send(f"[error: {err_str}]")
            if sent is not None:
                bot_msg_ids.append(str(sent.id))
            return
        finally:
            _stop_thinking(session_id)

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
        # Discord hard-caps content at 2000 chars. split_message targets
        # MAX_MSG_LEN (1900) but fence-carry edits can push a chunk slightly
        # over in pathological inputs — guard so the Last chunk with a View
        # attached doesn't silently fail on a length violation.
        DISCORD_CONTENT_HARD_CAP = 2000
        safe_chunks: list[str] = []
        for c in chunks:
            if len(c) > DISCORD_CONTENT_HARD_CAP:
                print(f"[worker-bot] chunk {len(c)}ch over {DISCORD_CONTENT_HARD_CAP} — splitting",
                      flush=True)
                # Hard-split at the cap. Leaves fence mismatch to Discord's
                # renderer, but at least the message sends.
                for j in range(0, len(c), DISCORD_CONTENT_HARD_CAP - 20):
                    safe_chunks.append(c[j:j + DISCORD_CONTENT_HARD_CAP - 20])
            else:
                safe_chunks.append(c)
        chunks = safe_chunks

        for i, chunk in enumerate(chunks):
            is_last = i == len(chunks) - 1
            if is_last and is_plan_output:
                view = PlanReviewView(answer, session_id)
                sent = await msg.channel.send(chunk, view=view)
                view.message = sent
            elif is_last:
                sent = await msg.channel.send(chunk, view=SpeakView(answer, msg.channel.id))
            else:
                sent = await msg.channel.send(chunk)
            if sent is not None:
                bot_msg_ids.append(str(sent.id))
    finally:
        _channel_in_flight.pop(msg.channel.id, None)
        # Index the bot's rendered messages against this turn so a native
        # Discord edit can later resolve which messages to delete on rewind.
        if bot_msg_ids and data is not None:
            turn_index = data.get("turn_index")
            if isinstance(turn_index, int):
                asyncio.create_task(
                    _register_bot_msgs(
                        session_id, turn_index, list(bot_msg_ids), str(msg.channel.id)
                    ),
                    name=f"idx_bot_msgs_{session_id}_{turn_index}",
                )

    # Queued injections replay as fresh turns. This MUST run after the in-flight
    # flag is cleared — the recursive on_message() call checks _channel_in_flight
    # and would otherwise re-enter the dispatcher instead of starting a worker.
    if data is not None:
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
                # Silent failure was the original sin — user thinks their
                # queued follow-up was delivered. Tell them to resend.
                preview = q_text.split("\n", 1)[0][:140]
                try:
                    await msg.channel.send(
                        f"⚠️ Queued follow-up couldn't replay: `{preview}` — please resend."
                    )
                except Exception:
                    pass


async def run():
    if not WORKER_TOKEN:
        print("[worker-bot] DISCORD_TOKEN_WORKER not set — skipping", flush=True)
        return
    await client.start(WORKER_TOKEN)
