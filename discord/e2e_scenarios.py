"""
Discord end-to-end scenario runner.

Drives the worker bot from the config bot's Discord client, so we can exercise
the whole stack (Discord → phoebe-api → sandbox → llm) the way a real user does.

Prerequisites:
  - DISCORD_TOKEN_CONFIG        (config bot credentials; already used by bot_config.py)
  - DISCORD_TEST_CHANNEL_ID     channel id in the guild where scenarios run
  - DISCORD_TEST_DRIVER_USER_ID user id of the config bot (must match the token)
  - PHOEBE_ENABLE_TEXT_COMMANDS=1  in the worker bot's env, so !mode/!btw/!stop work
  - The test channel id must be in DISCORD_WORKER_CHANNELS for the worker bot

Running:
  make e2e                   # all scenarios
  make e2e-one SCENARIO=scenario_tool_ordering

Exit code is non-zero if any scenario fails.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable

import discord
import httpx

PHOEBE_API_URL         = os.environ.get("PHOEBE_API_URL", "http://phoebe-api:8090")
BOT_GATEWAY_URL     = os.environ.get("PHOEBE_DISCORD_GATEWAY_URL", "http://localhost:4000")
CONFIG_TOKEN        = os.environ.get("DISCORD_TOKEN_CONFIG", "")
TEST_CHANNEL_ID     = int(os.environ.get("DISCORD_TEST_CHANNEL_ID", "0") or "0")
DRIVER_USER_ID      = int(os.environ.get("DISCORD_TEST_DRIVER_USER_ID", "0") or "0")
DEFAULT_TIMEOUT_S   = float(os.environ.get("PHOEBE_E2E_TIMEOUT_S", "90"))

# Default "1" = persist; scenario messages stay in the channel so you can scroll
# back and see what happened. Set to "0" for CI-style auto-cleanup.
PERSIST_MESSAGES    = os.environ.get("PHOEBE_E2E_PERSIST_MESSAGES", "1").strip() != "0"


# ── Scenario context ─────────────────────────────────────────────────────────

@dataclass
class ScenarioContext:
    channel: discord.TextChannel
    client: discord.Client
    collected: list[discord.Message] = field(default_factory=list)
    started_at: float = field(default_factory=time.time)

    def reset_collector(self) -> None:
        self.collected = []
        self.started_at = time.time()


async def send(ctx: ScenarioContext, text: str) -> discord.Message:
    return await ctx.channel.send(text)


def _is_worker_message(ctx: ScenarioContext, msg: discord.Message) -> bool:
    """True if the message is from a bot other than ourselves — i.e. the worker.

    Messages the driver bot (us) posts are echoed back by Discord and land in
    ctx.collected too; treating those as "replies" is the #1 way to get a
    false-positive match that races ahead of the worker."""
    if msg.author.id == ctx.client.user.id:
        return False
    if DRIVER_USER_ID and msg.author.id == DRIVER_USER_ID:
        return False
    return True


async def wait_for(
    ctx: ScenarioContext,
    predicate: Callable[[discord.Message], bool],
    timeout: float = DEFAULT_TIMEOUT_S,
    include_own: bool = False,
) -> discord.Message:
    """Poll ctx.collected for the first message matching predicate.

    By default ignores our own outgoing messages (see _is_worker_message)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        for msg in ctx.collected:
            if not include_own and not _is_worker_message(ctx, msg):
                continue
            if predicate(msg):
                return msg
        await asyncio.sleep(0.25)
    raise AssertionError(f"Timed out waiting for predicate after {timeout}s")


async def wait_for_text(ctx: ScenarioContext, substring: str, timeout: float = DEFAULT_TIMEOUT_S) -> discord.Message:
    return await wait_for(ctx, lambda m: substring.lower() in (m.content or "").lower(), timeout)


async def wait_for_subtext(ctx: ScenarioContext, substring: str, timeout: float = DEFAULT_TIMEOUT_S) -> discord.Message:
    return await wait_for(
        ctx,
        lambda m: (m.content or "").startswith("-# ") and substring.lower() in (m.content or "").lower(),
        timeout,
    )


async def wait_for_final_reply(ctx: ScenarioContext, timeout: float = DEFAULT_TIMEOUT_S) -> discord.Message:
    """Wait for the first non-subtext, non-embed-only message from the worker bot."""
    def is_final(m: discord.Message) -> bool:
        content = m.content or ""
        if content.startswith("-# "):
            return False
        stripped = content.strip()
        if not stripped:
            return False
        # A leaked tool-call marker is not a real reply; the retry usually
        # produces a proper reply next. Skipping these avoids asserting on
        # malformed output that the worker itself will correct.
        if stripped.startswith("<|tool_call|>") or stripped.startswith("<tool_call"):
            return False
        lower = stripped.lower()
        # Ignore the command-ack messages from the text-command handler —
        # they're not replies to the task, they're replies to !mode/!model.
        if lower.startswith("switched to"):
            return False
        if "no active run" in lower:
            return False
        return True
    return await wait_for(ctx, is_final, timeout)


async def wait_for_idle(ctx: ScenarioContext, timeout: float = 180.0) -> None:
    """Poll the bot-side /discord/in_flight endpoint until the channel is idle.

    The bot clears `_channel_in_flight` strictly AFTER the api clears its
    `_active_sessions` (the bot still has queued tool_trace edits to render).
    Polling the api side produces false-idle; the only authoritative readiness
    signal is the bot's own in-flight dict."""
    deadline = time.time() + timeout
    last_sid: str | None = None
    while time.time() < deadline:
        async with httpx.AsyncClient(timeout=5) as c:
            try:
                r = await c.get(
                    f"{BOT_GATEWAY_URL}/discord/in_flight",
                    params={"channel_id": ctx.channel.id},
                )
                if r.status_code == 200:
                    data = r.json()
                    if not data.get("in_flight"):
                        return
                    last_sid = data.get("session_id")
            except Exception:
                pass
        await asyncio.sleep(1.0)
    raise AssertionError(f"channel {ctx.channel.id} still in-flight after {timeout}s (sid={last_sid})")


async def channel_reset(channel_id: int) -> None:
    """Force-clear the bot's in-flight flag for a channel. Runner-only escape hatch."""
    async with httpx.AsyncClient(timeout=5) as c:
        try:
            await c.post(f"{BOT_GATEWAY_URL}/discord/channel_reset",
                         params={"channel_id": channel_id})
        except Exception as e:
            print(f"  [channel_reset warn: {e}]", flush=True)


async def session_reset(channel_id: int, user_id: int | None = None) -> dict | None:
    """Rotate the channel's session_id so the next message runs with zero prior
    context. Addresses the load-bearing context-bleed bug where a prior
    scenario's write_config critique leaked into the next scenario's "hi" reply."""
    async with httpx.AsyncClient(timeout=5) as c:
        try:
            params: dict = {"channel_id": channel_id}
            if user_id:
                params["user_id"] = user_id
            r = await c.post(f"{BOT_GATEWAY_URL}/discord/session_reset", params=params)
            if r.status_code == 200:
                return r.json()
        except Exception as e:
            print(f"  [session_reset warn: {e}]", flush=True)
    return None


async def purge_channel(channel_id: int, bot: str = "mod") -> dict | None:
    """Delete every bot-reachable message in the test channel. Called once at
    runner start — a week of prior scenario runs is not context we want leaking
    into the next run's session history."""
    async with httpx.AsyncClient(timeout=60) as c:
        try:
            r = await c.post(
                f"{BOT_GATEWAY_URL}/discord/purge_channel",
                json={"channel_id": channel_id, "bot": bot},
            )
            if r.status_code == 200:
                return r.json()
        except Exception as e:
            print(f"  [purge_channel warn: {e}]", flush=True)
    return None


async def kill_session(session_id: str) -> None:
    """Hard-cancel an active api-side task. No grace period."""
    async with httpx.AsyncClient(timeout=5) as c:
        try:
            await c.post(f"{PHOEBE_API_URL}/v1/sessions/{session_id}/kill")
        except Exception as e:
            print(f"  [kill_session warn: {e}]", flush=True)


async def _patch_config(patch: dict) -> dict:
    """PATCH /config; returns {'updated': True, 'config': {...}} on success."""
    async with httpx.AsyncClient(timeout=10) as c:
        r = await c.patch(f"{PHOEBE_API_URL}/config", json=patch)
        r.raise_for_status()
        return r.json()


async def _get_config() -> dict:
    async with httpx.AsyncClient(timeout=10) as c:
        r = await c.get(f"{PHOEBE_API_URL}/config")
        r.raise_for_status()
        return r.json()


# ── Supervisor gating ────────────────────────────────────────────────────────
# E2E scenarios test plumbing (injection, popups, tool ordering, skill
# discovery, schema rejection, …) — not answer quality. Running the supervisor
# adds latency and can mask real bugs via retries. So we default supervisor OFF
# and let individual scenarios opt in via `needs_supervisor = True`.

_saved_supervisor_enabled: bool | None = None
_current_supervisor_enabled: bool | None = None


def needs_supervisor(fn: Callable) -> Callable:
    """Mark a scenario as requiring the supervisor enabled for its session."""
    fn.needs_supervisor = True  # type: ignore[attr-defined]
    return fn


async def _snapshot_supervisor() -> None:
    global _saved_supervisor_enabled, _current_supervisor_enabled
    try:
        cfg = await _get_config()
        _saved_supervisor_enabled = bool(cfg.get("agent", {}).get("supervisor_enabled", True))
        _current_supervisor_enabled = _saved_supervisor_enabled
    except Exception as e:
        print(f"  [supervisor snapshot warn: {e}]", flush=True)


async def _set_supervisor(enabled: bool) -> None:
    global _current_supervisor_enabled
    if _current_supervisor_enabled == enabled:
        return
    try:
        await _patch_config({"agent": {"supervisor_enabled": enabled}})
        _current_supervisor_enabled = enabled
    except Exception as e:
        print(f"  [supervisor toggle warn: {e}]", flush=True)


async def _restore_supervisor() -> None:
    if _saved_supervisor_enabled is None:
        return
    try:
        await _patch_config({"agent": {"supervisor_enabled": _saved_supervisor_enabled}})
    except Exception as e:
        print(f"  [supervisor restore warn: {e}]", flush=True)


async def inject(session_id: str, text: str, mode: str) -> dict | None:
    async with httpx.AsyncClient(timeout=10) as client:
        try:
            r = await client.post(
                f"{PHOEBE_API_URL}/v1/sessions/{session_id}/inject",
                json={"text": text, "mode": mode},
            )
            r.raise_for_status()
            return r.json()
        except Exception as e:
            print(f"  [inject {mode} failed: {e}]", flush=True)
            return None


async def current_session_id(channel_id: int) -> str | None:
    """Resolve the session id currently running in a channel.

    Primary source: the bot-side /discord/in_flight endpoint — it's the same
    dict that `wait_for_idle` polls, so this stays consistent with readiness.
    Falls back to reading the state file only if the endpoint is unreachable
    (shouldn't happen inside the discord container)."""
    async with httpx.AsyncClient(timeout=5) as c:
        try:
            r = await c.get(
                f"{BOT_GATEWAY_URL}/discord/in_flight",
                params={"channel_id": channel_id},
            )
            if r.status_code == 200:
                sid = r.json().get("session_id")
                if sid:
                    return sid
        except Exception:
            pass
    # Fallback — persisted state file on disk.
    state_path = os.environ.get("STATE_DIR", "/app/state") + "/bot_worker_state.json"
    try:
        import json
        with open(state_path, "r") as f:
            state = json.load(f)
        ch_sessions = state.get("channel_sessions", {})
        sid = ch_sessions.get(str(channel_id))
        if sid:
            return sid
        session_ids = state.get("session_ids", {})
        if DRIVER_USER_ID:
            sid = session_ids.get(f"{channel_id}_{DRIVER_USER_ID}")
            if sid:
                return sid
        for key, sid in session_ids.items():
            if key.startswith(f"{channel_id}_"):
                return sid
    except Exception:
        pass
    return None


async def clear_channel(ctx: ScenarioContext) -> None:
    """Delete every message in the test channel authored by a bot or by us.

    Safety: never deletes messages from non-bot users."""
    try:
        async for msg in ctx.channel.history(limit=100):
            if msg.author.bot:
                try:
                    await msg.delete()
                except Exception:
                    pass
    except Exception as e:
        print(f"  [clear_channel warn: {e}]", flush=True)


# ── Scenarios ────────────────────────────────────────────────────────────────

async def scenario_converse_short_reply(ctx: ScenarioContext) -> None:
    await send(ctx, "!mode converse")
    await wait_for_text(ctx, "converse", timeout=15)
    ctx.reset_collector()
    await send(ctx, "hi")
    reply = await wait_for_final_reply(ctx, timeout=45)
    body = (reply.content or "").strip()
    assert body, "empty final reply"
    assert "<|end|>" not in body, f"raw end-marker leaked: {body!r}"


async def scenario_plan_specificity(ctx: ScenarioContext) -> None:
    await send(ctx, "!mode plan")
    await wait_for_text(ctx, "plan", timeout=15)
    ctx.reset_collector()
    await send(ctx, "add a logout button to the login page")
    reply = await wait_for_final_reply(ctx, timeout=120)
    body = (reply.content or "")
    assert any(ext in body for ext in (".py", ".ts", ".md", ".tsx", ".js", ".yaml")), \
        f"plan lacks concrete file anchors: {body[:200]}"


async def scenario_tool_ordering(ctx: ScenarioContext) -> None:
    """The critical invariant: tool_started subtext lands BEFORE any subsequent
    worker text. Regression if the synchronous emit is removed."""
    await send(ctx, "!mode build")
    await wait_for_text(ctx, "build", timeout=15)
    ctx.reset_collector()
    await send(ctx, "web_search for 'fastapi sse tutorial' and give me a one-sentence summary")
    # First subtext (tool_started) must appear before the final reply.
    started = await wait_for_subtext(ctx, "web_search", timeout=30)
    final = await wait_for_final_reply(ctx, timeout=120)
    assert started.created_at < final.created_at, \
        f"tool_started subtext {started.created_at} landed after final reply {final.created_at}"


async def scenario_end_marker_keeps_looping(ctx: ScenarioContext) -> None:
    """Multi-tool tasks should emit at least one worker_status subtext but still
    produce exactly one final reply at the end."""
    await send(ctx, "!mode build")
    await wait_for_text(ctx, "build", timeout=15)
    ctx.reset_collector()
    await send(
        ctx,
        "use file_list on /project/app, then file_read /project/app/main.py, "
        "then summarize in two sentences",
    )
    final = await wait_for_final_reply(ctx, timeout=180)
    # Final must come after any 💭 status if one fired.
    statuses = [m for m in ctx.collected if (m.content or "").startswith("-# 💭")]
    if statuses:
        latest_status = max(s.created_at for s in statuses)
        assert latest_status < final.created_at, "worker_status landed after final reply"


async def scenario_mid_flight_not_urgent(ctx: ScenarioContext) -> None:
    """Inject a not_urgent note mid-run; the note is stapled onto the *next*
    tool_result, so we need a task that makes more than one tool call."""
    await send(ctx, "!mode build")
    await wait_for_text(ctx, "build", timeout=15)
    ctx.reset_collector()
    await send(
        ctx,
        "web_fetch https://example.com, then web_fetch https://example.org, "
        "then tell me what differs",
    )
    await wait_for_subtext(ctx, "web_fetch", timeout=30)
    sid = await current_session_id(ctx.channel.id)
    assert sid, "could not resolve in-flight session id"
    await inject(sid, "also mention the HTTP status code", mode="not_urgent")
    final = await wait_for_final_reply(ctx, timeout=180)
    body = (final.content or "").lower()
    # The injection asks for the HTTP status code. Accept any clear
    # acknowledgement: the word "status", "http 200", or a bare "200".
    assert ("status" in body) or ("http 200" in body) or (" 200 " in f" {body} "), \
        f"reply did not incorporate injection: {body[:200]}"


async def scenario_mid_flight_stop(ctx: ScenarioContext) -> None:
    await send(ctx, "!mode build")
    await wait_for_text(ctx, "build", timeout=15)
    ctx.reset_collector()
    await send(
        ctx,
        "fetch https://httpbin.org/delay/5 three times in a row and then summarize",
    )
    await wait_for_subtext(ctx, "", timeout=30)  # wait for any subtext
    sid = await current_session_id(ctx.channel.id)
    assert sid, "could not resolve in-flight session id"
    await inject(sid, "", mode="stop")
    # Hard cancel kicks in after 15s if cooperative stop hasn't landed; the
    # final reply (or the reset) must arrive within the wait_for_idle window.
    final = await wait_for_final_reply(ctx, timeout=60)
    body = (final.content or "").lower()
    assert "stopped" in body or "[stopped by user]" in body, \
        f"final reply did not indicate stop: {body[:200]}"


async def scenario_dispatcher_popup(ctx: ScenarioContext) -> None:
    """A plain message to a channel with an in-flight session must trigger the
    InjectionView popup with four buttons (Immediate / Not urgent / Clarify / Queue)."""
    await send(ctx, "!mode build")
    await wait_for_text(ctx, "build", timeout=15)
    ctx.reset_collector()
    await send(ctx, "fetch https://httpbin.org/delay/5 three times and then summarize")
    await wait_for_subtext(ctx, "", timeout=30)
    # Now interject — expect the dispatcher popup.
    await send(ctx, "quick question mid-run")
    popup = await wait_for_text(ctx, "run is already in progress", timeout=15)
    assert popup.components, "popup missing action components (buttons)"
    # Total button count across all action rows should be 4.
    button_count = sum(len(row.children) for row in popup.components)
    assert button_count >= 4, f"expected ≥4 buttons, got {button_count}"


async def scenario_skill_discovery(ctx: ScenarioContext) -> None:
    """Write a SKILL.md directly, confirm the worker sees it in a subsequent turn."""
    import json, pathlib
    skill_dir = pathlib.Path("/config/skills/e2e-probe")
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: e2e-probe\ndescription: Temporary E2E test skill.\n"
        "when-to-trigger: TRIGGER when asked about e2e-probe.\n"
        "when-not-to-trigger: DO NOT TRIGGER otherwise.\n"
        "allowed-tools: [file_read]\n---\n\n## Purpose\nE2E test probe.\n"
    )
    try:
        await send(ctx, "!mode converse")
        await wait_for_text(ctx, "converse", timeout=15)
        ctx.reset_collector()
        await send(
            ctx,
            "list the names of skill playbooks from config/skills — "
            "name only, one per line",
        )
        final = await wait_for_final_reply(ctx, timeout=90)
        body = (final.content or "").lower()
        assert "e2e-probe" in body, f"skill not discovered: {body[:200]}"
    finally:
        try:
            (skill_dir / "SKILL.md").unlink(missing_ok=True)
            skill_dir.rmdir()
        except Exception:
            pass


async def scenario_config_schema_reject(ctx: ScenarioContext) -> None:
    """Ask the worker to patch an obviously-typo'd config key; the reply must
    reference the rejection and the real config must not contain the typo.

    write_config is in approval.build.ask_user by default, which would block
    the scenario waiting for a button click. We temporarily move it to
    auto_allow.tools for the duration of this scenario only."""
    # Snapshot the original approval.build config before patching.
    cfg_before = await _get_config()
    build_allow = list(cfg_before.get("approval", {}).get("build", {}).get("auto_allow", {}).get("tools", []))
    build_ask   = list(cfg_before.get("approval", {}).get("build", {}).get("ask_user", []))
    try:
        new_allow = build_allow + (["write_config"] if "write_config" not in build_allow else [])
        new_ask   = [t for t in build_ask if t != "write_config"]
        await _patch_config({
            "approval": {"build": {
                "auto_allow": {"tools": new_allow, "paths": []},
                "ask_user": new_ask,
            }},
        })
        await send(ctx, "!mode build")
        await wait_for_text(ctx, "build", timeout=15)
        ctx.reset_collector()
        await send(
            ctx,
            "use write_config to set agent.supervisor_pass_treshold to 0.8 (yes with that spelling)",
        )
        final = await wait_for_final_reply(ctx, timeout=180)
        body = (final.content or "").lower()
        assert "treshold" in body or "did you mean" in body or "unknown" in body or "rejected" in body, \
            f"schema rejection not surfaced: {body[:200]}"
        # Verify the live config doesn't contain the typo.
        cfg_after = await _get_config()
        agent = cfg_after.get("agent", {})
        assert "supervisor_pass_treshold" not in agent, "typo key leaked into config.yaml"
    finally:
        # Restore the original approval config.
        await _patch_config({
            "approval": {"build": {
                "auto_allow": {"tools": build_allow, "paths": []},
                "ask_user": build_ask,
            }},
        })


async def scenario_diagnostic_all_green(ctx: ScenarioContext) -> None:
    await send(ctx, "!mode build")
    await wait_for_text(ctx, "build", timeout=15)
    ctx.reset_collector()
    await send(
        ctx,
        "call the diagnostic_check tool and list any check names that start with "
        "'config_' or 'skills_' or 'inject_'",
    )
    final = await wait_for_final_reply(ctx, timeout=180)
    body = (final.content or "").lower()
    hits = sum(1 for k in (
        "config_schema", "skills_directory",
        "inject_endpoint_reachable", "supervisor_mode_overrides",
    ) if k in body)
    assert hits >= 2, f"diagnostic summary missing new checks (hits={hits}): {body[:300]}"


SCENARIOS: list[Callable[[ScenarioContext], Awaitable[None]]] = [
    scenario_converse_short_reply,
    scenario_plan_specificity,
    scenario_tool_ordering,
    scenario_end_marker_keeps_looping,
    scenario_mid_flight_not_urgent,
    scenario_mid_flight_stop,
    scenario_dispatcher_popup,
    scenario_skill_discovery,
    scenario_config_schema_reject,
    scenario_diagnostic_all_green,
]


# ── Runner ───────────────────────────────────────────────────────────────────

async def _run(selected: list[str] | None) -> int:
    if not CONFIG_TOKEN:
        print("DISCORD_TOKEN_CONFIG not set", file=sys.stderr)
        return 2
    if not TEST_CHANNEL_ID:
        print("DISCORD_TEST_CHANNEL_ID not set — refusing to run (would pollute default channel)",
              file=sys.stderr)
        return 2

    intents = discord.Intents.default()
    intents.message_content = True
    intents.guilds = True
    client = discord.Client(intents=intents)

    scenarios = SCENARIOS
    if selected:
        by_name = {s.__name__: s for s in SCENARIOS}
        missing = [n for n in selected if n not in by_name]
        if missing:
            print(f"Unknown scenarios: {missing}. Known: {list(by_name)}", file=sys.stderr)
            return 2
        scenarios = [by_name[n] for n in selected]

    results: list[tuple[str, str]] = []
    done = asyncio.Event()

    @client.event
    async def on_ready():
        try:
            channel = client.get_channel(TEST_CHANNEL_ID) or await client.fetch_channel(TEST_CHANNEL_ID)
            ctx = ScenarioContext(channel=channel, client=client)

            @client.event
            async def on_message(m: discord.Message):
                if m.channel.id == TEST_CHANNEL_ID:
                    ctx.collected.append(m)

            # One-time: purge the test channel before the very first scenario.
            # A week of prior runs with stale write_config approval popups and
            # leaked supervisor critiques is not the starting state we want.
            print("[setup] purging test channel…", flush=True)
            purge_result = await purge_channel(TEST_CHANNEL_ID)
            if purge_result:
                print(f"[setup] purged {purge_result.get('deleted', '?')} messages", flush=True)
            # And clear any stale in-flight flag or session mapping left over
            # from a prior runner process that crashed mid-scenario.
            await channel_reset(TEST_CHANNEL_ID)
            await session_reset(TEST_CHANNEL_ID, DRIVER_USER_ID or None)

            # Snapshot the current supervisor_enabled value so we can restore it
            # at teardown. We default to OFF for all scenarios and opt in per
            # scenario via @needs_supervisor.
            await _snapshot_supervisor()

            for scen in scenarios:
                # Pre-scenario isolation:
                #   1) Hard-kill any api-side task still attached to the prior
                #      scenario's session (stop mode's 15-s grace is too slow
                #      for tight e2e timing).
                #   2) Rotate the channel's session_id so prior turns' JSONL
                #      history doesn't get replayed — this was the real cause
                #      of the converse-mode "failed to use fetch_url" critiques
                #      appearing on simple "hi" replies.
                #   3) Confirm the bot-side flag is clear before proceeding.
                sid = await current_session_id(TEST_CHANNEL_ID)
                if sid:
                    await kill_session(sid)
                await session_reset(TEST_CHANNEL_ID, DRIVER_USER_ID or None)
                await channel_reset(TEST_CHANNEL_ID)
                try:
                    await wait_for_idle(ctx, timeout=30)
                except AssertionError:
                    sid = await current_session_id(TEST_CHANNEL_ID)
                    if sid:
                        await kill_session(sid)
                    await asyncio.sleep(2)
                    await channel_reset(TEST_CHANNEL_ID)
                ctx.reset_collector()
                # Gate supervisor per scenario: off unless explicitly opted in.
                await _set_supervisor(getattr(scen, "needs_supervisor", False))
                print(f"→ {scen.__name__}"
                      f"{' [supervisor ON]' if getattr(scen, 'needs_supervisor', False) else ''}",
                      flush=True)
                try:
                    await scen(ctx)
                    results.append((scen.__name__, "PASS"))
                except AssertionError as e:
                    results.append((scen.__name__, f"FAIL: {e}"))
                except Exception as e:
                    results.append((scen.__name__, f"ERROR: {type(e).__name__}: {e}"))
                # Post-scenario cleanup: kill the session outright (don't just
                # send stop — we've seen cooperative cancels hang mid-LLM-call)
                # and wait for the bot to finish rendering before moving on.
                sid = await current_session_id(TEST_CHANNEL_ID)
                if sid:
                    await kill_session(sid)
                try:
                    await wait_for_idle(ctx, timeout=180)
                except AssertionError as e:
                    print(f"  [warn: {e}; forcing channel_reset]", flush=True)
                    await channel_reset(TEST_CHANNEL_ID)
                if PERSIST_MESSAGES:
                    # Post a visible separator so scrolling back is easy.
                    try:
                        await ctx.channel.send(f"━━━ end of {scen.__name__} ━━━")
                    except Exception:
                        pass
                else:
                    await clear_channel(ctx)
        finally:
            # Restore the supervisor_enabled value we snapshotted at setup so
            # the e2e run leaves config.yaml exactly as it was found.
            await _restore_supervisor()
            done.set()

    task = asyncio.create_task(client.start(CONFIG_TOKEN))
    try:
        await done.wait()
    finally:
        await client.close()
        try:
            await task
        except Exception:
            pass

    width = max((len(n) for n, _ in results), default=10)
    print()
    for name, status in results:
        print(f"{name.ljust(width)}  {status}")
    bad = [r for r in results if not r[1].startswith("PASS")]
    print(f"\n{len(results) - len(bad)}/{len(results)} passed")
    return 1 if bad else 0


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", action="append", default=[],
                   help="Run only the named scenario (repeatable)")
    args = p.parse_args()
    selected = args.scenario or None
    code = asyncio.run(_run(selected))
    sys.exit(code)


if __name__ == "__main__":
    main()
