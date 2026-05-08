# Pycord migration container (A2 PR2)

Parallel to `phoebe-discord` (which keeps using `discord.py`). This container
runs the same Phoebe Discord surface but on **py-cord**, and is gated behind
a docker-compose **profile** so it does not auto-start.

> **Important:** Discord rejects two clients connected with the same bot token,
> so only one of `phoebe-discord` and `phoebe-discord-pycord` should be
> running at a time. Both containers persist to the same `state/discord` and
> `config/skills` volumes, so session IDs, plans, and skills carry across
> the swap with no data loss.

## Phase status

| Phase | Goal | Status |
|---|---|---|
| 1 | Skeleton container — proves py-cord image builds + FastAPI starts | done |
| 2a | `bot_config.py` migrated (smallest, 2 slash commands) | done |
| 2b | `bot_mod.py` migrated (no slash commands, just nickname/typing) | done |
| 2c | `views.py` migrated (6 Views with buttons + selects) | done |
| 2d | `bot_worker.py` migrated (largest — 24 slashes + on_message) | done |
| 2e | `main.py` migrated (FastAPI gateway + voice playback) | done |
| 3 | e2e suite green against pycord container, cutover decision | in progress |

## Library version: py-cord 2.8.0rc2

This image runs **py-cord 2.8.0rc2** (release candidate, not yet stable) +
**davey 0.1.5** (Discord's DAVE/E2EE protocol Python library). 2.8.0rc2 is
the first py-cord version with built-in DAVE support — `discord/voice/utils/
dependencies.py` checks for the `davey` import and enables E2EE when present.

Discord's DAVE/E2EE protocol was enforced globally on March 2 2026. Bots
without DAVE support get error code 4017 OR a silent voice handshake
stall. The 2.6.1 → 2.8.0rc2 + davey bump removes that library-level
blocker.

A separate UDP egress block on the user's network still prevents live
voice playback in some setups; the 🔊 Listen button in `views.py` falls
back to WAV file attachment automatically when voice handshake fails.

## Swap procedure

To start using the Pycord container:

```bash
cd /home/homelab/Phoebe
docker compose stop phoebe-discord
docker compose --profile pycord up -d phoebe-discord-pycord
docker logs -f phoebe-discord-pycord     # watch startup
```

To roll back to the legacy `discord.py` container:

```bash
cd /home/homelab/Phoebe
docker compose stop phoebe-discord-pycord
docker compose start phoebe-discord
docker logs -f phoebe-discord            # watch startup
```

Both directions are non-destructive — state lives on the bind mounts, not
inside the container.

## Health checks per phase

> Note: only `phoebe-discord-pycord` exposes a host port (4001 → 4000).
> The legacy `phoebe-discord` container is reachable only from inside the
> Docker network — production does not need host access. Use
> `docker exec phoebe-api curl http://phoebe-discord:4000/health` to
> probe the legacy container.

* **Phase 1**: `curl http://localhost:4001/health` returns
  `{"ok": true, "service": "phoebe-discord-pycord", ...}` with the py-cord
  version string visible in the JSON.
* **Phase 2a**: `/status` and `/help` slash commands work in `#phoebe-config`.
* **Phase 2b**: bot nickname flips to `🧠 thinking` while a worker run is
  active and `channel.typing()` is visible to the user.
* **Phase 2c**: clicking the `🔊 Listen` button on any worker reply
  flips it to `⏸ Pause` / `▶ Resume`. `/btw` injection popup shows 4
  buttons. Approval prompts (Yes/No/Always) render.
* **Phase 2d**: a normal chat conversation works end-to-end — `/mode plan`,
  `/new`, `/clear`, `/btw`, `/stop` all behave identically to the legacy
  container.
* **Phase 2e**: `/discord/speak` returns a WAV file message; `/discord/speak_voice`
  plays in the user's voice channel; `/speak` toggle persists across restarts.
* **Phase 3**: `make e2e` reports the same 6 reliable scenarios passing
  that the legacy container does.

## Why a separate container

`discord.py` and `py-cord` both occupy the `discord` Python namespace, so
they cannot coexist in one image. The two-container approach keeps the
legacy code path completely intact and gives operator-controlled cutover
+ rollback as the unit of safety.

## Operational gotcha: shared-netns drift after multiple recreates

`phoebe-api` and `phoebe-sandbox` both run with `network_mode: service:phoebe-vpn`,
sharing `phoebe-vpn`'s network namespace. When you `--force-recreate`
`phoebe-discord-pycord` repeatedly during testing OR restart only `phoebe-vpn`
on its own, the api/sandbox processes can keep running but lose their grip
on the live netns — they appear healthy from their (now-defunct) view, but
external traffic to `phoebe-vpn:8090` from the bridge gets `Connection refused`.

**Symptom**: bots logged in successfully, container healthy, but `[error: All
connection attempts failed]` on every message because the bots can't reach
the api over the bridge.

**Recovery procedure (in this order)**:
```bash
docker compose restart phoebe-vpn
docker compose restart phoebe-api phoebe-sandbox
# verify bridge route is alive:
docker run --rm --network phoebe_phoebe-net python:3.12-slim \
  python3 -c "import urllib.request; print(urllib.request.urlopen('http://phoebe-vpn:8090/health',timeout=3).status)"
# expect: 200
```

The pycord container does NOT need restarting — it just needs the api on
the other side of the bridge to be reachable again.

## Pycord-specific gotchas discovered during migration

* **`discord.ui.View.__init__` requires a running event loop.** discord.py's
  init falls back to `get_event_loop()` and creates one if needed; py-cord
  calls `get_running_loop()` and raises `RuntimeError` if you try to
  construct a View from sync code (e.g. a top-level test setup). Production
  message handlers always run inside the bot's event loop, so this only
  surfaces in tests / smoke scripts — wrap them in `asyncio.run(...)`.

* **`@discord.ui.button` callback signature is REVERSED.** discord.py uses
  `(self, interaction, button)`; py-cord uses `(self, button, interaction)`.
  Mixing the two produces "This interaction failed" in the Discord UI and
  `'Button' object has no attribute 'guild'` in the container logs (because
  the Button is being passed where Interaction was expected). All 16 button
  callbacks across SpeakView / CallbackApprovalView / PlanReviewView /
  InjectionView / DreamEditReviewView were flipped during phase 2c.

* **User-defined `@client.event on_ready` REPLACES the default handler that
  syncs slash commands.** Result: slash commands silently disappear after
  the next bot restart. Each migrated bot's `on_ready` must call
  `await client.sync_commands()` explicitly. Affects all three bots.

* **`debug_guilds=[GUILD_ID]` on `discord.Bot(...)`** is the py-cord
  equivalent of discord.py's manual `tree.copy_global_to(guild) + tree.sync(guild)`.
  Slashes appear instantly in the named guild instead of waiting for the
  ~1h global propagation.
