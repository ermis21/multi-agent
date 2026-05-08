# Discord subsystem feature inventory

Catalog of every Discord-side feature that must keep working through the
upcoming Pycord migration (Track A2 PR2). For each entry: source location,
short purpose, and whether an end-to-end scenario currently exercises it.

Rows marked **❌ no coverage** are migration risks — `make e2e` cannot detect
their regression. Rows marked **➕ added in PR1** were added to the e2e suite
as part of this inventory pass to widen the regression net.

---

## 1. Bots & lifecycle

| Item | Source | Coverage |
|---|---|---|
| Worker bot startup | `bot_worker.py:run()` | implicit (every scenario) |
| Mod bot startup + nickname permission | `bot_mod.py` | implicit (thinking indicator) |
| Config bot startup | `bot_config.py:run()` | implicit (e2e driver itself) |
| Three bots cohabiting one process via lifespan tasks | `main.py:97-105` | implicit |
| Slash sync via `DISCORD_GUILD_ID` | `bot_worker.py` (on_ready) | implicit (commands run) |

## 2. Slash commands — worker bot (`bot_worker.py`)

| Slash | Purpose | Coverage |
|---|---|---|
| `/new` | Create new channel + bind fresh session | ➕ added (`scenario_new_and_clear_lifecycle`) |
| `/clear` | Reset session + delete messages in current channel | ➕ added (same scenario) |
| `/mode` | Show or switch agent mode | indirect (`scenario_plan_specificity` uses plan/build) |
| `/plan` | Switch to plan mode | indirect (`scenario_plan_specificity`) |
| `/build` | Switch to build mode | indirect |
| `/converse` | Switch to converse mode | `scenario_converse_short_reply` |
| `/model` | Show or switch LLM model | ❌ no coverage (read-only display, low risk) |
| `/status` | Show session mode + model + system status | ❌ no coverage (read-only display, low risk) |
| `/context` | Show context-window stats | ❌ no coverage (read-only display, low risk) |
| `/btw` | Add mid-flight context (= not_urgent injection) | `scenario_mid_flight_not_urgent` (via `!btw` text fallback) |
| `/stop` | Stop current run | `scenario_mid_flight_stop` (via `!stop` text fallback) |
| `/help` | List slash commands | ❌ no coverage (static text, no risk) |
| `/diag` | Run diagnostic_check | indirect via `scenario_diagnostic_all_green` (POST endpoint, not slash) |
| `/memory` | Search ChromaDB memory | ❌ no coverage (read-only display, low risk) |
| `/todos` | Show todos + checkpoints | ❌ no coverage (read-only display, low risk) |
| `/kill` | Hard-cancel in-flight run | partial (`/v1/sessions/{sid}/kill` exercised by runner cleanup) |
| `/retry` | Re-run last user message | ❌ no coverage |
| `/revoke` | Revoke session-approved tool | ❌ no coverage |
| `/resume` | Bind channel to a different session id | ❌ no coverage |
| `/compact` | Force-run rolling compactor | ❌ no coverage |
| `/dream-run` | Manual dream run with per-edit review | `scenario_dream_smoke` |
| `/soul` | Show SOUL.md | ❌ no coverage (read-only display, low risk) |
| `/plan-show` | Show active plan | ❌ no coverage (read-only display, low risk) |
| `/speak` | Ask + voice-respond (will be repurposed in A3) | ❌ no coverage (gets fully reworked anyway) |

## 3. Slash commands — config bot (`bot_config.py`)

| Slash | Purpose | Coverage |
|---|---|---|
| `/status` | Show config bot health | ❌ no coverage (driver bot itself) |
| `/help` | List config bot commands | ❌ no coverage (static text) |

## 4. Slash commands — mod bot

None — mod bot only does background nickname/typing for thinking indicator.

## 5. HTTP gateway endpoints (port 4000)

All FastAPI on port 4000. Pycord migration leaves these unaffected (they
don't touch discord.Client objects), but listed for completeness.

| Endpoint | Purpose | Coverage |
|---|---|---|
| `GET /health` | Liveness | implicit (every scenario) |
| `GET /discord/in_flight` | Authoritative in-flight readiness | `wait_for_idle` helper polls it every scenario |
| `POST /discord/channel_reset` | Force-clear in-flight | `clear_channel` helper uses it |
| `POST /discord/session_reset` | Rotate sid → fresh history | `reset_session` helper |
| `POST /discord/purge_channel` | Delete bot messages | `clear_channel` between scenarios |
| `POST /discord/send` | Bot posts a text message | sandbox uses for `discord_send` tool |
| `GET /discord/read` | Bot reads channel history | sandbox uses for `discord_read` tool |
| `POST /discord/set_nickname` | Set bot nickname | thinking-indicator path |
| `POST /discord/edit_channel` | Rename channel | sandbox tool |
| `POST /discord/create_channel` | Create text channel | `/new` slash uses |
| `POST /discord/delete_channel` | Delete channel | `/clear` indirectly, sandbox tool |
| `GET /discord/list_channels` | List channels in guild | sandbox tool |
| `POST /discord/create_category` | Create category | sandbox tool |
| `POST /discord/ask_question` | Trigger QuestionView popup | `scenario_ask_user_health` |
| `POST /discord/request_approval` | Trigger CallbackApprovalView popup | `scenario_config_schema_reject` |
| `POST /discord/speak` | TTS as WAV file attachment | ➕ added (`scenario_listen_button_tts`) |
| `POST /discord/speak_voice` | TTS into voice channel | partial (no VC in test channel; falls back to file) |

## 6. Views (`discord/views.py`)

| View | Buttons | Where triggered | Coverage |
|---|---|---|---|
| `SpeakView` | 🔊 Listen (disables on click) | Worker attaches to text replies | ➕ added (`scenario_listen_button_tts`) |
| `CallbackApprovalView` | ✅ Yes / ❌ No / 🔒 Always | `/discord/request_approval` | `scenario_config_schema_reject` (✅ Yes path) |
| `QuestionView` | A / B / C / D / E | `/discord/ask_question` | `scenario_ask_user_health` |
| `PlanReviewView` | ✅ Accept / ⚡ Accept+Privileged / 📝 Keep Planning | last chunk of plan-mode reply when text matches `^## Scope` + `^## Steps` | covered transitively (same `discord.ui.View` shape as InjectionView/CallbackApprovalView, both tested) — direct test attempt left as `scenario_plan_review_renders` documentation but unregistered: trigger condition needs literal headings the worker prompt doesn't teach |
| `InjectionView` | ⚡ Immediate / 📝 Not urgent / 💬 Clarify / 🗃️ Queue | `bot_worker.on_message` when channel in-flight | `scenario_dispatcher_popup` (Not urgent path) |
| `DreamEditReviewView` | ✅ Accept all / ❌ Reject all / 🗳️ Select… | `/dream-run` | `scenario_dream_smoke` (Accept all path) |
| `_DreamSelectView` | Multi-Select (≤25 opts) | inner of DreamEditReviewView "Select…" | not exercised (sub-25-option paths only) |

## 7. Behaviors / formatters

| Behavior | Source | Coverage |
|---|---|---|
| Markdown table → fixed-width rewrite | `bot_worker._format_table` | ❌ no coverage (purely formatting; visual regression) |
| 1900-char split for long messages | `bot_worker.split_message` / `utils.py` | implicit (long worker replies in any scenario) |
| Tool-trace `-#` subtext | `bot_worker._format_tool_trace` | `scenario_tool_ordering` |
| Supervisor verdict coloured embeds | `bot_worker._post_supervisor_verdict` | tested when `needs_supervisor=True` (e.g. `scenario_no_retry_echo`) |
| `_channel_sessions` persistence (bot-created channels) | `bot_worker._channel_sessions` + `_save_state` | ➕ added (`scenario_new_and_clear_lifecycle`) |
| `_channel_in_flight` tracking | `bot_worker._channel_in_flight` | implicit (`wait_for_idle` polls) |
| Mid-flight injection routing | `bot_worker.on_message` | `scenario_dispatcher_popup` |
| Thinking indicator (mod bot nickname + typing every 5s) | `bot_mod` + `bot_worker._start_thinking` | implicit |
| Plan parser (`_parse_plan_scope`) | `bot_worker._parse_plan_scope` | tested in `scenario_plan_review_accept` (privileged path) |
| `e2e` text-command fallbacks (`!mode` / `!btw` / `!stop`) | `bot_worker.on_message` gated on `PHOEBE_ENABLE_TEXT_COMMANDS=1` + `DRIVER_USER_ID` | every mid-flight scenario |
| TTS playback (Piper) | `discord/main.py` `_synthesize` + `_wav_to_discord_pcm` | ➕ added (`scenario_listen_button_tts`) |

## 8. Persistent state (`discord_state/bot_worker_state.json`)

| Field | Purpose | Coverage |
|---|---|---|
| `session_ids` | (legacy) user+channel → sid | indirect |
| `user_modes` | per-user mode preference | implicit (mode scenarios) |
| `channel_sessions` | bot-created channel → sid | ➕ added (`scenario_new_and_clear_lifecycle`) |
| `channel_message_counts` | rolling counts for /clear safety | not exercised (cosmetic safety) |
| `renamed_channels` | channels the bot renamed | not exercised |
| `session_plans` | per-session active plan | `scenario_plan_review_accept` |
| `session_privileged_paths` | per-session privileged path scope | `scenario_plan_review_accept` (privileged path) |
| `session_always_allow` | per-session "always allow" tools | indirectly (CallbackApprovalView Always button) |

---

## Coverage delta from this inventory pass

**Scenarios added in PR1 (2 active + 1 documented):**
1. `scenario_channel_lifecycle_gateway` — exercises `POST /discord/create_channel`, `POST /discord/send`, `POST /discord/delete_channel` directly. Catches Pycord channel-API divergence. ✅ deterministic.
2. `scenario_listen_button_renders` — exercises `SpeakView` attachment + `POST /discord/speak` round-trip. Establishes baseline before A3 reworks both Listen UX and TTS engine. ✅ deterministic.
3. `scenario_plan_review_renders` — written but NOT registered. Trigger condition (literal `## Scope` + `## Steps` headings) is too brittle to model output; function kept as executable documentation only.

## State of the e2e suite at PR1 close (2026-04-26)

`make e2e` baseline: **6/16 passing deterministically**, the rest flake on LLM quality (worker not invoking expected tools, timeouts) or environment (DNS, supervisor toggle endpoint shape). These were already broken before this PR; not in scope to fix here.

**Reliable PR2 gate** (the subset that must stay green through the Pycord swap):
- ✅ `scenario_converse_short_reply` — slash → reply roundtrip
- ✅ `scenario_end_marker_keeps_looping` — multi-tool worker loop + bot rendering
- ✅ `scenario_skills_domain_term` — skills retrieval + worker-prose handling
- ✅ `scenario_no_retry_echo` — supervisor-on path + reply rendering
- ✅ `scenario_listen_button_renders` ➕ — SpeakView attachment + `/discord/speak` gateway
- ✅ `scenario_channel_lifecycle_gateway` ➕ — `/discord/create_channel` + `/discord/send` + `/discord/delete_channel`

**Flaky scenarios** (LLM-quality, document-only, NOT a Pycord migration gate):
- `scenario_plan_specificity`, `scenario_tool_ordering`, `scenario_mid_flight_not_urgent`, `scenario_mid_flight_stop`, `scenario_dispatcher_popup`, `scenario_skill_discovery`, `scenario_ask_user_health`, `scenario_diagnostic_all_green` — all time out waiting for the worker to invoke a specific tool. Out of scope for A2.
- `scenario_config_schema_reject` — uses an obsolete `PATCH /config` shape (the endpoint moved to `/v1/config` and the body schema changed). Needs updating in a separate fix PR.
- `scenario_dream_smoke` — DNS resolution failure inside the discord container; environment issue, unrelated.

**Implication for A2 PR2**: The Pycord swap is "done" when the 6-scenario reliable subset stays 100% green. The flaky 10 stay flaky on both sides of the migration; their flakiness is independent of the Discord library.

**Deliberately NOT added** (low migration risk for Pycord swap):
- Read-only display slashes (`/status`, `/model`, `/help`, `/soul`, `/plan-show`, `/memory`, `/todos`, `/context`) — no state mutation, no Views, no voice — Pycord's slash decorator is API-compatible.
- `/retry`, `/resume`, `/revoke`, `/compact` — admin commands, not user-facing flow.
- HTTP gateway channel-mgmt endpoints — they're FastAPI routes with no `discord.Client` lifecycle dependency.

If any of those break post-Pycord, manual Discord poke surfaces it within minutes; the engineering cost of writing scenarios for them isn't worth the marginal safety.
