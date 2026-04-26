# System prompt creation — end-to-end

How Phoebe assembles the system prompt for every LLM call, and where each
adaptation surface lives.

---

## 1. Inputs (read fresh on every `generate()` call; mtime-cached)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          SOURCES OF TRUTH                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│ config/prompts/                                                              │
│   ├── {role}.md                  ← dedicated roles (presence == dedicated)   │
│   ├── {role}_{agent_mode}_{mode}.md                                          │
│   ├── {role}_{agent_mode}.md     ← worker/supervisor only                    │
│   ├── {role}_{mode}.md                                                       │
│   ├── {role}_full.md             ← last-resort fallback                      │
│   ├── tools/{tool}.md            ← per-tool snippet, mtime-cached            │
│   └── overlays/{name}.md         ← per-model overlay paragraph, mtime-cached │
│                                                                              │
│ config/identity/{IDENTITY,USER}.md   state/soul/{SOUL,MEMORY}.md             │
│ config/skills/*/SKILL.md             ← YAML frontmatter, mtime-cached        │
│ config/config.yaml                   ← prompts.mode, soul.max_chars,         │
│                                        agent.mode.*, context.budgets.*       │
│ config/agents.yaml                   ← role_cfg, spawnable_agents            │
│ state/sessions/{sid}/state.json      ← tools.invoked counts, plan, perms     │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Call surface

```
loop.py / entrypoints.py
        │
        │ per attempt in retry loop:
        ▼
generate(role, allowed_tools, session_id, attempt,
         agent_mode="plan|build|converse",
         extra={ "{{AGENT_MODE}}":..., "{{PLAN_CONTEXT}}":...,
                 "{{SUPERVISOR_HANDLER}}":...,        # only attempt > 0
                 "{{TOOL_TRACES}}":..., "{{RUBRIC}}":..., ... },
         prompts_dir=None)            # dream runner overrides this
        │
        ▼
returns (prompt_text, agent_id="{role}_{sid[:8]}_{attempt}_{hex6}")
```

---

## 3. Internal pipeline — `app/prompt_generator.generate()`

```
┌────────────────── prompt_generator.generate() ───────────────────────────────┐
│                                                                              │
│ 1. Load cfg + agents_cfg                                                     │
│    mode := cfg.prompts.mode               ("concise" | "full")               │
│    ctx_on := cfg.context.enabled                                             │
│    elision := cfg.context.elision_strategy   (head|tail|head_tail|middle)    │
│    budgets := cfg.context.budgets.{soul,user,memory,identity,                │
│                                    tool_docs,skills,history,...}             │
│                                                                              │
│ 2. Load SessionState  → drives ranking + telemetry (best-effort)             │
│    state.skills.active ← _discover_skills() names                            │
│                                                                              │
│ 3. Template resolution  (_load_base_template)                                │
│    ┌──────────────────────────────────────────────┐                          │
│    │ tier 1: {role}.md           ── dedicated     │                          │
│    │ tier 2: {role}_{agent_mode}_{mode}.md        │ ←─ worker/supervisor     │
│    │ tier 3: {role}_{agent_mode}.md               │    only                  │
│    │ tier 4: {role}_{mode}.md                     │                          │
│    │ tier 5: {role}_full.md                       │                          │
│    └──────────────────────────────────────────────┘                          │
│    _strip_frontmatter(); raise FileNotFoundError if nothing matches.         │
│                                                                              │
│ 4. Curated section assembly  (per file: SOUL, USER, MEMORY, IDENTITY)        │
│       _read_curated() → char-cap to prompts.workspace_file_max_chars         │
│       if ctx_on:                                                             │
│           compress_section(text, budgets[<key>], elision, label)             │
│           ↳ if count(text) > budget: truncate(strategy)                      │
│              and prepend "[compressed <label>: N→budget tok]"                │
│                                                                              │
│ 5. Tool block  (_build_tool_block)                                           │
│       allowed_tools          ← role_cfg.allowed_tools − mode.excluded_tools  │
│       if ctx_on:                                                             │
│           filter_tool_docs(allowed, state, cfg, agent_mode)                  │
│             tier 1: _ALWAYS_TOOLS (file_read/write/edit, shell_exec,         │
│                                    memory_search, tool_result_recall)        │
│             tier 2: hot tools — sorted desc by state.tools.invoked[name]     │
│             tier 3: _MODE_SEEDS[agent_mode]  (per-mode high-signal set)      │
│             tier 4: remaining allowed, original order                        │
│             ↳ accumulate until count(TOOL_DOCS[t]) > budgets.tool_docs       │
│                always-tools never trimmed.                                   │
│       Render:  header + grammar reminder + inventory + each TOOL_DOCS[t]     │
│                                                                              │
│ 6. Skills block  (_build_skills_block)                                       │
│       skills_entries := _discover_skills() (config/skills/*/SKILL.md)        │
│       if ctx_on:                                                             │
│           filter_skills(user_msg, skills, cfg)                               │
│             greedy fit under budgets.skills using _format_skill_line()       │
│       + sub-agents table from spawnable_agents (agents.yaml)                 │
│                                                                              │
│ 6b. Overlay block  (_build_overlay_block)                                    │
│       name := role_cfg.model → cfg.models[name].prompt_overlay               │
│               or cfg.llm.prompt_overlay  (None → empty string)               │
│       text := _load_overlays()[name]   (mtime-cached)                        │
│       Substituted at {{PROMPT_OVERLAY}} — typically the FIRST line of the    │
│       template (worker_concise.md:1) so a per-model paragraph can sit        │
│       above the rules block. Default = empty.                                │
│                                                                              │
│ 7. Build subs map  (the substitution table)                                  │
│       see "Placeholder map" below                                            │
│       extra= overrides volatile slots from caller (loop.py / entrypoints.py) │
│                                                                              │
│ 8. Replace each "{{KEY}}" in template with subs[key]                         │
│                                                                              │
│ 9. Telemetry → state.context_stats.*                                         │
│       last_prompt_tokens   = count(prompt)                                   │
│       section_tokens       = {SOUL:..,USER:..,MEMORY:..,IDENTITY:..,         │
│                               ALLOWED_TOOLS:.., SKILLS:..}                   │
│       last_kv_prefix_hash  = sha1(prompt before "<|prefix_end|>")[:12]       │
│       soft_cap_exceeded    = count > cfg.context.total_soft_cap              │
│       state.save()  (best-effort)                                            │
│                                                                              │
│ 10. Audit write  → /cache/prompts/{agent_id}.md   (cleaned per-session)      │
│                                                                              │
│  → return (prompt, agent_id)                                                 │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Placeholder map (`subs`)

Every key is computed and substituted on every call, but **only those the
chosen template actually contains end up in the rendered prompt**. The
supervisor templates, for example, have no `{{SOUL}}/{{USER}}/{{MEMORY}}/
{{IDENTITY}}/{{ALLOWED_TOOLS}}/{{SKILLS}}` placeholders — the work is
computed and discarded.

```
                        SOURCE                                  ADAPTS WITH
─────────────────────── ─────────────────────────────────────── ────────────────
{{SOUL}}                state/soul/SOUL.md (compressed)         soul.max_chars
{{MEMORY}}              state/soul/MEMORY.md (compressed)       budgets.memory
{{USER}}                config/identity/USER.md (compressed)    budgets.user
{{IDENTITY}}            config/identity/IDENTITY.md (comp.)     budgets.identity
{{ALLOWED_TOOLS}}       _build_tool_block(filter_tool_docs(...)) state.tools.invoked,
                                                                 mode, agent_mode
{{SKILLS}}              _build_skills_block(filter_skills(...))  budgets.skills,
                                                                 spawnable_agents
{{AGENT_ID}}            role_sid_attempt_hex6                   per-call
{{AGENT_ROLE}}          role                                    fixed
{{SESSION_ID}}          session_id                              per-session
{{ATTEMPT}}             retry index (0..max_retries)            increases per retry
{{DATETIME}}            now (UTC)                               per-call (volatile)
{{HOST}}                $HOSTNAME                               container
{{THRESHOLD}}           supervisor_pass_threshold               per-mode override
{{SOUL_MAX_CHARS}}      soul.max_chars                          config
{{MODE}}                cfg.prompts.mode                        config
{{AGENT_MODE}}          worker:     _mode_context_string(...)  ← extra (long form)
                        supervisor: bare mode string ("plan"|"build"|"converse")
{{PLAN_CONTEXT}}        active plan transcript          ← extra session
{{PLAN_CONTEXT_SECTION}}supervisor plan-compliance block ← extra build+plan_active
{{SUPERVISOR_HANDLER}}  ACCEPTED/REJECTED preamble       ← extra ONLY when attempt>0
{{TOOL_TRACES}}         worker tool log (supervisor)     ← extra per attempt
{{WORKER_MODALITY}}     no_tool|tool_light|tool_heavy[_with_errors] ← extra
{{ERROR_RATE}}          errs/n  (supervisor)             ← extra
{{TOOL_COUNT}}          len(traces)                      ← extra
{{RUBRIC}}              _build_supervisor_rubric(modality, mode)   ← extra
{{APPROVAL_CONTEXT}}    "" (intentionally empty)         agent never sees gating
{{PROMPT_OVERLAY}}      _build_overlay_block(cfg, role_cfg) per-model overlay file
                        (cfg.models.<name>.prompt_overlay or cfg.llm.prompt_overlay)
discord_moderator only:
{{INACTIVE_DAYS}} {{ARCHIVE_CATEGORY}} {{CONVERSATIONS_CATEGORY}} {{THEMED_CATEGORIES}}
                        cfg.discord_moderator.*
```

**Note on `filter_skills(user_msg, ...)`** — `generate()` reads `user_msg`
from `extra["current_user_msg"]`, but no production callsite populates that
key (`loop.py`, `entrypoints.py` pass other extras only). So `user_msg` is
always `""` today and `filter_skills` is purely greedy budget-fit. The
`user_msg` parameter is the PR 3 / semantic-ranking placeholder.

---

## 5. KV-cache boundary (`<|prefix_end|>`)

The marker exists only in three templates today (and the cacheable region
differs per template):

```
worker_concise.md                       dreamer.md
────────────────────────────             ────────────────────────────
# Rules (static)                         (long static rules + opt-in
{{SOUL}}                                  deep-dive section, all static)
{{MEMORY}}
{{ALLOWED_TOOLS}}                        <|prefix_end|>
{{SKILLS}}                                {{IDENTITY}}
                                          {{SOUL}}
<|prefix_end|>                            {{USER}}
                                          {{MEMORY}}
{{DATETIME}} {{SESSION_ID}}               {{ALLOWED_TOOLS}}
{{ATTEMPT}}                               {{SKILLS}}
{{AGENT_MODE}}                            {{DATETIME}} {{SESSION_ID}}
{{APPROVAL_CONTEXT}}                      {{ATTEMPT}}
{{PLAN_CONTEXT_SECTION}}
```

- `worker_concise.md` caches SOUL/MEMORY/ALLOWED_TOOLS/SKILLS; volatile
  fields (DATETIME, SESSION_ID, ATTEMPT, AGENT_MODE, APPROVAL_CONTEXT,
  PLAN_CONTEXT_SECTION) sit below the marker.
- `dreamer.md` puts all curated/tool/skill substitutions **below** the
  marker on purpose — every dream wants fresh reads of IDENTITY/SOUL/USER/
  MEMORY, so cache reuse is intentionally forfeited.
- `dream_user_simulator.md` also has the marker (analogous design).
- **`worker_full.md`, `supervisor_full.md`, `supervisor_concise.md`, and
  every dedicated-role prompt have no marker at all** — `prefix_hash()`
  returns `None` for those calls, so KV-prefix caching is not asserted.
- Invariant carried by CLAUDE.md: when a template has a marker, content
  above it must be a pure function of `(role, mode, agent_mode)` so
  llama.cpp can reuse KV across turns.

Telemetry: `state.context_stats.last_kv_prefix_hash`.
Canary tests: `test_prompt_cache_boundary.py` + diagnostic probe
`prefix_marker_present`.

---

## 6. Adaptation axes (where the prompt morphs)

```
                                  ADAPTS                  CHANGED BY
──────────────────────────────── ───────────────────── ─────────────────────────
Role                              template selection    role_cfg.allowed_tools,
                                                        spawnable_agents
model                             {{PROMPT_OVERLAY}}    cfg.models.<name>.prompt_overlay
                                  paragraph             or cfg.llm.prompt_overlay
agent_mode (plan/build/converse) _MODE_SEEDS, excluded, _mode_tools, _mode_temperature,
                                  mode block,           agent.mode.{<m>}.excluded_tools,
                                  _MODE_SHORT/long      prompts.describe_mode_in_system_prompt
attempt #                         {{SUPERVISOR_HANDLER}} ACCEPTED/REJECTED preamble
                                  injected from attempt 1
session usage                     filter_tool_docs hot-rank per state.tools.invoked
context size                      compress_section,     cfg.context.budgets.* per slot,
                                  elision strategy      cfg.context.total_soft_cap
mode + modality (supervisor)      {{RUBRIC}} swaps      _classify_worker_modality(traces)
                                                        × {plan, build, converse}
                                                        × {no_tool, tool_light, tool_heavy,
                                                           *_with_errors}
threshold                         {{THRESHOLD}}         agent.supervisor_mode_overrides[mode]
plan-mode active plan             {{PLAN_CONTEXT}},     loop.py reads state.plan
                                  {{PLAN_CONTEXT_SECTION}} (build+plan only)
discord_moderator                 themed-categories     cfg.discord_moderator.*
                                  placeholders
shadow prompts (dream)            prompts_dir override   app.dream.runner sets alt root
```

---

## 7. Per-turn flow inside the retry loop

```
run_agent_loop  (app/loop.py)
  │
  ├─ for attempt in range(1 + max_retries):
  │     │
  │     ├─ supervisor_handler := "" if attempt==0 else ACCEPTED/REJECTED block
  │     │
  │     ├─ generate(role="worker", agent_mode=mode, attempt=N,
  │     │          extra={AGENT_MODE, PLAN_CONTEXT, SUPERVISOR_HANDLER}) ─┐
  │     │                                                                │
  │     ├─ _run_worker(prompt, ...) → worker_response, tool_traces  ◄────┘
  │     │     │
  │     │     └─ each iteration may also call generate() implicitly?  No —
  │     │        prompt is fixed for the turn. Volatile data flows in via
  │     │        user/tool_result messages, not by re-rendering the system prompt.
  │     │
  │     ├─ modality, error_rate, tool_count := _classify_worker_modality(traces)
  │     ├─ rubric_text := _build_supervisor_rubric(modality, mode)
  │     │
  │     ├─ generate(role="supervisor", agent_mode=mode, attempt=N,
  │     │          extra={AGENT_MODE, PLAN_CONTEXT, PLAN_CONTEXT_SECTION,
  │     │                 TOOL_TRACES, WORKER_MODALITY, ERROR_RATE,
  │     │                 TOOL_COUNT, RUBRIC, THRESHOLD})  ─────────────────┐
  │     │                                                                   │
  │     └─ _run_supervisor(...) → verdict (pass / score / issue arrays)  ◄──┘
  │           │
  │           └─ pass → "final" turn logged, _auto_store_memory fired
  │              fail → next attempt; verdict.feedback feeds SUPERVISOR_HANDLER
  └─ exhaust → best-scored result returned
```

---

## 8. Compactor effect on next-turn prompt

```
TRIGGER PATHS (current code)
───────────────────────────────────────────────────────────────────────────
manual:    POST /v1/sessions/{sid}/compact  (main.py:1044)
              └─► run_compaction(sid)
auto:      app/compactor.py:269 maybe_spawn(sid, cfg)
              └─ documented as "called by run_agent_loop's finally block"
                 BUT no production callsite exists today; only
                 test/test_compactor.py invokes it.  The auto-fire path
                 is currently DORMANT (loop.py:518-561 finally only calls
                 cleanup_generated and trace_queue 'done').

run_compaction(sid)
   │
   ├─ should_trigger(state, cfg) — three-way AND:
   │     turn_count - covers ≥ compaction_interval_turns (6)
   │     est_history_tokens ≥ budgets.history * 1.5
   │     now - last_compaction_ts ≥ compaction_churn_seconds (60)
   │
   ├─ spawns role="session_compactor" via entrypoints.run_agent_role
   │     plan_context = uncompacted transcript tail
   │
   ├─ output MUST contain "## RUNNING_SUMMARY" else
   │     {"error":"malformed_output"} and pointers stay put
   │
   └─ on success:
        append role=final pseudo-turn to state/sessions/{sid}/active.jsonl
        state.record_compaction(turn_count, "sessions/{sid}/active.jsonl")

NEXT TURN
   _rebuild_session_context() prefers state.history.active over the
   full turn log → fewer tokens in the message stream surrounding the
   system prompt. The system prompt itself is unchanged.
```

---

## 9. Outputs

- the rendered prompt is sent as `messages[0]` with `role="system"` —
  `app/worker.py:144` builds `[{"role":"system","content":system_prompt}] +
  messages` before each LLM call inside the worker iteration loop.
- a copy is written to `/cache/prompts/{agent_id}.md`. Cleanup:
  `cleanup_generated(sid)` runs in every loop/role finally
  (`loop.py:561`, `entrypoints.py:68`,`:211`); `cleanup_all_generated()`
  wipes the directory at API boot (`main.py:83`).
- telemetry persisted to `state.context_stats.{last_prompt_tokens,
  section_tokens, last_kv_prefix_hash, soft_cap_exceeded}` (best-effort —
  failures are swallowed so a state-write hiccup never breaks the turn).

---

## 10. Where to look

- `app/prompt_generator.py:519` — `generate()` (the orchestrator)
- `app/prompt_generator.py:436` — `_load_base_template()` (5-tier resolution)
- `app/prompt_generator.py:78` — `_load_tool_docs()` (mtime-cached tool snippets)
- `app/prompt_generator.py:139` — `_load_overlays()` (mtime-cached per-model overlays)
- `app/prompt_generator.py:174` — `_build_overlay_block()` (`{{PROMPT_OVERLAY}}`)
- `app/prompt_generator.py:306` — `_discover_skills()` (mtime-cached SKILL.md scan)
- `app/context_compressor.py:74` — `filter_tool_docs()` (4-tier ranking + budget)
- `app/context_compressor.py:148` — `filter_skills()` (greedy budget fit;
  `user_msg` parameter currently dormant)
- `app/context_compressor.py:57` — `compress_section()` (curated-file truncation)
- `app/context_compressor.py:259` — `prefix_hash()` (sha1 of bytes before
  `<|prefix_end|>`; returns None when marker absent)
- `app/mode.py:38` — `_mode_context_string()` (`{{AGENT_MODE}}` long-form
  block — worker only; supervisor passes the bare mode string)
- `app/supervisor.py:57` — `_classify_worker_modality()` and `:78`
  `_build_supervisor_rubric()` (mode × modality matrix)
- `app/loop.py:285` — worker `generate()` callsite per attempt
- `app/loop.py:368` — supervisor `generate()` callsite per attempt
- `app/entrypoints.py:44` (soul_updater), `:143` (`run_agent_role`) — other
  `generate()` callsites (no supervisor loop)
- `app/worker.py:154` — `[{"role":"system","content":prompt}] + messages`
  (where the prompt actually leaves the orchestrator; same `system_prompt`
  is reused across every iteration of the inner tool loop)
- `app/main.py:83` — `cleanup_all_generated()` on API boot
- `app/main.py:1044` — `POST /v1/sessions/{sid}/compact` →
  `run_compaction()` (the only production trigger today)
- `app/compactor.py:269` — `maybe_spawn()` (auto-trigger; orphaned —
  callsite documented but missing in `loop.py`'s finally)
- `config/prompts/worker_concise.md`, `dreamer.md`,
  `dream_user_simulator.md` — only templates carrying `<|prefix_end|>`
- `config/prompts/supervisor_full.md` / `supervisor_concise.md` — templates
  that intentionally use no curated/tool/skill substitutions
- `config/prompts/overlays/<name>.md` — per-model overlay paragraphs (e.g.
  `gemma_grammar.md` ships seeded but disabled)
- `config/config.yaml` — `prompts.mode`, `agent.mode.*`, `context.budgets.*`,
  `supervisor_pass_threshold`, `llm.prompt_overlay`,
  `models.<name>.prompt_overlay`
