"""
Dynamic per-agent prompt generator.

Each agent invocation gets a unique prompt assembled from:
  1. A base template (prompts/base/{role}_{mode}.md)
  2. Workspace file injection (SOUL, USER, MEMORY, IDENTITY)
  3. Dynamic context (agent_id, role, allowed tools, attempt, session_id, datetime)

The generated prompt is written to prompts/generated/{agent_id}.md as a
human-readable audit trail. Agents can read their own prompt via:
  file_read prompts/generated/{agent_id}.md

Generated files are gitignored and cleaned up on API startup.
"""

import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

from app.config_loader import get_config, get_agents_config

PROMPTS_DIR  = Path(os.environ.get("PROMPTS_DIR",  "/app/prompts"))
WORKSPACE    = Path(os.environ.get("WORKSPACE_DIR", "/workspace"))
GENERATED    = PROMPTS_DIR / "generated"

# ── Tool documentation registry ───────────────────────────────────────────────
# Each entry is injected only when the tool appears in the agent's allowed list.
# Keep these concise — they're repeated in every prompt.

TOOL_DOCS: dict[str, str] = {
    "file_read": """\
### file_read
Read a file. Prefix path with `project/` to read the system source code (read-only).
<|tool_call|>call: file_read, {"path": "notes.txt"}<|tool_call|>
<|tool_call|>call: file_read, {"path": "project/app/agents.py"}<|tool_call|>""",

    "file_write": """\
### file_write
Write content to a file in the workspace. Creates parent directories if needed.
Workspace only — cannot write under `project/` (read-only mount).
<|tool_call|>call: file_write, {"path": "relative/path", "content": "text content"}<|tool_call|>""",

    "file_edit": """\
### file_edit
Replace exactly one occurrence of `old_string` with `new_string` in a workspace file.
Fails if `old_string` is not found or matches more than once — make it specific.
Workspace only — cannot edit files under `project/` (read-only mount).
<|tool_call|>call: file_edit, {"path": "notes.txt", "old_string": "foo", "new_string": "bar"}<|tool_call|>""",

    "file_list": """\
### file_list
List files and directories. Prefix with `project/` to list source code.
<|tool_call|>call: file_list, {"path": "."}<|tool_call|>
<|tool_call|>call: file_list, {"path": "project/app"}<|tool_call|>""",

    "file_search": """\
### file_search
Recursive glob search for files. Prefix path with `project/` to search source code.
<|tool_call|>call: file_search, {"path": ".", "pattern": "*.py"}<|tool_call|>
<|tool_call|>call: file_search, {"path": "project/", "pattern": "*.md"}<|tool_call|>""",

    "directory_tree": """\
### directory_tree
Recursive directory tree (default depth 3, max 6). Prefix with `project/` for source code.
<|tool_call|>call: directory_tree, {"path": ".", "depth": 3}<|tool_call|>
<|tool_call|>call: directory_tree, {"path": "project/", "depth": 2}<|tool_call|>""",

    "file_move": """\
### file_move
Move or rename a file within the workspace.
<|tool_call|>call: file_move, {"source": "old/path.txt", "destination": "new/path.txt"}<|tool_call|>""",

    "create_dir": """\
### create_dir
Create a directory (and parents) in the workspace.
<|tool_call|>call: create_dir, {"path": "my/new/folder"}<|tool_call|>""",

    "file_info": """\
### file_info
Get size, type, and modification time for a file or directory. Supports `project/` prefix.
<|tool_call|>call: file_info, {"path": "project/app/agents.py"}<|tool_call|>""",

    "shell_exec": """\
### shell_exec
Execute a bash command in the workspace. Timeout max 120s.
A Python venv is pre-created at `venv/` — use `venv/bin/python` or `venv/bin/pip install`.
> `/project` is READ-ONLY. Any write there (including `sed -i`) will fail. Use `file_edit`/`file_write` for workspace files, or switch to `coding_agent` for project source.
> Always check `exit_code` in the result — non-zero means the command failed even if no exception was raised.
<|tool_call|>call: shell_exec, {"command": "venv/bin/python script.py", "timeout_ms": 10000, "cwd": "."}<|tool_call|>""",

    "execute_command": """\
### execute_command
Execute a shell command in the workspace (alias for shell_exec). Timeout max 120s.
A Python venv is pre-created at `venv/` — use `venv/bin/python` or `venv/bin/pip install`.
> `/project` is READ-ONLY. Any write there (including `sed -i`) will fail. Use `file_edit`/`file_write` for workspace files, or switch to `coding_agent` for project source.
> Always check `exit_code` in the result — non-zero means the command failed even if no exception was raised.
<|tool_call|>call: execute_command, {"command": "venv/bin/pip install requests", "timeout_ms": 30000}<|tool_call|>""",

    "git_status": """\
### git_status
Show current git status of the project repo (/project).
<|tool_call|>call: git_status, {}<|tool_call|>""",

    "git_commit": """\
### git_commit
Stage all changes and create a commit in /project.
<|tool_call|>call: git_commit, {"message": "agent: describe the change"}<|tool_call|>""",

    "git_rollback": """\
### git_rollback
Create a revert commit to undo the last commit (safe, non-destructive).
<|tool_call|>call: git_rollback, {}<|tool_call|>""",

    "git_log": """\
### git_log
Show recent commit history of /project.
<|tool_call|>call: git_log, {"n": 10}<|tool_call|>""",

    "docker_test_up": """\
### docker_test_up
Build and start the test stack (port 8091) using docker-compose.test.yml.
Wait ~30s for it to become healthy before calling docker_test_health.
<|tool_call|>call: docker_test_up, {}<|tool_call|>""",

    "docker_test_down": """\
### docker_test_down
Stop and remove the test stack containers.
<|tool_call|>call: docker_test_down, {}<|tool_call|>""",

    "docker_test_health": """\
### docker_test_health
Probe the test stack health endpoint (http://host.docker.internal:8091/health).
<|tool_call|>call: docker_test_health, {}<|tool_call|>""",

    "read_config": """\
### read_config
Read the current system configuration (config/config.yaml).
<|tool_call|>call: read_config, {}<|tool_call|>""",

    "write_config": """\
### write_config
Update configuration values. Provide only the keys you want to change.
<|tool_call|>call: write_config, {"prompts": {"mode": "concise"}, "agent": {"max_retries": 3}}<|tool_call|>""",

    "web_search": """\
### web_search
Search the web via Exa. Returns titles, URLs, and highlight snippets.
<|tool_call|>call: web_search, {"query": "latest news about X", "n": 5, "type": "auto"}<|tool_call|>""",

    "web_fetch": """\
### web_fetch
Fetch and extract the readable text from a URL. Returns `{url, status, text}` — `status` is the HTTP status code (e.g. 200), `text` is the extracted body (≤ 8000 chars).
<|tool_call|>call: web_fetch, {"url": "https://example.com/article"}<|tool_call|>""",

    "memory_add": """\
### memory_add
Store a piece of information in long-term memory with optional tags.
<|tool_call|>call: memory_add, {"content": "User prefers concise answers.", "tags": ["preference"]}<|tool_call|>""",

    "memory_search": """\
### memory_search
Semantic search over stored memories. Returns top-k matches with scores.
<|tool_call|>call: memory_search, {"query": "user preferences", "n": 5}<|tool_call|>""",

    "memory_list": """\
### memory_list
List the most recent stored memories.
<|tool_call|>call: memory_list, {"n": 20}<|tool_call|>""",

    "notion_search": """\
### notion_search
Search across the connected Notion workspace.
<|tool_call|>call: notion_search, {"query": "project roadmap"}<|tool_call|>""",

    "notion_get_page": """\
### notion_get_page
Retrieve a Notion page by its ID.
<|tool_call|>call: notion_get_page, {"page_id": "abc123"}<|tool_call|>""",

    "notion_create_page": """\
### notion_create_page
Create a new page in Notion under a parent page or database.
<|tool_call|>call: notion_create_page, {"parent": {"page_id": "abc123"}, "properties": {"title": {"title": [{"text": {"content": "New Page"}}]}}}<|tool_call|>""",

    "notion_update_page": """\
### notion_update_page
Update the content of an existing Notion page.
<|tool_call|>call: notion_update_page, {"page_id": "abc123", "properties": {"title": {"title": [{"text": {"content": "Updated Title"}}]}}}<|tool_call|>""",

    "discord_send": """\
### discord_send
Send a message to a Discord channel.
<|tool_call|>call: discord_send, {"channel_id": 123456789, "content": "Hello!", "bot": "worker"}<|tool_call|>""",

    "discord_read": """\
### discord_read
Read recent messages from a Discord channel.
<|tool_call|>call: discord_read, {"channel_id": 123456789, "limit": 20, "bot": "worker"}<|tool_call|>""",

    "discord_set_nickname": """\
### discord_set_nickname
Set a nickname for a guild member.
<|tool_call|>call: discord_set_nickname, {"guild_id": 123, "user_id": 456, "nickname": "NewName", "bot": "worker"}<|tool_call|>""",

    "discord_edit_channel": """\
### discord_edit_channel
Edit a Discord channel's name, topic, or category.
Use `category_id` to move a channel to a different category.
<|tool_call|>call: discord_edit_channel, {"channel_id": 123456789, "name": "new-name", "topic": "New topic"}<|tool_call|>
<|tool_call|>call: discord_edit_channel, {"channel_id": 123456789, "category_id": 987654321}<|tool_call|>""",

    "discord_create_channel": """\
### discord_create_channel
Create a new text channel in the Discord guild.
`category_id` is optional. Discover category IDs with `discord_list_channels` first.
<|tool_call|>call: discord_create_channel, {"name": "my-channel", "topic": "Optional topic", "category_id": 123456789}<|tool_call|>""",

    "discord_delete_channel": """\
### discord_delete_channel
Permanently delete a Discord channel. Cannot be undone.
<|tool_call|>call: discord_delete_channel, {"channel_id": 123456789}<|tool_call|>""",

    "discord_list_channels": """\
### discord_list_channels
List all channels in the Discord guild with metadata.
Returns: id, name, type, category_id, category_name, topic, position, last_message_ts.
`last_message_ts` is null if the channel has never had a message.
<|tool_call|>call: discord_list_channels, {}<|tool_call|>""",

    "discord_create_category": """\
### discord_create_category
Create a new category channel in the Discord guild.
<|tool_call|>call: discord_create_category, {"name": "🛠️ System Work"}<|tool_call|>""",

    "tts_speak": """\
### tts_speak
Generate a spoken audio response and send it as a WAV file to a Discord channel.
Use this when the user asks you to "speak", "say out loud", or "respond with voice".
`channel_id` is required. Keep `text` concise — long responses may take a few seconds to synthesise.
<|tool_call|>call: tts_speak, {"channel_id": 123456789, "text": "Hello! Here is your answer."}<|tool_call|>""",

    "diagnostic_check": """\
### diagnostic_check
Run a deterministic health check of all system components (filesystem, ChromaDB,
git, LLM, Notion, Discord, API keys, config files, prompt templates).
Returns JSON with pass/warn/fail per component and an overall status.
No LLM involvement — fully deterministic.
<|tool_call|>call: diagnostic_check, {}<|tool_call|>""",

    "run_agent": """\
### run_agent
Delegate a task to a specialized sub-agent. It runs independently with its own tools
and cannot see your conversation. Give it a self-contained task with all needed context.
<|tool_call|>call: run_agent, {"role": "coding_agent", "task": "Add clean_text_for_tts() to /workspace/bot_worker.py that strips markdown."}<|tool_call|>
- `role` (required): agent from the "Available sub-agents" list.
- `task` (required): complete instruction with file paths, constraints, expected outcome.
The sub-agent's final response is returned as the tool result.""",

    "deliberate": """\
### deliberate
Run a structured debate between two positions to resolve a genuine decision.
Two advocates argue point-by-point in short (1-2 sentence) exchanges.
After 4 messages, YOU receive the full transcript and decide which side wins.
You can continue the debate for more rounds or accept a side.

When to use:
- Two plausible approaches and you're genuinely unsure which is better
- A choice that would be costly to reverse
- Tradeoffs you're not sure how to weigh

Do NOT use for trivial decisions, obvious choices, or questions the user should answer.

Frame positions as STRONG, SPECIFIC arguments — not wishy-washy hedges.

**New debate:**
<|tool_call|>call: deliberate, {"question": "Regex vs AST for extracting function signatures?", "context": "Python files, well-formatted, 50-500 lines each. Need name + params + return type.", "position_a": "Regex: simpler, fast, sufficient for well-formatted code", "position_b": "AST: handles edge cases like decorators, multiline signatures, nested functions"}<|tool_call|>

**Continue a debate** (if status was "active" and you want more rounds):
<|tool_call|>call: deliberate, {"debate_id": "debate_abc12345", "question": "", "context": "", "position_a": "", "position_b": "", "max_exchanges": 4}<|tool_call|>

You receive both the transcript AND an independent judge's verdict.
Then you decide:
- Accept the judge's verdict -> proceed with the winning approach
- Continue -> call deliberate again with the debate_id for more rounds
- Override the judge -> if you have context the advocates/judge didn't (explain why)

Result includes: transcript, positions, concessions, judge_winner, judge_reason, judge_confidence.""",

    "ask_user": """\
### ask_user
Ask the user a multiple-choice clarification question. 2-5 options.
The system may answer on the user's behalf if the answer is obvious from context.

Use when:
- The user's intent is ambiguous (could mean A or B)
- You need a preference only the user can provide
- The choice depends on context you don't have

Do NOT ask questions you can answer by reading files, searching memory, or investigating.

<|tool_call|>call: ask_user, {"question": "Which database should I configure?", "options": ["PostgreSQL (recommended for production)", "SQLite (simpler, for development)", "Keep current setup"], "context": "User asked to set up the database. Workspace has both pg and sqlite configs."}<|tool_call|>""",
}


# ── Template helpers ───────────────────────────────────────────────────────────

def _read_workspace(name: str, max_chars: int) -> str:
    path = WORKSPACE / name
    if not path.exists():
        return f"[{name} not found — create it to provide context]"
    text = path.read_text(encoding="utf-8")
    if len(text) > max_chars:
        text = text[:max_chars] + f"\n\n... [{name} truncated at {max_chars} chars]"
    return text


def _build_approval_context(cfg: dict, allowed_tools: list[str], mode: str = "converse") -> str:
    """
    Build the approval context block injected as {{APPROVAL_CONTEXT}}.

    The agent always sees "all tools pre-approved" so it calls any tool
    directly without hesitation.  Actual approval gating (ask_user / auto_fail)
    is enforced externally by call_tool() in mcp_client.py, which pauses the
    worker loop and surfaces a Yes / No / Always prompt in Discord.  The agent
    never needs to know about this; from its perspective the tool call simply
    takes a little longer to return.
    """
    return ""


def _build_tool_block(allowed_tools: list[str]) -> str:
    if not allowed_tools:
        return "_No tools available for this agent role._"
    docs = [TOOL_DOCS[t] for t in allowed_tools if t in TOOL_DOCS]
    inventory = ", ".join(f"`{t}`" for t in allowed_tools)
    header = (
        "## Calling a tool\n\n"
        "Emit exactly one tool call on its own, in this format and nothing else:\n"
        '`<|tool_call|>call: TOOL_NAME, {param_json}<|tool_call|>`\n\n'
        "Example:\n"
        '`<|tool_call|>call: file_read, {"path": "example.txt"}<|tool_call|>`\n\n'
        "Rules:\n"
        "- No prose, no markdown fences, no nesting around the call.\n"
        "- `param_json` is a flat JSON object — do NOT wrap it in `{\"params\": ...}`.\n"
        "- Give your final plain-text answer only once you have enough information.\n\n"
        f"## Your tools ({len(allowed_tools)})\n\n"
        f"You have access to **exactly these tools** — do not claim otherwise, "
        f"do not invent tool names, and do not call anything outside this list:\n"
        f"{inventory}\n\n"
        "## Tool reference\n"
    )
    return header + "\n\n".join(docs)


def _build_skills_block(spawnable_agents: list[str], agents_cfg: dict) -> str:
    """Build the {{SKILLS}} block listing available sub-agents and skill playbooks."""
    parts: list[str] = []
    if spawnable_agents:
        rows = []
        for role in spawnable_agents:
            desc = agents_cfg.get(role, {}).get("description", "")
            rows.append(f"| `{role}` | {desc} |")
        table = "| Agent | What it does |\n|-------|-------------|\n" + "\n".join(rows)
        parts.append(
            "## Available sub-agents\n\n"
            "You can delegate work to specialized agents using the `run_agent` tool.\n"
            "Each sub-agent runs independently with its own tools — it cannot see your conversation.\n"
            "Give it a **self-contained** task with all file paths, context, and constraints it needs.\n\n"
            f"{table}\n"
        )

    skills = _discover_skills()
    if skills:
        rows = []
        for s in skills:
            trig = s.get("when", "").strip().replace("\n", " ")
            if len(trig) > 90:
                trig = trig[:89] + "…"
            rows.append(f"| `{s['name']}` | {trig or '_(no trigger listed)_'} | `{s['path']}` |")
        table = "| Skill | Trigger | Path |\n|-------|---------|------|\n" + "\n".join(rows)
        parts.append(
            "## Available skill playbooks\n\n"
            "These are procedures saved under `workspace/skills/*/SKILL.md`.\n"
            "Read the full file with `file_read` when the trigger matches what you're doing.\n\n"
            f"{table}\n"
        )

    return "\n".join(parts)


# ── Skill discovery ───────────────────────────────────────────────────────────
# Cached by mtime of the skills directory (cheap scan; only re-parse on change).
_SKILLS_CACHE: dict = {"mtime": 0.0, "entries": []}


def _discover_skills() -> list[dict]:
    """Scan workspace/skills/*/SKILL.md and return a list of skill metadata dicts.

    Each SKILL.md may start with YAML frontmatter:
        ---
        name: log-triage
        description: One-line summary
        when-to-trigger: When the user asks about error logs
        ---
    Fields we extract: name (fallback: directory name), description, when (trigger line).
    """
    skills_root = WORKSPACE / "skills"
    try:
        mtime = skills_root.stat().st_mtime
    except FileNotFoundError:
        _SKILLS_CACHE["mtime"] = 0.0
        _SKILLS_CACHE["entries"] = []
        return []
    if mtime == _SKILLS_CACHE["mtime"]:
        return _SKILLS_CACHE["entries"]

    entries: list[dict] = []
    for skill_md in sorted(skills_root.glob("*/SKILL.md")):
        try:
            text = skill_md.read_text(encoding="utf-8")
        except Exception:
            continue
        meta = _parse_frontmatter(text)
        name = meta.get("name") or skill_md.parent.name
        when = (
            meta.get("when-to-trigger")
            or meta.get("when_to_trigger")
            or meta.get("description")
            or ""
        )
        rel_path = f"workspace/skills/{skill_md.parent.name}/SKILL.md"
        entries.append({"name": name, "when": when, "path": rel_path})

    _SKILLS_CACHE["mtime"] = mtime
    _SKILLS_CACHE["entries"] = entries
    return entries


def _parse_frontmatter(text: str) -> dict:
    """Minimal `key: value` YAML frontmatter parser — avoids a yaml dependency here."""
    if not text.startswith("---"):
        return {}
    end = text.find("\n---", 3)
    if end == -1:
        return {}
    block = text[3:end].strip()
    out: dict[str, str] = {}
    for line in block.splitlines():
        line = line.rstrip()
        if not line or ":" not in line or line.lstrip().startswith("#"):
            continue
        key, _, value = line.partition(":")
        key = key.strip().lower()
        value = value.strip().strip('"').strip("'")
        if key:
            out[key] = value
    return out


def _load_base_template(role: str, mode: str, agent_mode: str = "") -> str:
    """
    Load the appropriate base template with 3-tier fallback for agent_mode.

    For worker/supervisor roles when agent_mode is set (plan/build/converse):
      1. {role}_{agent_mode}_{mode}.md  (e.g. worker_plan_concise.md)
      2. {role}_{agent_mode}.md         (e.g. worker_plan.md)
      3. {role}_{mode}.md               (e.g. worker_full.md — existing fallback)

    Dedicated roles (soul_updater, coding_agent, etc.) skip this and use {role}.md.
    """
    base = PROMPTS_DIR / "base"

    # Dedicated roles have their own single template — no mode variants
    if role in ("soul_updater", "config_agent", "improvement_agent", "discord_moderator",
                "agent_spawner", "session_compactor", "cron_creator", "research_agent",
                "coding_agent", "tool_builder", "skill_builder",
                "webfetch_summarizer", "prompt_suggester"):
        filename = f"{role}.md"
        path = base / filename
        if path.exists():
            return path.read_text(encoding="utf-8")
        raise FileNotFoundError(f"No prompt template found for role={role!r} at {path}")

    # 3-tier fallback for worker/supervisor with agent_mode
    if agent_mode and role in ("worker", "supervisor"):
        # Tier 1: role_agentmode_promptmode (e.g. worker_plan_concise.md)
        tier1 = base / f"{role}_{agent_mode}_{mode}.md"
        if tier1.exists():
            return tier1.read_text(encoding="utf-8")
        # Tier 2: role_agentmode (e.g. worker_plan.md)
        tier2 = base / f"{role}_{agent_mode}.md"
        if tier2.exists():
            return tier2.read_text(encoding="utf-8")

    # Tier 3 / default: role_promptmode (e.g. worker_full.md)
    path = base / f"{role}_{mode}.md"
    if path.exists():
        return path.read_text(encoding="utf-8")

    # Final fallback: try the full variant
    fallback = base / f"{role}_full.md"
    if fallback.exists():
        return fallback.read_text(encoding="utf-8")
    raise FileNotFoundError(f"No prompt template found for role={role!r} mode={mode!r} agent_mode={agent_mode!r}")


# ── Public API ─────────────────────────────────────────────────────────────────

def generate(
    role: str,
    allowed_tools: list[str],
    session_id: str = "",
    attempt: int = 0,
    extra: dict | None = None,
    agent_mode: str = "converse",
) -> tuple[str, str]:
    """
    Generate a prompt for the given agent role.

    Returns:
        (prompt_text, agent_id)

    The generated prompt is written to prompts/generated/{agent_id}.md.
    """
    cfg       = get_config()
    mode      = cfg["prompts"]["mode"]
    max_chars = cfg["prompts"]["workspace_file_max_chars"]

    agent_id = f"{role}_{session_id[:8]}_{attempt}_{uuid.uuid4().hex[:6]}"

    template = _load_base_template(role, mode, agent_mode=agent_mode)

    # Build substitution map
    soul_max = cfg["soul"]["max_chars"]
    agents_cfg = get_agents_config()
    role_cfg = agents_cfg.get(role, {})
    spawnable = role_cfg.get("spawnable_agents", [])

    # Mirror the discovered skill catalog onto session state (skills.active).
    if session_id:
        try:
            from app.session_state import SessionState
            _st = SessionState.load_or_create(session_id)
            _st.set("skills.active", [s["name"] for s in _discover_skills()])
            _st.save()
        except Exception:
            pass

    subs = {
        "{{SOUL}}":          _read_workspace("SOUL.md",     max_chars),
        "{{USER}}":          _read_workspace("USER.md",     max_chars),
        "{{MEMORY}}":        _read_workspace("MEMORY.md",   max_chars),
        "{{IDENTITY}}":      _read_workspace("IDENTITY.md", max_chars),
        "{{ALLOWED_TOOLS}}": _build_tool_block(allowed_tools),
        "{{SKILLS}}":        _build_skills_block(spawnable, agents_cfg),
        "{{AGENT_ID}}":      agent_id,
        "{{AGENT_ROLE}}":    role,
        "{{SESSION_ID}}":    session_id,
        "{{ATTEMPT}}":       str(attempt),
        "{{DATETIME}}":      datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "{{HOST}}":          os.environ.get("HOSTNAME", "unknown"),
        "{{THRESHOLD}}":     str(cfg["agent"]["supervisor_pass_threshold"]),
        "{{SOUL_MAX_CHARS}}": str(soul_max),
        "{{MODE}}":          mode,
        "{{AGENT_MODE}}":           "",   # overridden via extra= when a mode is active
        "{{PLAN_CONTEXT}}":         "",   # overridden via extra= when a plan is active
        "{{SUPERVISOR_HANDLER}}":   "",   # empty by default; injected via extra= on retries
        "{{PLAN_CONTEXT_SECTION}}": "",   # empty by default; injected via extra= in build mode
        "{{TOOL_TRACES}}":          "(no tools were called)",  # default; overridden via extra=
        "{{APPROVAL_CONTEXT}}": _build_approval_context(cfg, allowed_tools, agent_mode),
        **(extra or {}),
    }

    # Discord moderator — inject config-driven placeholders
    if role == "discord_moderator":
        dm_cfg = cfg.get("discord_moderator", {})
        subs["{{INACTIVE_DAYS}}"]           = str(dm_cfg.get("inactive_days", 7))
        subs["{{ARCHIVE_CATEGORY}}"]        = dm_cfg.get("archive_category", "📦 Archive")
        subs["{{CONVERSATIONS_CATEGORY}}"]  = dm_cfg.get("conversations_category", "Conversations")
        themed = dm_cfg.get("themed_categories", [])
        subs["{{THEMED_CATEGORIES}}"]       = ", ".join(f'"{c}"' for c in themed)

    prompt = template
    for key, value in subs.items():
        prompt = prompt.replace(key, value)

    # Write generated prompt to file (audit trail + agent self-inspection)
    GENERATED.mkdir(parents=True, exist_ok=True)
    out_path = GENERATED / f"{agent_id}.md"
    out_path.write_text(prompt, encoding="utf-8")

    return prompt, agent_id


def cleanup_generated(session_id: str) -> None:
    """Remove generated prompt files for a completed session."""
    if not GENERATED.exists():
        return
    for f in GENERATED.glob(f"*_{session_id[:8]}_*.md"):
        f.unlink(missing_ok=True)


def cleanup_all_generated() -> None:
    """Remove all generated prompt files. Called on API startup."""
    if not GENERATED.exists():
        return
    for f in GENERATED.glob("*.md"):
        f.unlink(missing_ok=True)
