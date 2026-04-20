"""
Dynamic per-agent prompt generator.

Each agent invocation gets a unique prompt assembled from:
  1. A base template (config/prompts/{role}_{mode}.md)
  2. Curated context files (SOUL, USER, MEMORY, IDENTITY)
  3. Dynamic context (agent_id, role, allowed tools, attempt, session_id, datetime)

The generated prompt is written to cache/prompts/{agent_id}.md as a
human-readable audit trail. Agents can read their own prompt via:
  file_read cache/prompts/{agent_id}.md

Generated files are gitignored and cleaned up on API startup.
"""

import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

from app.config_loader import get_config, get_agents_config
from app.context_compressor import (
    compress_section,
    filter_skills,
    filter_tool_docs,
    prefix_hash,
    section_tokens,
)
from app.tokenizer import ElisionStrategy, count

PROMPTS_DIR  = Path(os.environ.get("PROMPTS_DIR",  "/config/prompts"))
WORKSPACE    = Path(os.environ.get("WORKSPACE_DIR", "/workspace"))
CONFIG       = Path(os.environ.get("CONFIG_DIR",    "/config"))
STATE        = Path(os.environ.get("STATE_DIR",     "/state"))
GENERATED    = Path(os.environ.get("GENERATED_DIR", "/cache/prompts"))

# Canonical on-disk paths for curated files substituted into prompts.
_CURATED_PATHS: dict[str, Path] = {
    "SOUL.md":     STATE  / "soul" / "SOUL.md",
    "MEMORY.md":   STATE  / "soul" / "MEMORY.md",
    "USER.md":     CONFIG / "identity" / "USER.md",
    "IDENTITY.md": CONFIG / "identity" / "IDENTITY.md",
}

# Map curated file name → ContextBudgets field name used when cfg.context.enabled.
_CURATED_BUDGET_KEYS: dict[str, str] = {
    "SOUL.md":     "soul",
    "MEMORY.md":   "memory",
    "USER.md":     "user",
    "IDENTITY.md": "identity",
}

# ── Tool documentation registry ───────────────────────────────────────────────
# Each tool's prompt snippet lives at `{PROMPTS_DIR}/tools/{tool}.md`. The
# `TOOL_DOCS` mapping is rebuilt on directory mtime change, so editing a file
# takes effect without an app restart.
#
# Add a new tool: drop `{tool}.md` under `config/prompts/tools/`. Use the shape
#   ### tool_name
#   One-line description
#   <|tool_call|>call: tool_name, {...}<|tool_call|>
# so _build_tool_block can concatenate entries verbatim.

_TOOL_DOCS_CACHE: dict = {"mtime": 0.0, "docs": {}}


def _load_tool_docs() -> dict[str, str]:
    root = PROMPTS_DIR / "tools"
    try:
        mtime = root.stat().st_mtime
    except FileNotFoundError:
        _TOOL_DOCS_CACHE["mtime"] = 0.0
        _TOOL_DOCS_CACHE["docs"] = {}
        return _TOOL_DOCS_CACHE["docs"]
    if mtime == _TOOL_DOCS_CACHE["mtime"]:
        return _TOOL_DOCS_CACHE["docs"]

    docs: dict[str, str] = {}
    for f in sorted(root.glob("*.md")):
        try:
            docs[f.stem] = f.read_text(encoding="utf-8").rstrip()
        except Exception:
            continue
    _TOOL_DOCS_CACHE["mtime"] = mtime
    _TOOL_DOCS_CACHE["docs"] = docs
    return docs


class _ToolDocsMapping:
    """dict-like view over the on-disk tool doc registry; re-reads on mtime change."""

    def __getitem__(self, key: str) -> str:
        return _load_tool_docs()[key]

    def __contains__(self, key: object) -> bool:
        return key in _load_tool_docs()

    def __iter__(self):
        return iter(_load_tool_docs())

    def __len__(self) -> int:
        return len(_load_tool_docs())

    def get(self, key: str, default=None):
        return _load_tool_docs().get(key, default)

    def keys(self):
        return _load_tool_docs().keys()

    def values(self):
        return _load_tool_docs().values()

    def items(self):
        return _load_tool_docs().items()


TOOL_DOCS = _ToolDocsMapping()


# ── Template helpers ───────────────────────────────────────────────────────────

def _read_curated(path: Path, max_chars: int) -> str:
    if not path.exists():
        return f"[{path.name} not found — create it to provide context]"
    text = path.read_text(encoding="utf-8")
    if len(text) > max_chars:
        text = text[:max_chars] + f"\n\n... [{path.name} truncated at {max_chars} chars]"
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


def _build_skills_block(spawnable_agents: list[str], agents_cfg: dict,
                        skill_entries: list[dict] | None = None) -> str:
    """Build the {{SKILLS}} block listing available sub-agents and skill playbooks.

    When *skill_entries* is None the full discovered catalog is rendered (legacy
    behavior). Callers pass a filtered/ranked subset when context compression
    is active so the skills table stays inside its token budget.
    """
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

    skills = skill_entries if skill_entries is not None else _discover_skills()
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
            "These are procedures saved under `config/skills/*/SKILL.md`.\n"
            "Read the full file with `file_read` when the trigger matches what you're doing.\n\n"
            f"{table}\n"
        )

    return "\n".join(parts)


# ── Skill discovery ───────────────────────────────────────────────────────────
# Cached by mtime of the skills directory (cheap scan; only re-parse on change).
_SKILLS_CACHE: dict = {"mtime": 0.0, "entries": []}


def _discover_skills() -> list[dict]:
    """Scan config/skills/*/SKILL.md and return a list of skill metadata dicts.

    Each SKILL.md may start with YAML frontmatter:
        ---
        name: log-triage
        description: One-line summary
        when-to-trigger: When the user asks about error logs
        ---
    Fields we extract: name (fallback: directory name), description, when (trigger line).
    """
    skills_root = CONFIG / "skills"
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
        rel_path = f"config/skills/{skill_md.parent.name}/SKILL.md"
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


def _strip_frontmatter(text: str) -> str:
    """Return the file body with a leading `---\\n…\\n---\\n` block removed."""
    if not text.startswith("---"):
        return text
    end = text.find("\n---", 3)
    if end == -1:
        return text
    body_start = end + len("\n---")
    if body_start < len(text) and text[body_start] == "\n":
        body_start += 1
    return text[body_start:]


def _load_base_template(role: str, mode: str, agent_mode: str = "") -> str:
    """
    Load the base prompt template for `role`.

    Resolution order (first match wins):
      1. `{role}.md` — dedicated roles with a single prompt.
      2. `{role}_{agent_mode}_{mode}.md` — e.g. worker_plan_concise.md (worker/supervisor).
      3. `{role}_{agent_mode}.md`        — e.g. worker_plan.md         (worker/supervisor).
      4. `{role}_{mode}.md`              — e.g. worker_full.md.
      5. `{role}_full.md`                — last-resort variant.

    A role is "dedicated" iff `{role}.md` exists. No hardcoded allowlist — drop
    a new `{role}.md` into `config/prompts/` and it Just Works. Optional
    `kind: dedicated|worker|supervisor` frontmatter is documentation only.
    """
    base = PROMPTS_DIR

    def _load(p: Path) -> str:
        return _strip_frontmatter(p.read_text(encoding="utf-8"))

    # 1. Dedicated roles.
    dedicated = base / f"{role}.md"
    if dedicated.exists():
        return _load(dedicated)

    # 2/3. Mode-variant fallback (worker/supervisor and any role using the pattern).
    if agent_mode and role in ("worker", "supervisor"):
        tier1 = base / f"{role}_{agent_mode}_{mode}.md"
        if tier1.exists():
            return _load(tier1)
        tier2 = base / f"{role}_{agent_mode}.md"
        if tier2.exists():
            return _load(tier2)

    # 4. Prompt-mode fallback.
    path = base / f"{role}_{mode}.md"
    if path.exists():
        return _load(path)

    # 5. Last-resort.
    fallback = base / f"{role}_full.md"
    if fallback.exists():
        return _load(fallback)

    raise FileNotFoundError(
        f"No prompt template found for role={role!r} mode={mode!r} agent_mode={agent_mode!r}. "
        f"Expected one of: {role}.md, {role}_{agent_mode}_{mode}.md, {role}_{agent_mode}.md, "
        f"{role}_{mode}.md, {role}_full.md."
    )


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

    # Content-aware compression (PR 1). Master rollback: cfg.context.enabled=false
    # short-circuits to legacy char-capped behavior for every code path below.
    ctx_cfg: dict = cfg.get("context", {}) or {}
    ctx_on: bool  = bool(ctx_cfg.get("enabled", True))
    budgets: dict = (ctx_cfg.get("budgets", {}) or {}) if ctx_on else {}
    elision_name: str = str(ctx_cfg.get("elision_strategy", "head_tail"))
    try:
        elision = ElisionStrategy(elision_name)
    except ValueError:
        elision = ElisionStrategy.HEAD_TAIL

    agent_id = f"{role}_{session_id[:8]}_{attempt}_{uuid.uuid4().hex[:6]}"

    template = _load_base_template(role, mode, agent_mode=agent_mode)

    # Build substitution map
    soul_max = cfg["soul"]["max_chars"]
    agents_cfg = get_agents_config()
    role_cfg = agents_cfg.get(role, {})
    spawnable = role_cfg.get("spawnable_agents", [])

    # Load session state once so filter_tool_docs can rank by prior usage, and
    # so telemetry can be persisted without a second round-trip to disk.
    _state = None
    if session_id:
        try:
            from app.sessions.state import SessionState
            _state = SessionState.load_or_create(session_id)
            _state.set("skills.active", [s["name"] for s in _discover_skills()])
        except Exception:
            _state = None

    # Curated sections: read raw, then compress to per-section budget when ctx_on.
    curated: dict[str, str] = {}
    for fname in ("SOUL.md", "USER.md", "MEMORY.md", "IDENTITY.md"):
        raw = _read_curated(_CURATED_PATHS[fname], max_chars)
        if ctx_on:
            budget = int(budgets.get(_CURATED_BUDGET_KEYS[fname], 0) or 0)
            curated[fname] = compress_section(raw, budget, elision, label=fname.split(".")[0])
        else:
            curated[fname] = raw

    # Tool-doc + skill filtering (budget-capped). Falls through to full list
    # when ctx_on=false or filter returns empty.
    tool_list_for_block = allowed_tools
    skills_entries_for_block = None  # None → legacy _build_skills_block uses _discover_skills()
    if ctx_on:
        try:
            tool_list_for_block = filter_tool_docs(allowed_tools, _state, cfg, agent_mode=agent_mode)
        except Exception:
            tool_list_for_block = allowed_tools
        try:
            user_msg = (extra or {}).get("current_user_msg", "") if extra else ""
            skills_entries_for_block = filter_skills(user_msg, _discover_skills(), cfg)
        except Exception:
            skills_entries_for_block = None

    subs = {
        "{{SOUL}}":          curated["SOUL.md"],
        "{{USER}}":          curated["USER.md"],
        "{{MEMORY}}":        curated["MEMORY.md"],
        "{{IDENTITY}}":      curated["IDENTITY.md"],
        "{{ALLOWED_TOOLS}}": _build_tool_block(tool_list_for_block),
        "{{SKILLS}}":        _build_skills_block(spawnable, agents_cfg, skills_entries_for_block),
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

    # Telemetry — count the rendered prompt + per-section sub-blocks so we can
    # detect cache-prefix drift and compression regressions without replaying
    # the whole prompt. Best-effort; a counting or state-write failure must
    # never break the request path.
    if _state is not None:
        try:
            total = count(prompt)
            secs = section_tokens(subs, keys=(
                "{{SOUL}}", "{{USER}}", "{{MEMORY}}", "{{IDENTITY}}",
                "{{ALLOWED_TOOLS}}", "{{SKILLS}}",
            ))
            soft_cap = int(ctx_cfg.get("total_soft_cap", 12000) or 0)
            _state.set("context_stats.last_prompt_tokens", total)
            _state.set("context_stats.section_tokens", secs)
            _state.set("context_stats.last_kv_prefix_hash", prefix_hash(prompt))
            _state.set("context_stats.soft_cap_exceeded", bool(soft_cap and total > soft_cap))
            _state.save()
        except Exception:
            pass
    elif session_id:
        # State load failed earlier; still persist skills.active if we can.
        try:
            from app.sessions.state import SessionState
            _st = SessionState.load_or_create(session_id)
            _st.set("skills.active", [s["name"] for s in _discover_skills()])
            _st.save()
        except Exception:
            pass

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
