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

from app.config_loader import get_config

PROMPTS_DIR  = Path(os.environ.get("PROMPTS_DIR",  "/app/prompts"))
WORKSPACE    = Path(os.environ.get("WORKSPACE_DIR", "/workspace"))
GENERATED    = PROMPTS_DIR / "generated"

# ── Tool documentation registry ───────────────────────────────────────────────
# Each entry is injected only when the tool appears in the agent's allowed list.
# Keep these concise — they're repeated in every prompt.

TOOL_DOCS: dict[str, str] = {
    "file_read": """\
### file_read
Read a file from the workspace.
```json
{"tool": "file_read", "params": {"path": "relative/path/to/file"}}
```""",

    "file_write": """\
### file_write
Write content to a file in the workspace. Creates parent directories if needed.
```json
{"tool": "file_write", "params": {"path": "relative/path", "content": "text content"}}
```""",

    "file_list": """\
### file_list
List files and directories in a workspace path.
```json
{"tool": "file_list", "params": {"path": "."}}
```""",

    "shell_exec": """\
### shell_exec
Execute a bash command in the workspace. Timeout max 120s.
```json
{"tool": "shell_exec", "params": {"command": "ls -la", "timeout_ms": 10000, "cwd": "."}}
```""",

    "git_status": """\
### git_status
Show current git status of the project repo (/project).
```json
{"tool": "git_status", "params": {}}
```""",

    "git_commit": """\
### git_commit
Stage all changes and create a commit in /project.
```json
{"tool": "git_commit", "params": {"message": "agent: describe the change"}}
```""",

    "git_rollback": """\
### git_rollback
Create a revert commit to undo the last commit (safe, non-destructive).
```json
{"tool": "git_rollback", "params": {}}
```""",

    "git_log": """\
### git_log
Show recent commit history of /project.
```json
{"tool": "git_log", "params": {"n": 10}}
```""",

    "docker_test_up": """\
### docker_test_up
Build and start the test stack (port 8091) using docker-compose.test.yml.
Wait ~30s for it to become healthy before calling docker_test_health.
```json
{"tool": "docker_test_up", "params": {}}
```""",

    "docker_test_down": """\
### docker_test_down
Stop and remove the test stack containers.
```json
{"tool": "docker_test_down", "params": {}}
```""",

    "docker_test_health": """\
### docker_test_health
Probe the test stack health endpoint (http://host.docker.internal:8091/health).
```json
{"tool": "docker_test_health", "params": {}}
```""",

    "read_config": """\
### read_config
Read the current system configuration (config/config.yaml).
```json
{"tool": "read_config", "params": {}}
```""",

    "write_config": """\
### write_config
Update configuration values. Provide only the keys you want to change.
```json
{"tool": "write_config", "params": {"prompts": {"mode": "concise"}, "agent": {"max_retries": 3}}}
```""",
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


def _build_tool_block(allowed_tools: list[str]) -> str:
    if not allowed_tools:
        return "_No tools available for this agent role._"
    docs = [TOOL_DOCS[t] for t in allowed_tools if t in TOOL_DOCS]
    header = (
        "Call a tool by responding with **only** a JSON object — no prose, "
        "no markdown fences, just the raw JSON. Example:\n"
        '```json\n{"tool": "file_read", "params": {"path": "example.txt"}}\n```\n'
        "Give your final answer once you have enough information.\n\n"
        "## Available Tools\n"
    )
    return header + "\n\n".join(docs)


def _load_base_template(role: str, mode: str) -> str:
    """
    Load the appropriate base template.
    Falls back to the full template if the concise one doesn't exist.
    """
    # soul_updater and config_agent have their own dedicated templates
    if role in ("soul_updater", "config_agent", "improvement_agent"):
        filename = f"{role}.md"
    else:
        filename = f"{role}_{mode}.md"

    path = PROMPTS_DIR / "base" / filename
    if not path.exists():
        # Fallback: try the full variant
        fallback = PROMPTS_DIR / "base" / f"{role}_full.md"
        if fallback.exists():
            return fallback.read_text(encoding="utf-8")
        raise FileNotFoundError(f"No prompt template found for role={role!r} mode={mode!r} at {path}")
    return path.read_text(encoding="utf-8")


# ── Public API ─────────────────────────────────────────────────────────────────

def generate(
    role: str,
    allowed_tools: list[str],
    session_id: str = "",
    attempt: int = 0,
    extra: dict | None = None,
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

    template = _load_base_template(role, mode)

    # Build substitution map
    soul_max = cfg["soul"]["max_chars"]
    subs = {
        "{{SOUL}}":          _read_workspace("SOUL.md",     max_chars),
        "{{USER}}":          _read_workspace("USER.md",     max_chars),
        "{{MEMORY}}":        _read_workspace("MEMORY.md",   max_chars),
        "{{IDENTITY}}":      _read_workspace("IDENTITY.md", max_chars),
        "{{ALLOWED_TOOLS}}": _build_tool_block(allowed_tools),
        "{{AGENT_ID}}":      agent_id,
        "{{AGENT_ROLE}}":    role,
        "{{SESSION_ID}}":    session_id,
        "{{ATTEMPT}}":       str(attempt),
        "{{DATETIME}}":      datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "{{HOST}}":          os.environ.get("HOSTNAME", "unknown"),
        "{{THRESHOLD}}":     str(cfg["agent"]["supervisor_pass_threshold"]),
        "{{SOUL_MAX_CHARS}}": str(soul_max),
        "{{MODE}}":          mode,
        **(extra or {}),
    }

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
