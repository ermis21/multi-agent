"""
Content-aware compression for prompt assembly.

Pure functions — no I/O except reading already-loaded session state. Called by
`prompt_generator.generate()` and the tool-result append site to keep high-signal
tokens inside Gemma's sliding window while shrinking or relocating filler.

All functions are no-ops when `cfg.context.enabled` is false — the caller
short-circuits for a single rollback switch.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Iterable

from app.tokenizer import ElisionStrategy, count, truncate

logger = logging.getLogger("phoebe.context")

# Always keep these in the tool block regardless of state: they anchor the
# tool-call grammar and cover the minimum viable toolbelt. If an agent lacks one
# of them its `allowed_tools` won't contain it — `filter_tool_docs` only ever
# returns a subset of the input list.
_ALWAYS_TOOLS = (
    "file_read",
    "file_write",
    "file_edit",
    "shell_exec",
    "memory_search",
    "tool_result_recall",
)

# Per-mode seed sets — high-signal tools for the mode even if never invoked
# this session. Keeps a fresh converse session from seeing an unusably thin
# tool block while still pruning 29 tools in other modes.
_MODE_SEEDS: dict[str, tuple[str, ...]] = {
    "plan": ("file_read", "file_search", "file_list", "directory_tree",
             "web_search", "web_fetch", "memory_search", "deliberate",
             "ask_user", "tool_result_recall"),
    "build": ("file_read", "file_write", "file_edit", "shell_exec",
              "git_status", "git_commit", "docker_test_up", "docker_test_down",
              "docker_test_health", "memory_add", "memory_search",
              "tool_result_recall"),
    "converse": ("file_read", "memory_search", "memory_list",
                 "web_search", "web_fetch", "discord_read",
                 "ask_user", "tool_result_recall"),
}


# ── Section-level ────────────────────────────────────────────────────────────

def compress_section(text: str, budget: int, strategy: ElisionStrategy, label: str) -> str:
    """
    Keep *text* verbatim when under budget; otherwise truncate with the given
    strategy and prepend a one-line compressed-size marker. *label* is a short
    identifier used both for the elision marker and for telemetry.
    """
    if budget <= 0 or not text:
        return text
    n = count(text)
    if n <= budget:
        return text
    compressed = truncate(text, budget, strategy)
    return f"[compressed {label}: {n}→{budget} tok]\n{compressed}"


# ── Tool-doc filter ──────────────────────────────────────────────────────────

def filter_tool_docs(allowed: list[str],
                     state,
                     cfg: dict,
                     agent_mode: str = "converse") -> list[str]:
    """
    Rank and prune *allowed* tool list so the rendered tool block fits the
    `context.budgets.tool_docs` budget.

    Ranking:
      1. Tools in _ALWAYS_TOOLS (grammar anchors).
      2. Tools invoked this session (state.tools.invoked count > 0), hot first.
      3. Mode seed set.
      4. Remaining allowed tools.

    Stops adding once the estimated docs would exceed the budget. Always at
    least the always-set is rendered (even past budget) so the model never
    loses `file_read`/`shell_exec` access.
    """
    if not allowed:
        return []

    budget = int(cfg.get("context", {}).get("budgets", {}).get("tool_docs", 1500))
    allowed_set = set(allowed)

    # Ordered unique accumulator
    ranked: list[str] = []
    seen: set[str] = set()

    def _add(name: str) -> None:
        if name in allowed_set and name not in seen:
            ranked.append(name)
            seen.add(name)

    # Tier 1 — always tools (as present in allowed)
    for t in _ALWAYS_TOOLS:
        _add(t)

    # Tier 2 — hot tools from session state
    invoked: dict[str, int] = {}
    if state is not None:
        try:
            invoked = state.get("tools.invoked", {}) or {}
        except Exception:
            invoked = {}
    hot = sorted(invoked.items(), key=lambda kv: -int(kv[1] or 0))
    for name, _n in hot:
        _add(name)

    # Tier 3 — mode seeds
    for t in _MODE_SEEDS.get(agent_mode, ()):
        _add(t)

    # Tier 4 — remaining allowed, in the order given
    for t in allowed:
        _add(t)

    # Enforce the budget. We need access to TOOL_DOCS to estimate sizes; import
    # lazily to avoid a circular dependency with prompt_generator.
    from app.prompt_generator import TOOL_DOCS

    always_keep = {t for t in _ALWAYS_TOOLS if t in allowed_set}
    out: list[str] = []
    running = 0
    for t in ranked:
        doc = TOOL_DOCS.get(t, "")
        tokens = count(doc) if doc else 40  # reasonable default for undocumented tools
        if t in always_keep or running + tokens <= budget:
            out.append(t)
            running += tokens
    return out


# ── Skills filter (PR 1: budget-only; PR 3 swaps in semantic) ─────────────────

def filter_skills(user_msg: str, skills: list[dict], cfg: dict) -> list[dict]:
    """
    Keep as many skills as fit in `context.budgets.skills`. PR 1 keeps them in
    the order they were discovered; PR 3 replaces this with semantic ranking
    against `user_msg`.
    """
    if not skills:
        return []
    budget = int(cfg.get("context", {}).get("budgets", {}).get("skills", 800))
    out: list[dict] = []
    running = 0
    for s in skills:
        row = f"| `{s.get('name','')}` | {s.get('when','')} | `{s.get('path','')}` |"
        tokens = count(row)
        if running + tokens > budget and out:
            break
        out.append(s)
        running += tokens
    return out


# ── Tool-result compaction + handle store ────────────────────────────────────

def _handle_id(tool: str, params: dict, body: str) -> str:
    h = hashlib.sha1()
    h.update(tool.encode("utf-8", "ignore"))
    try:
        h.update(json.dumps(params, sort_keys=True, ensure_ascii=False).encode("utf-8", "ignore"))
    except Exception:
        h.update(str(params).encode("utf-8", "ignore"))
    h.update(body.encode("utf-8", "ignore"))
    return f"rf-{h.hexdigest()[:6]}"


def compact_tool_result(tool: str, params: dict, body: str, budget: int) -> tuple[str, str | None]:
    """
    Return `(inline_text, handle_id)`.

    If the body fits within *budget* tokens, `inline_text == body` and
    `handle_id` is `None`. Otherwise the inline text is a head+tail preview
    with an elision marker telling the model it can call `tool_result_recall`
    with the returned handle.
    """
    if budget <= 0 or not body:
        return body, None
    n = count(body)
    if n <= budget:
        return body, None

    hid = _handle_id(tool, params or {}, body)
    head_budget = max(1, int(budget * 0.4))
    tail_budget = max(1, int(budget * 0.4))
    head = truncate(body, head_budget, ElisionStrategy.HEAD)
    tail = truncate(body, tail_budget, ElisionStrategy.TAIL)
    elided = max(0, n - head_budget - tail_budget)
    preview = (
        f"{head}\n"
        f"…[{elided} tok elided — recall with tool_result_recall id=\"{hid}\"]…\n"
        f"{tail}"
    )
    return preview, hid


# ── Handle storage ────────────────────────────────────────────────────────────
# Sandbox handler (tool_result_recall) reads these files; we write them here so
# the orchestrator never crosses the sandbox boundary with tool-result payloads.

_STATE_DIR = Path(os.environ.get("STATE_DIR", "/state"))


def tool_results_dir(session_id: str) -> Path:
    return _STATE_DIR / "sessions" / session_id / "tool_results"


def store_tool_result(session_id: str, handle_id: str, tool: str,
                      params: dict, body: str, *, tokens: int | None = None) -> None:
    """
    Append-once storage for an elided tool-result body. Best-effort: failures
    are logged and swallowed so the turn never breaks on storage errors.
    """
    if not session_id or not handle_id:
        return
    try:
        d = tool_results_dir(session_id)
        d.mkdir(parents=True, exist_ok=True)
        target = d / f"{handle_id}.txt"
        if not target.exists():
            target.write_text(body, encoding="utf-8")
        idx = d / "index.jsonl"
        preview = json.dumps(params, ensure_ascii=False, sort_keys=True)[:160]
        entry = {
            "handle_id": handle_id,
            "tool": tool,
            "params_preview": preview,
            "bytes": len(body.encode("utf-8", "ignore")),
            "tokens": tokens if tokens is not None else count(body),
        }
        with idx.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.warning("tool_result handle store failed for %s: %s", handle_id, e)


# ── Telemetry helpers ────────────────────────────────────────────────────────

def prefix_hash(rendered_prompt: str, marker: str = "<|prefix_end|>") -> str | None:
    """Return a short hash of the cacheable prefix (everything up to *marker*)."""
    if marker not in rendered_prompt:
        return None
    prefix = rendered_prompt.split(marker, 1)[0]
    return "sha1:" + hashlib.sha1(prefix.encode("utf-8", "ignore")).hexdigest()[:12]


def section_tokens(subs: dict[str, str], keys: Iterable[str] | None = None) -> dict[str, int]:
    """Count tokens for a set of template substitutions — used by telemetry."""
    if keys is None:
        keys = subs.keys()
    out: dict[str, int] = {}
    for k in keys:
        v = subs.get(k)
        if isinstance(v, str) and v:
            out[k.strip("{}")] = count(v)
    return out
