"""
Persistent per-session state — `sessions/{sid}/state.json`.

Holds durable session config that outlives a single request: mode, model,
permissions, plan, supervisor overrides + last verdict, skills in play, tool
stats, workflow goal/checkpoints/todos, and pointers to the full and active
history JSONLs.

Every session owns a directory `sessions/{sid}/` containing:
  sessions/{sid}/turns.jsonl        — full turn log
  sessions/{sid}/state.json         — this file
  sessions/{sid}/active.jsonl       — compacted view the model sees, or absent
  sessions/{sid}/approvals.jsonl    — approval request/response audit
  sessions/{sid}/tool_errors.jsonl  — tool error audit

Write rules:
  - State file is rewritten atomically (tmp + rename) once per user-turn boundary.
  - `skills.invoked` is set-semantic (dedupe on add).
  - `context_audit.*` and `tools.invoked` are accumulated via `TurnAccumulator`
    in memory during the turn and merged via `flush_turn(...)` at turn end.
  - Sidecar `.jsonl` logs append live via `log_approval()` / `log_tool_error()`.

Runtime-only state (cancel event, asyncio task, pending injections,
current_iteration/attempt/tool_in_flight) lives in `app.main._active_sessions`
and is never persisted.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_SKILL_PATH_RE = re.compile(r"^/?config/skills/([^/]+)/SKILL\.md$")

SESSIONS_DIR = Path(os.environ.get("SESSIONS_DIR", "/state/sessions"))

_CACHE: dict[str, "SessionState"] = {}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _session_dir(session_id: str) -> Path:
    return SESSIONS_DIR / session_id


def _default_state(session_id: str) -> dict[str, Any]:
    full = f"sessions/{session_id}/turns.jsonl"
    return {
        "session_id": session_id,
        "parent_session_id": None,
        "agent_role": "worker",
        "created_at": _now(),
        "updated_at": _now(),
        "channel_id": None,
        "user_id": None,
        "source_trigger": {"type": "user", "ref": None},

        "mode": None,
        "model": None,
        "current_goal": None,
        "blocked_on": None,
        "checkpoints": [],
        "todos": [],

        "history": {
            "full": full,
            "active": None,
            "compaction_covers_up_to_turn": None,
            "last_compaction_ts": None,
        },

        "permissions": {
            "approved_tools": [],
            "denied_tools": [],
            "privileged_paths": [],
            "always_deny_paths": [],
        },
        "plan": None,

        "skills": {
            "active": [],
            "invoked": [],
        },
        "tools": {"invoked": {}},
        "context_audit": {
            "files_read": [],
            "files_written": [],
            "web_fetches": [],
            "web_searches": [],
            "memory_writes": [],
        },
        "sub_sessions": {"active": [], "completed": []},

        "supervisor": {
            "enabled": None,
            "threshold": None,
            "max_retries": None,
            "response_format": "json",
            "prompt_overrides": {},
            "last_verdict": None,
            "fail_count": 0,
        },

        "stats": {
            "turn_count": 0,
            "llm_call_count": 0,
            "token_usage": {"input": 0, "output": 0, "thinking": 0},
            "total_duration_ms": 0,
            "tool_error_count": 0,
        },

        # Telemetry populated by prompt_generator.generate(); used by diagnostics
        # and future cache-reuse heuristics.
        "context_stats": {
            "last_prompt_tokens": 0,
            "last_prompt_tokens_pre_compression": 0,
            "last_compression_ratio": 1.0,
            "last_kv_prefix_hash": None,
            "soft_cap_exceeded": False,
            "section_tokens": {},
            "handles_created": 0,
            "compression_triggers": {
                "soul": 0, "memory": 0, "user": 0, "identity": 0,
                "tool_docs": 0, "skills": 0, "history": 0, "tool_result": 0,
            },
        },
        # Handle store index for elided tool-result bodies. On-disk files live
        # under state/sessions/{sid}/tool_results/{handle_id}.txt — this dict
        # only tracks metadata so session_compactor / recall diagnostics can
        # enumerate without re-reading every blob.
        "tool_results": {},

        "debate_id": None,
        "pending_injections": [],
    }


@dataclass
class TurnAccumulator:
    """In-memory deltas gathered during a turn. Merged into state at turn end."""
    tools_invoked: dict[str, int] = field(default_factory=dict)
    files_read: list[str] = field(default_factory=list)
    files_written: list[str] = field(default_factory=list)
    web_fetches: list[str] = field(default_factory=list)
    web_searches: list[str] = field(default_factory=list)
    memory_writes: list[str] = field(default_factory=list)
    skills_invoked: list[str] = field(default_factory=list)
    tool_error_count: int = 0
    llm_call_count: int = 0
    token_usage: dict[str, int] = field(
        default_factory=lambda: {"input": 0, "output": 0, "thinking": 0}
    )
    duration_ms: int = 0

    def record_tool(self, tool: str, params: dict | None, error: bool) -> None:
        self.tools_invoked[tool] = self.tools_invoked.get(tool, 0) + 1
        if error:
            self.tool_error_count += 1
        p = params or {}
        # Opportunistic audit capture — best-effort, cheap
        if tool in ("file_read", "file_search", "file_list", "directory_tree", "file_info"):
            path = p.get("path")
            if path:
                self.files_read.append(str(path))
                if tool == "file_read":
                    m = _SKILL_PATH_RE.match(str(path))
                    if m:
                        self.record_skill(m.group(1))
        elif tool in ("file_write", "file_edit", "file_move", "create_dir"):
            path = p.get("path") or p.get("destination")
            if path:
                self.files_written.append(str(path))
        elif tool == "web_fetch":
            url = p.get("url")
            if url:
                self.web_fetches.append(str(url))
        elif tool == "web_search":
            q = p.get("query")
            if q:
                self.web_searches.append(str(q))
        elif tool == "memory_add":
            tags = p.get("tags") or []
            self.memory_writes.append(",".join(str(t) for t in tags) or "memory_add")

    def record_skill(self, name: str) -> None:
        if name and name not in self.skills_invoked:
            self.skills_invoked.append(name)


class SessionState:
    """Loadable, mutable-in-memory, atomically-persisted session state."""

    def __init__(self, session_id: str, data: dict[str, Any]):
        self.session_id = session_id
        self.data = data
        self.path = _session_dir(session_id) / "state.json"

    # ── Loading / saving ───────────────────────────────────────────────

    @classmethod
    def load_or_create(cls, session_id: str) -> "SessionState":
        if session_id in _CACHE:
            return _CACHE[session_id]
        sdir = _session_dir(session_id)
        sdir.mkdir(parents=True, exist_ok=True)
        path = sdir / "state.json"
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                _migrate_in_place(data, session_id)
            except Exception:
                data = _default_state(session_id)
        else:
            data = _default_state(session_id)
        st = cls(session_id, data)
        _CACHE[session_id] = st
        return st

    def save(self) -> None:
        """Atomic write: tmp file in same dir, then os.replace()."""
        self.data["updated_at"] = _now()
        sdir = _session_dir(self.session_id)
        sdir.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(
            prefix=".state.", suffix=".tmp", dir=str(sdir)
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(self.data, f, ensure_ascii=False, indent=2)
            os.replace(tmp_name, self.path)
            os.chmod(self.path, 0o644)
        except Exception:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
            raise

    # ── Turn lifecycle ─────────────────────────────────────────────────

    def flush_turn(self, acc: TurnAccumulator, verdict: dict | None = None) -> None:
        """Merge an in-memory turn accumulator into the persistent state and save.

        Called once per completed user turn (after the final response is logged).
        `verdict` is the supervisor's last result dict (or None if disabled / errored).
        """
        # Stats
        stats = self.data["stats"]
        stats["turn_count"] = int(stats.get("turn_count", 0)) + 1
        stats["llm_call_count"] = int(stats.get("llm_call_count", 0)) + acc.llm_call_count
        stats["tool_error_count"] = int(stats.get("tool_error_count", 0)) + acc.tool_error_count
        stats["total_duration_ms"] = int(stats.get("total_duration_ms", 0)) + acc.duration_ms
        tu = stats.setdefault("token_usage", {"input": 0, "output": 0, "thinking": 0})
        for k in ("input", "output", "thinking"):
            tu[k] = int(tu.get(k, 0)) + int(acc.token_usage.get(k, 0))

        # Tools invoked (accumulate counts)
        tmap = self.data["tools"].setdefault("invoked", {})
        for name, count in acc.tools_invoked.items():
            tmap[name] = int(tmap.get(name, 0)) + int(count)

        # Context audit (append, dedupe preserving order, cap to last 200 per list)
        ca = self.data["context_audit"]
        for key, items in (
            ("files_read", acc.files_read),
            ("files_written", acc.files_written),
            ("web_fetches", acc.web_fetches),
            ("web_searches", acc.web_searches),
            ("memory_writes", acc.memory_writes),
        ):
            existing = ca.setdefault(key, [])
            seen = set(existing)
            for item in items:
                if item not in seen:
                    existing.append(item)
                    seen.add(item)
            if len(existing) > 200:
                del existing[: len(existing) - 200]

        # Skills invoked (unique id list)
        inv = self.data["skills"].setdefault("invoked", [])
        seen = set(inv)
        for name in acc.skills_invoked:
            if name not in seen:
                inv.append(name)
                seen.add(name)

        # Supervisor verdict
        if verdict is not None:
            sup = self.data["supervisor"]
            sup["last_verdict"] = {
                "pass": bool(verdict.get("pass", False)),
                "score": float(verdict.get("score", 0.0)),
                "feedback": verdict.get("feedback", ""),
                "attempt": int(verdict.get("attempt", 0)),
                "ts": _now(),
            }
            if not sup["last_verdict"]["pass"]:
                sup["fail_count"] = int(sup.get("fail_count", 0)) + 1

        self.save()

    # ── Field-level updates (rare; save() called by caller if needed) ──

    def set(self, dotted_key: str, value: Any) -> None:
        """`state.set("supervisor.threshold", 0.75)` → nested write."""
        parts = dotted_key.split(".")
        node: Any = self.data
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = value

    def get(self, dotted_key: str, default: Any = None) -> Any:
        node: Any = self.data
        for p in dotted_key.split("."):
            if not isinstance(node, dict) or p not in node:
                return default
            node = node[p]
        return node

    def record_compaction(self, covers_up_to_turn: int, active_rel_path: str) -> None:
        """Atomically update the three history fields after a successful compaction.

        Called by app.compactor.run_compaction once the compactor role has
        produced a well-formed body and the pseudo-turn has been appended to
        active.jsonl. Caller owns the save() — we only mutate in memory so the
        set is visible as a single flush.
        """
        hist = self.data.setdefault("history", {})
        hist["active"] = active_rel_path
        hist["compaction_covers_up_to_turn"] = int(covers_up_to_turn)
        hist["last_compaction_ts"] = _now()

    def add_sub_session(self, child_sid: str) -> None:
        sub = self.data["sub_sessions"]
        if child_sid not in sub.get("active", []):
            sub.setdefault("active", []).append(child_sid)

    def complete_sub_session(self, child_sid: str) -> None:
        sub = self.data["sub_sessions"]
        if child_sid in sub.get("active", []):
            sub["active"].remove(child_sid)
        if child_sid not in sub.get("completed", []):
            sub.setdefault("completed", []).append(child_sid)


def _migrate_in_place(data: dict, session_id: str) -> None:
    """Best-effort merge of missing keys onto an older state file."""
    defaults = _default_state(session_id)

    def _fill(dst: dict, src: dict) -> None:
        for k, v in src.items():
            if k not in dst:
                dst[k] = v
            elif isinstance(v, dict) and isinstance(dst[k], dict):
                _fill(dst[k], v)

    _fill(data, defaults)


# ── Sidecar log helpers (append-only) ─────────────────────────────────

def _sidecar_path(session_id: str, suffix: str) -> Path:
    sdir = _session_dir(session_id)
    sdir.mkdir(parents=True, exist_ok=True)
    return sdir / f"{suffix}.jsonl"


def _append_jsonl(path: Path, entry: dict) -> None:
    entry.setdefault("ts", _now())
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def log_approval(session_id: str, tool: str, status: str, extra: dict | None = None) -> None:
    """status ∈ {requested, approved, denied, timeout, auto_allowed, auto_failed}."""
    entry = {"tool": tool, "status": status}
    if extra:
        entry.update(extra)
    _append_jsonl(_sidecar_path(session_id, "approvals"), entry)


def log_tool_error(session_id: str, tool: str, error: str, params_preview: str = "") -> None:
    _append_jsonl(
        _sidecar_path(session_id, "tool_errors"),
        {"tool": tool, "error": error[:500], "params_preview": params_preview},
    )


def log_supervisor_override(session_id: str, attempt: int, reason: str,
                            original: dict, overridden: dict) -> None:
    """Record a programmatic override of a supervisor verdict (e.g. hallucination guard)."""
    _append_jsonl(
        _sidecar_path(session_id, "supervisor_overrides"),
        {
            "attempt": attempt,
            "reason": reason,
            "original": {
                "pass":  original.get("pass"),
                "score": original.get("score"),
                "feedback": original.get("feedback", "")[:500],
            },
            "overridden": {
                "pass":  overridden.get("pass"),
                "score": overridden.get("score"),
            },
        },
    )


# ── Cache control ─────────────────────────────────────────────────────

def drop_from_cache(session_id: str) -> None:
    _CACHE.pop(session_id, None)


def cache_size() -> int:
    return len(_CACHE)
