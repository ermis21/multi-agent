"""
MCP Tool Server — runs in phoebe-sandbox container.

Exposes tools over HTTP JSON-RPC (POST /mcp).
NOT exposed publicly — internal Docker network only.

Tools:
  Workspace:   file_read, file_write, file_list, file_search, directory_tree,
               file_move, create_dir, file_info
  Shell:       shell_exec, execute_command  (resource-limited, cwd=/workspace)
  Git:         git_status, git_commit, git_rollback, git_log
  Docker:      docker_test_up, docker_test_down, docker_test_health
  Web:         web_search, web_fetch
  Memory:      memory_add, memory_search, memory_list
  Notion:      notion_search, notion_get_page, notion_create_page, notion_update_page
  Discord:     discord_send, discord_read, discord_set_nickname, discord_edit_channel
  Diagnostics: diagnostic_check

File path prefixes:
  (default)  → /workspace  (read + write; agent scratch)
  config/    → /config     (read + write; identity/, prompts/, skills/, *.yaml)
  state/     → /state      (read + write; soul/, memory/, sessions/, discord/, chroma/)
  cache/     → /cache      (read + write; regenerable audit trails)
  project/   → /project    (read-only; .env blocked)
"""

import json
import os
import re
import resource
import stat as _stat
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import uuid

import chromadb
import httpx
import yaml
from bs4 import BeautifulSoup
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
from exa_py import Exa
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="phoebe-sandbox-mcp", docs_url=None, redoc_url=None)

WORKSPACE       = Path(os.environ.get("WORKSPACE_DIR",   "/workspace")).resolve()
CONFIG          = Path(os.environ.get("CONFIG_DIR",      "/config")).resolve()
STATE           = Path(os.environ.get("STATE_DIR",       "/state")).resolve()
CACHE           = Path(os.environ.get("CACHE_DIR",       "/cache")).resolve()
PROJECT         = Path(os.environ.get("PROJECT_DIR",     "/project")).resolve()
MEMPALACE_HOME  = Path(os.environ.get("MEMPALACE_HOME",  "/state/chroma"))
EXA_API_KEY     = os.environ.get("EXA_API_KEY", "")
NOTION_URL      = os.environ.get("NOTION_URL",   "http://phoebe-notion:3000")
DISCORD_URL     = os.environ.get("DISCORD_URL",  "http://phoebe-discord:4000")
LLAMA_URL       = os.environ.get("LLAMA_URL",    "http://host.docker.internal:8080")
NOTION_TOKEN    = os.environ.get("NOTION_TOKEN", "")
PROMPTS_DIR     = Path(os.environ.get("PROMPTS_DIR",     "/config/prompts"))

# Config files — accessed via /project mount (sandbox has .:/project)
_CONFIG_YAML    = PROJECT / "config" / "config.yaml"
_AGENTS_YAML    = PROJECT / "config" / "agents.yaml"

# Shared subprocess environment — defined after WORKSPACE/PROJECT are resolved
_SUBPROCESS_ENV = {
    "PATH":      "/usr/local/bin:/usr/bin:/bin",
    "HOME":      "/tmp",
    "WORKSPACE": str(WORKSPACE),
    "CONFIG":    str(CONFIG),
    "STATE":     str(STATE),
    "CACHE":     str(CACHE),
    "PROJECT":   str(PROJECT),
}

# Shell execution limits
MAX_SHELL_TIMEOUT_S = 120
SHELL_MEM_LIMIT     = 512 * 1024 * 1024  # 512 MB address space
SHELL_PROC_LIMIT    = 64

# Output truncation
MAX_STDOUT_CHARS = 8000
MAX_STDERR_CHARS = 2000


# ── Safety helpers ─────────────────────────────────────────────────────────

def _safe_path(rel: str, root: Path) -> Path:
    """Resolve path and reject anything escaping root."""
    target = (root / rel).resolve()
    if not str(target).startswith(str(root)):
        root_name = {
            WORKSPACE: "workspace",
            CONFIG:    "config",
            STATE:     "state",
            CACHE:     "cache",
            PROJECT:   "project",
        }.get(root, str(root))
        hint = (
            f"Path {rel!r} is outside the {root_name} root ({root}). "
            f"Writable roots: /workspace (default), /config, /state, /cache. "
            f"/project is read-only. "
            f"Retry with a path under /workspace/ (e.g. /workspace/{Path(rel).name})."
        )
        raise HTTPException(status_code=400, detail=hint)
    return target


_BLOCKED_FILES = {".env", ".env.local", ".env.production", ".env.staging"}


def _strip_prefix(rel: str, prefix: str) -> str:
    """Strip leading `prefix/` or `/prefix/` (or bare `prefix` / `/prefix`) from rel."""
    return (
        rel.removeprefix(f"/{prefix}")
        .removeprefix(prefix)
        .lstrip("/")
    )


_PREFIX_ROOTS = (
    ("project", PROJECT),
    ("config",  CONFIG),
    ("state",   STATE),
    ("cache",   CACHE),
    ("workspace", WORKSPACE),
)


def _match_prefix(rel: str) -> tuple[str, Path] | None:
    for name, root in _PREFIX_ROOTS:
        if rel == name or rel == f"/{name}" or rel.startswith(f"{name}/") or rel.startswith(f"/{name}/"):
            return name, root
    return None


def _resolve_read_path(rel: str) -> Path:
    """
    Resolve a path for reading.
      project/  → /project  (read-only; .env* blocked)
      config/   → /config   (rw; identity/, skills/, prompts/, *.yaml)
      state/    → /state    (rw; soul/, memory/, sessions/, discord/, chroma/)
      cache/    → /cache    (rw; regenerable)
      workspace/ or no prefix → /workspace (agent scratch)
    """
    hit = _match_prefix(rel)
    if hit:
        name, root = hit
        inner = _strip_prefix(rel, name)
        p = _safe_path(inner or ".", root)
        if root == PROJECT and p.name in _BLOCKED_FILES:
            raise HTTPException(status_code=403, detail=f"Access to {p.name!r} is not permitted")
        return p
    return _safe_path(rel, WORKSPACE)


def _resolve_write_path(rel: str) -> Path:
    """
    Resolve a path for writing.
      config/ state/ cache/ workspace/ → respective root (rw)
      project/ → rejected (read-only mount)
      system/  → rejected (deprecated; use config/identity, config/skills, state/soul, state/memory)
      no prefix → /workspace (agent scratch, default)
    """
    if rel.startswith("system/") or rel == "system" or rel.startswith("/system"):
        raise HTTPException(
            status_code=400,
            detail=(
                "`system/` is deprecated — use `config/identity/`, `config/skills/`, "
                "`state/soul/`, or `state/memory/` depending on the target file."
            ),
        )
    hit = _match_prefix(rel)
    if hit:
        name, root = hit
        if root == PROJECT:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Cannot write under project/ — /project is mounted read-only. "
                    "Writable roots: /workspace (default), /config, /state, /cache."
                ),
            )
        inner = _strip_prefix(rel, name)
        return _safe_path(inner or ".", root)
    return _safe_path(rel, WORKSPACE)


def _set_limits():
    """Called as preexec_fn for shell subprocesses."""
    resource.setrlimit(resource.RLIMIT_AS,    (SHELL_MEM_LIMIT, SHELL_MEM_LIMIT))
    resource.setrlimit(resource.RLIMIT_NPROC, (SHELL_PROC_LIMIT, SHELL_PROC_LIMIT))


# ── Request / Response models ──────────────────────────────────────────────

class MCPRequest(BaseModel):
    method: str
    params: dict = {}


# ── Tool implementations ───────────────────────────────────────────────────

def _file_read(params: dict) -> dict:
    try:
        path = _resolve_read_path(params["path"])
        return {"content": path.read_text(encoding="utf-8")}
    except HTTPException:
        raise
    except FileNotFoundError:
        return {"error": (
            f"file not found: {params['path']}. "
            f"Use file_list or directory_tree to verify the path. "
            f"Paths under /project are read-only source; /workspace is writable."
        )}
    except IsADirectoryError:
        return {"error": (
            f"{params['path']} is a directory, not a file. "
            f"Use file_list or directory_tree to inspect directory contents."
        )}
    except Exception as e:
        return {"error": f"file_read failed: {e}"}


def _file_write(params: dict) -> dict:
    try:
        path = _resolve_write_path(params["path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(params["content"], encoding="utf-8")
        root = next((r for r in (CONFIG, STATE, CACHE, WORKSPACE) if str(path).startswith(str(r))), WORKSPACE)
        return {"written": len(params["content"]), "path": str(path.relative_to(root))}
    except HTTPException:
        raise
    except Exception as e:
        return {"error": f"file_write failed: {e}"}


def _file_edit(params: dict) -> dict:
    """
    Replace exactly one occurrence of old_string with new_string in a writable file.
    Routes via the `system/` prefix like file_write; defaults to /workspace.
    Fails if old_string is not found or matches more than once — forces the caller
    to pick an anchor that uniquely identifies the edit site.
    /project is mounted read-only and cannot be edited.
    """
    try:
        path = _resolve_write_path(params["path"])
        old  = params["old_string"]
        new  = params["new_string"]
        if not path.exists():
            return {"error": (
                f"file not found in workspace: {params['path']}. "
                f"file_edit only modifies existing files — to create a new file, use file_write "
                f"(excluded in plan mode; ask the user to switch with /mode build). "
                f"To verify the path, use file_list or directory_tree."
            )}
        content = path.read_text(encoding="utf-8")
        count = content.count(old)
        if count == 0:
            return {"error": (
                f"old_string not found in {params['path']}. "
                f"Read the current file contents with file_read before retrying — "
                f"old_string must match the file byte-for-byte, including whitespace."
            )}
        if count > 1:
            return {"error": (
                f"old_string matches {count} places in {params['path']} — make it more specific "
                f"by including surrounding context so it appears exactly once."
            )}
        path.write_text(content.replace(old, new, 1), encoding="utf-8")
        return {"ok": True, "path": params["path"], "replaced": 1}
    except HTTPException:
        raise
    except Exception as e:
        return {"error": f"file_edit failed: {e}"}


def _file_list(params: dict) -> dict:
    path = _resolve_read_path(params.get("path", "."))
    try:
        entries = []
        for e in sorted(path.iterdir()):
            st = e.stat()
            entries.append({"name": e.name, "is_dir": _stat.S_ISDIR(st.st_mode), "size": st.st_size})
        root = next(
            (r for r in (PROJECT, CONFIG, STATE, CACHE, WORKSPACE) if str(path).startswith(str(r))),
            WORKSPACE,
        )
        return {"entries": entries, "path": str(path.relative_to(root))}
    except (FileNotFoundError, NotADirectoryError):
        return {"entries": [], "error": f"Path not found: {params.get('path', '.')}"}


def _file_search(params: dict) -> dict:
    """Recursive glob search under workspace or project/."""
    root_rel = params.get("path", ".")
    pattern  = params["pattern"]
    root     = _resolve_read_path(root_rel)
    if not root.is_dir():
        return {"matches": [], "count": 0, "error": f"Not a directory: {root_rel}"}
    matches = [str(p.relative_to(root)) for p in sorted(root.rglob(pattern))][:200]
    return {"matches": matches, "count": len(matches)}


def _directory_tree(params: dict) -> dict:
    """Recursive directory tree up to depth N (max 6)."""
    root  = _resolve_read_path(params.get("path", "."))
    depth = min(int(params.get("depth", 3)), 6)

    def _tree(p: Path, d: int) -> dict:
        node: dict = {"name": p.name, "type": "dir" if p.is_dir() else "file"}
        if p.is_dir() and d > 0:
            try:
                node["children"] = [
                    _tree(c, d - 1)
                    for c in sorted(p.iterdir())
                    if not c.name.startswith(".")
                ]
            except PermissionError:
                node["children"] = []
        return node

    return _tree(root, depth)


def _file_move(params: dict) -> dict:
    """Move or rename a file within a writable root (workspace or system)."""
    try:
        src  = _resolve_write_path(params["source"])
        dest = _resolve_write_path(params["destination"])
        if not src.exists():
            return {
                "error": (
                    f"Source not found: '{params['source']}'. "
                    "file_move only operates within writable roots (/workspace, /config, /state, /cache). "
                    "To move project files, use shell_exec with 'mv' — but note /project is read-only."
                )
            }
        dest.parent.mkdir(parents=True, exist_ok=True)
        src.rename(dest)
        dest_root = next(
            (r for r in (CONFIG, STATE, CACHE, WORKSPACE) if str(dest).startswith(str(r))),
            WORKSPACE,
        )
        return {"ok": True, "destination": str(dest.relative_to(dest_root))}
    except HTTPException:
        raise
    except Exception as e:
        return {"error": f"file_move failed: {e}"}


def _create_dir(params: dict) -> dict:
    """Create a directory (and parents) in a writable root."""
    try:
        path = _resolve_write_path(params["path"])
        path.mkdir(parents=True, exist_ok=True)
        return {"ok": True, "path": params["path"]}
    except HTTPException:
        raise
    except Exception as e:
        return {"error": f"create_dir failed: {e}"}


def _file_info(params: dict) -> dict:
    """Return stat metadata for a file or directory."""
    try:
        p = _resolve_read_path(params["path"])
        s = p.stat()
        return {
            "path":     params["path"],
            "size":     s.st_size,
            "is_dir":   p.is_dir(),
            "modified": datetime.fromtimestamp(s.st_mtime, tz=timezone.utc).isoformat(),
        }
    except HTTPException:
        raise
    except Exception as e:
        return {"error": f"file_info failed: {e}"}


_TMP_HINT = (
    "[note] Command touches /tmp. Files written there do not persist across tool calls "
    "and are invisible to other tools (file_read, file_list, etc.). For anything you "
    "want to keep or hand off, use /workspace/.\n"
)


def _shell_exec(params: dict) -> dict:
    command     = params["command"]
    timeout_ms  = params.get("timeout_ms", 30_000)
    cwd_rel     = params.get("cwd", ".")
    timeout_s   = min(timeout_ms / 1000.0, MAX_SHELL_TIMEOUT_S)

    # Allow cwd inside workspace or project, but not arbitrary paths
    if cwd_rel.startswith("/project"):
        cwd = PROJECT
    else:
        cwd = _safe_path(cwd_rel, WORKSPACE)

    try:
        result = subprocess.run(
            ["sh", "-c", command],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            cwd=str(cwd),
            preexec_fn=_set_limits,
            env=_SUBPROCESS_ENV,
        )
        stdout = result.stdout[-MAX_STDOUT_CHARS:]
        # /tmp tripwire: non-blocking nudge prepended when commands touch /tmp
        if "/tmp" in command:
            stdout = _TMP_HINT + stdout
        return {
            "exit_code": result.returncode,
            "stdout":    stdout,
            "stderr":    result.stderr[-MAX_STDERR_CHARS:],
        }
    except subprocess.TimeoutExpired:
        return {"exit_code": -1, "stdout": "", "stderr": f"Timed out after {timeout_s:.0f}s"}
    except Exception as e:
        return {"exit_code": -1, "stdout": "", "stderr": str(e)}


# ── Git tools (cwd = /project) ─────────────────────────────────────────────

def _git_run(args: list[str]) -> dict:
    try:
        result = subprocess.run(
            ["git", "-c", f"safe.directory={PROJECT}"] + args,
            capture_output=True, text=True,
            timeout=30, cwd=str(PROJECT),
            env={**_SUBPROCESS_ENV, "GIT_TERMINAL_PROMPT": "0"},
        )
        return {
            "exit_code": result.returncode,
            "stdout":    result.stdout.strip(),
            "stderr":    result.stderr.strip(),
        }
    except subprocess.TimeoutExpired:
        return {"exit_code": -1, "stdout": "", "stderr": "git timed out"}
    except Exception as e:
        return {"exit_code": -1, "stdout": "", "stderr": f"git failed: {e}"}


def _git_status(_params: dict) -> dict:
    return _git_run(["status", "--short"])


def _git_commit(params: dict) -> dict:
    message = params.get("message", "agent: automated update")
    _git_run(["add", "-A"])
    return _git_run(["commit", "-m", message])


def _git_rollback(_params: dict) -> dict:
    """Safe rollback: creates a revert commit rather than destroying history."""
    return _git_run(["revert", "HEAD", "--no-edit"])


def _git_log(params: dict) -> dict:
    n = int(params.get("n", 10))
    return _git_run(["log", f"--max-count={n}", "--oneline"])


# ── Docker test-stack tools ────────────────────────────────────────────────

def _docker_compose(args: list[str]) -> dict:
    try:
        result = subprocess.run(
            ["docker", "compose", "-f", str(PROJECT / "docker-compose.test.yml")] + args,
            capture_output=True, text=True,
            timeout=MAX_SHELL_TIMEOUT_S, cwd=str(PROJECT),
            env=_SUBPROCESS_ENV,
        )
        return {
            "exit_code": result.returncode,
            "stdout":    result.stdout[-MAX_STDOUT_CHARS:],
            "stderr":    result.stderr[-MAX_STDERR_CHARS:],
        }
    except subprocess.TimeoutExpired:
        return {"exit_code": -1, "stdout": "", "stderr": "docker compose timed out"}
    except Exception as e:
        return {"exit_code": -1, "stdout": "", "stderr": f"docker compose failed: {e}"}


def _docker_test_up(_params: dict) -> dict:
    return _docker_compose(["up", "-d", "--build"])


def _docker_test_down(_params: dict) -> dict:
    return _docker_compose(["down", "--remove-orphans"])


def _docker_test_health(_params: dict) -> dict:
    """Probe the test stack's health endpoint on host port 8091."""
    try:
        r = httpx.get("http://host.docker.internal:8091/health", timeout=5)
        return {"status_code": r.status_code, "body": r.json()}
    except Exception as e:
        return {"status_code": -1, "error": str(e)}


# ── Web tools ─────────────────────────────────────────────────────────────

def _web_search(params: dict) -> dict:
    if not EXA_API_KEY:
        return {"error": "EXA_API_KEY not set"}
    query = params["query"]
    n     = int(params.get("n", 5))
    kind  = params.get("type", "auto")
    exa   = Exa(api_key=EXA_API_KEY)
    resp  = exa.search_and_contents(query, num_results=n, type=kind, highlights=True)
    results = []
    for r in resp.results:
        results.append({
            "url":        r.url,
            "title":      r.title,
            "highlights": r.highlights or [],
        })
    return {"results": results}


def _web_fetch(params: dict) -> dict:
    url = params["url"]
    try:
        r    = httpx.get(url, timeout=10, follow_redirects=True,
                         headers={"User-Agent": "phoebe-agent/1.0"})
        soup = BeautifulSoup(r.text, "lxml")
        for tag in soup(["script", "style", "nav", "footer"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)
        return {"url": url, "status": r.status_code, "text": text[:MAX_STDOUT_CHARS]}
    except Exception as e:
        return {"url": url, "error": str(e)}


# ── Memory tools (ChromaDB) ────────────────────────────────────────────────

_chroma_client = None  # chromadb.PersistentClient
_chroma_ef = DefaultEmbeddingFunction()


def _get_collection() -> chromadb.Collection:
    global _chroma_client
    if _chroma_client is None:
        MEMPALACE_HOME.mkdir(parents=True, exist_ok=True)
        _chroma_client = chromadb.PersistentClient(path=str(MEMPALACE_HOME))
    return _chroma_client.get_or_create_collection(
        "memories", embedding_function=_chroma_ef
    )


def _memory_add(params: dict) -> dict:
    content = params["content"]
    tags    = params.get("tags", [])
    col     = _get_collection()
    col.add(
        documents=[content],
        ids=[str(uuid.uuid4())],
        metadatas=[{"tags": ",".join(tags) if tags else ""}],
    )
    return {"added": True}


def _memory_search(params: dict) -> dict:
    query = params["query"]
    n     = int(params.get("n", 5))
    col   = _get_collection()
    try:
        results = col.query(query_texts=[query], n_results=n,
                            include=["documents", "metadatas", "distances"])
        hits = []
        for doc, meta, dist in zip(results["documents"][0],
                                   results["metadatas"][0],
                                   results["distances"][0]):
            hits.append({
                "content": doc,
                "score":   round(1 - dist, 4),
                "tags":    [t for t in meta.get("tags", "").split(",") if t],
            })
        return {"results": hits}
    except Exception as e:
        return {"results": [], "error": str(e)}


def _memory_list(params: dict) -> dict:
    n   = int(params.get("n", 20))
    col = _get_collection()
    try:
        result = col.get(limit=n, include=["documents", "metadatas"])
        memories = [
            {"content": doc, "tags": [t for t in meta.get("tags", "").split(",") if t]}
            for doc, meta in zip(result["documents"], result["metadatas"])
        ]
        return {"memories": memories}
    except Exception as e:
        return {"memories": [], "error": str(e)}


# ── Notion proxy ───────────────────────────────────────────────────────────

# Maps our tool names → Notion MCP method names
_NOTION_METHOD_MAP = {
    "notion_search":       "search",
    "notion_get_page":     "retrieve-a-page",
    "notion_create_page":  "create-a-page",
    "notion_update_page":  "update-page-content",
}


def _notion_call(tool_name: str, params: dict) -> dict:
    method = _NOTION_METHOD_MAP[tool_name]
    try:
        r = httpx.post(
            f"{NOTION_URL}/mcp",
            json={"jsonrpc": "2.0", "id": 1, "method": method, "params": params},
            timeout=15,
        )
        r.raise_for_status()
        data = r.json()
        if data.get("error"):
            return {"error": data["error"]}
        return data.get("result", {})
    except Exception as e:
        return {"error": f"Notion proxy failed: {e}"}




# ── Discord proxy ──────────────────────────────────────────────────────────

def _discord_proxy(endpoint: str, payload: dict) -> dict:
    try:
        r = httpx.post(f"{DISCORD_URL}/discord/{endpoint}", json=payload, timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": f"Discord proxy failed: {e}"}


def _discord_send(params: dict) -> dict:
    return _discord_proxy("send", params)

def _discord_read(params: dict) -> dict:
    try:
        r = httpx.get(f"{DISCORD_URL}/discord/read", params=params, timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": f"Discord proxy failed: {e}"}

def _discord_set_nickname(params: dict) -> dict:
    return _discord_proxy("set_nickname", params)

def _discord_edit_channel(params: dict) -> dict:
    return _discord_proxy("edit_channel", params)

def _discord_create_channel(params: dict) -> dict:
    return _discord_proxy("create_channel", params)

def _discord_delete_channel(params: dict) -> dict:
    return _discord_proxy("delete_channel", params)

def _discord_list_channels(params: dict) -> dict:
    try:
        r = httpx.get(f"{DISCORD_URL}/discord/list_channels", params=params, timeout=30)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": f"Discord proxy failed: {e}"}

def _discord_create_category(params: dict) -> dict:
    return _discord_proxy("create_category", params)

def _tts_speak(params: dict) -> dict:
    return _discord_proxy("speak", params)


# ── Diagnostics ────────────────────────────────────────────────────────────

def _diagnostic_check(_params: dict) -> dict:
    """
    Deterministic health check — no LLM involved.
    Probes every subsystem and returns pass/warn/fail per component.
    """
    checks: dict[str, dict] = {}

    def ok(detail=""):   return {"status": "pass", "detail": detail}
    def warn(detail):    return {"status": "warn", "detail": detail}
    def fail(detail):    return {"status": "fail", "detail": detail}

    # readable + writable probes for all four rw roots
    for label, root in (
        ("workspace", WORKSPACE),
        ("config",    CONFIG),
        ("state",     STATE),
        ("cache",     CACHE),
    ):
        try:
            checks[f"{label}_readable"] = ok() if root.is_dir() else fail("not a directory")
        except Exception as e:
            checks[f"{label}_readable"] = fail(str(e))
        probe = root / ".diag_probe"
        try:
            probe.write_text("diag-canary")
            content = probe.read_text()
            probe.unlink()
            checks[f"{label}_writable"] = ok() if content == "diag-canary" else fail("content mismatch")
        except Exception as e:
            checks[f"{label}_writable"] = fail(str(e))
            try:
                probe.unlink(missing_ok=True)
            except Exception:
                pass

    # mempalace_accessible — open chroma collection and list
    try:
        _get_collection().get(limit=1)
        checks["mempalace_accessible"] = ok()
    except Exception as e:
        checks["mempalace_accessible"] = fail(str(e))

    # git_repo — check /project is a git repo
    try:
        r = subprocess.run(
            ["git", "-c", f"safe.directory={PROJECT}", "rev-parse", "--is-inside-work-tree"],
            capture_output=True, text=True, timeout=5, cwd=str(PROJECT),
            env={**_SUBPROCESS_ENV, "GIT_TERMINAL_PROMPT": "0"},
        )
        if r.returncode == 0 and r.stdout.strip() == "true":
            sha = subprocess.run(
                ["git", "-c", f"safe.directory={PROJECT}", "rev-parse", "--short", "HEAD"],
                capture_output=True, text=True, timeout=5, cwd=str(PROJECT),
                env={**_SUBPROCESS_ENV, "GIT_TERMINAL_PROMPT": "0"},
            )
            checks["git_repo"] = ok(f"HEAD: {sha.stdout.strip()}")
        else:
            checks["git_repo"] = fail(r.stderr.strip() or "not a git repo")
    except Exception as e:
        checks["git_repo"] = fail(str(e))

    # llm_api — GET /health on LLM
    try:
        r = httpx.get(f"{LLAMA_URL}/health", timeout=5)
        checks["llm_api"] = ok(f"HTTP {r.status_code}") if r.status_code == 200 \
            else warn(f"HTTP {r.status_code}")
    except httpx.TimeoutException:
        checks["llm_api"] = fail(f"timeout reaching {LLAMA_URL}")
    except Exception as e:
        checks["llm_api"] = fail(str(e))

    # notion_container — try reaching the Notion MCP server
    try:
        r = httpx.get(NOTION_URL, timeout=5)
        checks["notion_container"] = ok(f"HTTP {r.status_code}")
    except httpx.TimeoutException:
        checks["notion_container"] = warn(f"timeout reaching {NOTION_URL}")
    except Exception as e:
        checks["notion_container"] = fail(str(e))

    # discord_container — GET /health
    try:
        r = httpx.get(f"{DISCORD_URL}/health", timeout=5)
        checks["discord_container"] = ok(f"HTTP {r.status_code}") if r.status_code == 200 \
            else warn(f"HTTP {r.status_code}")
    except httpx.TimeoutException:
        checks["discord_container"] = warn(f"timeout reaching {DISCORD_URL}")
    except Exception as e:
        checks["discord_container"] = fail(str(e))

    # exa_api_key
    checks["exa_api_key"] = ok() if EXA_API_KEY else fail("EXA_API_KEY not set")

    # notion_token
    checks["notion_token"] = ok() if NOTION_TOKEN \
        else warn("NOTION_TOKEN not set — notion_* tools will fail")

    # discord_tokens
    dw = bool(os.environ.get("DISCORD_TOKEN_WORKER", ""))
    dc = bool(os.environ.get("DISCORD_TOKEN_CONFIG", ""))
    if dw and dc:
        checks["discord_tokens"] = ok()
    elif not dw and not dc:
        checks["discord_tokens"] = warn("both DISCORD_TOKEN_WORKER and DISCORD_TOKEN_CONFIG missing")
    else:
        missing = "DISCORD_TOKEN_WORKER" if not dw else "DISCORD_TOKEN_CONFIG"
        checks["discord_tokens"] = warn(f"{missing} missing")

    # config_yaml + agents_yaml
    for label, path in [("config_yaml", _CONFIG_YAML), ("agents_yaml", _AGENTS_YAML)]:
        try:
            data = yaml.safe_load(path.read_text())
            checks[label] = ok() if isinstance(data, dict) else fail("parsed but not a dict")
        except FileNotFoundError:
            checks[label] = fail(f"not found: {path}")
        except Exception as e:
            checks[label] = fail(str(e))

    # prompt_templates — all required base templates must exist, with expected {{vars}}
    required_with_checks = [
        # (filename, [required regex patterns])
        ("worker_full.md",           [r"\{\{ALLOWED_TOOLS\}\}", r"<\|end\|>"]),
        ("worker_concise.md",        [r"\{\{ALLOWED_TOOLS\}\}", r"<\|end\|>"]),
        ("supervisor_full.md",       [r"\{\{RUBRIC\}\}",        r"\{\{THRESHOLD\}\}"]),
        ("supervisor_concise.md",    [r"\{\{RUBRIC\}\}",        r"\{\{THRESHOLD\}\}"]),
        ("soul_updater.md",          []),
        ("config_agent.md",          []),
        ("improvement_agent.md",     []),
        ("testing_agent_full.md",    []),
        ("testing_agent_concise.md", []),
        ("research_agent.md",        []),
        ("coding_agent.md",          []),
        ("tool_builder.md",          []),
        ("skill_builder.md",         [r"/config/skills/"]),
        ("webfetch_summarizer.md",   []),
        ("prompt_suggester.md",      []),
        ("session_compactor.md",     []),
        ("cron_creator.md",          []),
        ("agent_spawner.md",         []),
    ]
    missing_tpl: list[str] = []
    missing_var: list[str] = []
    for fname, patterns in required_with_checks:
        path = PROMPTS_DIR / fname
        if not path.exists():
            missing_tpl.append(fname)
            continue
        try:
            text = path.read_text()
        except Exception:
            missing_tpl.append(fname)
            continue
        for pat in patterns:
            if not re.search(pat, text):
                missing_var.append(f"{fname}:missing {pat}")
    if not missing_tpl and not missing_var:
        checks["prompt_templates"] = ok(f"all {len(required_with_checks)} templates present and valid")
    elif missing_tpl:
        checks["prompt_templates"] = warn(f"missing files: {', '.join(missing_tpl[:3])}")
    else:
        checks["prompt_templates"] = warn(f"template-var gaps: {', '.join(missing_var[:3])}")

    # config_schema — reach the API inside the shared netns (localhost:8090)
    try:
        r = httpx.get("http://localhost:8090/v1/config/validate", timeout=5)
        if r.status_code == 200:
            body = r.json()
            if body.get("valid"):
                checks["config_schema"] = ok("no schema drift")
            else:
                errs = body.get("errors", [])
                checks["config_schema"] = warn(f"{len(errs)} issue(s): {'; '.join(errs[:3])}")
        else:
            checks["config_schema"] = warn(f"API HTTP {r.status_code}")
    except Exception as e:
        checks["config_schema"] = fail(f"API unreachable: {e}")

    # supervisor_mode_overrides — structural check on parsed config
    try:
        cfg_data = yaml.safe_load(_CONFIG_YAML.read_text()) or {}
        overrides = (cfg_data.get("agent") or {}).get("supervisor_mode_overrides") or {}
        if not overrides:
            default_thresh = (cfg_data.get("agent") or {}).get("supervisor_pass_threshold", 0.7)
            checks["supervisor_mode_overrides"] = warn(
                f"no mode overrides set — all modes fall back to supervisor_pass_threshold={default_thresh}"
            )
        else:
            bad = []
            for mode in ("plan", "build", "converse"):
                val = overrides.get(mode)
                if val is None:
                    continue
                if not isinstance(val, (int, float)) or val < 0.0 or val > 1.0:
                    bad.append(f"{mode}={val!r}")
            if bad:
                checks["supervisor_mode_overrides"] = fail(f"invalid values: {', '.join(bad)}")
            else:
                present = [m for m in ("plan", "build", "converse") if m in overrides]
                checks["supervisor_mode_overrides"] = ok(f"configured: {', '.join(present)}")
    except Exception as e:
        checks["supervisor_mode_overrides"] = fail(str(e))

    # skills_directory — scan config/skills/*/SKILL.md and sanity-check frontmatter
    skills_root = CONFIG / "skills"
    try:
        if not skills_root.is_dir():
            checks["skills_directory"] = ok("no skills directory (0 skills)")
        else:
            skill_files = sorted(skills_root.glob("*/SKILL.md"))
            if not skill_files:
                checks["skills_directory"] = ok("0 skills")
            else:
                malformed: list[str] = []
                for sf in skill_files:
                    try:
                        text = sf.read_text(encoding="utf-8")
                        if not text.lstrip().startswith("---"):
                            malformed.append(sf.parent.name)
                            continue
                        # quick frontmatter parse — find the closing ---
                        body = text.lstrip()
                        end_idx = body.find("\n---", 3)
                        if end_idx < 0:
                            malformed.append(sf.parent.name)
                            continue
                        fm_text = body[3:end_idx].strip()
                        meta = yaml.safe_load(fm_text) or {}
                        if not isinstance(meta, dict) or not meta.get("name"):
                            malformed.append(sf.parent.name)
                    except Exception:
                        malformed.append(sf.parent.name)
                if not malformed:
                    checks["skills_directory"] = ok(f"{len(skill_files)} skill(s), all parseable")
                else:
                    checks["skills_directory"] = warn(
                        f"{len(skill_files)} skill(s), {len(malformed)} malformed: {', '.join(malformed[:3])}"
                    )
    except Exception as e:
        checks["skills_directory"] = fail(str(e))

    # inject_endpoint_reachable — POST to a non-existent session; expect 404
    try:
        r = httpx.post(
            "http://localhost:8090/v1/sessions/__diag__/inject",
            json={"text": "", "mode": "queue"},
            timeout=5,
        )
        if r.status_code == 404:
            checks["inject_endpoint_reachable"] = ok("route mounted")
        elif r.status_code in (400, 422):
            # Endpoint present but rejected payload — still proves routing
            checks["inject_endpoint_reachable"] = ok(f"route mounted (HTTP {r.status_code})")
        else:
            checks["inject_endpoint_reachable"] = warn(f"unexpected HTTP {r.status_code}")
    except Exception as e:
        checks["inject_endpoint_reachable"] = fail(f"API unreachable: {e}")

    # session_state_integrity — glob sessions/*.state.json, verify schema fields parse
    try:
        import glob as _glob
        # /project is the repo bind-mount; sessions/ lives at the repo root
        candidates = _glob.glob("/project/state/sessions/*/state.json")
        if not candidates:
            checks["session_state_integrity"] = ok("no state files yet")
        else:
            corrupt: list[str] = []
            missing_jsonl: list[str] = []
            for p in candidates:
                sid = Path(p).parent.name
                try:
                    data = json.loads(Path(p).read_text(encoding="utf-8"))
                except Exception:
                    corrupt.append(sid)
                    continue
                for key in ("session_id", "created_at", "history"):
                    if key not in data:
                        corrupt.append(f"{sid}:missing {key}")
                        break
                else:
                    jsonl = data.get("history", {}).get("full") or ""
                    jsonl_path = Path(jsonl if jsonl.startswith("/") else f"/project/{jsonl}")
                    if not jsonl_path.exists():
                        missing_jsonl.append(sid)
            total = len(candidates)
            bad = len(corrupt)
            if bad == 0 and not missing_jsonl:
                checks["session_state_integrity"] = ok(f"{total} state file(s), all parseable")
            elif bad == 0:
                checks["session_state_integrity"] = warn(
                    f"{total} state file(s) parse; {len(missing_jsonl)} missing matching JSONL: "
                    + ", ".join(missing_jsonl[:3])
                )
            else:
                checks["session_state_integrity"] = warn(
                    f"{bad}/{total} state files corrupt: " + ", ".join(corrupt[:3])
                )
    except Exception as e:
        checks["session_state_integrity"] = fail(f"integrity probe failed: {e}")

    # ── PR 1 context-compression probes ──────────────────────────────────────

    # compressor_budgets_valid: every budget > 0 and total_soft_cap >= sum(budgets)
    try:
        import yaml as _yaml
        with (CONFIG / "config.yaml").open() as f:
            _cfg = _yaml.safe_load(f) or {}
        _ctx = _cfg.get("context", {}) or {}
        _budgets = _ctx.get("budgets", {}) or {}
        _bad_keys = [k for k, v in _budgets.items() if not isinstance(v, int) or v < 0]
        _soft_cap = int(_ctx.get("total_soft_cap", 0) or 0)
        _sum = sum(int(v or 0) for v in _budgets.values())
        if _bad_keys:
            checks["compressor_budgets_valid"] = fail(f"invalid budget values: {_bad_keys}")
        elif _soft_cap and _sum > _soft_cap:
            checks["compressor_budgets_valid"] = warn(
                f"sum of budgets ({_sum}) exceeds total_soft_cap ({_soft_cap})"
            )
        else:
            checks["compressor_budgets_valid"] = ok(
                f"{len(_budgets)} budgets, sum={_sum}, soft_cap={_soft_cap}"
            )
    except Exception as e:
        checks["compressor_budgets_valid"] = fail(f"budget probe failed: {e}")

    # tokenizer_reachable: llama.cpp /tokenize must return 200 within 2s OR
    # tiktoken must import cleanly (fallback is acceptable).
    try:
        _llama = os.environ.get("LLAMA_URL", "http://llama-api-manager:8081")
        try:
            _r = httpx.post(f"{_llama}/tokenize", json={"content": "ping"}, timeout=2.0)
            _j = _r.json() if _r.status_code == 200 else {}
            if isinstance(_j, dict) and _j.get("tokens") is not None:
                checks["tokenizer_reachable"] = ok("llama.cpp /tokenize returns tokens")
            else:
                raise RuntimeError(f"HTTP {_r.status_code}, no tokens field")
        except Exception as _le:
            try:
                import tiktoken  # noqa: F401
                checks["tokenizer_reachable"] = warn(
                    f"llama /tokenize unreachable ({_le}); tiktoken fallback available"
                )
            except Exception:
                checks["tokenizer_reachable"] = fail(
                    f"both llama ({_le}) and tiktoken unavailable — using char/4 heuristic only"
                )
    except Exception as e:
        checks["tokenizer_reachable"] = fail(f"tokenizer probe failed: {e}")

    # tool_result_recall_roundtrip: write a synthetic handle, recall it, delete.
    try:
        import uuid as _uuid
        _sid = f"__diag_{_uuid.uuid4().hex[:6]}__"
        _hid = "rf-" + _uuid.uuid4().hex[:6]
        _d = STATE / "sessions" / _sid / "tool_results"
        _d.mkdir(parents=True, exist_ok=True)
        _payload = "diag-canary-" + _uuid.uuid4().hex
        (_d / f"{_hid}.txt").write_text(_payload)
        resp = _tool_result_recall({"id": _hid, "session_id": _sid, "offset": 0, "limit": 200})
        _ok = resp.get("chunk") == _payload and resp.get("chars_total") == len(_payload)
        # cleanup
        try:
            (_d / f"{_hid}.txt").unlink()
            _d.rmdir()
            _d.parent.rmdir()
        except Exception:
            pass
        checks["tool_result_recall_roundtrip"] = ok("handle store/recall round-trips") if _ok \
            else fail(f"recall mismatch: {resp}")
    except Exception as e:
        checks["tool_result_recall_roundtrip"] = fail(f"recall probe failed: {e}")

    # prefix_marker_present: the latest worker_* template must contain <|prefix_end|>
    try:
        _tp = CONFIG / "prompts"
        found: list[str] = []
        for name in ("worker_full.md", "worker_concise.md"):
            p = _tp / name
            if p.exists() and "<|prefix_end|>" in p.read_text(encoding="utf-8", errors="ignore"):
                found.append(name)
        if len(found) == 2:
            checks["prefix_marker_present"] = ok("both worker templates carry <|prefix_end|>")
        elif found:
            checks["prefix_marker_present"] = warn(
                f"only {found} carry <|prefix_end|> — KV cache boundary partial"
            )
        else:
            checks["prefix_marker_present"] = fail(
                "no worker template contains <|prefix_end|> — prefix cache won't reuse"
            )
    except Exception as e:
        checks["prefix_marker_present"] = fail(f"marker probe failed: {e}")

    # Compute overall
    statuses   = [c["status"] for c in checks.values()]
    fail_count = statuses.count("fail")
    warn_count = statuses.count("warn")
    pass_count = statuses.count("pass")
    overall    = "fail" if fail_count else ("warn" if warn_count else "pass")

    return {
        "checks":     checks,
        "overall":    overall,
        "pass_count": pass_count,
        "warn_count": warn_count,
        "fail_count": fail_count,
    }


# ── Handles ────────────────────────────────────────────────────────────────

def _tool_result_recall(params: dict) -> dict:
    """
    Read a previously-elided tool-result body by its handle id.

    Bodies are written once by the worker's tool-result append site under
    `/state/sessions/{sid}/tool_results/{handle_id}.txt`. This handler returns
    a character-indexed slice so the model can scan a large result in chunks
    without re-inflating the whole thing into the context window.

    Params:
      id         — required, e.g. "rf-7c29"
      session_id — required, owning session (path scoped here)
      offset     — char offset (default 0)
      limit      — max chars to return (default 4000, capped at 16000)
    """
    hid = str(params.get("id") or "").strip()
    sid = str(params.get("session_id") or "").strip()
    if not hid or not sid:
        raise HTTPException(status_code=400, detail="tool_result_recall requires both 'id' and 'session_id'")
    if not hid.startswith("rf-") or any(c in hid for c in "/\\.") or len(hid) > 32:
        raise HTTPException(status_code=400, detail=f"invalid handle id: {hid!r}")
    # session_id is baked into file-name so we sanity-check shape, too.
    if any(c in sid for c in "/\\") or len(sid) > 128:
        raise HTTPException(status_code=400, detail=f"invalid session id: {sid!r}")

    offset = max(0, int(params.get("offset") or 0))
    limit  = max(1, min(16000, int(params.get("limit") or 4000)))

    target = STATE / "sessions" / sid / "tool_results" / f"{hid}.txt"
    if not target.exists():
        return {
            "id": hid,
            "session_id": sid,
            "offset": offset,
            "limit": limit,
            "chars_total": 0,
            "chunk": "",
            "has_more": False,
            "error": "handle not found (may have been cleaned up or never stored)",
        }
    try:
        body = target.read_text(encoding="utf-8")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"failed to read handle {hid}: {e}")

    total = len(body)
    chunk = body[offset:offset + limit]
    return {
        "id": hid,
        "session_id": sid,
        "offset": offset,
        "limit": limit,
        "chars_total": total,
        "chunk": chunk,
        "has_more": (offset + len(chunk)) < total,
    }


# ── Dispatch table ─────────────────────────────────────────────────────────

HANDLERS = {
    # Workspace
    "file_read":              _file_read,
    "file_write":             _file_write,
    "file_edit":              _file_edit,
    "file_list":              _file_list,
    "file_search":            _file_search,
    "directory_tree":         _directory_tree,
    "file_move":              _file_move,
    "create_dir":             _create_dir,
    "file_info":              _file_info,
    # Shell
    "shell_exec":             _shell_exec,
    "execute_command":        _shell_exec,   # alias — same impl, clearer name
    # Git
    "git_status":             _git_status,
    "git_commit":             _git_commit,
    "git_rollback":           _git_rollback,
    "git_log":                _git_log,
    # Docker
    "docker_test_up":         _docker_test_up,
    "docker_test_down":       _docker_test_down,
    "docker_test_health":     _docker_test_health,
    # Web
    "web_search":             _web_search,
    "web_fetch":              _web_fetch,
    # Memory
    "memory_add":             _memory_add,
    "memory_search":          _memory_search,
    "memory_list":            _memory_list,
    # Notion (proxied through phoebe-notion container)
    "notion_search":          lambda p: _notion_call("notion_search", p),
    "notion_get_page":        lambda p: _notion_call("notion_get_page", p),
    "notion_create_page":     lambda p: _notion_call("notion_create_page", p),
    "notion_update_page":     lambda p: _notion_call("notion_update_page", p),
    # Discord
    "discord_send":            _discord_send,
    "discord_read":            _discord_read,
    "discord_set_nickname":    _discord_set_nickname,
    "discord_edit_channel":    _discord_edit_channel,
    "discord_create_channel":  _discord_create_channel,
    "discord_delete_channel":  _discord_delete_channel,
    "discord_list_channels":   _discord_list_channels,
    "discord_create_category": _discord_create_category,
    "tts_speak":               _tts_speak,
    # Diagnostics
    "diagnostic_check":       _diagnostic_check,
    # Handles
    "tool_result_recall":     _tool_result_recall,
}


# ── Routes ─────────────────────────────────────────────────────────────────

@app.post("/mcp")
def dispatch(req: MCPRequest):
    if req.method not in HANDLERS:
        return {"result": None, "error": f"Unknown method: {req.method!r}. Available: {sorted(HANDLERS)}"}
    try:
        result = HANDLERS[req.method](req.params)
        return {"result": result, "error": None}
    except HTTPException as e:
        return {"result": None, "error": e.detail}
    except Exception as e:
        return {"result": None, "error": str(e)}


@app.get("/health")
def health():
    return {
        "ok": True,
        "workspace": str(WORKSPACE),
        "config":    str(CONFIG),
        "state":     str(STATE),
        "cache":     str(CACHE),
        "project":   str(PROJECT),
    }


@app.get("/tools")
def list_tools():
    """Returns available tool names — useful for debugging."""
    return {"tools": sorted(HANDLERS.keys())}
