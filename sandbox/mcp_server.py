"""
MCP Tool Server — runs in mab-sandbox container.

Exposes tools over HTTP JSON-RPC (POST /mcp).
NOT exposed publicly — internal Docker network only.

Tools:
  Workspace:   file_read, file_write, file_list
  Shell:       shell_exec  (resource-limited, cwd=/workspace)
  Git:         git_status, git_commit, git_rollback, git_log
  Docker:      docker_test_up, docker_test_down, docker_test_health
  Web:         web_search, web_fetch
  Memory:      memory_add, memory_search, memory_list
  Notion:      notion_search, notion_get_page, notion_create_page, notion_update_page
  Discord:     discord_send, discord_read, discord_set_nickname, discord_edit_channel
  Diagnostics: diagnostic_check
"""

import os
import resource
import stat as _stat
import subprocess
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

app = FastAPI(title="mab-sandbox-mcp", docs_url=None, redoc_url=None)

WORKSPACE       = Path(os.environ.get("WORKSPACE_DIR",   "/workspace")).resolve()
PROJECT         = Path(os.environ.get("PROJECT_DIR",     "/project")).resolve()
MEMPALACE_HOME  = Path(os.environ.get("MEMPALACE_HOME",  "/mempalace"))
EXA_API_KEY     = os.environ.get("EXA_API_KEY", "")
NOTION_URL      = os.environ.get("NOTION_URL",   "http://mab-notion:3000")
DISCORD_URL     = os.environ.get("DISCORD_URL",  "http://mab-discord:4000")
LLAMA_URL       = os.environ.get("LLAMA_URL",    "http://host.docker.internal:8080")
NOTION_TOKEN    = os.environ.get("NOTION_TOKEN", "")
PROMPTS_DIR     = Path(os.environ.get("PROMPTS_DIR",     "/app/prompts"))

# Config files — accessed via /project mount (sandbox has .:/project)
_CONFIG_YAML    = PROJECT / "config" / "config.yaml"
_AGENTS_YAML    = PROJECT / "config" / "agents.yaml"

# Shared subprocess environment — defined after WORKSPACE/PROJECT are resolved
_SUBPROCESS_ENV = {
    "PATH":      "/usr/local/bin:/usr/bin:/bin",
    "HOME":      "/tmp",
    "WORKSPACE": str(WORKSPACE),
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
        raise HTTPException(status_code=400, detail=f"Path traversal rejected: {rel!r}")
    return target


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
    path = _safe_path(params["path"], WORKSPACE)
    return {"content": path.read_text(encoding="utf-8")}


def _file_write(params: dict) -> dict:
    path = _safe_path(params["path"], WORKSPACE)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(params["content"], encoding="utf-8")
    return {"written": len(params["content"]), "path": str(path.relative_to(WORKSPACE))}


def _file_list(params: dict) -> dict:
    path = _safe_path(params.get("path", "."), WORKSPACE)
    try:
        entries = []
        for e in sorted(path.iterdir()):
            st = e.stat()
            entries.append({"name": e.name, "is_dir": _stat.S_ISDIR(st.st_mode), "size": st.st_size})
        return {"entries": entries, "path": str(path.relative_to(WORKSPACE))}
    except (FileNotFoundError, NotADirectoryError):
        return {"entries": [], "error": f"Path not found: {params.get('path', '.')}"}


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
        return {
            "exit_code": result.returncode,
            "stdout":    result.stdout[-MAX_STDOUT_CHARS:],
            "stderr":    result.stderr[-MAX_STDERR_CHARS:],
        }
    except subprocess.TimeoutExpired:
        return {"exit_code": -1, "stdout": "", "stderr": f"Timed out after {timeout_s:.0f}s"}


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
                         headers={"User-Agent": "mab-agent/1.0"})
        soup = BeautifulSoup(r.text, "lxml")
        for tag in soup(["script", "style", "nav", "footer"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)
        return {"url": url, "text": text[:MAX_STDOUT_CHARS]}
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

    # workspace_readable
    try:
        checks["workspace_readable"] = ok() if WORKSPACE.is_dir() else fail("not a directory")
    except Exception as e:
        checks["workspace_readable"] = fail(str(e))

    # workspace_writable — write/read/unlink canary
    probe = WORKSPACE / ".diag_probe"
    try:
        probe.write_text("diag-canary")
        content = probe.read_text()
        probe.unlink()
        checks["workspace_writable"] = ok() if content == "diag-canary" else fail("content mismatch")
    except Exception as e:
        checks["workspace_writable"] = fail(str(e))
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

    # prompt_templates — all required base templates must exist
    required = [
        "worker_full.md", "worker_concise.md",
        "supervisor_full.md", "supervisor_concise.md",
        "soul_updater.md", "config_agent.md", "improvement_agent.md",
        "testing_agent_full.md", "testing_agent_concise.md",
    ]
    missing_tpl = [t for t in required if not (PROMPTS_DIR / "base" / t).exists()]
    if not missing_tpl:
        checks["prompt_templates"] = ok(f"all {len(required)} templates present")
    else:
        checks["prompt_templates"] = warn(f"missing: {', '.join(missing_tpl)}")

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


# ── Dispatch table ─────────────────────────────────────────────────────────

HANDLERS = {
    # Workspace
    "file_read":              _file_read,
    "file_write":             _file_write,
    "file_list":              _file_list,
    # Shell
    "shell_exec":             _shell_exec,
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
    # Notion (proxied through mab-notion container)
    "notion_search":          lambda p: _notion_call("notion_search", p),
    "notion_get_page":        lambda p: _notion_call("notion_get_page", p),
    "notion_create_page":     lambda p: _notion_call("notion_create_page", p),
    "notion_update_page":     lambda p: _notion_call("notion_update_page", p),
    # Discord
    "discord_send":           _discord_send,
    "discord_read":           _discord_read,
    "discord_set_nickname":   _discord_set_nickname,
    "discord_edit_channel":   _discord_edit_channel,
    # Diagnostics
    "diagnostic_check":       _diagnostic_check,
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
    return {"ok": True, "workspace": str(WORKSPACE), "project": str(PROJECT)}


@app.get("/tools")
def list_tools():
    """Returns available tool names — useful for debugging."""
    return {"tools": sorted(HANDLERS.keys())}
