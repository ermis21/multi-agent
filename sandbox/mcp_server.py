"""
MCP Tool Server — runs in mab-sandbox container.

Exposes tools over HTTP JSON-RPC (POST /mcp).
NOT exposed publicly — internal Docker network only.

Tools:
  Workspace:   file_read, file_write, file_list
  Shell:       shell_exec  (resource-limited, cwd=/workspace)
  Git:         git_status, git_commit, git_rollback, git_log
  Docker:      docker_test_up, docker_test_down, docker_test_health
"""

import os
import resource
import subprocess
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="mab-sandbox-mcp", docs_url=None, redoc_url=None)

WORKSPACE = Path(os.environ.get("WORKSPACE_DIR", "/workspace")).resolve()
PROJECT   = Path(os.environ.get("PROJECT_DIR", "/project")).resolve()

# Shell execution limits
MAX_SHELL_TIMEOUT_S = 120
SHELL_MEM_LIMIT     = 512 * 1024 * 1024  # 512 MB address space
SHELL_PROC_LIMIT    = 64


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
    if not path.exists():
        return {"entries": [], "error": f"Path not found: {params.get('path', '.')}"}
    entries = [
        {"name": e.name, "is_dir": e.is_dir(), "size": e.stat().st_size if e.is_file() else 0}
        for e in sorted(path.iterdir())
    ]
    return {"entries": entries, "path": str(path.relative_to(WORKSPACE))}


def _shell_exec(params: dict) -> dict:
    command     = params["command"]
    timeout_ms  = params.get("timeout_ms", 30_000)
    cwd_rel     = params.get("cwd", ".")
    timeout_s   = min(timeout_ms / 1000.0, MAX_SHELL_TIMEOUT_S)

    # Allow cwd inside workspace or project, but not arbitrary paths
    if cwd_rel.startswith("/project") or cwd_rel == "/project":
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
            env={
                "PATH": "/usr/local/bin:/usr/bin:/bin",
                "HOME": "/tmp",
                "WORKSPACE": str(WORKSPACE),
                "PROJECT": str(PROJECT),
            },
        )
        return {
            "exit_code": result.returncode,
            "stdout":    result.stdout[-8000:] if len(result.stdout) > 8000 else result.stdout,
            "stderr":    result.stderr[-2000:] if len(result.stderr) > 2000 else result.stderr,
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
            env={"PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": "/tmp", "GIT_TERMINAL_PROMPT": "0"},
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
            env={"PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": "/tmp"},
        )
        return {
            "exit_code": result.returncode,
            "stdout":    result.stdout[-4000:],
            "stderr":    result.stderr[-2000:],
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


# ── Dispatch table ─────────────────────────────────────────────────────────

HANDLERS = {
    # Workspace
    "file_read":          _file_read,
    "file_write":         _file_write,
    "file_list":          _file_list,
    # Shell
    "shell_exec":         _shell_exec,
    # Git
    "git_status":         _git_status,
    "git_commit":         _git_commit,
    "git_rollback":       _git_rollback,
    "git_log":            _git_log,
    # Docker
    "docker_test_up":     _docker_test_up,
    "docker_test_down":   _docker_test_down,
    "docker_test_health": _docker_test_health,
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
