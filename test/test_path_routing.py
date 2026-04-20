"""Pure-unit tests for path-prefix routing in sandbox/mcp_server.py.

Covers:
  - `_resolve_read_path` for all five roots: workspace/, config/, state/, cache/, project/.
  - `_resolve_write_path` rejects the read-only project/ root and the deprecated system/ prefix.
  - Path traversal attempts are rejected.
  - `.env*` reads are blocked under project/.
"""

from __future__ import annotations

import pytest
from fastapi import HTTPException

import sandbox.mcp_server as mcp


@pytest.fixture
def tmp_roots(tmp_path, monkeypatch):
    """Remap all five roots to tmp subdirs and rebuild `_PREFIX_ROOTS`."""
    roots = {}
    for name in ("workspace", "config", "state", "cache", "project"):
        d = tmp_path / name
        d.mkdir()
        roots[name] = d.resolve()
    monkeypatch.setattr(mcp, "WORKSPACE", roots["workspace"])
    monkeypatch.setattr(mcp, "CONFIG",    roots["config"])
    monkeypatch.setattr(mcp, "STATE",     roots["state"])
    monkeypatch.setattr(mcp, "CACHE",     roots["cache"])
    monkeypatch.setattr(mcp, "PROJECT",   roots["project"])
    monkeypatch.setattr(mcp, "_PREFIX_ROOTS", (
        ("project",   roots["project"]),
        ("config",    roots["config"]),
        ("state",     roots["state"]),
        ("cache",     roots["cache"]),
        ("workspace", roots["workspace"]),
    ))
    yield roots


# ── Read-path routing ───────────────────────────────────────────────────────

def test_read_workspace_is_default(tmp_roots):
    assert mcp._resolve_read_path("foo.txt") == tmp_roots["workspace"] / "foo.txt"


def test_read_workspace_explicit_prefix(tmp_roots):
    assert mcp._resolve_read_path("workspace/foo.txt") == tmp_roots["workspace"] / "foo.txt"
    assert mcp._resolve_read_path("/workspace/foo.txt") == tmp_roots["workspace"] / "foo.txt"


def test_read_config_prefix(tmp_roots):
    assert mcp._resolve_read_path("config/identity/USER.md") == tmp_roots["config"] / "identity" / "USER.md"
    assert mcp._resolve_read_path("/config/skills/foo/SKILL.md") == tmp_roots["config"] / "skills" / "foo" / "SKILL.md"


def test_read_state_prefix(tmp_roots):
    assert mcp._resolve_read_path("state/soul/SOUL.md") == tmp_roots["state"] / "soul" / "SOUL.md"
    assert mcp._resolve_read_path("/state/memory/2026-04-20.md") == tmp_roots["state"] / "memory" / "2026-04-20.md"


def test_read_cache_prefix(tmp_roots):
    assert mcp._resolve_read_path("cache/prompts/abc.md") == tmp_roots["cache"] / "prompts" / "abc.md"


def test_read_project_prefix(tmp_roots):
    assert mcp._resolve_read_path("project/app/main.py") == tmp_roots["project"] / "app" / "main.py"
    assert mcp._resolve_read_path("/project/README.md") == tmp_roots["project"] / "README.md"


def test_read_project_blocks_dotenv(tmp_roots):
    with pytest.raises(HTTPException) as exc:
        mcp._resolve_read_path("project/.env")
    assert exc.value.status_code == 403


def test_read_traversal_rejected(tmp_roots):
    with pytest.raises(HTTPException) as exc:
        mcp._resolve_read_path("workspace/../etc/passwd")
    assert exc.value.status_code == 400


# ── Write-path routing ──────────────────────────────────────────────────────

def test_write_workspace_is_default(tmp_roots):
    assert mcp._resolve_write_path("notes.md") == tmp_roots["workspace"] / "notes.md"


def test_write_all_writable_roots(tmp_roots):
    assert mcp._resolve_write_path("config/skills/new/SKILL.md") == tmp_roots["config"] / "skills" / "new" / "SKILL.md"
    assert mcp._resolve_write_path("state/soul/SOUL.md") == tmp_roots["state"] / "soul" / "SOUL.md"
    assert mcp._resolve_write_path("cache/prompts/foo.md") == tmp_roots["cache"] / "prompts" / "foo.md"
    assert mcp._resolve_write_path("workspace/scratch.txt") == tmp_roots["workspace"] / "scratch.txt"


def test_write_project_rejected(tmp_roots):
    with pytest.raises(HTTPException) as exc:
        mcp._resolve_write_path("project/app/main.py")
    assert exc.value.status_code == 400
    assert "read-only" in exc.value.detail.lower()


def test_write_system_prefix_rejected(tmp_roots):
    """The legacy `system/` prefix is deprecated; writes must pick the new root explicitly."""
    for candidate in ("system/SOUL.md", "/system/SOUL.md", "system/skills/foo/SKILL.md"):
        with pytest.raises(HTTPException) as exc:
            mcp._resolve_write_path(candidate)
        assert exc.value.status_code == 400
        assert "deprecated" in exc.value.detail.lower()


def test_write_traversal_rejected(tmp_roots):
    with pytest.raises(HTTPException) as exc:
        mcp._resolve_write_path("config/../etc/passwd")
    assert exc.value.status_code == 400
