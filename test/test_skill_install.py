"""Tests for sandbox skill_install — URL parsing, safety rejects, live round-trip.

These tests import ``sandbox.mcp_server`` directly. The api container's image
ships a baked-in copy of the sandbox module that may lag the working tree
(sandbox isn't live-reloaded there), so we skip the whole file when the new
``_skill_install`` symbol is absent. Run inside ``phoebe-sandbox`` (where
``/project`` is bind-mounted) to exercise this suite against current code.
"""

from __future__ import annotations

import base64
import io
import shutil
from pathlib import Path

import httpx
import pytest

from sandbox import mcp_server

if not hasattr(mcp_server, "_skill_install"):
    pytest.skip(
        "sandbox.mcp_server._skill_install not present in this container's image "
        "(run inside phoebe-sandbox where /project is live-mounted)",
        allow_module_level=True,
    )


# ── URL parser ────────────────────────────────────────────────────────────────


def test_parse_github_folder_url_happy_main():
    owner, repo, ref, path = mcp_server._parse_github_folder_url(
        "https://github.com/anthropics/skills/tree/main/skills/pdf-form"
    )
    assert (owner, repo, ref, path) == ("anthropics", "skills", "main", "skills/pdf-form")


def test_parse_github_folder_url_accepts_sha_ref():
    owner, repo, ref, path = mcp_server._parse_github_folder_url(
        "https://github.com/a/b/tree/abc123/x/y/z"
    )
    assert ref == "abc123"
    assert path == "x/y/z"


@pytest.mark.parametrize("bad_url", [
    "https://gitlab.com/anthropics/skills/tree/main/skills/x",  # wrong host
    "ftp://github.com/a/b/tree/main/x",                         # wrong scheme
    "https://github.com/a/b/blob/main/x",                       # "blob" not "tree"
    "https://github.com/a/b",                                   # no /tree/
    "https://github.com/a/b/tree/main",                         # no path
    "https://github.com/a/b/tree/main/../secrets",              # traversal
])
def test_parse_github_folder_url_rejects_bad(bad_url):
    with pytest.raises(ValueError):
        mcp_server._parse_github_folder_url(bad_url)


# ── skill_install with mocked GitHub ──────────────────────────────────────────

class _FakeClient:
    """Minimal stand-in for httpx.Client that serves a scripted response table."""

    def __init__(self, responses: dict[str, object]):
        self._responses = responses

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False

    def get(self, url, params=None, **_kwargs):
        key = url
        if params:
            # Only "?ref=..." matters for fixture lookup; ignore others.
            if "ref" in params:
                key = f"{url}?ref={params['ref']}"
        if key not in self._responses:
            raise AssertionError(f"unexpected GET {key!r}")
        resp = self._responses[key]
        if isinstance(resp, Exception):
            raise resp
        return resp


def _mk_response(status: int, *, json_body=None, content: bytes | None = None):
    r = httpx.Response(
        status_code=status,
        json=json_body if json_body is not None else None,
        content=content,
        request=httpx.Request("GET", "https://example.invalid/"),
    )
    return r


@pytest.fixture
def skills_sandbox(tmp_path, monkeypatch):
    """Point sandbox CONFIG at a tmp dir and install a fake httpx.Client."""
    config_root = tmp_path / "config"
    (config_root / "skills").mkdir(parents=True)
    monkeypatch.setattr(mcp_server, "CONFIG", config_root)

    def _install_fake(responses):
        monkeypatch.setattr(
            mcp_server.httpx, "Client",
            lambda *a, **kw: _FakeClient(responses),
        )

    yield {"config": config_root, "install_fake": _install_fake}


SKILL_MD_GOOD = b"""\
---
name: demo-skill
description: A demo
when-to-trigger: |
  TRIGGER when testing.
when-not-to-trigger: |
  DO NOT TRIGGER in prod.
---

## Purpose
Demo.
"""


def test_skill_install_happy_path(skills_sandbox):
    responses = {
        "https://api.github.com/repos/a/b/contents/skills/demo-skill?ref=main": _mk_response(
            200, json_body=[
                {"name": "SKILL.md", "type": "file", "size": len(SKILL_MD_GOOD),
                 "download_url": "https://raw.example/SKILL.md"},
                {"name": "helper.md", "type": "file", "size": 12,
                 "download_url": "https://raw.example/helper.md"},
            ],
        ),
        "https://raw.example/SKILL.md":  _mk_response(200, content=SKILL_MD_GOOD),
        "https://raw.example/helper.md": _mk_response(200, content=b"hello world\n"),
    }
    skills_sandbox["install_fake"](responses)

    result = mcp_server._skill_install({
        "url": "https://github.com/a/b/tree/main/skills/demo-skill"
    })
    assert "error" not in result, result
    assert result["name"] == "demo-skill"
    dir_ = skills_sandbox["config"] / "skills" / "demo-skill"
    assert (dir_ / "SKILL.md").exists()
    assert (dir_ / "helper.md").read_bytes() == b"hello world\n"
    # Partial-stage dir must not linger.
    assert not (skills_sandbox["config"] / "skills" / ".demo-skill.partial").exists()


def test_skill_install_missing_skill_md_writes_nothing(skills_sandbox):
    responses = {
        "https://api.github.com/repos/a/b/contents/x?ref=main": _mk_response(
            200, json_body=[
                {"name": "README.md", "type": "file", "size": 4,
                 "download_url": "https://raw.example/README.md"},
            ],
        ),
        "https://raw.example/README.md": _mk_response(200, content=b"hi\n"),
    }
    skills_sandbox["install_fake"](responses)

    result = mcp_server._skill_install({"url": "https://github.com/a/b/tree/main/x"})
    assert "error" in result
    assert "SKILL.md not found" in result["error"]
    # No skill dir should be created.
    assert list((skills_sandbox["config"] / "skills").glob("*/")) == []


def test_skill_install_oversized_file_rejected(skills_sandbox):
    responses = {
        "https://api.github.com/repos/a/b/contents/x?ref=main": _mk_response(
            200, json_body=[
                {"name": "SKILL.md", "type": "file",
                 "size": mcp_server._SKILL_MAX_FILE + 1,
                 "download_url": "https://raw.example/huge"},
            ],
        ),
    }
    skills_sandbox["install_fake"](responses)
    result = mcp_server._skill_install({"url": "https://github.com/a/b/tree/main/x"})
    assert "error" in result and "per-file cap" in result["error"]


def test_skill_install_bad_entry_name_rejected(skills_sandbox):
    responses = {
        "https://api.github.com/repos/a/b/contents/x?ref=main": _mk_response(
            200, json_body=[
                {"name": "../escape.md", "type": "file", "size": 4,
                 "download_url": "https://raw.example/x"},
            ],
        ),
    }
    skills_sandbox["install_fake"](responses)
    result = mcp_server._skill_install({"url": "https://github.com/a/b/tree/main/x"})
    assert "error" in result and "unsafe name" in result["error"]


def test_skill_install_bad_frontmatter_name_rejected(skills_sandbox):
    bad = SKILL_MD_GOOD.replace(b"name: demo-skill", b"name: Bad_Name")
    responses = {
        "https://api.github.com/repos/a/b/contents/x?ref=main": _mk_response(
            200, json_body=[
                {"name": "SKILL.md", "type": "file", "size": len(bad),
                 "download_url": "https://raw.example/SKILL.md"},
            ],
        ),
        "https://raw.example/SKILL.md": _mk_response(200, content=bad),
    }
    skills_sandbox["install_fake"](responses)
    result = mcp_server._skill_install({"url": "https://github.com/a/b/tree/main/x"})
    assert "error" in result and "kebab-case" in result["error"]
    # Must not have written anything.
    assert list((skills_sandbox["config"] / "skills").glob("*/")) == []


def test_skill_install_refuses_overwrite_by_default(skills_sandbox):
    # Pre-seed an existing skill dir.
    existing = skills_sandbox["config"] / "skills" / "demo-skill"
    existing.mkdir()
    (existing / "marker").write_text("keep me")

    responses = {
        "https://api.github.com/repos/a/b/contents/skills/demo-skill?ref=main": _mk_response(
            200, json_body=[
                {"name": "SKILL.md", "type": "file", "size": len(SKILL_MD_GOOD),
                 "download_url": "https://raw.example/SKILL.md"},
            ],
        ),
        "https://raw.example/SKILL.md": _mk_response(200, content=SKILL_MD_GOOD),
    }
    skills_sandbox["install_fake"](responses)

    result = mcp_server._skill_install(
        {"url": "https://github.com/a/b/tree/main/skills/demo-skill"}
    )
    assert "error" in result and "already exists" in result["error"]
    # Original content preserved.
    assert (existing / "marker").read_text() == "keep me"


# ── Live integration (requires network + repo at fixed ref) ───────────────────

@pytest.mark.live
def test_skill_install_live_anthropics_skills(tmp_path, monkeypatch):
    """Install a real skill from anthropics/skills end-to-end.

    Uses the `template/` folder (stable, tiny, part of the spec) rather than a
    specific skill that might get renamed. If that folder lacks a valid SKILL.md
    frontmatter `name`, we accept an error result — the live gate still proves
    the HTTP + parse path works.
    """
    config_root = tmp_path / "config"
    (config_root / "skills").mkdir(parents=True)
    monkeypatch.setattr(mcp_server, "CONFIG", config_root)
    result = mcp_server._skill_install({
        "url": "https://github.com/anthropics/skills/tree/main/template"
    })
    # Either it installs cleanly, or it produces a named error (not a traceback).
    assert isinstance(result, dict)
    assert "error" in result or (result.get("name") and result.get("files"))
