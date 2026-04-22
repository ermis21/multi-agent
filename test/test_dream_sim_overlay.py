"""Unit tests for sandbox.sim_overlay helpers + a few sandbox-handler integration
scenarios that exercise the `_simulate` branches in mcp_server.

Pure-unit only — no chromadb network, no LLM. The tests monkeypatch the
handler's writable-root constants to point into ``tmp_path``, simulating the
sandbox's /workspace + /config + /state + /cache mounts."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from sandbox import sim_overlay


# ── sim_overlay pure-helpers ────────────────────────────────────────────────

@pytest.fixture
def fake_roots(tmp_path, monkeypatch):
    """Point WORKSPACE/CONFIG/STATE/CACHE env vars at tmp_path subtrees so
    `classify_root` / `overlay_path_for` resolve deterministically."""
    ws = tmp_path / "ws"; ws.mkdir()
    cf = tmp_path / "cf"; cf.mkdir()
    st = tmp_path / "st"; st.mkdir()
    ca = tmp_path / "ca"; ca.mkdir()
    monkeypatch.setenv("WORKSPACE_DIR", str(ws))
    monkeypatch.setenv("CONFIG_DIR",    str(cf))
    monkeypatch.setenv("STATE_DIR",     str(st))
    monkeypatch.setenv("CACHE_DIR",     str(ca))
    overlay = tmp_path / "overlay"
    overlay.mkdir()
    return {
        "workspace": ws,
        "config":    cf,
        "state":     st,
        "cache":     ca,
        "overlay":   overlay,
    }


def test_is_sim_accepts_well_formed_marker():
    params = {
        "path": "workspace/foo.md",
        "_simulate": {
            "sim_sid": "abc",
            "overlay_root": "/tmp/x",
            "memory_collection": "sim_abc_memories",
        },
    }
    assert sim_overlay.is_sim(params) is not None


def test_is_sim_rejects_missing_overlay_root():
    assert sim_overlay.is_sim({"_simulate": {}}) is None
    assert sim_overlay.is_sim({"_simulate": None}) is None
    assert sim_overlay.is_sim({}) is None


def test_classify_root_for_each_writable(fake_roots):
    for name in ("workspace", "config", "state", "cache"):
        root = fake_roots[name]
        name_out, rel = sim_overlay.classify_root(root / "sub" / "x.md")
        assert name_out == name
        assert rel == Path("sub") / "x.md"


def test_classify_root_outside_returns_none(fake_roots, tmp_path):
    outside = tmp_path / "elsewhere" / "x.md"
    assert sim_overlay.classify_root(outside) == (None, None)


def test_overlay_path_mirrors_writable_root(fake_roots):
    real = fake_roots["workspace"] / "a" / "b.md"
    twin = sim_overlay.overlay_path_for(real, fake_roots["overlay"])
    assert twin == fake_roots["overlay"] / "workspace" / "a" / "b.md"


def test_overlay_path_for_project_returns_none(fake_roots):
    # /project is not writable; no overlay twin.
    outside = Path("/project") / "app" / "x.py"
    assert sim_overlay.overlay_path_for(outside, fake_roots["overlay"]) is None


def test_prepare_write_mkdirs_parent(fake_roots):
    real = fake_roots["workspace"] / "deeply" / "nested" / "f.md"
    twin = sim_overlay.prepare_write(real, fake_roots["overlay"])
    assert twin is not None
    assert twin.parent.exists()


def test_resolve_read_falls_through_when_no_twin(fake_roots):
    real = fake_roots["workspace"] / "absent.md"
    resolved = sim_overlay.resolve_read_with_overlay(real, fake_roots["overlay"])
    assert resolved == real


def test_resolve_read_uses_twin_when_exists(fake_roots):
    real = fake_roots["workspace"] / "f.md"
    real.write_text("original", encoding="utf-8")
    twin = sim_overlay.overlay_path_for(real, fake_roots["overlay"])
    twin.parent.mkdir(parents=True, exist_ok=True)
    twin.write_text("overlaid", encoding="utf-8")
    got = sim_overlay.resolve_read_with_overlay(real, fake_roots["overlay"])
    assert got == twin
    assert got.read_text(encoding="utf-8") == "overlaid"


def test_tombstone_marks_and_hides(fake_roots):
    real = fake_roots["workspace"] / "gone.md"
    real.write_text("still here", encoding="utf-8")
    assert not sim_overlay.is_tombstoned(fake_roots["overlay"], real)
    sim_overlay.mark_deleted(fake_roots["overlay"], real)
    assert sim_overlay.is_tombstoned(fake_roots["overlay"], real)
    with pytest.raises(FileNotFoundError):
        sim_overlay.resolve_read_with_overlay(real, fake_roots["overlay"])


def test_list_merged_combines_real_and_overlay(fake_roots):
    # Real content
    (fake_roots["workspace"] / "a.md").write_text("A", encoding="utf-8")
    (fake_roots["workspace"] / "b.md").write_text("B", encoding="utf-8")
    # Overlay adds c and overrides a
    twin_dir = fake_roots["overlay"] / "workspace"
    twin_dir.mkdir(parents=True)
    (twin_dir / "a.md").write_text("A-overlay", encoding="utf-8")
    (twin_dir / "c.md").write_text("C", encoding="utf-8")
    names = sim_overlay.list_merged(fake_roots["workspace"], fake_roots["overlay"])
    assert names == ["a.md", "b.md", "c.md"]


def test_list_merged_honors_tombstones(fake_roots):
    (fake_roots["workspace"] / "keep.md").write_text("K", encoding="utf-8")
    (fake_roots["workspace"] / "gone.md").write_text("G", encoding="utf-8")
    sim_overlay.mark_deleted(fake_roots["overlay"], fake_roots["workspace"] / "gone.md")
    names = sim_overlay.list_merged(fake_roots["workspace"], fake_roots["overlay"])
    assert "keep.md" in names
    assert "gone.md" not in names


# ── mcp_server handler integration ──────────────────────────────────────────

@pytest.fixture
def sandbox_env(tmp_path, monkeypatch):
    """Point the sandbox handler module at tmp_path roots.

    Requires re-importing mcp_server after env vars are set because the
    module-level path constants resolve at import time.
    """
    ws = tmp_path / "workspace"; ws.mkdir()
    cf = tmp_path / "config"; cf.mkdir()
    st = tmp_path / "state"; st.mkdir()
    ca = tmp_path / "cache"; ca.mkdir()
    pj = tmp_path / "project"; pj.mkdir()
    monkeypatch.setenv("WORKSPACE_DIR", str(ws))
    monkeypatch.setenv("CONFIG_DIR",    str(cf))
    monkeypatch.setenv("STATE_DIR",     str(st))
    monkeypatch.setenv("CACHE_DIR",     str(ca))
    monkeypatch.setenv("PROJECT_DIR",   str(pj))
    monkeypatch.setenv("MEMPALACE_HOME", str(st / "chroma"))

    # Re-import to pick up env vars for the module constants.
    import importlib
    if "sandbox.mcp_server" in __import__("sys").modules:
        del __import__("sys").modules["sandbox.mcp_server"]
    mcp = importlib.import_module("sandbox.mcp_server")

    overlay = tmp_path / "overlay"
    overlay.mkdir()

    return {
        "workspace": ws, "config": cf, "state": st, "cache": ca, "project": pj,
        "overlay": overlay,
        "mcp": mcp,
    }


def _sim(marker_root: Path) -> dict:
    return {
        "sim_sid": "abc",
        "overlay_root": str(marker_root),
        "memory_collection": "sim_abc_memories",
    }


def test_file_write_lands_in_overlay_real_path_untouched(sandbox_env):
    mcp = sandbox_env["mcp"]
    ws = sandbox_env["workspace"]
    overlay = sandbox_env["overlay"]
    params = {"path": "workspace/foo.md", "content": "sim-only",
              "_simulate": _sim(overlay)}
    result = mcp._file_write(params)
    assert "error" not in result
    # Real workspace must still be empty
    assert not (ws / "foo.md").exists()
    # Overlay should have the content
    assert (overlay / "workspace" / "foo.md").read_text(encoding="utf-8") == "sim-only"


def test_file_read_after_sim_write_sees_overlay_content(sandbox_env):
    mcp = sandbox_env["mcp"]
    overlay = sandbox_env["overlay"]
    # Pre-populate real workspace
    (sandbox_env["workspace"] / "foo.md").write_text("old real", encoding="utf-8")
    # Sim writes an overlay version
    mcp._file_write({"path": "workspace/foo.md", "content": "new sim",
                     "_simulate": _sim(overlay)})
    # Sim read sees overlay
    r = mcp._file_read({"path": "workspace/foo.md", "_simulate": _sim(overlay)})
    assert r["content"] == "new sim"
    # Non-sim read sees real
    r_real = mcp._file_read({"path": "workspace/foo.md"})
    assert r_real["content"] == "old real"


def test_file_list_merges_real_and_overlay(sandbox_env):
    mcp = sandbox_env["mcp"]
    overlay = sandbox_env["overlay"]
    (sandbox_env["workspace"] / "a.md").write_text("A", encoding="utf-8")
    mcp._file_write({"path": "workspace/b.md", "content": "B",
                     "_simulate": _sim(overlay)})
    result = mcp._file_list({"path": "workspace", "_simulate": _sim(overlay)})
    names = [e["name"] for e in result["entries"]]
    assert "a.md" in names and "b.md" in names


def test_file_move_tombstones_source(sandbox_env):
    mcp = sandbox_env["mcp"]
    overlay = sandbox_env["overlay"]
    (sandbox_env["workspace"] / "src.md").write_text("X", encoding="utf-8")
    r = mcp._file_move({
        "source": "workspace/src.md",
        "destination": "workspace/dst.md",
        "_simulate": _sim(overlay),
    })
    assert r.get("ok") is True
    # Real source still exists (we didn't actually move it)
    assert (sandbox_env["workspace"] / "src.md").exists()
    # Overlay has destination
    assert (overlay / "workspace" / "dst.md").exists()
    # List should hide src.md and show dst.md
    result = mcp._file_list({"path": "workspace", "_simulate": _sim(overlay)})
    names = [e["name"] for e in result["entries"]]
    assert "src.md" not in names
    assert "dst.md" in names


def test_shell_exec_under_sim_returns_fake_success(sandbox_env):
    mcp = sandbox_env["mcp"]
    overlay = sandbox_env["overlay"]
    r = mcp._shell_exec({
        "command": "echo please-do-not-run",
        "_simulate": _sim(overlay),
    })
    assert r["exit_code"] == 0
    assert r["stdout"] == ""
    assert r.get("_simulated") == "shell_exec_faked"


def test_git_commit_under_sim_fakes_hash(sandbox_env):
    mcp = sandbox_env["mcp"]
    overlay = sandbox_env["overlay"]
    r = mcp._git_commit({
        "message": "wip",
        "_simulate": _sim(overlay),
    })
    assert r["exit_code"] == 0
    assert r["stdout"].startswith("[simulated] sim-")
    assert r.get("_simulated") == "git_commit_faked"


def test_docker_test_health_under_sim_reports_healthy(sandbox_env):
    mcp = sandbox_env["mcp"]
    overlay = sandbox_env["overlay"]
    r = mcp._docker_test_health({"_simulate": _sim(overlay)})
    assert r["status_code"] == 200
    assert r["body"] == {"status": "healthy"}
    assert r.get("_simulated") == "docker_test_health_faked"


def test_create_dir_under_sim_writes_to_overlay(sandbox_env):
    mcp = sandbox_env["mcp"]
    overlay = sandbox_env["overlay"]
    r = mcp._create_dir({
        "path": "workspace/newdir/nested",
        "_simulate": _sim(overlay),
    })
    assert r.get("ok") is True
    assert not (sandbox_env["workspace"] / "newdir").exists()
    assert (overlay / "workspace" / "newdir" / "nested").exists()


def test_file_edit_under_sim_reads_overlay_first(sandbox_env):
    mcp = sandbox_env["mcp"]
    overlay = sandbox_env["overlay"]
    # Real has one version, overlay has another
    (sandbox_env["workspace"] / "f.md").write_text("real=OLD", encoding="utf-8")
    mcp._file_write({"path": "workspace/f.md", "content": "overlay=OLD",
                     "_simulate": _sim(overlay)})
    r = mcp._file_edit({
        "path": "workspace/f.md",
        "old_string": "overlay=OLD",
        "new_string": "overlay=NEW",
        "_simulate": _sim(overlay),
    })
    assert r.get("ok") is True
    # Overlay was updated, real untouched
    assert (overlay / "workspace" / "f.md").read_text(encoding="utf-8") == "overlay=NEW"
    assert (sandbox_env["workspace"] / "f.md").read_text(encoding="utf-8") == "real=OLD"


def test_file_write_real_path_when_sim_absent(sandbox_env):
    mcp = sandbox_env["mcp"]
    ws = sandbox_env["workspace"]
    # No _simulate marker → normal write path
    r = mcp._file_write({"path": "workspace/real.md", "content": "hi"})
    assert "error" not in r
    assert (ws / "real.md").read_text(encoding="utf-8") == "hi"
