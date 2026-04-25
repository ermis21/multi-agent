"""Tests for the per-model prompt overlay system.

Overlays live at `config/prompts/overlays/<name>.md` and get injected into
the system prompt at the `{{PROMPT_OVERLAY}}` marker. Activated by setting
`prompt_overlay: <name>` on a model entry in `cfg.models.<name>` (per-role)
or `cfg.llm` (default).
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from app import prompt_generator


@pytest.fixture
def overlay_workspace(tmp_path, monkeypatch):
    """Redirect OVERLAYS_DIR to a tmp dir; reset cache so each test sees fresh world."""
    overlays_dir = tmp_path / "overlays"
    overlays_dir.mkdir()
    monkeypatch.setattr(prompt_generator, "OVERLAYS_DIR", overlays_dir)
    prompt_generator._OVERLAYS_CACHE["mtime"] = 0.0
    prompt_generator._OVERLAYS_CACHE["docs"] = {}
    yield overlays_dir


def _write_overlay(root: Path, name: str, body: str) -> Path:
    p = root / f"{name}.md"
    p.write_text(body, encoding="utf-8")
    return p


def test_no_overlay_dir(tmp_path, monkeypatch):
    """Missing overlays dir → empty docs, never raises."""
    monkeypatch.setattr(prompt_generator, "OVERLAYS_DIR", tmp_path / "does_not_exist")
    prompt_generator._OVERLAYS_CACHE["mtime"] = 0.0
    prompt_generator._OVERLAYS_CACHE["docs"] = {}
    assert prompt_generator._load_overlays() == {}


def test_loads_single_overlay(overlay_workspace):
    _write_overlay(overlay_workspace, "gemma_grammar", "Use <|tool_call|>...")
    docs = prompt_generator._load_overlays()
    assert "gemma_grammar" in docs
    assert "Use <|tool_call|>" in docs["gemma_grammar"]


def test_resolve_overlay_from_model_override(overlay_workspace):
    """role_cfg.model → cfg.models[name].prompt_overlay wins over cfg.llm."""
    cfg = {
        "llm": {"prompt_overlay": "default_overlay"},
        "models": {"local_llama": {"prompt_overlay": "gemma_grammar"}},
    }
    role_cfg = {"model": "local_llama"}
    assert prompt_generator._resolve_overlay_name(cfg, role_cfg) == "gemma_grammar"


def test_resolve_overlay_falls_back_to_llm_default(overlay_workspace):
    """No override on the named model → fall back to cfg.llm.prompt_overlay."""
    cfg = {
        "llm": {"prompt_overlay": "default_overlay"},
        "models": {"local_llama": {}},
    }
    role_cfg = {"model": "local_llama"}
    assert prompt_generator._resolve_overlay_name(cfg, role_cfg) == "default_overlay"


def test_resolve_overlay_none_when_unset(overlay_workspace):
    cfg = {"llm": {}, "models": {}}
    role_cfg = {}
    assert prompt_generator._resolve_overlay_name(cfg, role_cfg) is None


def test_resolve_overlay_no_role_model(overlay_workspace):
    """Role with no model override → falls back to cfg.llm."""
    cfg = {"llm": {"prompt_overlay": "global"}, "models": {}}
    role_cfg = {}
    assert prompt_generator._resolve_overlay_name(cfg, role_cfg) == "global"


def test_build_overlay_block_returns_content(overlay_workspace):
    _write_overlay(overlay_workspace, "gemma_grammar", "OVERLAY BODY")
    cfg = {"llm": {}, "models": {"m": {"prompt_overlay": "gemma_grammar"}}}
    role_cfg = {"model": "m"}
    assert prompt_generator._build_overlay_block(cfg, role_cfg) == "OVERLAY BODY"


def test_build_overlay_block_returns_empty_when_unset(overlay_workspace):
    cfg = {"llm": {}, "models": {}}
    role_cfg = {}
    assert prompt_generator._build_overlay_block(cfg, role_cfg) == ""


def test_build_overlay_block_returns_empty_when_file_missing(overlay_workspace):
    """Configured overlay name but file doesn't exist → empty (graceful)."""
    cfg = {"llm": {"prompt_overlay": "ghost"}, "models": {}}
    role_cfg = {}
    assert prompt_generator._build_overlay_block(cfg, role_cfg) == ""


def test_overlay_mtime_cache_invalidates_on_change(overlay_workspace):
    import os
    _write_overlay(overlay_workspace, "alpha", "v1")
    assert prompt_generator._load_overlays()["alpha"] == "v1"
    _write_overlay(overlay_workspace, "alpha", "v2")
    # Force mtime bump for filesystems with 1s granularity
    now = os.path.getmtime(overlay_workspace) + 5
    os.utime(overlay_workspace, (now, now))
    assert prompt_generator._load_overlays()["alpha"] == "v2"
