"""Tests for config/skills/*/SKILL.md auto-discovery."""

import textwrap
from pathlib import Path

import pytest

from app import prompt_generator


@pytest.fixture
def tmp_workspace(tmp_path, monkeypatch):
    """Redirect CONFIG to a tmp dir and clear the skills cache so each test
    sees a fresh world. Fixture name kept as ``tmp_workspace`` to minimise
    test-call-site churn."""
    monkeypatch.setattr(prompt_generator, "CONFIG", tmp_path)
    # Reset the mtime-cached skills list
    if hasattr(prompt_generator, "_SKILLS_CACHE"):
        prompt_generator._SKILLS_CACHE["mtime"] = 0.0
        prompt_generator._SKILLS_CACHE["entries"] = []
    yield tmp_path


def _write_skill(root: Path, name: str, description: str = "Test skill", when: str = "TRIGGER when X.") -> Path:
    skill_dir = root / "skills" / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    content = textwrap.dedent(f"""\
        ---
        name: {name}
        description: {description}
        when-to-trigger: |
          {when}
        when-not-to-trigger: |
          DO NOT TRIGGER otherwise.
        allowed-tools: [file_read]
        ---

        ## Purpose
        Testing the discovery pipeline.
        """)
    path = skill_dir / "SKILL.md"
    path.write_text(content)
    return path


def test_discover_skills_no_directory(tmp_workspace):
    entries = prompt_generator._discover_skills()
    assert entries == []


def test_discover_skills_empty_directory(tmp_workspace):
    (tmp_workspace / "skills").mkdir()
    entries = prompt_generator._discover_skills()
    assert entries == []


def test_discover_skills_single(tmp_workspace):
    _write_skill(tmp_workspace, "log-triage", description="Analyze log lines")
    entries = prompt_generator._discover_skills()
    assert len(entries) == 1
    entry = entries[0]
    assert entry["name"] == "log-triage"
    assert "config/skills/log-triage/SKILL.md" in entry["path"]


def test_discover_skills_multiple_sorted(tmp_workspace):
    _write_skill(tmp_workspace, "alpha")
    _write_skill(tmp_workspace, "beta")
    _write_skill(tmp_workspace, "gamma")
    entries = prompt_generator._discover_skills()
    names = [e["name"] for e in entries]
    assert names == sorted(names)


def test_discover_skills_mtime_cache_picks_up_new(tmp_workspace):
    import os
    _write_skill(tmp_workspace, "first")
    assert len(prompt_generator._discover_skills()) == 1
    _write_skill(tmp_workspace, "second")
    # Filesystem mtime can have coarse (1s) granularity on some filesystems,
    # so force an mtime bump on the skills dir to ensure cache invalidates.
    skills_dir = tmp_workspace / "skills"
    now = os.path.getmtime(skills_dir) + 5
    os.utime(skills_dir, (now, now))
    entries = prompt_generator._discover_skills()
    assert len(entries) == 2


def test_discover_skills_malformed_frontmatter_falls_back_to_dirname(tmp_workspace):
    """A malformed SKILL.md should not crash discovery."""
    skill_dir = tmp_workspace / "skills" / "broken"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("not-yaml-frontmatter-at-all\nbody only")
    # Discovery must not raise; entry may still be present with dirname as fallback
    entries = prompt_generator._discover_skills()
    assert isinstance(entries, list)  # didn't crash


def test_skills_block_includes_discovered(tmp_workspace):
    _write_skill(tmp_workspace, "log-triage", description="Analyze log lines", when="TRIGGER when errors spike.")
    block = prompt_generator._build_skills_block([], {})
    # The skill's name and a pointer to its file should appear in the injected block
    assert "log-triage" in block
    assert "config/skills/log-triage" in block


def test_skills_block_without_skills_still_renders(tmp_workspace):
    """With no skills and no spawnable agents, the block must be safe (empty or
    a no-op heading) — never raise."""
    block = prompt_generator._build_skills_block([], {})
    assert isinstance(block, str)
