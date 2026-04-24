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
    # The skill's name + description both surface. The block's preamble
    # tells the agent *how* to load the file (generic `<name>` template);
    # per-row path duplication is intentionally dropped per the Agent Skills
    # spec (progressive disclosure).
    assert "- **log-triage**:" in block
    assert "Analyze log lines" in block
    assert "config/skills/<name>/SKILL.md" in block


def test_skills_block_without_skills_still_renders(tmp_workspace):
    """With no skills and no spawnable agents, the block must be safe (empty or
    a no-op heading) — never raise."""
    block = prompt_generator._build_skills_block([], {})
    assert isinstance(block, str)


def test_parse_frontmatter_multiline_trigger():
    """Regression: the old hand-rolled parser silently dropped `key: |` blocks,
    causing `when-to-trigger` to vanish from every authored skill. yaml.safe_load
    must preserve the multiline body."""
    text = textwrap.dedent("""\
        ---
        name: log-triage
        description: Analyze log lines
        when-to-trigger: |
          TRIGGER when the user pastes a stack trace.
          TRIGGER when logs show repeated errors.
        when-not-to-trigger: |
          DO NOT TRIGGER on compile errors.
        allowed-tools: [file_read, shell_exec]
        ---

        body
        """)
    meta = prompt_generator._parse_frontmatter(text)
    assert meta["name"] == "log-triage"
    assert "stack trace" in meta["when-to-trigger"]
    assert "repeated errors" in meta["when-to-trigger"]
    assert meta["allowed-tools"] == ["file_read", "shell_exec"]


def test_parse_frontmatter_malformed_returns_empty():
    assert prompt_generator._parse_frontmatter("no frontmatter at all\nhello") == {}
    assert prompt_generator._parse_frontmatter("---\nkey: [unterminated\n") == {}
    # non-mapping (scalar or list at top level) → empty
    assert prompt_generator._parse_frontmatter("---\n- item1\n- item2\n---\n") == {}


def test_discover_skills_captures_multiline_when(tmp_workspace):
    """End-to-end: a skill with a multiline `when-to-trigger: |` block should
    preserve those triggers as `when_to_trigger` in the discovery entry,
    alongside the spec-canonical `description`."""
    _write_skill(
        tmp_workspace,
        "log-triage",
        description="fallback description",
        when="TRIGGER when errors appear in the log stream.",
    )
    entries = prompt_generator._discover_skills()
    assert len(entries) == 1
    entry = entries[0]
    # Spec-canonical field round-trips verbatim.
    assert entry["description"] == "fallback description"
    # Phoebe extension survives the multiline block → single-line normalisation.
    when = entry["when_to_trigger"]
    assert "errors appear in the log stream" in when
    assert "\n" not in when


def test_discover_skills_list_when_is_joined(tmp_workspace):
    """If an author writes `when-to-trigger` as a YAML list, it should be
    flattened to a single `; `-joined string on the entry."""
    skill_dir = tmp_workspace / "skills" / "multi-trigger"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(textwrap.dedent("""\
        ---
        name: multi-trigger
        when-to-trigger:
          - TRIGGER when A.
          - TRIGGER when B.
        ---
        body
        """))
    entries = prompt_generator._discover_skills()
    assert len(entries) == 1
    when = entries[0]["when_to_trigger"]
    assert "TRIGGER when A." in when
    assert "TRIGGER when B." in when
    assert ";" in when


def test_discover_skills_keeps_description_for_minimal_spec_skills(tmp_workspace):
    """A minimal SKILL.md (spec-compliant, no Phoebe extensions) must still
    produce a usable entry — `description` populated, triggers None."""
    skill_dir = tmp_workspace / "skills" / "pdf"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(textwrap.dedent("""\
        ---
        name: pdf
        description: Extract text and forms from PDF files. Use when the user mentions PDFs.
        ---
        body
        """))
    entries = prompt_generator._discover_skills()
    assert len(entries) == 1
    e = entries[0]
    assert e["name"] == "pdf"
    assert "Extract text and forms" in e["description"]
    assert e["when_to_trigger"] is None
    assert e["when_not_to_trigger"] is None


def test_format_skill_line_concise_spec_style(tmp_workspace):
    """Rendered metadata line should be a single bullet with name + description,
    no table scaffolding — matches the Agent Skills progressive-disclosure
    convention."""
    line = prompt_generator._format_skill_line({
        "name": "pdf",
        "description": "Extract text and forms from PDF files. Use when the user mentions PDFs.",
        "when_to_trigger": None,
        "when_not_to_trigger": None,
        "path": "config/skills/pdf/SKILL.md",
    })
    assert line.startswith("- **pdf**:")
    assert "Extract text and forms" in line
    # Per spec, path is deterministic from name — don't spend tokens on it.
    assert "config/skills" not in line


def test_format_skill_line_appends_phoebe_extensions():
    line = prompt_generator._format_skill_line({
        "name": "log-triage",
        "description": "Triage error logs.",
        "when_to_trigger": "logs spike with errors",
        "when_not_to_trigger": "compile errors",
        "path": "config/skills/log-triage/SKILL.md",
    })
    assert "Triage error logs" in line
    assert "Use when logs spike with errors" in line
    assert "Skip when compile errors" in line


def test_format_skill_line_truncates_runaway_descriptions():
    huge = "x" * 2000
    line = prompt_generator._format_skill_line({
        "name": "noisy",
        "description": huge,
        "when_to_trigger": None,
        "when_not_to_trigger": None,
        "path": "config/skills/noisy/SKILL.md",
    })
    assert line.endswith("…")
    assert len(line) < 500  # well under the 1024-char spec ceiling
