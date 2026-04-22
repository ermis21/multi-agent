"""Regression tests for the `prompts_dir` override on prompt_generator.

The dream simulator replays a session under a proposed prompt by writing
the candidate text to a temporary directory and asking prompt_generator to
load templates from *there* instead of `/config/prompts/`. These tests pin
that override contract so future refactors don't silently dissolve it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app import prompt_generator


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_load_base_template_honors_override_for_dedicated_role(tmp_path):
    """Dedicated roles (`{role}.md`) must resolve inside the override dir."""
    _write(tmp_path / "dreamer.md", "# SHADOW DREAMER\n\nbody\n")
    got = prompt_generator._load_base_template(
        "dreamer", mode="full", prompts_dir=tmp_path,
    )
    assert "SHADOW DREAMER" in got


def test_load_base_template_honors_override_for_mode_variant(tmp_path):
    """Mode-variant fallback (`{role}_{mode}.md`) must also read from the override."""
    _write(tmp_path / "worker_full.md", "# SHADOW WORKER FULL\n")
    got = prompt_generator._load_base_template(
        "worker", mode="full", prompts_dir=tmp_path,
    )
    assert "SHADOW WORKER FULL" in got


def test_load_base_template_override_isolated_from_live_prompts(tmp_path):
    """Override must not fall back to the real /config/prompts tree.

    If the override is ignored the loader would find the live `worker_full.md`
    and the FileNotFoundError below would never fire.
    """
    # Empty override dir — no templates inside.
    with pytest.raises(FileNotFoundError):
        prompt_generator._load_base_template(
            "worker", mode="full", prompts_dir=tmp_path,
        )


def test_load_base_template_without_override_reads_default_dir(tmp_path, monkeypatch):
    """Passing prompts_dir=None keeps the original behavior."""
    fake = tmp_path / "fake_prompts"
    fake.mkdir()
    _write(fake / "dreamer.md", "# REAL DREAMER\n")
    monkeypatch.setattr(prompt_generator, "PROMPTS_DIR", fake)
    got = prompt_generator._load_base_template("dreamer", mode="full")
    assert "REAL DREAMER" in got


def test_generate_propagates_prompts_dir_to_loader(tmp_path, monkeypatch):
    """The top-level `generate()` entrypoint must forward the override."""
    # Seed a minimal shadow prompt tree with just the template we care about.
    _write(tmp_path / "worker_full.md",
           "# shadow\n\n{{ALLOWED_TOOLS}}\n\n<|end|>\n")
    # Provide curated files inside the live STATE/CONFIG roots (empty is fine).
    monkeypatch.setattr(prompt_generator, "GENERATED", tmp_path / "generated")

    prompt_text, agent_id = prompt_generator.generate(
        role="worker",
        allowed_tools=[],
        session_id="s_override",
        attempt=0,
        agent_mode="converse",
        prompts_dir=tmp_path,
    )
    # The shadow template's unique header survives substitution.
    assert "# shadow" in prompt_text
    assert agent_id.startswith("worker_s_overri_0_")
