"""Phase 1 — the worker prompt's Subagents matrix must list every role the
worker is permitted to spawn. If `agents.yaml` adds or removes a spawnable
agent and the matrix doesn't keep up, the prompt becomes a lie. This is the
guard against silent drift.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml


def _read(name: str) -> str:
    for base in (Path("/app/config"), Path(__file__).resolve().parent.parent / "config"):
        p = base / name
        if p.exists():
            return p.read_text()
    pytest.skip(f"{name} not found")


def test_matrix_lists_all_worker_spawnable():
    worker_md = _read("prompts/worker_concise.md")
    agents = yaml.safe_load(_read("agents.yaml"))
    spawnable = agents["worker"]["spawnable_agents"]
    assert spawnable, "worker has no spawnable_agents — matrix would be empty"
    for role in spawnable:
        assert role in worker_md, (
            f"worker_concise.md is missing matrix row for spawnable role '{role}'"
        )


def test_matrix_does_not_advertise_unspawnable():
    """Don't tell the worker it can spawn roles it's not permitted to spawn."""
    worker_md = _read("prompts/worker_concise.md")
    agents = yaml.safe_load(_read("agents.yaml"))
    spawnable = set(agents["worker"]["spawnable_agents"])
    # Heuristic: any `*_agent` or `*_builder` token immediately under "## Subagents"
    # block must be in the spawnable set.
    if "# Subagents" not in worker_md:
        pytest.fail("Subagents section missing from worker_concise.md")
    section = worker_md.split("# Subagents", 1)[1].split("---", 1)[0]
    for role in ("coding_agent", "research_agent", "tool_builder",
                 "skill_builder", "webfetch_summarizer", "improvement_agent",
                 "discord_moderator"):
        if role in section and role not in spawnable:
            pytest.fail(f"matrix advertises '{role}' but worker cannot spawn it")
