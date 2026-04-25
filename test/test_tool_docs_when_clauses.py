"""Phase 1 — assert the curated set of tool docs that benefit from explicit
selection guidance carry `Use when:` / `Not when:` lines. We don't blanket
every tool: only ones with real selection ambiguity (file_read vs file_search,
web_fetch vs web_search, memory ops, shell vs file_edit, handle recall).
"""

from __future__ import annotations

from pathlib import Path

import pytest

CURATED_TOOLS = [
    "file_read",
    "file_search",
    "file_list",
    "directory_tree",
    "web_fetch",
    "web_search",
    "memory_search",
    "memory_add",
    "tool_result_recall",
    "shell_exec",
]


def _tools_dir() -> Path:
    for p in (Path("/app/config/prompts/tools"),
              Path(__file__).resolve().parent.parent / "config/prompts/tools"):
        if p.exists():
            return p
    pytest.skip("tools dir not found")


@pytest.mark.parametrize("tool", CURATED_TOOLS)
def test_curated_tool_has_when_clauses(tool):
    body = (_tools_dir() / f"{tool}.md").read_text()
    assert "Use when:" in body, f"{tool}.md missing 'Use when:' line"
    assert "Not when:" in body, f"{tool}.md missing 'Not when:' line"


def test_when_clauses_above_tool_call_example():
    """Selection guidance must come BEFORE the example payloads, otherwise the
    model sees the example first and fixates on it."""
    body = (_tools_dir() / "file_read.md").read_text()
    when_idx = body.index("Use when:")
    examples_idx = body.index("Examples:")
    assert when_idx < examples_idx, "guidance must precede the example"
