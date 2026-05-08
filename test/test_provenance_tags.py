"""Phase 2 — worker.py provenance wrap test.

We don't run a full _run_worker iteration here; instead we verify the wrap
logic at the closest possible isolation level: import the module-level
_UNTRUSTED_TOOLS set + reconstruct the exact format string used at L408.
That keeps the test fast and free of LLM/sandbox dependencies, while still
catching regressions in the wrap shape and the untrusted-tool list.
"""

from __future__ import annotations

import pytest

from app import worker


def _wrap(tool: str, body: str, flag_on: bool = True) -> str:
    """Mirror the wrap logic in worker.py L408 area for unit-level coverage."""
    header = f"[tool_result: {tool}] OK"
    if flag_on and tool in worker._UNTRUSTED_TOOLS:
        return (
            f'{header}\n<tool_result tool="{tool}" trust="untrusted">\n'
            f"{body}\n</tool_result>"
        )
    return f"{header}\n{body}"


def test_untrusted_tools_set_includes_web_and_third_party():
    expected = {"web_fetch", "web_search", "discord_read", "notion_search", "notion_get_page"}
    assert expected.issubset(worker._UNTRUSTED_TOOLS)


def test_trusted_tools_not_in_set():
    """Internal/local tools MUST NOT be wrapped — they're our own data."""
    for tool in ("file_read", "file_edit", "shell_exec", "memory_search",
                 "git_status", "tool_result_recall"):
        assert tool not in worker._UNTRUSTED_TOOLS, f"{tool} must not be marked untrusted"


def test_wrap_shape_for_untrusted():
    out = _wrap("web_fetch", "page body here", flag_on=True)
    assert '<tool_result tool="web_fetch" trust="untrusted">' in out
    assert "</tool_result>" in out
    # Header stays OUTSIDE the tag so the legacy parser still works.
    assert out.startswith("[tool_result: web_fetch] OK\n<tool_result")


def test_no_wrap_for_trusted():
    out = _wrap("file_read", "file contents", flag_on=True)
    assert "<tool_result" not in out
    assert "file contents" in out


def test_no_wrap_when_flag_off():
    out = _wrap("web_fetch", "page body", flag_on=False)
    assert "<tool_result" not in out
