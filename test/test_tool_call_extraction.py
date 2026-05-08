"""Unit tests for `_extract_tool_call` and its `json_repair` fallback (A1).

Covers all four parser branches (Gemma native, bare JSON, fenced, in-prose) on
both well-formed and malformed input. Malformed cases must be rescued by the
`_try_repair_json` helper instead of returning None.
"""
import json

from app.mcp_client import _extract_tool_call, _try_repair_json


# ── well-formed cases (regression) ──────────────────────────────────────────

def test_bare_json_clean():
    assert _extract_tool_call('{"tool": "git_status", "params": {}}') == {
        "tool": "git_status", "params": {},
    }


def test_gemma_native_clean():
    raw = '<|tool_call|>call: file_read, {"path": "foo"}<|tool_call|>'
    assert _extract_tool_call(raw) == {"tool": "file_read", "params": {"path": "foo"}}


def test_fenced_clean():
    raw = '```json\n{"tool": "shell_exec", "params": {"cmd": "ls"}}\n```'
    assert _extract_tool_call(raw) == {"tool": "shell_exec", "params": {"cmd": "ls"}}


def test_in_prose_clean():
    raw = 'Let me check:\n{"tool": "git_log", "params": {"n": 3}}\nDone.'
    assert _extract_tool_call(raw) == {"tool": "git_log", "params": {"n": 3}}


def test_prose_only_returns_none():
    assert _extract_tool_call("Just thinking about it.") is None


def test_gemma_nested_params_unwrap():
    raw = '<|tool_call|>call: file_read, {"params": {"path": "foo"}}<|tool_call|>'
    assert _extract_tool_call(raw) == {"tool": "file_read", "params": {"path": "foo"}}


# ── malformed cases (json_repair fallback) ──────────────────────────────────

def test_bare_json_trailing_comma_repaired():
    raw = '{"tool": "file_read", "params": {"path": "foo",}}'
    assert _extract_tool_call(raw) == {"tool": "file_read", "params": {"path": "foo"}}


def test_gemma_unquoted_keys_repaired():
    raw = '<|tool_call|>call: file_read, {path: "foo"}<|tool_call|>'
    assert _extract_tool_call(raw) == {"tool": "file_read", "params": {"path": "foo"}}


def test_fenced_unquoted_keys_repaired():
    raw = '```json\n{"tool": "shell_exec", "params": {cmd: "ls"}}\n```'
    assert _extract_tool_call(raw) == {"tool": "shell_exec", "params": {"cmd": "ls"}}


def test_in_prose_missing_quote_repaired():
    raw = 'Plan:\n{"tool": "git_log, "params": {"n": 3}}\nThat is the goal.'
    out = _extract_tool_call(raw)
    assert out is not None and out["tool"].startswith("git_log")


def test_in_prose_trailing_comma_repaired():
    raw = 'Step 1:\n{"tool": "shell_exec", "params": {"cmd": "ls",},}\nNext step.'
    assert _extract_tool_call(raw) == {"tool": "shell_exec", "params": {"cmd": "ls"}}


# ── helper unit ─────────────────────────────────────────────────────────────

def test_repair_returns_none_on_garbage():
    assert _try_repair_json("not json at all", "test") is None


def test_repair_logs_branch(capsys):
    _try_repair_json('{"a": 1,}', "regression_branch")
    out = capsys.readouterr().out
    assert "[tool_call_repair] branch=regression_branch" in out
