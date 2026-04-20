"""Pure-unit tests for app/context_compressor.py."""

from app.context_compressor import (
    _ALWAYS_TOOLS,
    _handle_id,
    compact_tool_result,
    compress_section,
    filter_skills,
    filter_tool_docs,
)
from app.tokenizer import ElisionStrategy


# ── compress_section ─────────────────────────────────────────────────────────

def test_compress_section_noop_under_budget():
    assert compress_section("short", 1000, ElisionStrategy.HEAD_TAIL, "x") == "short"


def test_compress_section_empty_stays_empty():
    assert compress_section("", 100, ElisionStrategy.HEAD_TAIL, "x") == ""


def test_compress_section_over_budget_prepends_marker():
    text = "word " * 4000
    out = compress_section(text, 40, ElisionStrategy.HEAD_TAIL, "MEMORY")
    assert out.startswith("[compressed MEMORY:")
    assert "tok]" in out.splitlines()[0]
    assert len(out) < len(text)


def test_compress_section_zero_budget_is_noop():
    text = "foo"
    assert compress_section(text, 0, ElisionStrategy.HEAD_TAIL, "x") == text


# ── filter_tool_docs ─────────────────────────────────────────────────────────

class _StubState:
    def __init__(self, invoked=None):
        self._invoked = invoked or {}

    def get(self, key, default=None):
        if key == "tools.invoked":
            return self._invoked
        return default


def test_filter_tool_docs_always_keeps_always_set():
    allowed = list(_ALWAYS_TOOLS) + ["git_status", "web_search"]
    cfg = {"context": {"budgets": {"tool_docs": 0}}}  # 0 budget → only always
    out = filter_tool_docs(allowed, _StubState(), cfg, agent_mode="build")
    for t in _ALWAYS_TOOLS:
        assert t in out


def test_filter_tool_docs_promotes_hot_tools():
    allowed = ["file_read", "shell_exec", "memory_search", "web_search", "git_status"]
    state = _StubState(invoked={"web_search": 8, "git_status": 3})
    cfg = {"context": {"budgets": {"tool_docs": 99999}}}
    out = filter_tool_docs(allowed, state, cfg, agent_mode="build")
    # All are kept when budget is generous; ordering isn't part of the contract
    assert set(out) == set(allowed)


def test_filter_tool_docs_empty_allowed_returns_empty():
    out = filter_tool_docs([], _StubState(), {"context": {"budgets": {"tool_docs": 1000}}})
    assert out == []


# ── filter_skills ────────────────────────────────────────────────────────────

def test_filter_skills_budget_cap_drops_overflow():
    skills = [
        {"name": f"skill-{i}", "when": "triggered when X"*8, "path": f"config/skills/s{i}/SKILL.md"}
        for i in range(40)
    ]
    cfg = {"context": {"budgets": {"skills": 30}}}  # tiny budget
    out = filter_skills("some query", skills, cfg)
    assert 0 < len(out) < len(skills)


def test_filter_skills_empty_list_is_empty():
    assert filter_skills("q", [], {"context": {"budgets": {"skills": 100}}}) == []


# ── compact_tool_result + handles ────────────────────────────────────────────

def test_handle_id_is_deterministic():
    a = _handle_id("web_fetch", {"url": "x"}, "body")
    b = _handle_id("web_fetch", {"url": "x"}, "body")
    assert a == b and a.startswith("rf-") and len(a) == 9


def test_handle_id_differs_on_body_change():
    a = _handle_id("web_fetch", {"url": "x"}, "body-a")
    b = _handle_id("web_fetch", {"url": "x"}, "body-b")
    assert a != b


def test_compact_tool_result_inlines_under_budget():
    body = "small body"
    inline, hid = compact_tool_result("file_read", {"path": "x"}, body, 1000)
    assert inline == body
    assert hid is None


def test_compact_tool_result_elides_over_budget():
    body = "word " * 4000
    inline, hid = compact_tool_result("web_fetch", {"url": "x"}, body, 60)
    assert hid is not None and hid.startswith("rf-")
    assert "tool_result_recall" in inline
    assert len(inline) < len(body)


def test_compact_tool_result_zero_budget_is_noop():
    body = "anything"
    inline, hid = compact_tool_result("x", {}, body, 0)
    assert inline == body and hid is None
