"""Pure-unit tests for supervisor modality classifier + rubric branching."""

import pytest

from app.agents import (
    _build_supervisor_rubric,
    _classify_worker_modality,
    _effective_threshold,
)


# ── _classify_worker_modality ────────────────────────────────────────────────

def test_classify_no_tool():
    modality, rate, count = _classify_worker_modality([])
    assert modality == "no_tool"
    assert rate == 0.0
    assert count == 0


def test_classify_tool_light_one():
    modality, rate, count = _classify_worker_modality([{"tool": "web_search"}])
    assert modality == "tool_light"
    assert rate == 0.0
    assert count == 1


def test_classify_tool_light_two():
    traces = [{"tool": "file_read"}, {"tool": "web_fetch"}]
    modality, _, count = _classify_worker_modality(traces)
    assert modality == "tool_light"
    assert count == 2


def test_classify_tool_heavy_three_plus():
    traces = [{"tool": f"t{i}"} for i in range(5)]
    modality, rate, count = _classify_worker_modality(traces)
    assert modality == "tool_heavy"
    assert count == 5
    assert rate == 0.0


def test_classify_with_errors_suffix():
    traces = [
        {"tool": "file_read"},
        {"tool": "shell_exec", "error": "boom"},
        {"tool": "file_write"},
        {"tool": "web_fetch", "error": "timeout"},
    ]
    modality, rate, _ = _classify_worker_modality(traces)
    assert modality == "tool_heavy_with_errors"
    assert rate == 0.5


def test_classify_error_rate_below_threshold():
    # 1/5 = 20% — below the 30% error-rate threshold
    traces = [{"tool": "t", "error": "x"}] + [{"tool": f"t{i}"} for i in range(4)]
    modality, rate, _ = _classify_worker_modality(traces)
    assert modality == "tool_heavy"
    assert rate == pytest.approx(0.2)


# ── _build_supervisor_rubric ──────────────────────────────────────────────────

def test_rubric_converse_no_tool_retires_research():
    body = _build_supervisor_rubric("no_tool", "converse")
    assert "conversational" in body.lower()
    assert "lack of research" in body.lower() or "no tool calls" in body.lower()
    # Should explicitly retire the research/source arrays
    assert "source_gaps" in body


def test_rubric_plan_demands_specificity():
    body = _build_supervisor_rubric("tool_heavy", "plan")
    assert "specific" in body.lower() or "specificity" in body.lower()


def test_rubric_build_with_errors_grades_handling_not_presence():
    body = _build_supervisor_rubric("tool_heavy_with_errors", "build")
    assert "error handling" in body.lower() or "recover" in body.lower()
    # The rubric explicitly instructs NOT to penalise error presence
    assert "do not penalise" in body.lower() or "do not count error presence" in body.lower()


def test_rubric_tool_light_does_not_demand_more():
    # The tool_light branch only fires when the mode is not plan/build/converse.
    # In production this is an edge case; the test documents the intended rubric.
    body = _build_supervisor_rubric("tool_light", "other")
    assert "do not demand more" in body.lower() or "do not demand" in body.lower()


def test_rubric_build_overrides_tool_light():
    """build mode wins over tool_light — build has its own rubric regardless of tool count."""
    body = _build_supervisor_rubric("tool_light", "build")
    assert "build-mode response" in body.lower()


def test_rubric_default_fallback_includes_all_five():
    # modality=tool_heavy + mode=build hits the build branch, so force a combo
    # that falls through to the default. Safest: an unexpected mode label.
    body = _build_supervisor_rubric("tool_heavy", "unknown_mode")
    assert "Tool Usage" in body
    assert "Source Verification" in body


# ── _effective_threshold ──────────────────────────────────────────────────────

def test_threshold_mode_override_wins():
    cfg = {"agent": {"supervisor_pass_threshold": 0.7, "supervisor_mode_overrides": {"converse": 0.5}}}
    assert _effective_threshold(cfg, "converse") == 0.5


def test_threshold_fallback_to_default():
    cfg = {"agent": {"supervisor_pass_threshold": 0.8, "supervisor_mode_overrides": {"plan": 0.9}}}
    # build has no override → falls back
    assert _effective_threshold(cfg, "build") == 0.8


def test_threshold_missing_agent_uses_07():
    assert _effective_threshold({}, "converse") == 0.7


def test_threshold_bad_override_ignored():
    cfg = {"agent": {"supervisor_pass_threshold": 0.7, "supervisor_mode_overrides": {"plan": "bad"}}}
    assert _effective_threshold(cfg, "plan") == 0.7


# ── _detect_hallucinated_zero_tool_claim (hallucination guard) ───────────────

from app.supervisor import _detect_hallucinated_zero_tool_claim  # noqa: E402


def test_hallucination_guard_fires_on_feedback():
    verdict = {
        "pass": False, "score": 0.0,
        "feedback": "The worker failed. No tools were actually called.",
        "tool_issues": [], "source_gaps": [], "research_gaps": [],
        "accuracy_issues": [], "completeness_issues": [],
    }
    reason = _detect_hallucinated_zero_tool_claim(verdict, tool_count=3)
    assert reason is not None
    assert "3 call" in reason


def test_hallucination_guard_fires_on_issue_array():
    verdict = {
        "feedback": "ok",
        "tool_issues": ["worker did not use any tools"],
        "accuracy_issues": [],
    }
    reason = _detect_hallucinated_zero_tool_claim(verdict, tool_count=2)
    assert reason is not None


def test_hallucination_guard_quiet_when_zero_tools_real():
    """A genuine zero-tool turn should not trigger the guard."""
    verdict = {"feedback": "no tools were called", "tool_issues": []}
    assert _detect_hallucinated_zero_tool_claim(verdict, tool_count=0) is None


def test_hallucination_guard_quiet_when_feedback_is_accurate():
    verdict = {
        "feedback": "worker used web_search but the result was sparse",
        "tool_issues": ["web_search query was underspecified"],
    }
    assert _detect_hallucinated_zero_tool_claim(verdict, tool_count=1) is None


def test_hallucination_guard_matches_various_phrasings():
    for phrase in (
        "no tools were called",
        "no tools actually called",
        "tool log shows no calls",
        "did not use any tools",
        "did not make any tools",
        "zero tools used",
        "no tool calls were made",
        "without using any tools",
    ):
        verdict = {"feedback": f"failing because: {phrase}."}
        assert _detect_hallucinated_zero_tool_claim(verdict, tool_count=1) is not None, phrase
