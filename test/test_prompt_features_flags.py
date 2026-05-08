"""Pure-unit tests for the prompt_features schema section + ModeToolsConfig.effort.

These flags are kill switches for the prompting-system overhaul (provenance
tags, Anthropic native tool-use, playbook, etc.). All default ON; the schema
must reject typos and constrain `effort` to the four named tiers.
"""

import pytest

from app.config_schema import (
    ConfigPatchError,
    PromptFeaturesConfig,
    RootConfig,
    validate_full,
    validate_patch,
)

EXPECTED_FLAGS = {
    "handle_recall_rule",
    "tool_when_lines",
    "subagent_matrix",
    "provenance_tags",
    "injection_sibling_message",
    "turn_summary_block",
    "playbook_enabled",
    "anthropic_native_tools",
    "supervisor_tool_use",
}


def test_every_flag_defaults_on():
    cfg = PromptFeaturesConfig()
    for flag in EXPECTED_FLAGS:
        assert getattr(cfg, flag) is True, f"{flag} must default True"


def test_root_config_includes_prompt_features():
    root = RootConfig()
    assert isinstance(root.prompt_features, PromptFeaturesConfig)


def test_unknown_flag_rejected_with_suggestion():
    with pytest.raises(ConfigPatchError) as exc:
        validate_patch({}, {"prompt_features": {"provenence_tags": False}})
    msg = str(exc.value)
    assert "provenence_tags" in msg
    assert "provenance_tags" in msg


def test_effort_accepts_tiers():
    for tier in ("low", "med", "high", "xhigh"):
        validate_patch({}, {"agent": {"mode": {"build": {"effort": tier}}}})


def test_effort_default_is_none():
    root = RootConfig()
    assert root.agent.mode.build.effort is None


def test_effort_rejects_unknown():
    with pytest.raises(ConfigPatchError):
        validate_patch({}, {"agent": {"mode": {"plan": {"effort": "extreme"}}}})


def test_validate_full_clean_with_features():
    issues = validate_full({"prompt_features": {f: True for f in EXPECTED_FLAGS}})
    assert issues == []
