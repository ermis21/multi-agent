"""Pure-unit tests for app.config_schema — no stack required."""

from pathlib import Path

import pytest
import yaml

from app.config_schema import (
    ConfigPatchError,
    RootConfig,
    validate_full,
    validate_patch,
)


@pytest.fixture(scope="module")
def real_config() -> dict:
    """Load the actual config/config.yaml so we're testing against reality."""
    for path in (Path("/app/config/config.yaml"), Path(__file__).resolve().parent.parent / "config/config.yaml"):
        if path.exists():
            return yaml.safe_load(path.read_text())
    pytest.skip("config/config.yaml not found")


def test_valid_config_parses(real_config):
    """The shipped config/config.yaml must validate cleanly."""
    assert validate_full(real_config) == []


def test_unknown_agent_key_rejected(real_config):
    with pytest.raises(ConfigPatchError) as exc:
        validate_patch(real_config, {"agent": {"supervisor_pass_treshold": 0.8}})
    msg = str(exc.value)
    assert "supervisor_pass_treshold" in msg
    # difflib suggestion points at the real key
    assert "supervisor_pass_threshold" in msg


def test_mode_override_range(real_config):
    with pytest.raises(ConfigPatchError):
        validate_patch(real_config, {"agent": {"supervisor_mode_overrides": {"plan": 1.5}}})


def test_mode_override_negative(real_config):
    with pytest.raises(ConfigPatchError):
        validate_patch(real_config, {"agent": {"supervisor_mode_overrides": {"converse": -0.1}}})


def test_inflection_mode_literal(real_config):
    with pytest.raises(ConfigPatchError):
        validate_patch(real_config, {"agent": {"inflection_mode": "bogus"}})


def test_agent_mode_default_literal(real_config):
    with pytest.raises(ConfigPatchError):
        validate_patch(real_config, {"agent": {"mode": {"default": "not_a_mode"}}})


def test_prompts_mode_literal(real_config):
    with pytest.raises(ConfigPatchError):
        validate_patch(real_config, {"prompts": {"mode": "verbose"}})


def test_tools_allows_extras(real_config):
    """tools.* is intentionally open — users add new integration blocks."""
    out = validate_patch(real_config, {"tools": {"custom_integration": {"key": "value"}}})
    assert out["tools"]["custom_integration"]["key"] == "value"


def test_models_allows_named_overrides(real_config):
    out = validate_patch(real_config, {"models": {"my_model": {"provider": "anthropic", "model": "foo"}}})
    assert "my_model" in out["models"]


def test_dotted_key_patch(real_config):
    out = validate_patch(real_config, {"agent.supervisor_pass_threshold": 0.8})
    assert out["agent"]["supervisor_pass_threshold"] == 0.8


def test_validate_full_reports_multiple_issues():
    """validate_full returns a list, never raises."""
    bad = {"agent": {"inflection_mode": "bogus", "max_retries": -1}}
    issues = validate_full(bad)
    assert issues  # non-empty
    assert any("inflection_mode" in i for i in issues)
    assert any("max_retries" in i for i in issues)


def test_validate_full_on_empty_dict():
    """Empty config should parse (all fields have defaults)."""
    assert validate_full({}) == []


def test_soul_schedule_shape():
    """Cron strings are sanity-checked for 5-field shape."""
    # Valid
    RootConfig.model_validate({"soul": {"enabled": True, "schedule": "0 5 * * *", "max_chars": 1000}})
    # Invalid — too few fields
    with pytest.raises(Exception):
        RootConfig.model_validate({"soul": {"enabled": True, "schedule": "0 5", "max_chars": 1000}})


def test_approval_bucket_forbids_extras(real_config):
    """approval.build.new_key should be rejected (closed schema)."""
    with pytest.raises(ConfigPatchError):
        validate_patch(real_config, {"approval": {"build": {"new_key": []}}})


# ── Context (PR 1 — Gemma-aware budgets) ─────────────────────────────────────

def test_context_defaults_valid():
    """RootConfig with no `context` block must fall back to defaults cleanly."""
    cfg = RootConfig.model_validate({})
    assert cfg.context.enabled is True
    assert cfg.context.budgets.soul == 512
    assert cfg.context.total_soft_cap == 12000
    assert cfg.context.tokenizer_backend == "llama"
    assert cfg.context.elision_strategy == "head_tail"


def test_context_unknown_key_rejected(real_config):
    with pytest.raises(ConfigPatchError) as exc:
        validate_patch(real_config, {"context": {"enabeld": True}})  # typo
    msg = str(exc.value)
    assert "enabeld" in msg and "enabled" in msg


def test_context_budgets_unknown_key_rejected(real_config):
    with pytest.raises(ConfigPatchError):
        validate_patch(real_config, {"context": {"budgets": {"made_up_slot": 100}}})


def test_context_budget_negative_rejected(real_config):
    with pytest.raises(ConfigPatchError):
        validate_patch(real_config, {"context": {"budgets": {"soul": -1}}})


def test_context_soft_cap_negative_rejected(real_config):
    with pytest.raises(ConfigPatchError):
        validate_patch(real_config, {"context": {"total_soft_cap": -100}})


def test_context_tokenizer_backend_literal(real_config):
    with pytest.raises(ConfigPatchError):
        validate_patch(real_config, {"context": {"tokenizer_backend": "sentencepiece"}})


def test_context_elision_strategy_literal(real_config):
    with pytest.raises(ConfigPatchError):
        validate_patch(real_config, {"context": {"elision_strategy": "random"}})


def test_context_disabled_toggle_accepted(real_config):
    """The master feature flag must accept `false` for rollback."""
    out = validate_patch(real_config, {"context": {"enabled": False}})
    assert out["context"]["enabled"] is False
