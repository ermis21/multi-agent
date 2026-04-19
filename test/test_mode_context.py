"""Pure-unit tests for _mode_context_string — config-driven mode capability block."""

from app.agents import _mode_context_string


def _cfg(excluded_plan=None, excluded_build=None, excluded_converse=None, describe=True):
    return {
        "prompts": {"describe_mode_in_system_prompt": describe},
        "agent": {
            "mode": {
                "plan": {"excluded_tools": excluded_plan or []},
                "build": {"excluded_tools": excluded_build or []},
                "converse": {"excluded_tools": excluded_converse or []},
            }
        },
        "models": {},
    }


def test_short_form_when_cfg_missing():
    out = _mode_context_string("plan")
    assert "Mode: PLAN" in out
    # Short form is a single line
    assert "\n\n" not in out


def test_long_form_lists_excluded_tools():
    cfg = _cfg(excluded_plan=["file_write", "shell_exec"])
    out = _mode_context_string("plan", cfg=cfg)
    assert "`file_write`" in out
    assert "`shell_exec`" in out
    assert "Excluded tools" in out


def test_research_rule_present_in_long_form():
    cfg = _cfg(excluded_plan=["file_write"])
    for mode in ("plan", "build", "converse"):
        out = _mode_context_string(mode, cfg=cfg)
        assert "web_search" in out, f"{mode} should mention web_search in research rule"


def test_plan_mentions_mode_switch_instruction():
    cfg = _cfg(excluded_plan=["file_write"])
    out = _mode_context_string("plan", cfg=cfg)
    assert "/mode build" in out


def test_plan_warns_against_file_edit_substitute():
    cfg = _cfg(excluded_plan=["file_write"])
    out = _mode_context_string("plan", cfg=cfg)
    assert "file_edit" in out
    assert "file_write" in out


def test_global_toggle_off_returns_short_form():
    cfg = _cfg(excluded_plan=["file_write"], describe=False)
    out = _mode_context_string("plan", cfg=cfg)
    assert "Excluded tools" not in out
    assert "Mode: PLAN" in out


def test_per_model_override_wins_off():
    cfg = _cfg(excluded_plan=["file_write"])
    cfg["models"] = {"claude_opus": {"describe_mode_in_system_prompt": False}}
    role_cfg = {"model": "claude_opus"}
    out = _mode_context_string("plan", cfg=cfg, role_cfg=role_cfg)
    assert "Excluded tools" not in out


def test_per_model_override_wins_on_when_global_off():
    cfg = _cfg(excluded_plan=["file_write"], describe=False)
    cfg["models"] = {"gemma": {"describe_mode_in_system_prompt": True}}
    role_cfg = {"model": "gemma"}
    out = _mode_context_string("plan", cfg=cfg, role_cfg=role_cfg)
    assert "Excluded tools" in out


def test_unknown_mode_falls_back_to_empty():
    out = _mode_context_string("unknown_mode")
    assert out == ""


def test_empty_excluded_renders_none_placeholder():
    cfg = _cfg()  # all empty
    out = _mode_context_string("build", cfg=cfg)
    assert "_(none)_" in out
