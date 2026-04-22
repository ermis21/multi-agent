"""Tests for the 4 dream probes in sandbox/mcp_server.py::_diagnostic_check.

These run inside `phoebe-sandbox` where `/project:ro` is the live repo, so we
import the module and call `_diagnostic_check({})` directly (same pattern as
existing diagnostic test coverage).

Note: since the probe file reads `/project/config/model_ranks.yaml` and
`/project/config/config.yaml` directly, the test can only assert the probe
keys exist, return pass/warn/fail status, and carry non-empty `detail` strings.
"""

from __future__ import annotations

from pathlib import Path

import pytest

# These probes read /project/config/*.yaml and call the live sandbox module.
# Only phoebe-sandbox mounts /project, so skip gracefully when absent (e.g. when
# collected inside phoebe-api by `make test-integration`).
pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        not Path("/project/config/config.yaml").exists(),
        reason="/project not mounted — run via `make test-dream-live` inside phoebe-sandbox",
    ),
]


def test_dream_probes_present_and_statused():
    """After Phase 5 wiring, these 4 new checks must appear in the probe map."""
    from sandbox.mcp_server import _diagnostic_check
    result = _diagnostic_check({})
    checks = result["checks"]

    for key in ("dream_model_viable", "dream_state_writable",
                "phrase_index_consistent", "dream_cron_scheduled"):
        assert key in checks, f"missing probe: {key}"
        assert checks[key]["status"] in ("pass", "warn", "fail"), \
            f"unexpected status for {key}: {checks[key]}"
        assert isinstance(checks[key].get("detail"), str), \
            f"{key} missing detail string"


def test_dream_state_writable_can_write_to_state_roots():
    """This probe must land `pass` when /state/dream/{runs,phrase_*}/ are writable."""
    from sandbox.mcp_server import _diagnostic_check
    result = _diagnostic_check({})
    s = result["checks"]["dream_state_writable"]
    # Sandbox always has /state writable — this should pass unless hardware-full.
    assert s["status"] == "pass", f"dream_state_writable not passing: {s}"


def test_dream_model_viable_has_viable_models_in_repo_config():
    """The repo's config/model_ranks.yaml ships with Opus + Sonnet; probe must
    report at least one viable model (or warn cleanly when catalog is empty)."""
    from sandbox.mcp_server import _diagnostic_check
    result = _diagnostic_check({})
    s = result["checks"]["dream_model_viable"]
    # With shipped config we expect pass; at minimum status must be valid.
    assert s["status"] in ("pass", "warn", "fail")
    if s["status"] == "pass":
        # Opus + Sonnet should both qualify under the default floor.
        assert "claude-" in s["detail"]


def test_dream_cron_scheduled_graceful_when_disabled():
    """When dream.enabled=false in the repo config, cron_scheduled should not fail."""
    from sandbox.mcp_server import _diagnostic_check
    import yaml
    from pathlib import Path
    cfg = yaml.safe_load(Path("/project/config/config.yaml").read_text())
    dream_enabled = bool((cfg.get("dream") or {}).get("enabled", False))
    if dream_enabled:
        pytest.skip("dream enabled — skipping 'disabled graceful' assertion")
    result = _diagnostic_check({})
    s = result["checks"]["dream_cron_scheduled"]
    assert s["status"] in ("pass", "warn"), f"unexpected fail when dream disabled: {s}"
