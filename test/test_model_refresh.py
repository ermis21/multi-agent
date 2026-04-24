"""Unit tests for app.model_refresh — endpoint probing is stubbed."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from app import model_refresh


@pytest.fixture
def ranks_file(tmp_path, monkeypatch):
    header = (
        "# Test header comment. This must survive round-trip.\n"
        "# Second line of the header.\n\n"
    )
    body = {
        "models": [
            {
                "name": "claude-opus-4-7",
                "provider": "anthropic",
                "model_id": "claude-opus-4-7",
                "tier": "frontier",
                "context_window": 200000,
                "rank": 1,
                "capabilities": ["destructive_edit_safe"],
            },
            {
                "name": "local_a",
                "provider": "local",
                "model_id": "stale-a.gguf",
                "tier": "medium",
                "context_window": 1000,
                "rank": 5,
                "capabilities": ["long_context_reasoning"],
            },
            {
                "name": "local_b",
                "provider": "local",
                "model_id": "fresh-b.gguf",
                "tier": "small",
                "context_window": 4096,
                "rank": 10,
                "capabilities": [],
            },
        ]
    }
    path = tmp_path / "model_ranks.yaml"
    path.write_text(header + yaml.dump(body, sort_keys=False), encoding="utf-8")
    monkeypatch.setattr(model_refresh, "_CATALOG_PATH", path)
    return path


@pytest.fixture
def state_dir(tmp_path, monkeypatch):
    sd = tmp_path / "state"
    (sd / "model_metadata").mkdir(parents=True)
    monkeypatch.setattr(model_refresh, "_METADATA_PATH", sd / "model_metadata" / "local_models.json")
    return sd


@pytest.fixture
def cfg():
    return {
        "llm": {"provider": "local", "base_url": "http://default:8081"},
        "models": {
            "local_a": {"provider": "local", "url": "http://a.invalid:8000"},
            "local_b": {"provider": "local", "base_url": "http://b.invalid:8000"},
        },
    }


def _stub_probe(responses):
    """Return an async stub for `_probe_endpoint` scripted by endpoint URL."""
    async def _probe(url, timeout_s):
        if url not in responses:
            raise AssertionError(f"unexpected probe {url!r}")
        return responses[url]
    return _probe


def test_drift_detected_and_patched(ranks_file, state_dir, cfg, monkeypatch):
    responses = {
        "http://a.invalid:8000": ("actual-a.gguf", 262144, None),
        "http://b.invalid:8000": ("fresh-b.gguf", 4096, None),   # unchanged
    }
    monkeypatch.setattr(model_refresh, "_probe_endpoint", _stub_probe(responses))

    import asyncio
    summary = asyncio.run(model_refresh.refresh_local_models(cfg, auto_patch=True))

    assert summary["patched"] is True
    assert "local_a" in summary["drifted"]
    assert "local_b" in summary["unchanged"]
    assert summary["errors"] == []

    # File was rewritten with new values; header comments survived.
    text = ranks_file.read_text()
    assert text.startswith("# Test header comment.")
    assert "# Second line of the header." in text
    parsed = yaml.safe_load(text)
    names = {e["name"]: e for e in parsed["models"]}
    assert names["local_a"]["model_id"] == "actual-a.gguf"
    assert names["local_a"]["context_window"] == 262144
    assert names["local_b"]["model_id"] == "fresh-b.gguf"
    # Non-local entry untouched.
    assert names["claude-opus-4-7"]["model_id"] == "claude-opus-4-7"

    # State file written and parseable.
    state = json.loads(Path(model_refresh._METADATA_PATH).read_text())
    assert state["entries"]["local_a"]["drift"]["context_window"] == {"from": 1000, "to": 262144}
    assert state["entries"]["local_b"]["drift"] == {}


def test_auto_patch_false_is_dry_run(ranks_file, state_dir, cfg, monkeypatch):
    responses = {
        "http://a.invalid:8000": ("actual-a.gguf", 262144, None),
        "http://b.invalid:8000": ("fresh-b.gguf", 4096, None),
    }
    monkeypatch.setattr(model_refresh, "_probe_endpoint", _stub_probe(responses))

    import asyncio
    summary = asyncio.run(model_refresh.refresh_local_models(cfg, auto_patch=False))

    assert summary["patched"] is False
    assert "local_a" in summary["drifted"]
    # File unchanged — drift recorded but not applied.
    parsed = yaml.safe_load(ranks_file.read_text())
    names = {e["name"]: e for e in parsed["models"]}
    assert names["local_a"]["model_id"] == "stale-a.gguf"
    assert names["local_a"]["context_window"] == 1000


def test_endpoint_error_recorded_not_raised(ranks_file, state_dir, cfg, monkeypatch):
    responses = {
        "http://a.invalid:8000": (None, None, "ConnectError: refused"),
        "http://b.invalid:8000": ("fresh-b.gguf", 4096, None),
    }
    monkeypatch.setattr(model_refresh, "_probe_endpoint", _stub_probe(responses))

    import asyncio
    summary = asyncio.run(model_refresh.refresh_local_models(cfg, auto_patch=True))

    assert "local_a" in summary["errors"]
    assert summary["entries"]["local_a"]["error"].startswith("ConnectError")
    # Errored entry not marked drifted; local_b unchanged → nothing to patch.
    assert "local_a" not in summary["drifted"]
    assert summary["patched"] is False


def test_parse_llama_cpp_response():
    payload = {
        "object": "list",
        "data": [{
            "id": "gemma-4-E4B-it-Q8_0.gguf",
            "meta": {"n_ctx_train": 131072, "n_params": 7518069290},
        }],
    }
    model_id, ctx = model_refresh._parse_models_response(payload)
    assert model_id == "gemma-4-E4B-it-Q8_0.gguf"
    assert ctx == 131072


def test_parse_legacy_models_list():
    payload = {"models": [{"name": "foo.gguf"}]}
    model_id, ctx = model_refresh._parse_models_response(payload)
    assert model_id == "foo.gguf"
    assert ctx is None


def test_parse_empty_response():
    assert model_refresh._parse_models_response({}) == (None, None)


def test_endpoint_falls_back_to_llm_base_url(cfg):
    entry_cfg = {"llm": cfg["llm"], "models": {"other": {"url": "http://x"}}}
    assert model_refresh._resolve_endpoint("orphan", entry_cfg) == "http://default:8081"


def test_no_endpoint_when_nothing_set():
    assert model_refresh._resolve_endpoint("x", {}) is None
