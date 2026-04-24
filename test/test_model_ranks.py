"""Pure-unit tests for app.model_ranks — no stack, no network."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from app.model_ranks import (
    ModelCatalog,
    ModelNotViableError,
    TIER_ORDER,
    select_model_for,
)


@pytest.fixture
def real_catalog() -> ModelCatalog:
    """Load the shipped config/model_ranks.yaml as the baseline catalog."""
    for p in (
        Path("/app/config/model_ranks.yaml"),
        Path(__file__).resolve().parent.parent / "config/model_ranks.yaml",
    ):
        if p.exists():
            return ModelCatalog.model_validate(yaml.safe_load(p.read_text()))
    pytest.skip("config/model_ranks.yaml not found")


def _dream_cfg(**overrides) -> dict:
    base = {
        "min_tier": "large",
        "min_context_window": 200000,
        "required_capabilities": [
            "destructive_edit_safe",
            "long_context_reasoning",
            "prompt_self_critique",
        ],
        "model": None,
    }
    base.update(overrides)
    return {"dream": base, "models": {}}


def test_tier_ordering_is_monotonic():
    assert TIER_ORDER["small"] < TIER_ORDER["medium"] < TIER_ORDER["large"] < TIER_ORDER["frontier"]


def test_default_floor_selects_opus(real_catalog):
    sel = select_model_for("dream", _dream_cfg(), catalog=real_catalog)
    assert sel.entry.name == "claude-opus-4-7"
    assert sel.entry.rank == 1


def test_pinned_model_wins(real_catalog):
    cfg = _dream_cfg(model="claude-sonnet-4-6")
    sel = select_model_for("dream", cfg, catalog=real_catalog)
    assert sel.entry.name == "claude-sonnet-4-6"


def test_pinned_model_below_floor_raises(real_catalog):
    cfg = _dream_cfg(model="claude-haiku-4-5")
    with pytest.raises(ModelNotViableError) as exc:
        select_model_for("dream", cfg, catalog=real_catalog)
    assert "does not meet floor" in str(exc.value)


def test_pinned_model_absent_raises(real_catalog):
    cfg = _dream_cfg(model="gpt-5-imaginary")
    with pytest.raises(ModelNotViableError) as exc:
        select_model_for("dream", cfg, catalog=real_catalog)
    assert "not found in model_ranks.yaml" in str(exc.value)


def test_empty_catalog_raises(real_catalog):
    empty = ModelCatalog(models=[])
    with pytest.raises(ModelNotViableError):
        select_model_for("dream", _dream_cfg(), catalog=empty)


def test_lower_min_tier_admits_haiku(real_catalog):
    """Dropping the floor to 'medium' + relaxing caps → Haiku qualifies."""
    cfg = _dream_cfg(min_tier="medium", required_capabilities=["long_context_reasoning"])
    sel = select_model_for("dream", cfg, catalog=real_catalog)
    # Opus still ranks lower → Opus still wins. But Haiku is now *eligible*.
    eligible = [m for m in real_catalog.models if "long_context_reasoning" in m.capabilities
                and TIER_ORDER[m.tier] >= TIER_ORDER["medium"]
                and m.context_window >= 200000]
    assert any(m.name == "claude-haiku-4-5" for m in eligible)
    assert sel.entry.name == "claude-opus-4-7"


def test_higher_context_floor_rejects_everything(real_catalog):
    # Ceiling must sit above the largest context in the catalog — Claude 1M
    # plus headroom for whatever replaces it next.
    cfg = _dream_cfg(min_context_window=10_000_000)
    with pytest.raises(ModelNotViableError) as exc:
        select_model_for("dream", cfg, catalog=real_catalog)
    assert "meets dream floor" in str(exc.value)


def test_capability_requires_all_not_any(real_catalog):
    """simulation_judge is declared by Opus+Sonnet; requiring it must still select Opus."""
    caps = [
        "destructive_edit_safe",
        "long_context_reasoning",
        "prompt_self_critique",
        "simulation_judge",
    ]
    sel = select_model_for("dream", _dream_cfg(required_capabilities=caps), catalog=real_catalog)
    assert sel.entry.name == "claude-opus-4-7"
    # Capability missing anywhere → no match.
    with pytest.raises(ModelNotViableError):
        select_model_for(
            "dream",
            _dream_cfg(required_capabilities=caps + ["nonexistent_capability"]),
            catalog=real_catalog,
        )


def test_bogus_tier_raises(real_catalog):
    cfg = _dream_cfg(min_tier="godlike")
    with pytest.raises(ModelNotViableError) as exc:
        select_model_for("dream", cfg, catalog=real_catalog)
    assert "min_tier" in str(exc.value)


def test_catalog_rejects_bad_tier_literal():
    bad = {"models": [{
        "name": "x", "provider": "anthropic", "model_id": "x",
        "tier": "ultra", "context_window": 200000, "rank": 1, "capabilities": [],
    }]}
    with pytest.raises(Exception):
        ModelCatalog.model_validate(bad)


def test_catalog_rejects_extras():
    bad = {"models": [{
        "name": "x", "provider": "anthropic", "model_id": "x",
        "tier": "large", "context_window": 200000, "rank": 1, "capabilities": [],
        "mystery_field": True,
    }]}
    with pytest.raises(Exception):
        ModelCatalog.model_validate(bad)


def test_overlay_prefers_declared_models_block(real_catalog):
    cfg = _dream_cfg()
    cfg["models"] = {
        "claude-opus-4-7": {"provider": "anthropic", "model": "override-id", "url": "https://x"},
    }
    sel = select_model_for("dream", cfg, catalog=real_catalog)
    # Overlay from cfg.models wins over synthesized entry from catalog.
    assert sel.llm_override["model"] == "override-id"
    assert sel.llm_override["url"] == "https://x"


def test_overlay_synthesized_when_not_in_models_block(real_catalog):
    cfg = _dream_cfg()
    sel = select_model_for("dream", cfg, catalog=real_catalog)
    assert sel.llm_override["provider"] == "anthropic"
    assert sel.llm_override["model"] == sel.entry.model_id
