"""
Model capability catalog + role-scoped selection.

`config/model_ranks.yaml` declares which models exist and what they can do
(tier, context window, capability tags). Role configs in `config.yaml`
(currently only `dream.*`) set a floor; `select_model_for(role, cfg)` returns
the lowest-rank model that clears the floor, or raises `ModelNotViableError`
if none do.

Decoupled from `config.yaml:models.*` (which carries endpoint/URL/API params)
so capability metadata and transport details can evolve independently.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

Tier = Literal["small", "medium", "large", "frontier"]
TIER_ORDER: dict[str, int] = {"small": 0, "medium": 1, "large": 2, "frontier": 3}


class ModelNotViableError(RuntimeError):
    """No model in the catalog meets the configured floor for the requested role."""


class ModelEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    provider: str
    model_id: str
    tier: Tier
    context_window: int = Field(ge=0)
    rank: int = Field(ge=1)
    capabilities: list[str] = Field(default_factory=list)
    # Independent benchmark score (Artificial Analysis Intelligence Index, 0–100).
    # Informational only — selection still keys off `rank` — but carries the
    # provenance for why rank N sits where it does so edits are auditable.
    intelligence_index: float | None = Field(default=None, ge=0.0, le=100.0)
    notes: str | None = None


class ModelCatalog(BaseModel):
    model_config = ConfigDict(extra="forbid")
    models: list[ModelEntry] = Field(default_factory=list)


@dataclass
class Selection:
    """Resolved model + the LLM-config overlay to feed `app/llm.py:_llm_call`."""
    entry: ModelEntry
    llm_override: dict[str, Any]


_CATALOG_PATH = Path(os.environ.get("MODEL_RANKS_PATH", "/config/model_ranks.yaml"))
_cache: tuple[float, ModelCatalog] | None = None


def _load_catalog(path: Path = _CATALOG_PATH) -> ModelCatalog:
    """Mtime-cached catalog load — matches the pattern used for config.yaml."""
    global _cache
    try:
        mtime = path.stat().st_mtime
    except FileNotFoundError as exc:
        raise ModelNotViableError(f"model_ranks.yaml not found at {path}") from exc
    if _cache is not None and _cache[0] == mtime:
        return _cache[1]
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    try:
        cat = ModelCatalog.model_validate(raw)
    except ValidationError as e:
        raise ModelNotViableError(f"model_ranks.yaml failed schema validation: {e}") from e
    _cache = (mtime, cat)
    return cat


def _invalidate_cache() -> None:
    """For tests."""
    global _cache
    _cache = None


def _eligible(entry: ModelEntry, min_tier: str, min_ctx: int, required_caps: list[str]) -> bool:
    if TIER_ORDER[entry.tier] < TIER_ORDER[min_tier]:
        return False
    if entry.context_window < min_ctx:
        return False
    return set(required_caps).issubset(set(entry.capabilities))


def select_model_for(role: str, cfg: dict[str, Any], *, catalog: ModelCatalog | None = None) -> Selection:
    """
    Return the lowest-rank model meeting the floor for `role` (e.g. "dream").
    Honors `cfg[role].model` as an explicit pin. Raises `ModelNotViableError`
    when nothing qualifies.
    """
    role_cfg = (cfg or {}).get(role, {})
    if not isinstance(role_cfg, dict):
        raise ModelNotViableError(f"cfg.{role} is not a mapping")

    min_tier = role_cfg.get("min_tier", "large")
    min_ctx = int(role_cfg.get("min_context_window", 200000))
    required_caps = list(role_cfg.get("required_capabilities", []))
    pinned = role_cfg.get("model")

    if min_tier not in TIER_ORDER:
        raise ModelNotViableError(
            f"cfg.{role}.min_tier={min_tier!r} not in {list(TIER_ORDER)}"
        )

    cat = catalog or _load_catalog()
    eligible = [m for m in cat.models if _eligible(m, min_tier, min_ctx, required_caps)]

    if pinned:
        hit = next((m for m in eligible if m.name == pinned), None)
        if hit is None:
            if not any(m.name == pinned for m in cat.models):
                raise ModelNotViableError(
                    f"cfg.{role}.model={pinned!r} not found in model_ranks.yaml"
                )
            raise ModelNotViableError(
                f"cfg.{role}.model={pinned!r} does not meet floor "
                f"(tier>={min_tier}, ctx>={min_ctx}, caps>={required_caps})"
            )
        entry = hit
    else:
        if not eligible:
            raise ModelNotViableError(
                f"No model in model_ranks.yaml meets {role} floor "
                f"(tier>={min_tier}, ctx>={min_ctx}, caps>={required_caps})"
            )
        entry = min(eligible, key=lambda m: m.rank)

    return Selection(entry=entry, llm_override=_overlay_for(entry, cfg))


def _overlay_for(entry: ModelEntry, cfg: dict[str, Any]) -> dict[str, Any]:
    """
    Build the LLM override dict that `app/llm.py:_llm_call` overlays onto `cfg.llm`.
    Prefers a pre-declared `cfg.models.<name>` when present (lets the user set
    url/api-specific knobs); otherwise synthesizes from the catalog entry.
    """
    models_cfg = (cfg or {}).get("models") or {}
    if entry.name in models_cfg:
        return dict(models_cfg[entry.name])
    return {"provider": entry.provider, "model": entry.model_id}
