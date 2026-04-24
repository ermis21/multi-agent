"""
Nightly sweep that reconciles `config/model_ranks.yaml` against the models
actually served by every `provider=local` endpoint.

For each catalog entry with `provider: local`, this module:

1. Resolves the endpoint by looking up `cfg.models.<name>.url` or `.base_url`
   (falling back to `cfg.llm.base_url`).
2. GETs `{endpoint}/v1/models`, reads the first entry's `meta.n_ctx_train`
   (llama.cpp-server shape) and `id` / top-level `name`.
3. Records observed vs catalog values to
   `/state/model_metadata/local_models.json` (for audit + diagnostics).
4. If `local_models_refresh.auto_patch` is true and drift is detected, rewrites
   `config/model_ranks.yaml` in place — preserving the leading comment header —
   so the catalog stays honest without hand-edits.

Non-goals: discovering new local endpoints (we only refresh ones already in the
catalog), rebalancing ranks/tiers (those are user judgment calls), or touching
remote providers.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import yaml

from app.config_loader import get_config
from app.model_ranks import _CATALOG_PATH, _invalidate_cache

_STATE_DIR = Path(os.environ.get("STATE_DIR", "/state"))
_METADATA_PATH = _STATE_DIR / "model_metadata" / "local_models.json"


def _resolve_endpoint(name: str, cfg: dict[str, Any]) -> str | None:
    """Pick the best URL for a local catalog entry. cfg.models.<name> wins over
    cfg.llm; `url` wins over `base_url` (matches app/llm.py precedence)."""
    models = (cfg or {}).get("models") or {}
    entry = models.get(name) or {}
    for key in ("url", "base_url"):
        val = entry.get(key)
        if val:
            return str(val).rstrip("/")
    llm = (cfg or {}).get("llm") or {}
    for key in ("url", "base_url"):
        val = llm.get(key)
        if val:
            return str(val).rstrip("/")
    return None


def _parse_models_response(payload: dict[str, Any]) -> tuple[str | None, int | None]:
    """Extract (model_id, context_window) from an OpenAI-style /v1/models reply.

    llama.cpp-server returns ctx via `data[0].meta.n_ctx_train`; ollama
    sometimes uses `models[0].details.parameter_size` only. We prefer the
    OpenAI-ish `data[]` list; fall back to the legacy `models[]` list.
    """
    model_id: str | None = None
    ctx: int | None = None

    data = payload.get("data") or []
    if isinstance(data, list) and data:
        first = data[0] or {}
        model_id = first.get("id") or first.get("name")
        meta = first.get("meta") or {}
        raw_ctx = meta.get("n_ctx_train") or meta.get("n_ctx")
        if isinstance(raw_ctx, int) and raw_ctx > 0:
            ctx = raw_ctx

    if model_id is None:
        models = payload.get("models") or []
        if isinstance(models, list) and models:
            first = models[0] or {}
            model_id = first.get("model") or first.get("name")

    return model_id, ctx


async def _probe_endpoint(url: str, timeout_s: float) -> tuple[str | None, int | None, str | None]:
    async with httpx.AsyncClient(timeout=timeout_s) as client:
        try:
            r = await client.get(f"{url}/v1/models")
            r.raise_for_status()
            payload = r.json()
        except Exception as e:
            return None, None, f"{type(e).__name__}: {e}"
    model_id, ctx = _parse_models_response(payload)
    if model_id is None:
        return None, None, "no model entry in /v1/models response"
    return model_id, ctx, None


def _load_ranks_yaml() -> tuple[str, dict[str, Any]]:
    """Return (header_comments, parsed_body) so we can round-trip the file
    without obliterating the documentation header."""
    text = _CATALOG_PATH.read_text(encoding="utf-8")
    header_lines: list[str] = []
    body_start = 0
    for i, line in enumerate(text.splitlines(keepends=True)):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            header_lines.append(line)
            continue
        body_start = i
        break
    header = "".join(header_lines)
    body_text = "".join(text.splitlines(keepends=True)[body_start:])
    parsed = yaml.safe_load(body_text) or {}
    return header, parsed


def _write_ranks_yaml(header: str, body: dict[str, Any]) -> None:
    dumped = yaml.dump(body, default_flow_style=False, sort_keys=False, allow_unicode=True)
    # Ensure there's a blank line between header comments and the first key.
    if header and not header.endswith("\n\n"):
        header = header.rstrip("\n") + "\n\n"
    _CATALOG_PATH.write_text(header + dumped, encoding="utf-8")


async def refresh_local_models(
    cfg: dict[str, Any] | None = None,
    *,
    auto_patch: bool | None = None,
    timeout_s: float | None = None,
) -> dict[str, Any]:
    """Probe every provider=local entry; write state file; optionally patch the
    catalog. Returns a summary dict suitable for logging and /internal replies.
    """
    cfg = cfg or get_config()
    refresh_cfg = (cfg or {}).get("local_models_refresh") or {}
    if auto_patch is None:
        auto_patch = bool(refresh_cfg.get("auto_patch", True))
    if timeout_s is None:
        timeout_s = float(refresh_cfg.get("timeout_s", 8.0))

    header, body = _load_ranks_yaml()
    entries = body.get("models") or []

    observations: dict[str, dict[str, Any]] = {}
    drifted: list[str] = []
    errors: list[str] = []
    unchanged: list[str] = []

    for entry in entries:
        if entry.get("provider") != "local":
            continue
        name = entry.get("name")
        url = _resolve_endpoint(name, cfg)
        if not url:
            observations[name] = {
                "endpoint": None,
                "ok": False,
                "error": "no endpoint resolvable from cfg.models or cfg.llm",
            }
            errors.append(name)
            continue

        observed_id, observed_ctx, err = await _probe_endpoint(url, timeout_s)
        catalog_id = entry.get("model_id")
        catalog_ctx = entry.get("context_window")

        record: dict[str, Any] = {
            "endpoint": url,
            "ok": err is None,
            "error": err,
            "observed_model_id": observed_id,
            "observed_context_window": observed_ctx,
            "catalog_model_id": catalog_id,
            "catalog_context_window": catalog_ctx,
        }

        if err:
            errors.append(name)
            observations[name] = record
            continue

        changes: dict[str, Any] = {}
        if observed_id and observed_id != catalog_id:
            changes["model_id"] = {"from": catalog_id, "to": observed_id}
            entry["model_id"] = observed_id
        if observed_ctx and observed_ctx != catalog_ctx:
            changes["context_window"] = {"from": catalog_ctx, "to": observed_ctx}
            entry["context_window"] = observed_ctx

        record["drift"] = changes
        observations[name] = record
        if changes:
            drifted.append(name)
        else:
            unchanged.append(name)

    patched = False
    if auto_patch and drifted:
        _write_ranks_yaml(header, body)
        _invalidate_cache()
        patched = True

    summary = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "patched": patched,
        "drifted": drifted,
        "unchanged": unchanged,
        "errors": errors,
        "entries": observations,
    }

    _METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = _METADATA_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, _METADATA_PATH)

    return summary
