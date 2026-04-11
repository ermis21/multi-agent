"""
Configuration loader with mtime-based cache.

Files are bind-mounted from the host, so changes are picked up on next request
without restarting the container.
"""

import os
from pathlib import Path

import yaml

CONFIG_PATH  = Path(os.environ.get("CONFIG_PATH",        "/app/config/config.yaml"))
AGENTS_PATH  = Path(os.environ.get("AGENTS_CONFIG_PATH", "/app/config/agents.yaml"))

# Module-level cache: (mtime, parsed_dict)
_config_cache: tuple[float, dict] = (0.0, {})
_agents_cache: tuple[float, dict] = (0.0, {})


def get_config() -> dict:
    """Return parsed config.yaml, re-reading only when the file changes."""
    global _config_cache
    mtime = CONFIG_PATH.stat().st_mtime
    if mtime != _config_cache[0]:
        _config_cache = (mtime, yaml.safe_load(CONFIG_PATH.read_text()))
    return _config_cache[1]


def get_agents_config() -> dict:
    """Return parsed agents.yaml, re-reading only when the file changes."""
    global _agents_cache
    mtime = AGENTS_PATH.stat().st_mtime
    if mtime != _agents_cache[0]:
        _agents_cache = (mtime, yaml.safe_load(AGENTS_PATH.read_text()))
    return _agents_cache[1]


def patch_config(patch: dict) -> dict:
    """
    Deep-merge patch into config.yaml and write it back.
    Returns the updated config dict.
    """
    cfg = yaml.safe_load(CONFIG_PATH.read_text())
    _deep_merge(cfg, patch)
    CONFIG_PATH.write_text(yaml.dump(cfg, default_flow_style=False, sort_keys=False, allow_unicode=True))
    # Invalidate cache
    global _config_cache
    _config_cache = (0.0, {})
    return get_config()


def _deep_merge(base: dict, override: dict) -> None:
    """Recursively merge override into base in-place."""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
