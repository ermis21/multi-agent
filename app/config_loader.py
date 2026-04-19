"""
Configuration loader with mtime-based cache.

Files are bind-mounted from the host, so changes are picked up on next request
without restarting the container.
"""

import os
from pathlib import Path
from typing import Callable

import yaml

CONFIG_PATH = Path(os.environ.get("CONFIG_PATH",        "/app/config/config.yaml"))
AGENTS_PATH = Path(os.environ.get("AGENTS_CONFIG_PATH", "/app/config/agents.yaml"))


def _make_yaml_loader(path: Path) -> Callable[[], dict]:
    """Return a getter that re-reads path only when its mtime changes."""
    cache: list = [0.0, {}]  # [mtime, parsed_dict] — list so closure can mutate

    def getter() -> dict:
        mtime = path.stat().st_mtime
        if mtime != cache[0]:
            cache[0] = mtime
            cache[1] = yaml.safe_load(path.read_text())
        return cache[1]

    getter._cache = cache  # expose for patch_config to invalidate
    return getter


get_config       = _make_yaml_loader(CONFIG_PATH)
get_agents_config = _make_yaml_loader(AGENTS_PATH)


def patch_config(patch: dict) -> dict:
    """
    Deep-merge patch into config.yaml and write it back.
    Returns the updated config dict.

    Accepts both nested dicts and dotted-key notation:
      {"tools": {"discord": {"default_channel_id": "x"}}}
      {"tools.discord.default_channel_id": "x"}   ← expanded automatically

    The merged result is validated against RootConfig before the file is
    written — schema drift raises ConfigPatchError and the file is left
    untouched.
    """
    from app.config_schema import validate_patch  # lazy — avoids import cycle on module load

    current = yaml.safe_load(CONFIG_PATH.read_text()) or {}
    merged = validate_patch(current, patch)  # raises ConfigPatchError on drift
    CONFIG_PATH.write_text(yaml.dump(merged, default_flow_style=False, sort_keys=False, allow_unicode=True))
    get_config._cache[0] = 0.0  # invalidate cache
    return get_config()


def _expand_dotted_keys(patch: dict) -> dict:
    """
    Expand any top-level dotted keys into nested dicts.
    {"a.b.c": 1, "x": {"y": 2}} → {"a": {"b": {"c": 1}}, "x": {"y": 2}}
    """
    expanded: dict = {}
    for key, value in patch.items():
        if isinstance(key, str) and "." in key:
            parts = key.split(".")
            node = expanded
            for part in parts[:-1]:
                node = node.setdefault(part, {})
            node[parts[-1]] = value
        else:
            expanded[key] = value
    return expanded


def _deep_merge(base: dict, override: dict) -> None:
    """Recursively merge override into base in-place."""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
