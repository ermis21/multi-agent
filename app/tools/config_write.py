"""
write_config — patch the configuration.

Deep-merges the provided params into the current config.
Validates via config_schema. Invalidates mtime cache.
"""

from __future__ import annotations

from typing import Any

from app.config_loader import patch_config
from app.config_schema import ConfigPatchError
from app.tools.registry import ToolDef


async def handle(
    params: dict[str, Any],
    session_id: str,
    mode: str,
    state: Any | None,
) -> dict[str, Any]:
    """Patch the configuration with the provided params."""
    try:
        return {"updated": True, "config": patch_config(params)}
    except ConfigPatchError as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": f"Config write failed: {e}"}


TOOL = ToolDef(
    name="write_config",
    category="config",
    description="Patch the Phoebe configuration (deep-merge).",
    slow=False,
    handler=handle,
)
