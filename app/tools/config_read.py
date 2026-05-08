"""
read_config — return the current configuration.

Returns the full config dict as read by config_loader.
"""

from __future__ import annotations

from typing import Any

from app.config_loader import get_config
from app.tools.registry import ToolDef


async def handle(
    params: dict[str, Any],
    session_id: str,
    mode: str,
    state: Any | None,
) -> dict[str, Any]:
    """Return the current configuration."""
    return {"config": get_config()}


TOOL = ToolDef(
    name="read_config",
    category="config",
    description="Read the current Phoebe configuration.",
    slow=False,
    handler=handle,
)
