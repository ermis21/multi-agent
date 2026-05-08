"""
Tool package — local tool handlers.

Bootstraps the ToolRegistry at import time.
"""

import os

from .registry import ToolRegistry, registry


def _bootstrap() -> None:
    """Initialize the global registry singleton."""
    global registry
    if registry is not None:
        return

    discord_notify_url = os.environ.get(
        "PHOEBE_DISCORD_NOTIFY_URL",
        os.environ.get("PHOEBE_DISCORD_URL", "http://phoebe-discord:4000") + "/discord/send",
    )

    registry = ToolRegistry(
        scan_packages=["app.tools", "app.dream"],
        discord_notify_url=discord_notify_url,
    )


_bootstrap()

__all__ = ["registry"]
