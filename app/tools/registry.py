"""
Tool registry — auto-discovers and registers local tool handlers.

Scans configured packages for modules exporting TOOL or TOOL_* constants.
Each constant must be a ToolDef with a callable `handler` field.

Usage:
    from app.tools.registry import registry

    # Dispatch a tool call
    result = await registry.dispatch("read_config", params, session_id, mode, state)

    # Derive LOCAL_TOOLS and SLOW_TOOLS
    local_names = registry.local_tool_names()  # frozenset[str]
    slow_names = registry.slow_tool_names()    # frozenset[str]
"""

from __future__ import annotations

import importlib
import logging
import pkgutil
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

import httpx

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ToolDef:
    """Metadata for a single local tool."""
    name: str
    category: str       # "config" | "sub_agent" | "dream" | "diagnostic"
    description: str    # one-line, for registry listing
    slow: bool = False  # per-tool, maps to SLOW_TOOLS classification
    handler: Callable[..., Awaitable[dict[str, Any]]] | None = None


class ToolRegistry:
    """Auto-discovers and manages local tool handlers.

    Scans packages listed in `scan_packages` at construction time.
    Each module exporting a TOOL or TOOL_* constant (ToolDef instance)
    is registered automatically.
    """

    def __init__(
        self,
        scan_packages: list[str] | None = None,
        discord_notify_url: str | None = None,
    ):
        self._tools: dict[str, ToolDef] = {}
        self._scan_packages = scan_packages or [
            "app.tools",
            "app.dream",
        ]
        self._discord_notify_url = discord_notify_url
        self._discover()

    def _discover(self) -> None:
        """Scan packages for tool modules and register them."""
        failures: list[str] = []

        for package_name in self._scan_packages:
            try:
                package = importlib.import_module(package_name)
            except ImportError as e:
                logger.error("Failed to import tool package %s: %s", package_name, e)
                failures.append(f"Package {package_name}: {e}")
                continue

            if not hasattr(package, "__path__"):
                logger.warning("Package %s has no __path__, skipping", package_name)
                continue

            for _, module_name, _ in pkgutil.iter_modules(package.__path__):
                module_path = f"{package_name}.{module_name}"
                try:
                    module = importlib.import_module(module_path)
                except Exception as e:
                    logger.error("Failed to import tool module %s: %s", module_path, e)
                    failures.append(f"Module {module_path}: {e}")
                    continue

                # Find all TOOL or TOOL_* exports
                for attr_name in dir(module):
                    if not attr_name.startswith("TOOL"):
                        continue
                    attr = getattr(module, attr_name, None)
                    if not isinstance(attr, ToolDef):
                        continue

                    if attr.handler is None:
                        logger.warning(
                            "Tool %s in %s has no handler, skipping",
                            attr.name,
                            module_path,
                        )
                        continue

                    self._tools[attr.name] = attr
                    logger.debug("Registered tool: %s (category=%s)", attr.name, attr.category)

        # Notify Discord of any failures
        if failures:
            self._notify_failures(failures)

    def _notify_failures(self, failures: list[str]) -> None:
        """Send failure report to Discord with retries."""
        if not self._discord_notify_url:
            logger.warning("Tool load failures (no Discord URL configured): %s", failures)
            return

        import asyncio

        async def _send() -> None:
            report = (
                "⚠️ **Phoebe Tool Load Failures**\n\n"
                + "\n".join(f"- {f}" for f in failures)
            )
            async with httpx.AsyncClient(timeout=10) as client:
                for attempt in range(1, 4):
                    try:
                        r = await client.post(
                            self._discord_notify_url,
                            json={"content": report},
                            timeout=5,
                        )
                        if r.status_code < 400:
                            logger.info("Sent tool failure report to Discord")
                            return
                        logger.warning(
                            "Discord notification failed (attempt %d): %s",
                            attempt,
                            r.status_code,
                        )
                    except Exception as e:
                        logger.warning(
                            "Discord notification failed (attempt %d): %s",
                            attempt,
                            e,
                        )
                    if attempt < 3:
                        await asyncio.sleep(2)
                logger.error("Failed to send tool failure report after 3 retries")

        try:
            loop = asyncio.get_running_loop()
            loop.create_task(_send())
        except RuntimeError:
            # No running loop (startup edge case) — log only
            logger.error("Tool load failures (no event loop): %s", failures)

    def get(self, name: str) -> ToolDef | None:
        """Return ToolDef by name, or None if not found."""
        return self._tools.get(name)

    async def dispatch(
        self,
        name: str,
        params: dict[str, Any],
        session_id: str,
        mode: str,
        state: Any | None,
    ) -> dict[str, Any] | None:
        """Dispatch a tool call to its handler.

        Returns the handler's result dict, or None if the tool is not registered.
        Wraps handler call in uniform error boundary.
        """
        tool_def = self._tools.get(name)
        if tool_def is None:
            return None

        try:
            return await tool_def.handler(params, session_id, mode, state)
        except Exception as e:
            logger.error("Tool %s failed: %s", name, e, exc_info=True)
            return {"error": f"{name} failed: {e}"}

    def local_tool_names(self) -> frozenset[str]:
        """Return frozenset of all registered local tool names."""
        return frozenset(self._tools.keys())

    def slow_tool_names(self) -> frozenset[str]:
        """Return frozenset of slow tool names."""
        return frozenset(
            name for name, tool in self._tools.items() if tool.slow
        )

    def is_dream_tool(self, name: str) -> bool:
        """Check if a tool belongs to the dream category."""
        tool = self._tools.get(name)
        return tool is not None and tool.category == "dream"

    def __repr__(self) -> str:
        return f"<ToolRegistry tools={len(self._tools)}>"


# Singleton instance — created by app/tools/__init__.py
registry: ToolRegistry | None = None
