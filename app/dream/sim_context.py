"""Per-replay simulation context plumbing.

The dream simulator runs the agent under a silent overlay: writes go to an
ephemeral `/cache/dream_sim_overlay/{sim_sid}/fs/` tree, and sandbox handlers
branch on a `_simulate` marker in their params to route I/O through the
overlay. This module owns the ContextVar that carries the per-replay marker
across `await` boundaries.

`app/mcp_client.py:call_tool` reads `current()` just before forwarding a tool
invocation to the sandbox; when set, it injects `_simulate` into `params`.
The overlay root + ephemeral chroma collection name live in the context so
sandbox handlers can address both.
"""

from __future__ import annotations

import contextvars
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SimContext:
    sim_sid: str
    overlay_root: Path       # /cache/dream_sim_overlay/{sim_sid}/fs
    memory_collection: str   # ephemeral chroma collection name (sim_{sim_sid}_memories)


CTX: contextvars.ContextVar[SimContext | None] = contextvars.ContextVar(
    "dream_sim_ctx", default=None
)


def enter(ctx: SimContext) -> contextvars.Token:
    """Bind `ctx` as the active simulation context for the current task."""
    return CTX.set(ctx)


def exit(token: contextvars.Token) -> None:
    """Restore the prior simulation context (or None)."""
    CTX.reset(token)


def current() -> SimContext | None:
    """Return the active simulation context, or None when no sim is in flight."""
    return CTX.get()


def as_sandbox_marker(ctx: SimContext) -> dict:
    """Shape the context as the `_simulate` dict that the sandbox handlers read."""
    return {
        "sim_sid": ctx.sim_sid,
        "overlay_root": str(ctx.overlay_root),
        "memory_collection": ctx.memory_collection,
    }
