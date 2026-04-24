"""Per-edit review gate for manually-triggered dream runs.

Process-local registry keyed by dreamer session_id. When a dream run is
started with `review_required=True`, the runner registers the dreamer's
trace queue here for each conversation. `dream_finalize` consults
`is_active(sid)` and, if set, emits a `dream_finalize_review` event into
that queue and awaits `resolve(sid, decisions)` from the API endpoint.

Resolution sources:
  - POST /v1/dream/review_response (CLI + Discord both POST here)
  - timeout (defaults to drop-all; caller can pass a different default)
  - unregister() — clears without resolving; pending future rolls back

Keep small and allocation-light; one active review per dreamer session.
"""

from __future__ import annotations

import asyncio
import logging

logger = logging.getLogger("dream.review_bus")

_active: dict[str, dict] = {}


def register(sid: str, queue: asyncio.Queue) -> None:
    """Arm a review channel for this dreamer session."""
    _active[sid] = {"queue": queue, "future": None}


def unregister(sid: str) -> None:
    """Clear the review channel. If a future is pending, resolve it with {}
    (drop-all) so `dream_finalize` unblocks cleanly on early exit."""
    slot = _active.pop(sid, None)
    if slot is None:
        return
    fut = slot.get("future")
    if fut is not None and not fut.done():
        fut.set_result({})


def is_active(sid: str) -> bool:
    return sid in _active


async def request_decisions(
    sid: str,
    batch_summary: dict,
    *,
    timeout_s: float = 660.0,
) -> dict[str, str]:
    """Emit dream_finalize_review on the registered queue and await decisions.

    Returns a {phrase_id: "keep"|"drop"} map. On timeout, returns {} — the
    caller should treat missing phrase_ids as drop.
    """
    slot = _active.get(sid)
    if slot is None:
        return {}
    loop = asyncio.get_running_loop()
    fut: asyncio.Future = loop.create_future()
    slot["future"] = fut
    try:
        slot["queue"].put_nowait({
            "event": "dream_finalize_review",
            "data": batch_summary,
        })
    except Exception:
        logger.exception("failed to enqueue dream_finalize_review event for %s", sid)
        return {}
    try:
        return await asyncio.wait_for(fut, timeout=timeout_s)
    except asyncio.TimeoutError:
        try:
            slot["queue"].put_nowait({
                "event": "dream_finalize_review_timeout",
                "data": {"dreamer_sid": sid},
            })
        except Exception:
            pass
        return {}
    finally:
        if slot.get("future") is fut:
            slot["future"] = None


def resolve(sid: str, decisions: dict[str, str]) -> bool:
    """Endpoint-side resolution. Returns True if an awaiter was unblocked."""
    slot = _active.get(sid)
    if slot is None:
        return False
    fut = slot.get("future")
    if fut is None or fut.done():
        return False
    fut.set_result(decisions)
    return True


def active_sids() -> list[str]:
    """Diagnostic helper — list currently-armed dreamer sessions."""
    return list(_active.keys())


# `dream_skip` is now emitted by the runner after run_agent_role returns,
# so it fires regardless of whether review_bus is armed. The runner reads
# the skip rationale off the dreamer's session state (stamped inside
# `dream_finalize`) and pushes the event onto the trace queue itself.
