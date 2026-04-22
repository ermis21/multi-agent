"""Auto-sim state-machine hook for the dreamer role's worker loop.

Wires `dream_state` + `simulator` into `_run_worker` so the dreamer never has
to invoke `simulate_conversation` explicitly. After each worker iteration
(tool call, `<|end|>`, or mid-turn scaffolding), `after_iteration` checks
whether a pending batch is in `submit` phase and the latest LLM response did
NOT call `dream_submit` / `edit_revise`. If so it runs the simulator and
returns a synthetic tool-result message the worker splices into context.

The runner also rolls back any pending batch if the dreamer exits without
finalizing — see `rollback_if_unfinalized`.
"""

from __future__ import annotations

import json
from typing import Any, Awaitable, Callable

from app.dream import dream_state, simulator

# Tool names that count as "still revising" — no auto-sim fires in the turn
# immediately following them.
_REVISE_TOOLS = frozenset({"dream_submit", "edit_revise"})


def _last_tool_name(full_messages: list[dict]) -> str | None:
    """Parse the tool name from the most recent `[tool_result: NAME]` user msg."""
    for msg in reversed(full_messages):
        if msg.get("role") != "user":
            continue
        content = msg.get("content", "") or ""
        if content.startswith("[tool_result: "):
            head = content.split("\n", 1)[0]
            # "[tool_result: NAME] OK" or "[tool_result: NAME] ERROR"
            try:
                name = head.split("[tool_result: ", 1)[1].split("]", 1)[0]
                return name.strip() or None
            except IndexError:
                return None
    return None


def _format_sim_body(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)


HookFn = Callable[[list[dict]], Awaitable[str | None]]


def make_dream_hook(conversation_sid: str, cfg: dict) -> HookFn:
    """Build the `after_iteration_hook` for a dreamer `_run_worker` run.

    Returns an async callable that, given the current `full_messages` list,
    decides whether to fire auto-sim. When it does, it returns the synthetic
    tool-result string the worker should splice into context. Otherwise None.
    """
    async def hook(full_messages: list[dict]) -> str | None:
        if not dream_state.has_pending_batch(conversation_sid):
            return None
        try:
            batch = dream_state.load_pending(conversation_sid)
        except dream_state.NoPendingBatch:
            return None

        last = _last_tool_name(full_messages)
        just_revised = last in _REVISE_TOOLS
        if not dream_state.should_auto_sim(
            batch, dreamer_just_called_submit_or_revise=just_revised
        ):
            return None

        try:
            result = await simulator.run_simulation(conversation_sid, cfg)
            payload = result.to_payload()
            return (
                "[tool_result: simulate_conversation] OK\n"
                f"{_format_sim_body(payload)}"
            )
        except simulator.SimulatorError as e:
            return (
                "[tool_result: simulate_conversation] ERROR\n"
                f"{e}\n\n"
                "The sim could not run. End your turn with dream_finalize."
            )
        except Exception as e:  # pragma: no cover — last-ditch safety
            return (
                "[tool_result: simulate_conversation] ERROR\n"
                f"{type(e).__name__}: {e}\n\n"
                "Proceed to dream_finalize."
            )

    return hook


def rollback_if_unfinalized(conversation_sid: str) -> bool:
    """Drop any still-pending batch for `conversation_sid`.

    Called in the dreamer runner's `finally` when the agent exits without
    calling `dream_finalize`. Returns True if a batch was dropped.
    """
    try:
        return dream_state.delete_pending(conversation_sid)
    except Exception:
        return False
