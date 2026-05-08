"""Auto-sim state-machine hook for the dreamer role's worker loop.

Wires `dream_state` + `simulator` into `_run_worker` so the dreamer never has
to invoke `simulate_conversation` explicitly. After each worker iteration
(tool call, `<|end|>`, or mid-turn scaffolding), `after_iteration` checks
whether a pending batch is in `submit` phase and the iteration that just
completed did NOT call `dream_submit` / `edit_revise`. If so it runs the
simulator and returns a synthetic tool-result message the worker splices
into context.

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


def _format_sim_body(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)


# Hook signature: `_run_worker` passes the current message list AND an
# explicit per-iteration flag saying whether the iteration that just
# completed executed a revise tool. The flag is the honest signal —
# inferring it from the last `[tool_result: ...]` string fails on the
# prose-without-end branch where the last tool_result is stale (from a
# prior iteration) and gives a false "still revising" reading.
HookFn = Callable[[list[dict], bool], Awaitable[str | None]]


def make_dream_hook(conversation_sid: str, cfg: dict) -> HookFn:
    """Build the `after_iteration_hook` for a dreamer `_run_worker` run.

    Returns an async callable: `hook(full_messages, just_revised)`. When it
    returns a string, the worker splices it in as a synthetic tool_result
    and continues the loop (lets the dreamer react to the sim result).

    When `cfg.dream.auto_sim_enabled` is False (the default), the hook
    short-circuits the simulator entirely: it transitions the pending batch
    to `finalize_only` (so `edit_revise` / `dream_submit` are gated off) and
    returns a synthetic "sim DISABLED" tool_result telling the dreamer to
    proceed directly to `dream_finalize`. This avoids the Opus-stall
    failure mode where a large sim-result payload gets dumped into the
    dreamer's context and the post-sim LLM call never returns.
    """
    auto_sim_enabled = bool(((cfg or {}).get("dream") or {}).get("auto_sim_enabled", False))

    async def hook(full_messages: list[dict], just_revised: bool) -> str | None:
        if not dream_state.has_pending_batch(conversation_sid):
            return None
        try:
            batch = dream_state.load_pending(conversation_sid)
        except dream_state.NoPendingBatch:
            return None

        if not dream_state.should_auto_sim(
            batch, dreamer_just_called_submit_or_revise=just_revised
        ):
            return None

        # Auto-sim disabled → advance state to finalize_only and tell the
        # dreamer to commit directly. No simulator run, no LLM calls, no
        # Opus-stall risk.
        if not auto_sim_enabled:
            # `on_simulation_complete(model_match=False, simulations_cap=0)`
            # unconditionally transitions to finalize_only.
            dream_state.on_simulation_complete(
                batch, model_match=False, simulations_cap=0,
            )
            dream_state.save_pending(batch)
            return (
                "[tool_result: simulate_conversation] DISABLED\n"
                "Auto-simulation is disabled in this deployment "
                "(cfg.dream.auto_sim_enabled=false). You cannot iterate on "
                "simulation feedback; call `dream_finalize(keep, drop)` now "
                "to commit or abandon the staged edits. The user will review "
                "each edit via the per-edit review TUI before anything lands "
                "on disk."
            )

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
