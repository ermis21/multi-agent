"""
deliberate — run a structured debate between two positions.

Uses the debate module to run advocates and a judge.
Stores debate_id in session state for continuation.
"""

from __future__ import annotations

from typing import Any

from app.sessions.state import SessionState
from app.tools.registry import ToolDef


async def handle(
    params: dict[str, Any],
    session_id: str,
    mode: str,
    state: Any | None,
) -> dict[str, Any]:
    """Run a structured debate."""
    from app.debate import run_debate

    result = await run_debate(
        question=params.get("question", ""),
        context=params.get("context", ""),
        position_a=params.get("position_a", ""),
        position_b=params.get("position_b", ""),
        session_id=session_id,
        debate_id=params.get("debate_id", ""),
        max_exchanges=params.get("max_exchanges", 0),
    )

    # Store debate_id in session state for continuation
    if session_id and isinstance(result, dict) and result.get("debate_id"):
        try:
            dst = SessionState.load_or_create(session_id)
            dst.set("debate_id", result["debate_id"])
            dst.save()
        except Exception:
            pass

    return result


TOOL = ToolDef(
    name="deliberate",
    category="sub_agent",
    description="Run a structured debate between two positions.",
    slow=True,
    handler=handle,
)
