"""
ask_user — present multi-choice options to the user via Discord.

Blocks until the user responds. Returns the selected option.
"""

from __future__ import annotations

from typing import Any

from app.ask_user import ask_user_question
from app.tools.registry import ToolDef


async def handle(
    params: dict[str, Any],
    session_id: str,
    mode: str,
    state: Any | None,
) -> dict[str, Any]:
    """Ask the user a multi-choice question."""
    return await ask_user_question(
        question=params.get("question", ""),
        options=params.get("options", []),
        context=params.get("context", ""),
        session_id=session_id,
    )


TOOL = ToolDef(
    name="ask_user",
    category="sub_agent",
    description="Present multi-choice options to the user via Discord.",
    slow=True,
    handler=handle,
)
