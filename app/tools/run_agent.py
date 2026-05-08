"""
run_agent — spawn a sub-agent to run a task independently.

Creates a child session, runs the agent role, returns the result.
Tracks sub-session in parent state.
"""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from app.sessions.state import SessionState
from app.tools.registry import ToolDef


async def handle(
    params: dict[str, Any],
    session_id: str,
    mode: str,
    state: Any | None,
) -> dict[str, Any]:
    """Spawn a sub-agent to run a task."""
    # Coerce common misspellings silently — weak models reach for `agent_name`
    # because `run_agent` reads naturally with agent_name. Intent is unambiguous.
    if not params.get("role"):
        for alt in ("agent_name", "agent", "sub_agent", "name"):
            if params.get(alt):
                params["role"] = params[alt]
                break

    role = params.get("role", "")
    task = params.get("task", "")
    spawnable = params.get("spawnable_agents") or []

    if not role or not task:
        return {
            "error": (
                f"run_agent needs 'role' (options: {spawnable}) and 'task' (full instruction string). "
                f'Example: {{"role": "{spawnable[0] if spawnable else "coding_agent"}", '
                '"task": "Draft …"}}'
            )
        }

    if role not in spawnable:
        return {"error": f"Cannot spawn '{role}'. Available sub-agents: {spawnable}"}

    child_sid = f"{session_id or 'sub'}_{role}_{uuid4().hex[:6]}"

    # Track sub-session in parent state
    parent_state: SessionState | None = None
    if session_id:
        try:
            parent_state = SessionState.load_or_create(session_id)
            parent_state.add_sub_session(child_sid)
            parent_state.save()
        except Exception:
            parent_state = None

    # Initialize child state
    try:
        child_state = SessionState.load_or_create(child_sid)
        child_state.set("parent_session_id", session_id or None)
        child_state.set("agent_role", role)
        child_state.set("source_trigger", {"type": "sub_agent", "ref": session_id or None})
        child_state.save()
    except Exception:
        pass

    try:
        # Lazy import to avoid circular dependency
        from app.agents import run_agent_role

        result = await run_agent_role(
            role,
            {"messages": [{"role": "user", "content": task}]},
            child_sid,
        )
        text = result.get("choices", [{}])[0].get("message", {}).get("content", "")
        return {"role": role, "response": text, "sub_session_id": child_sid}
    except Exception as e:
        return {"error": f"Sub-agent '{role}' failed: {e}"}
    finally:
        if parent_state is not None:
            try:
                parent_state.complete_sub_session(child_sid)
                parent_state.save()
            except Exception:
                pass


TOOL = ToolDef(
    name="run_agent",
    category="sub_agent",
    description="Spawn a sub-agent to run a task independently.",
    slow=True,
    handler=handle,
)
