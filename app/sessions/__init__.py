"""Session subsystem — per-session state, turn log, and legacy migration.

Public API is re-exported here so callers only import from `app.sessions`.
See individual submodules for the implementation:
  app.sessions.logger   — SessionLogger, turns.jsonl I/O
  app.sessions.state    — SessionState, TurnAccumulator, sidecar audit logs
  app.sessions.migrate  — flat → folder layout migration
"""

from app.sessions.logger import (
    SESSIONS_DIR,
    SessionLogger,
    get_session,
    list_sessions,
)
from app.sessions.migrate import migrate_flat_to_folders
from app.sessions.state import (
    SessionState,
    TurnAccumulator,
    log_approval,
    log_tool_error,
)

__all__ = [
    "SESSIONS_DIR",
    "SessionLogger",
    "SessionState",
    "TurnAccumulator",
    "get_session",
    "list_sessions",
    "log_approval",
    "log_tool_error",
    "migrate_flat_to_folders",
]
