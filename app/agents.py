"""
Supervisor / worker orchestration — facade re-exporting names from the split
submodules. See:
  app.llm          — provider I/O (_llm_call_*, _content, Anthropic client)
  app.mode         — plan/build/converse helpers
  app.supervisor   — modality, rubric, threshold, _run_supervisor
  app.worker       — inner tool loop (_run_worker, _short_params, END_MARKER)
  app.loop         — run_agent_loop + session-context helpers
  app.entrypoints  — run_soul_update, run_config_agent, run_agent_role
"""

from app.entrypoints import (
    run_agent_role,
    run_config_agent,
    run_soul_update,
)
from app.llm import (
    LLAMA_URL,
    _anthropic_client,
    _content,
    _extract_logprobs,
    _get_anthropic_client,
    _llm_call,
    _llm_call_anthropic,
    _llm_call_local,
    _llm_call_openai_compat,
    _OPENAI_COMPAT,
    _read_oauth_token,
)
from app.loop import (
    _auto_store_memory,
    _format_response,
    _rebuild_session_context,
    run_agent_loop,
)
from app.mode import (
    _MODE_SHORT,
    _RESEARCH_RULE,
    _mode_context_string,
    _mode_temperature,
    _mode_tools,
)
from app.supervisor import (
    _build_supervisor_rubric,
    _classify_worker_modality,
    _effective_threshold,
    _run_supervisor,
)
from app.worker import (
    END_MARKER,
    _SHELL_TOOLS,
    _TOOL_CALL_HINTS,
    _promote_shell_error,
    _run_worker,
    _short_params,
    _split_peer_review,
)

__all__ = [
    "END_MARKER",
    "LLAMA_URL",
    "run_agent_loop",
    "run_agent_role",
    "run_config_agent",
    "run_soul_update",
    "_build_supervisor_rubric",
    "_classify_worker_modality",
    "_effective_threshold",
    "_mode_context_string",
    "_mode_temperature",
    "_mode_tools",
    "_run_supervisor",
    "_run_worker",
    "_short_params",
    "_split_peer_review",
]
