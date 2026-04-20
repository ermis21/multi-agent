"""
Supervisor grading — modality classification, rubric selection, threshold resolution,
and the adversarial JSON-scoring LLM call.

`_run_supervisor` returns a structured verdict with issue arrays; the worker
loop in app.loop injects those arrays as retry feedback.
"""

import json

from app.llm import _content, _llm_call
from app.mcp_client import strip_json_fences


def _classify_worker_modality(tool_traces: list[dict]) -> tuple[str, float, int]:
    """Classify the worker turn by its tool-use profile.

    Returns (modality, error_rate, tool_count).
    modality ∈ {"no_tool", "tool_light", "tool_heavy"} with an optional
    "_with_errors" suffix when at least 30% of tool calls errored.
    """
    n = len(tool_traces)
    errs = sum(1 for t in tool_traces if t.get("error"))
    rate = (errs / n) if n else 0.0
    if n == 0:
        base = "no_tool"
    elif n < 3:
        base = "tool_light"
    else:
        base = "tool_heavy"
    if n and rate >= 0.3:
        base += "_with_errors"
    return base, rate, n


def _build_supervisor_rubric(modality: str, mode: str) -> str:
    """Build the grading rubric text for the given mode+modality combination.

    Emits a short Markdown block listing only the dimensions that apply to this
    turn. The supervisor is told to score against these dimensions and ignore
    others. Each rubric ends with the same issue-array mapping so the worker's
    retry feedback stays structured.
    """
    has_errors = modality.endswith("_with_errors")
    base = modality.replace("_with_errors", "")

    # converse + no tools: conversational coherence only
    if mode == "converse" and base == "no_tool":
        body = (
            "This was a conversational reply with no tools used. Grade **only**:\n"
            "- **Coherence** — does the answer address what the user asked?\n"
            "- **Accuracy** — are any factual claims reasonable given the conversation context?\n\n"
            "**Do NOT** score down for 'lack of research', 'no tool calls', or 'unverified claims' — "
            "the user's message did not require investigation. A short correct answer scores ≥ 0.8.\n"
            "Retire: tool_issues, source_gaps, research_gaps. Leave those arrays empty."
        )
    # plan mode: specificity is everything
    elif mode == "plan":
        body = (
            "This is a plan-mode response. Grade on:\n"
            "- **Specificity** — concrete file paths, function names, line numbers? Vague language "
            "('update the relevant module', 'refactor as needed') is a hard fail.\n"
            "- **Feasibility** — does the plan actually solve the user's ask?\n"
            "- **Tool grounding** — did the worker read the real files before proposing changes, "
            "or did it guess? Zero reads for a non-trivial plan is a source_gap.\n"
            "- **Scope discipline** — plans that sprawl beyond the ask should lose points.\n\n"
            "Populate: tool_issues, source_gaps (missing reads), research_gaps, completeness_issues."
        )
    # build mode: did the tools move us toward the goal?
    elif mode == "build":
        if has_errors:
            body = (
                "This is a build-mode response with tool errors in the trace. Grade on:\n"
                "- **Error handling** — did the worker recover from each error, or did it give up / "
                "ignore them? Unhandled errors = accuracy_issues. Errors that were diagnosed and "
                "worked around are **fine** — do not penalise the existence of errors.\n"
                "- **Goal progress** — did the executed tools (successful or not) advance the task?\n"
                "- **Completeness** — is the user's ask addressed at the end?\n\n"
                "Do NOT count error presence as a tool_issue. Count only missing recovery."
            )
        else:
            body = (
                "This is a build-mode response. Grade on:\n"
                "- **Goal alignment** — did the tools executed actually move toward the user's goal, "
                "or was the worker spelunking?\n"
                "- **Completeness** — is the asked change fully applied (file writes, commits, tests)?\n"
                "- **Accuracy** — do the claimed outcomes match the tool results?\n\n"
                "Populate: tool_issues (wrong tool for the job), accuracy_issues, completeness_issues."
            )
    # light tool use in any mode: don't demand more
    elif base == "tool_light":
        body = (
            "The worker used a small number of tools. Grade on whether those tools were "
            "**sufficient for the specific ask**. Do NOT demand more tools unless the answer is "
            "missing a concrete detail the user explicitly requested.\n"
            "- **Answer quality** — does it address the ask?\n"
            "- **Tool fit** — were the tools chosen appropriate?\n\n"
            "Retire source_gaps unless a specific claim in the answer is unsubstantiated."
        )
    # tool_heavy in converse: still weigh the answer, not the process
    elif mode == "converse" and base == "tool_heavy":
        body = (
            "The worker used many tools for a conversational ask. Grade on:\n"
            "- **Answer quality** — does the final reply actually answer the user?\n"
            "- **Accuracy** — are claims grounded in the tool output?\n"
            "- **Efficiency** — excessive tool use is worth flagging as a tool_issue, but not a hard fail."
        )
    # default fallback — original full rubric
    else:
        body = (
            "Grade on:\n"
            "1. **Tool Usage** — right tools? enough tools? verified before concluding?\n"
            "2. **Source Verification** — claims backed by actual tool output? any fabricated details?\n"
            "3. **Research Thoroughness** — enough investigation? cross-referenced where needed?\n"
            "4. **Factual Accuracy** — conclusions correct given the evidence gathered?\n"
            "5. **Completeness** — every part of the ask addressed?"
        )

    return body


def _effective_threshold(cfg: dict, mode: str) -> float:
    """Resolve the supervisor pass threshold for the given mode."""
    agent_cfg = cfg.get("agent", {})
    overrides = agent_cfg.get("supervisor_mode_overrides", {}) or {}
    if mode in overrides:
        try:
            return float(overrides[mode])
        except (TypeError, ValueError):
            pass
    return float(agent_cfg.get("supervisor_pass_threshold", 0.7))


async def _run_supervisor(
    worker_response:    str,
    original_messages:  list[dict],
    system_prompt:      str,
    cfg:                dict,
    include_history:    bool,
    role_cfg:           dict | None = None,
) -> dict:
    """
    Grade the worker response — process audit with structured issue arrays.
    Returns dict with pass, score, feedback, issue arrays, alternative, suggest_spawn, suggest_debate.
    Falls back to pass=True on JSON parse errors to avoid infinite error loops.
    """
    context_messages = original_messages if include_history else []
    messages = (
        [{"role": "system", "content": system_prompt}]
        + context_messages
        + [{"role": "assistant", "content": worker_response}]
        + [{"role": "user", "content": (
            "Audit the worker's process. Focus on: Did it use the right tools? "
            "Did it verify claims with actual data? Did it investigate enough? "
            "Respond ONLY with JSON."
        )}]
    )

    _fallback = {
        "pass": True, "score": 0.5, "feedback": "supervisor parse error — treating as pass",
        "alternative": "", "suggest_spawn": "", "suggest_debate": "",
        "tool_issues": [], "source_gaps": [], "research_gaps": [],
        "accuracy_issues": [], "completeness_issues": [],
    }

    try:
        resp   = await _llm_call(messages, cfg, role_cfg=role_cfg)
        raw    = strip_json_fences(_content(resp))
        result = json.loads(raw)
        for k in ("pass", "score", "feedback"):
            if k not in result:
                raise ValueError(f"supervisor missing key: {k}")
        result["score"] = float(result["score"])
        # Default optional keys
        result.setdefault("alternative", "")
        result.setdefault("suggest_spawn", "")
        result.setdefault("suggest_debate", "")
        result.setdefault("tool_issues", [])
        result.setdefault("source_gaps", [])
        result.setdefault("research_gaps", [])
        result.setdefault("accuracy_issues", [])
        result.setdefault("completeness_issues", [])
        return result
    except Exception:
        return _fallback
