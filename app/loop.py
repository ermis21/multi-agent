"""
Main supervisor/worker orchestration loop.

`run_agent_loop` is the handler behind POST /v1/chat/completions: it loads
session state, rebuilds conversation context, runs the worker, grades with
the supervisor, injects structured feedback, retries up to `max_retries`,
and flushes per-turn state on exit.
"""

import asyncio
import json
from pathlib import Path

import httpx

from app.config_loader import get_agents_config, get_config
from app.mcp_client import _session_plan_path, call_tool
from app.llm import _content  # noqa: F401 — kept for potential downstream use
from app.mode import _mode_context_string, _mode_temperature, _mode_tools
from app.prompt_generator import cleanup_generated, generate
from app.sessions.logger import SessionLogger, get_session
from app.sessions.state import SessionState, TurnAccumulator, log_supervisor_override
from app.supervisor import (
    _build_supervisor_rubric,
    _classify_worker_modality,
    _detect_hallucinated_zero_tool_claim,
    _effective_threshold,
    _run_supervisor,
)
from app.worker import _run_worker, _split_peer_review


def _rebuild_session_context(session_id: str, raw_input: list[dict], cfg: dict) -> list[dict]:
    """Reconstruct conversation history from prior "final" turns only.

    "final" turns hold the raw user input + the clean response, so replaying
    just those gives a stack without supervisor retry noise. Windowed by
    max_context_turns, then trimmed by max_context_messages (always starting
    on a user turn).

    Prefers `state.history.active` (compacted view written by session_compactor)
    if it exists; otherwise falls back to the full session JSONL.
    """
    max_ctx = cfg.get("agent", {}).get("max_context_turns", 20)

    turns: list[dict] | None = None
    try:
        _st = SessionState.load_or_create(session_id)
        active_path = _st.get("history.active")
        if active_path:
            p = Path(active_path if Path(active_path).is_absolute() else f"/{active_path}")
            if p.exists():
                turns = []
                for line in p.read_text(encoding="utf-8").strip().splitlines():
                    try:
                        turns.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    except Exception:
        turns = None

    if turns is None:
        turns = get_session(session_id)
    final_turns = [t for t in turns if t.get("role") == "final"]
    if not final_turns:
        return raw_input
    ctx: list[dict] = []
    for turn in final_turns[-max_ctx:]:
        ctx.extend(turn.get("messages", []))
        ctx.append({"role": "assistant", "content": turn.get("response", "")})
    max_msgs = cfg.get("agent", {}).get("max_context_messages", 0)
    if max_msgs and len(ctx) > max_msgs:
        ctx = ctx[-max_msgs:]
        while ctx and ctx[0]["role"] != "user":
            ctx = ctx[1:]
    return ctx + raw_input


async def _auto_store_memory(
    raw_input: list[dict],
    response: str,
    session_id: str,
    mode: str,
    allowed_tools: list[str],
) -> None:
    """
    Fire-and-forget: store a brief turn summary in MemPalace so the agent can
    search conversation history across sessions via memory_search.
    Only runs when memory_add is in the agent's allowed tool list.
    """
    if "memory_add" not in allowed_tools:
        return
    try:
        user_text = " ".join(
            m.get("content", "") for m in raw_input if m.get("role") == "user"
        )[:300]
        summary = (
            f"Session turn [{session_id}]:\n"
            f"User: {user_text}\n"
            f"Agent: {response[:500]}"
        )
        await call_tool(
            "memory_add",
            {"content": summary, "tags": ["session_history", session_id, mode]},
            ["memory_add"],
            mode="converse",
            approved_tools=["memory_add"],
        )
    except Exception:
        pass  # non-fatal; never block the response


def _format_response(content: str, session_id: str) -> dict:
    """Wrap a final answer in OpenAI chat completion format."""
    return {
        "id":      f"phoebe-{session_id}",
        "object":  "chat.completion",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
        "session_id": session_id,
    }


async def run_agent_loop(body: dict, session_id: str, trace_queue: asyncio.Queue | None = None,
                          session_state: dict | None = None) -> dict:
    """
    Supervisor / worker loop for a single chat completion request.

    Flow:
      1. Generate worker prompt (dynamic, role=worker)
      2. Worker produces response (with inner tool loop)
      3. If supervisor disabled → return immediately
      4. Generate supervisor prompt (dynamic, role=supervisor)
      5. Supervisor grades; if pass → return
      6. Inject feedback into messages; retry up to max_retries times
      7. Return best-scored response after exhausting retries
    """
    cfg        = get_config()
    agents_cfg = get_agents_config()
    logger     = SessionLogger(session_id)
    raw_input  = body.get("messages", [])  # unmodified user input — stored in "final" turns

    # Mode — sent by Discord bot per user preference, defaulting to config default
    mode_cfg = cfg.get("agent", {}).get("mode", {})
    mode     = body.get("mode", mode_cfg.get("default", "converse"))

    # Persistent per-session state (mode, model, overrides, stats, verdict, ...)
    state = SessionState.load_or_create(session_id)
    state.set("mode", mode)
    if body.get("model"):
        state.set("model", body.get("model"))
    else:
        # No explicit per-request override → snapshot the config default so
        # retroactive tools (the dreamer) can show "which model handled this
        # session". Last-writer-wins if the default changes mid-session; that
        # is acceptable — the dreamer wants a rough attribution, not a ledger.
        _default_model = (cfg.get("llm") or {}).get("model")
        if _default_model and not state.get("model"):
            state.set("model", _default_model)
    if body.get("channel_id") is not None:
        state.set("channel_id", body.get("channel_id"))
    if body.get("user_id") is not None:
        state.set("user_id", body.get("user_id"))
    if body.get("_source_trigger") is not None and state.get("source_trigger") == {"type": "user", "ref": None}:
        # Only overwrite the default; never clobber a previously-set trigger.
        state.set("source_trigger", body.get("_source_trigger"))

    # Register the Discord user message that seeded this turn, so a later
    # /rewind (triggered by native Discord message-edit) can locate the turn
    # index by discord_msg_id. Idempotent; safe across retries. turn_count has
    # not yet been incremented for this turn — flush_turn() in the finally
    # block does that — so it equals the ordinal of the final we are about to
    # produce.
    _disc_msg_id = body.get("discord_msg_id")
    _chan_id = body.get("channel_id")
    if _disc_msg_id:
        try:
            state.append_user_msg(
                turn_index=int(state.get("stats.turn_count", 0) or 0),
                discord_msg_id=str(_disc_msg_id),
                channel_id=str(_chan_id) if _chan_id is not None else None,
            )
        except Exception as e:
            print(f"[message_index] append_user_msg failed for {session_id}: {e}", flush=True)

    turn_acc = TurnAccumulator()

    # Plan/approval/privileged-path context — state is authoritative, body overlays.
    # Rationale: restarts (api or Discord bot) must not forget user-granted privileges.
    _state_plan        = state.get("plan") or {}
    _state_plan_ctx    = _state_plan.get("context") if isinstance(_state_plan, dict) else None
    _state_approved    = state.get("permissions.approved_tools") or []
    _state_privileged  = state.get("permissions.privileged_paths") or []

    plan_context = body.get("plan_context") or _state_plan_ctx or ""
    privileged_paths = list({*(_state_privileged or []), *(body.get("privileged_paths") or [])})
    if mode == "build" and not plan_context:
        plan_context = "[No active plan. Accept single-action requests. For multi-step tasks, suggest /mode plan.]"

    # Plan mode writes go only to this session-scoped artifact (enforced in
    # call_tool). Persist on state so restarts / replay can locate it.
    plan_file = _session_plan_path(session_id) if mode == "plan" else None
    if plan_file:
        try:
            state.set("plan.path", plan_file)
        except Exception:
            pass

    # Persist any new privileged paths granted by this request back onto state.
    if body.get("privileged_paths"):
        state.set("permissions.privileged_paths", privileged_paths)

    messages = _rebuild_session_context(session_id, raw_input, cfg)

    w_cfg  = dict(agents_cfg.get("worker", {}))  # shallow copy — may override model per-session
    s_cfg  = agents_cfg.get("supervisor", {})
    _state_model = state.get("model")
    if _state_model:
        w_cfg["model"] = _state_model
    w_tools        = _mode_tools(cfg, mode, w_cfg.get("allowed_tools", []))
    w_max_iter     = w_cfg.get("max_tool_iterations", 10)
    # Per-session supervisor overrides fall back to config when null
    sup_state      = state.get("supervisor", {}) or {}
    max_retries    = sup_state.get("max_retries") if sup_state.get("max_retries") is not None \
                     else cfg["agent"]["max_retries"]
    sup_enabled    = sup_state.get("enabled") if sup_state.get("enabled") is not None \
                     else cfg["agent"]["supervisor_enabled"]
    temperature    = _mode_temperature(cfg, mode)
    approved_tools = list({*(_state_approved or []), *(body.get("approved_tools") or [])})
    # `call_tool` appends to this list when the user clicks "Always" — mirror those
    # additions back to state so the next request sees them without re-approval.
    if body.get("approved_tools") or _state_approved:
        state.set("permissions.approved_tools", approved_tools)

    best_response = ""
    best_score    = -1.0
    tool_traces: list[dict] = []
    effective_threshold = (
        float(sup_state["threshold"])
        if sup_state.get("threshold") is not None
        else _effective_threshold(cfg, mode)
    )

    total_attempts = 1 + max_retries if sup_enabled else 1

    verbose_tools = cfg["logging"].get("verbose_tools", False)

    # Gate tool-trace streaming on verbose_tools; the done sentinel always uses trace_queue directly.
    _trace_q = trace_queue if (trace_queue is not None and verbose_tools) else None

    def _with_traces(resp: dict) -> dict:
        if verbose_tools and tool_traces:
            resp["tool_trace"] = tool_traces
        return resp

    inflection_mode = cfg.get("agent", {}).get("inflection_mode", "none")

    result_holder: dict | None = None
    last_verdict: dict | None = None
    try:
        for attempt in range(total_attempts):
            # Build supervisor handler — only injected on retries
            supervisor_handler = ""
            if attempt > 0:
                supervisor_handler = (
                    "\n\n## Handling Supervisor Feedback\n\n"
                    "You are receiving feedback from an adversarial process auditor. "
                    "This auditor is deliberately aggressive and skeptical — that is its job. "
                    "It focuses on your tool usage, source verification, and research thoroughness.\n\n"
                    "**Do NOT:**\n"
                    "- Accept every criticism uncritically — the auditor is sometimes wrong\n"
                    "- Rewrite your entire response because of one valid point\n"
                    "- Become defensive or dismissive\n\n"
                    "**DO:**\n"
                    "- Evaluate each critique point on its merits\n"
                    "- If a TOOL_ISSUE or SOURCE_GAP is valid, make the missing tool call NOW\n"
                    "- If a critique is unreasonable, explain why in one sentence and move on\n"
                    "- Accept valid points, reject bad ones, and provide a focused revision\n\n"
                    "Start your revision with:\n"
                    "ACCEPTED: <points you're incorporating>\n"
                    "REJECTED: <points you're pushing back on, with reason>\n"
                    "Then your revised answer."
                )

            worker_prompt, _ = generate(
                role="worker",
                allowed_tools=w_tools,
                session_id=session_id,
                attempt=attempt,
                agent_mode=mode,
                extra={
                    "{{AGENT_MODE}}": _mode_context_string(mode, cfg=cfg, role_cfg=w_cfg, plan_file=plan_file),
                    "{{PLAN_CONTEXT}}": plan_context,
                    "{{SUPERVISOR_HANDLER}}": supervisor_handler,
                },
            )

            if trace_queue is not None and attempt > 0:
                trace_queue.put_nowait({"event": "retry", "data": {"attempt": attempt}})

            try:
                worker_response, _, tool_traces = await _run_worker(
                    messages, worker_prompt, w_tools, w_max_iter, cfg, temperature, mode, approved_tools,
                    role_cfg=w_cfg,
                    session_id=session_id,
                    trace_queue=_trace_q,
                    extra_auto_allow_paths=privileged_paths or None,
                    inflection_mode=inflection_mode,
                    session_state=session_state,
                    turn_acc=turn_acc,
                )
            except httpx.HTTPError as e:
                worker_response = f"[LLM error: {e}]"

            # User-initiated stop: bail out of the supervisor retry loop too.
            if session_state is not None and session_state.get("cancel") is not None \
               and session_state["cancel"].is_set():
                clean_body, _ = _split_peer_review(worker_response)
                logger.log_turn(0, "final", raw_input, clean_body)
                if trace_queue is not None:
                    trace_queue.put_nowait({"event": "stopped", "data": {"session_id": session_id}})
                result_holder = _with_traces({
                    "id": f"chatcmpl-{session_id}",
                    "object": "chat.completion",
                    "model": cfg["llm"]["model"],
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": clean_body},
                                  "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                    "session_id": session_id, "stopped": True,
                })
                return result_holder

            if cfg["logging"]["log_supervisor_turns"]:
                logger.log_turn(attempt, "worker", messages, worker_response)

            if not sup_enabled:
                clean_body, review_block = _split_peer_review(worker_response)
                if review_block and trace_queue is not None:
                    trace_queue.put_nowait({"event": "worker_review", "data": {"review": review_block}})
                logger.log_turn(0, "final", raw_input, clean_body)
                await _auto_store_memory(raw_input, clean_body, session_id, mode, w_tools)
                result_holder = _with_traces(_format_response(clean_body, session_id))
                return result_holder

            # Format tool traces for supervisor visibility
            if tool_traces:
                trace_lines = []
                for ti, t in enumerate(tool_traces, 1):
                    status = f"ERROR: {t['error']}" if t.get("error") else f"OK ({t['lines']} lines)"
                    trace_lines.append(f"{ti}. {t['tool']} — {t['duration_s']}s — {status}")
                trace_text = "\n".join(trace_lines)
            else:
                trace_text = "(no tools were called)"

            # Classify worker behaviour this turn for rubric selection
            modality, error_rate, tool_count = _classify_worker_modality(tool_traces)
            rubric_text = _build_supervisor_rubric(modality, mode)

            # Build plan compliance section conditionally (added after the rubric)
            plan_section = ""
            if mode == "build" and plan_context and not plan_context.startswith("[No active plan"):
                plan_section = (
                    "\n\n**Plan Compliance** — Steps followed in order? Scope respected? "
                    "No unauthorized deviations? Scope violation = hard fail (score <= 0.4).\n\n"
                    "### Active Plan\n\n" + plan_context
                )

            sup_prompt, _ = generate(
                role="supervisor",
                allowed_tools=s_cfg.get("allowed_tools", []),
                session_id=session_id,
                attempt=attempt,
                agent_mode=mode,
                extra={
                    "{{AGENT_MODE}}":           mode,
                    "{{PLAN_CONTEXT}}":         plan_context,
                    "{{PLAN_CONTEXT_SECTION}}": plan_section,
                    "{{TOOL_TRACES}}":          trace_text,
                    "{{WORKER_MODALITY}}":      modality,
                    "{{ERROR_RATE}}":           f"{error_rate:.0%}",
                    "{{TOOL_COUNT}}":           str(tool_count),
                    "{{RUBRIC}}":               rubric_text,
                    "{{THRESHOLD}}":            f"{effective_threshold:.2f}",
                },
            )

            supervisor_result = await _run_supervisor(
                worker_response,
                messages,
                sup_prompt,
                cfg,
                s_cfg.get("include_conversation_history", True),
                role_cfg=s_cfg,
            )

            # Safety net: if the supervisor hallucinated "no tools were called"
            # despite the trace showing calls, override to pass. Weak models
            # routinely make this mistake under the plan-mode rubric, which
            # punishes zero reads as a hard source-gap.
            override_reason = _detect_hallucinated_zero_tool_claim(supervisor_result, tool_count)
            if override_reason:
                original_verdict = dict(supervisor_result)
                supervisor_result["pass"] = True
                supervisor_result["score"] = max(
                    float(supervisor_result.get("score", 0.0)), effective_threshold
                )
                supervisor_result["supervisor_override_reason"] = override_reason
                try:
                    log_supervisor_override(
                        session_id, attempt, override_reason,
                        original_verdict, supervisor_result,
                    )
                except Exception:
                    pass  # best-effort sidecar; never break the turn

            if cfg["logging"]["log_supervisor_turns"]:
                logger.log_turn(attempt, "supervisor", messages, worker_response, supervisor_result)

            score = supervisor_result.get("score", 0.0)
            last_verdict = {**supervisor_result, "attempt": attempt + 1}
            prev_best = best_score
            if score > best_score:
                best_score    = score
                best_response = worker_response

            # Trust the LLM's pass flag, but also honour our mode-based threshold as a floor.
            supervisor_pass = bool(supervisor_result.get("pass", False)) or score >= effective_threshold

            if supervisor_pass:
                clean_body, review_block = _split_peer_review(worker_response)
                if review_block and trace_queue is not None:
                    trace_queue.put_nowait({"event": "worker_review", "data": {"review": review_block}})
                logger.log_turn(0, "final", raw_input, clean_body)
                await _auto_store_memory(raw_input, clean_body, session_id, mode, w_tools)
                result_holder = _with_traces(_format_response(clean_body, session_id))
                return result_holder

            # Surface the supervisor's structured verdict to the user for debugging.
            # Fires on every failed attempt; mod bot renders it as a Discord embed.
            if trace_queue is not None:
                trace_queue.put_nowait({
                    "event": "supervisor_verdict",
                    "data": {
                        "attempt":              attempt + 1,
                        "max_attempts":         total_attempts,
                        "score":                score,
                        "pass_threshold":       effective_threshold,
                        "feedback":             supervisor_result.get("feedback", ""),
                        "tool_issues":          supervisor_result.get("tool_issues", []),
                        "source_gaps":          supervisor_result.get("source_gaps", []),
                        "research_gaps":        supervisor_result.get("research_gaps", []),
                        "accuracy_issues":      supervisor_result.get("accuracy_issues", []),
                        "completeness_issues":  supervisor_result.get("completeness_issues", []),
                        "suggest_spawn":        supervisor_result.get("suggest_spawn", ""),
                        "suggest_debate":       supervisor_result.get("suggest_debate", ""),
                    },
                })

            # Cost-capped retries: if this attempt was a retry and its score barely moved
            # vs. the prior best (< 0.05 gain), further retries are unlikely to cross the
            # threshold. Bail out with the best response we have.
            if attempt > 0 and prev_best >= 0 and score < prev_best + 0.05:
                break

            # Prepare retry: inject structured supervisor feedback
            critique_parts = []
            for label, key in [
                ("TOOL_ISSUES", "tool_issues"),
                ("SOURCE_GAPS", "source_gaps"),
                ("RESEARCH_GAPS", "research_gaps"),
                ("ACCURACY_ISSUES", "accuracy_issues"),
                ("COMPLETENESS_ISSUES", "completeness_issues"),
            ]:
                items = supervisor_result.get(key, [])
                if items:
                    critique_parts.append(f"{label}:\n" + "\n".join(f"  - {item}" for item in items))

            critique_text = "\n".join(critique_parts) if critique_parts else "No specific issues listed."

            alt = supervisor_result.get("alternative", "")
            alt_text = f"\nSUGGESTED_REVISION:\n{alt}" if alt else ""

            spawn_hint = supervisor_result.get("suggest_spawn", "")
            spawn_text = f"\nSPAWN_HINT: Consider delegating to `{spawn_hint}` via run_agent." if spawn_hint else ""

            debate_hint = supervisor_result.get("suggest_debate", "")
            debate_text = (
                f"\nDEBATE_SUGGESTED: The supervisor flagged an unresolved decision: {debate_hint}\n"
                "Before revising, use the `deliberate` tool to resolve this. Frame both sides as strong positions."
            ) if debate_hint else ""

            feedback_message = (
                f"[supervisor_feedback] Score: {score:.2f}\n"
                f"SUMMARY: {supervisor_result.get('feedback', '')}\n\n"
                f"{critique_text}"
                f"{alt_text}"
                f"{spawn_text}"
                f"{debate_text}\n\n"
                "Review each point above. Accept valid criticisms and incorporate them. "
                "Reject unreasonable ones with a brief reason. Then provide your revised response.\n\n"
                "Format your revision start:\n"
                "ACCEPTED: <points you're incorporating, comma-separated>\n"
                "REJECTED: <points you're pushing back on, with reason>\n"
                "Then your revised response."
            )

            messages = messages + [
                {"role": "assistant", "content": worker_response},
                {"role": "user",      "content": feedback_message},
            ]

        clean_body, review_block = _split_peer_review(best_response)
        if review_block and trace_queue is not None:
            trace_queue.put_nowait({"event": "worker_review", "data": {"review": review_block}})
        logger.log_turn(0, "final", raw_input, clean_body)
        await _auto_store_memory(raw_input, clean_body, session_id, mode, w_tools)
        result_holder = _with_traces(_format_response(clean_body, session_id))
        return result_holder
    finally:
        # Collect any queued mid-flight messages so the bot can replay them as a new turn.
        queued_texts: list[str] = []
        if session_state is not None:
            queued_texts = [
                (p.get("text") or "")
                for p in (session_state.get("pending") or [])
                if p.get("mode") == "queue"
            ]
        # If a hard-cancel (task.cancel) tore us down before result_holder was
        # built, synthesize a minimal "stopped" response so queued injections
        # still ride the SSE done event back to the bot. Without this, a user
        # who clicks "Immediate" while the LLM is mid-call would lose their
        # redirected message entirely.
        if result_holder is None and queued_texts:
            result_holder = _with_traces({
                "id": f"chatcmpl-{session_id}",
                "object": "chat.completion",
                "model": (cfg.get("llm") or {}).get("model", "unknown"),
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "[stopped by user]"},
                    "finish_reason": "stop",
                }],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                "session_id": session_id,
                "stopped": True,
            })
        if result_holder is not None and queued_texts:
            result_holder["queued_injections"] = queued_texts
        # Flush the turn accumulator + latest verdict to the persistent state file.
        try:
            state.flush_turn(turn_acc, verdict=last_verdict)
        except Exception as e:
            print(f"[session_state] flush_turn failed for {session_id}: {e}", flush=True)
        # Stamp the final turn's 0-based index onto the result so the Discord
        # bot can bind its rendered message_ids to this turn for edit-rewind.
        if isinstance(result_holder, dict):
            result_holder["turn_index"] = max(
                0, int(state.get("stats.turn_count", 0) or 0) - 1
            )
        if trace_queue is not None:
            trace_queue.put_nowait({"event": "done", "data": result_holder or {"error": "agent loop failed"}})
        cleanup_generated(session_id)
