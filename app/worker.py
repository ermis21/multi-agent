"""
Worker inner tool-call loop.

Runs one worker turn to completion: LLM → tool call or final answer → tool
dispatch → loop. Handles mid-flight user injections, inflection nudging,
end-of-turn marker parsing, and max-iteration summarisation.
"""

import asyncio
import json
import time
import uuid

from app.config_loader import get_config
from app.context_compressor import compact_tool_result, store_tool_result
from app.llm import _content, _extract_logprobs, _llm_call
from app.mcp_client import call_tool, _extract_tool_call
from app.sessions.state import SessionState, TurnAccumulator, log_tool_error

# Shell-style tools: a non-zero exit_code is promoted to a real error so the
# model sees ERROR instead of a misleading OK when commands fail.
_SHELL_TOOLS = frozenset({
    "shell_exec", "execute_command",
    "git_status", "git_commit", "git_rollback", "git_log",
    "docker_test_up", "docker_test_down", "docker_test_health",
})

# Tokens/substrings that look like an attempted tool call — used to detect
# parse failures so we can correct the model instead of accepting prose as final.
_TOOL_CALL_HINTS = ("<|tool_call", "<tool_call", '"tool":', "call:")

# Explicit end-of-turn marker. A non-tool-call message is only treated as the
# final answer when it contains this sentinel; otherwise it's mid-turn scaffolding.
END_MARKER = "<|end|>"


def _short_params(params: dict, max_chars: int = 60) -> str:
    """Human-skimmable one-line summary of tool params for the `tool_started` event."""
    if not params:
        return ""
    # Prefer common "headline" keys first
    for key in ("query", "command", "path", "url", "name", "text", "method", "tool", "role"):
        if key in params and params[key]:
            val = str(params[key]).strip().replace("\n", " ")
            if len(val) > max_chars:
                val = val[:max_chars - 1] + "…"
            return f'{key}="{val}"'
    # Fallback: compact JSON
    try:
        val = json.dumps(params, ensure_ascii=False)
    except Exception:
        val = str(params)
    val = val.replace("\n", " ")
    if len(val) > max_chars:
        val = val[:max_chars - 1] + "…"
    return val


def _promote_shell_error(method: str, result: dict) -> dict:
    """If a shell-style tool returned exit_code != 0, surface it as an error."""
    if method not in _SHELL_TOOLS or "error" in result:
        return result
    code = result.get("exit_code", 0)
    if code == 0:
        return result
    stderr = (result.get("stderr") or "").strip()
    out = dict(result)
    out["error"] = f"exit {code}" + (f": {stderr}" if stderr else "")
    return out


def _split_peer_review(text: str) -> tuple[str, str]:
    """Split a worker retry response into (clean_body, review_block).

    On retries, the worker is instructed to prefix its response with
    ACCEPTED: <...> and optionally REJECTED: <...> lines. These are useful
    for debugging but ugly in the main chat. Strip them for the user and
    surface the block separately via the worker_review SSE event.

    Returns (text, "") if no leading ACCEPTED:/REJECTED: header is found.
    """
    stripped = text.lstrip()
    if not stripped.startswith("ACCEPTED:") and not stripped.startswith("REJECTED:"):
        return text, ""

    # Count leading whitespace we stripped, so we can reassemble body offsets
    leading_ws = text[: len(text) - len(stripped)]
    lines = stripped.split("\n")
    header_lines: list[str] = []
    i = 0
    # Consume contiguous ACCEPTED:/REJECTED: lines (plus continuation lines
    # until a blank line or a line that doesn't fit the header pattern)
    while i < len(lines):
        line = lines[i]
        if line.startswith("ACCEPTED:") or line.startswith("REJECTED:"):
            header_lines.append(line)
            i += 1
            continue
        # Continuation line (indented or starts a sub-point) belongs to header
        # only if we already saw a header line and this line isn't blank
        if header_lines and line.strip() and (line.startswith(" ") or line.startswith("\t") or line.startswith("-")):
            header_lines.append(line)
            i += 1
            continue
        break

    review_block = "\n".join(header_lines).strip()
    # Skip any blank lines before the body
    while i < len(lines) and not lines[i].strip():
        i += 1
    body = "\n".join(lines[i:])
    # Preserve any leading whitespace from the original text
    if leading_ws and body:
        body = leading_ws + body
    return body, review_block


async def _run_worker(
    messages:       list[dict],
    system_prompt:  str,
    allowed_tools:  list[str],
    max_iterations: int,
    cfg:            dict,
    temperature:    float | None = None,
    mode:           str = "converse",
    approved_tools: list[str] | None = None,
    role_cfg:       dict | None = None,
    session_id:     str = "",
    trace_queue:    asyncio.Queue | None = None,
    extra_auto_allow_paths: list[str] | None = None,
    inflection_mode: str = "none",
    session_state:  dict | None = None,
    turn_acc:       TurnAccumulator | None = None,
    after_iteration_hook = None,
) -> tuple[str, list[dict], list[dict]]:
    """
    Run the worker agent with its inner tool-call loop.

    Returns (final_answer, updated_messages_list, tool_traces).
    tool_traces is a list of dicts: {tool, duration_s, lines, error}.
    """
    tool_traces: list[dict] = []
    spawnable = (role_cfg or {}).get("spawnable_agents", [])
    full_messages = [{"role": "system", "content": system_prompt}] + messages
    request_logprobs = inflection_mode in ("logprobs", "both")
    inflection_cfg = cfg.get("inflection", {})
    max_nudges = inflection_cfg.get("max_nudges_per_turn", 2)
    nudge_count = 0
    # Deferred injections to staple onto the next tool_result (not_urgent / clarify).
    deferred_injections: list[dict] = []

    content = ""
    for i in range(max_iterations):
        # --- Cancellation + mid-flight injection drain ---
        if session_state is not None:
            if session_state.get("cancel") is not None and session_state["cancel"].is_set():
                return "[stopped by user]", full_messages, tool_traces
            pending = session_state.get("pending") or []
            remaining: list[dict] = []
            for inj in pending:
                m = inj.get("mode")
                text = (inj.get("text") or "").strip()
                if not text:
                    continue
                if m == "immediate":
                    full_messages.append({
                        "role": "user",
                        "content": f"[user_interjection]\n{text}",
                    })
                    if trace_queue is not None:
                        trace_queue.put_nowait({"event": "injection", "data": {"mode": m, "text": text}})
                elif m in ("not_urgent", "clarify"):
                    deferred_injections.append({"mode": m, "text": text})
                    if trace_queue is not None:
                        trace_queue.put_nowait({"event": "injection", "data": {"mode": m, "text": text}})
                elif m == "queue":
                    remaining.append(inj)  # leave queue items for the API layer
                # "stop" is handled by cancel event above
            session_state["pending"] = remaining
            if pending and session_id:
                # Mirror drain to disk so a crash mid-iteration doesn't re-apply injections.
                try:
                    _st = SessionState.load_or_create(session_id)
                    _st.set("pending_injections", list(remaining))
                    _st.save()
                except Exception:
                    pass

        resp    = await _llm_call(full_messages, cfg, temperature, role_cfg=role_cfg,
                                   request_logprobs=request_logprobs)
        content = _content(resp)
        if turn_acc is not None:
            turn_acc.llm_call_count += 1
            _usage = resp.get("usage") or {}
            turn_acc.token_usage["input"] += int(_usage.get("prompt_tokens") or _usage.get("input_tokens") or 0)
            turn_acc.token_usage["output"] += int(_usage.get("completion_tokens") or _usage.get("output_tokens") or 0)

        inflection_nudge = ""
        if inflection_mode != "none" and nudge_count < max_nudges:
            try:
                from app.inflection import (
                    detect_inflection_points, format_inflection_nudge,
                    detect_linguistic_markers, format_linguistic_nudge,
                )

                if inflection_mode in ("logprobs", "both"):
                    logprobs_data = _extract_logprobs(resp)
                    if logprobs_data:
                        inflections = detect_inflection_points(
                            content, logprobs_data,
                            entropy_threshold=inflection_cfg.get("entropy_threshold", 1.5),
                            logprob_gap_threshold=inflection_cfg.get("logprob_gap_threshold", 0.5),
                        )
                        if inflections:
                            inflection_nudge = format_inflection_nudge(inflections)

                if inflection_mode in ("linguistic", "both"):
                    signals, should_nudge = detect_linguistic_markers(
                        content,
                        strong_threshold=inflection_cfg.get("strong_marker_threshold", 1),
                        weak_threshold=inflection_cfg.get("weak_marker_threshold", 3),
                    )
                    if should_nudge and not inflection_nudge:
                        inflection_nudge = format_linguistic_nudge(signals)
                    elif should_nudge and inflection_nudge:
                        inflection_nudge += "\n(Confirmed by linguistic markers in your response.)"
            except Exception:
                pass  # inflection detection is non-fatal

        tc = _extract_tool_call(content)

        if tc is None:
            # Explicit end-of-turn marker: worker is done.
            if END_MARKER in content:
                # Guard: model sometimes emits just `<|end|>` with no content
                # for short conversational prompts. Treat as malformed
                # termination and re-prompt — same pattern as the malformed
                # tool-call correction below. Bounded by max_tool_iterations.
                if content.replace(END_MARKER, "").strip() == "":
                    full_messages.append({"role": "assistant", "content": content})
                    full_messages.append({
                        "role": "user",
                        "content": (
                            "You ended the turn with no content. Provide your "
                            "final answer in plain text, then end with <|end|>."
                        ),
                    })
                    continue
                # Before exiting, sweep up any injections that arrived mid-LLM-call
                # (still in session_state["pending"], never drained to deferred)
                # plus any already-deferred notes that never got stapled to a
                # tool_result. Without this, a `not_urgent` inject sent between
                # the last tool dispatch and the final answer silently vanishes.
                if session_state is not None:
                    late_pending = session_state.get("pending") or []
                    late_remaining: list[dict] = []
                    for inj in late_pending:
                        m = inj.get("mode")
                        text = (inj.get("text") or "").strip()
                        if text and m in ("not_urgent", "clarify", "immediate"):
                            deferred_injections.append({"mode": m if m != "immediate" else "not_urgent", "text": text})
                        else:
                            late_remaining.append(inj)
                    session_state["pending"] = late_remaining
                if deferred_injections:
                    notes = []
                    for inj in deferred_injections:
                        prefix = "[user_note]" if inj["mode"] == "not_urgent" else "[user_clarification]"
                        notes.append(f"{prefix} {inj['text']}")
                    deferred_injections.clear()
                    full_messages.append({"role": "assistant", "content": content})
                    full_messages.append({"role": "user", "content": "\n\n".join(notes)})
                    continue
                # Dream auto-sim splice: if the hook returns a synthetic tool-result
                # (e.g. sim result), keep looping instead of returning so the
                # dreamer can react. `just_revised=False` — reaching this branch
                # means the current iteration emitted prose + end, not a revise
                # tool. (The tool-result branch below passes its own flag.)
                if after_iteration_hook is not None:
                    hook_msg = await after_iteration_hook(full_messages, False)
                    if hook_msg:
                        full_messages.append({"role": "assistant", "content": content})
                        full_messages.append({"role": "user", "content": hook_msg})
                        continue
                clean = content.replace(END_MARKER, "").rstrip()
                return clean, full_messages, tool_traces
            # If the content *looks* like a malformed tool call, correct the
            # model instead of accepting it as the final answer — this is what
            # causes hallucinated "command executed successfully" cascades.
            if any(h in content for h in _TOOL_CALL_HINTS):
                full_messages.append({"role": "assistant", "content": content})
                full_messages.append({
                    "role": "user",
                    "content": (
                        "Your tool call could not be parsed. Emit exactly:\n"
                        "<|tool_call|>call: TOOL_NAME, {param_json}<|tool_call|>\n"
                        "One line. No prose, no markdown, no nesting."
                    ),
                })
                continue
            # Before returning the final answer, check if we should nudge
            if inflection_nudge:
                nudge_count += 1
                full_messages.append({"role": "assistant", "content": content})
                full_messages.append({"role": "user", "content": inflection_nudge})
                continue  # give the worker a chance to reconsider / call deliberate
            # No tool call and no end marker → mid-turn scaffolding.
            # Surface to the user as live status, then keep looping.
            full_messages.append({"role": "assistant", "content": content})
            if trace_queue is not None and content.strip():
                trace_queue.put_nowait({"event": "worker_status", "data": {"text": content.strip()}})
            # Dream auto-sim splice: the dreamer sometimes stops calling
            # revise tools and just narrates ("now waiting for simulation"),
            # which used to trap it in a scaffold loop until max_iter. Fire
            # the hook here too — if a submit-phase batch is staged and the
            # dreamer is no longer revising, run the sim and splice the
            # result in so the dreamer can react and call dream_finalize.
            if after_iteration_hook is not None:
                hook_msg = await after_iteration_hook(full_messages, False)
                if hook_msg:
                    full_messages.append({"role": "user", "content": hook_msg})
                    continue
            full_messages.append({
                "role": "user",
                "content": (
                    "Continue. Either call a tool or end your turn with <|end|> "
                    "on its own line when you are done."
                ),
            })
            continue

        # Emit a synchronous tool_started event BEFORE dispatch so the Discord
        # bot can anchor the "⏳ tool …" message above any subsequent worker
        # text. call_id threads start → complete so the bot edits in place.
        call_id = uuid.uuid4().hex[:8]
        params_preview = _short_params(tc.get("params", {}))
        if trace_queue is not None:
            trace_queue.put_nowait({
                "event": "tool_started",
                "data": {"call_id": call_id, "tool": tc["tool"], "params_preview": params_preview},
            })

        # Execute the tool
        t0 = time.time()
        tool_result = await call_tool(tc["tool"], tc.get("params", {}), allowed_tools, mode, approved_tools,
                                      session_id=session_id, spawnable_agents=spawnable,
                                      extra_auto_allow_paths=extra_auto_allow_paths,
                                      trace_queue=trace_queue)
        tool_result = _promote_shell_error(tc["tool"], tool_result)
        duration = time.time() - t0

        # Format result clearly so the model can distinguish success from failure
        if "error" in tool_result:
            result_text = (
                f"[tool_result: {tc['tool']}] ERROR\n"
                f"{tool_result['error']}\n\n"
                "This tool call failed. Do NOT stop — keep working on the user's request. "
                "Pick one: retry with corrected parameters, try a different tool, or try a different approach. "
                "Only give a final plain-text answer if every reasonable approach has been exhausted."
            )
            trace = {"call_id": call_id, "tool": tc["tool"], "duration_s": round(duration, 2),
                     "lines": 0, "error": tool_result["error"], "params_preview": params_preview}
            tool_traces.append(trace)
            if trace_queue is not None:
                trace_queue.put_nowait({"event": "tool_trace", "data": trace})
            if turn_acc is not None:
                turn_acc.record_tool(tc["tool"], tc.get("params", {}), error=True)
                turn_acc.duration_ms += int(duration * 1000)
            if session_id:
                try:
                    log_tool_error(session_id, tc["tool"], str(tool_result["error"]), params_preview)
                except Exception:
                    pass
        else:
            raw_body = json.dumps(tool_result, ensure_ascii=False, indent=2)
            # Compact large tool results — stash the full body under a handle
            # and inline only a head+tail preview so we keep within the sliding
            # window. Worker can `tool_result_recall` the stashed body on demand.
            handle_id: str | None = None
            try:
                _cfg = get_config()
                _ctx = (_cfg.get("context") or {})
                if _ctx.get("enabled", True):
                    _budget = int((_ctx.get("budgets") or {}).get("tool_result_inline", 1500))
                    inline_body, handle_id = compact_tool_result(
                        tc["tool"], tc.get("params", {}), raw_body, _budget,
                    )
                else:
                    inline_body = raw_body
            except Exception:
                inline_body = raw_body
                handle_id = None

            header = f"[tool_result: {tc['tool']}] OK"
            if handle_id:
                header += f"  #{handle_id}"
                if session_id:
                    try:
                        store_tool_result(session_id, handle_id, tc["tool"],
                                          tc.get("params", {}), raw_body)
                    except Exception:
                        pass
                if turn_acc is not None:
                    try:
                        turn_acc.handles_created = getattr(turn_acc, "handles_created", 0) + 1
                    except Exception:
                        pass
            result_text = f"{header}\n{inline_body}"

            trace = {"call_id": call_id, "tool": tc["tool"], "duration_s": round(duration, 2),
                     "lines": result_text.count("\n"), "error": None, "params_preview": params_preview,
                     "handle_id": handle_id}
            tool_traces.append(trace)
            if trace_queue is not None:
                trace_queue.put_nowait({"event": "tool_trace", "data": trace})
            if turn_acc is not None:
                turn_acc.record_tool(tc["tool"], tc.get("params", {}), error=False)
                turn_acc.duration_ms += int(duration * 1000)

        # Staple any not_urgent / clarify injections onto this tool result so
        # the worker sees them at a natural boundary without restarting its plan.
        if deferred_injections:
            notes = []
            for inj in deferred_injections:
                prefix = "[user_note]" if inj["mode"] == "not_urgent" else "[user_clarification]"
                suffix = ""
                if inj["mode"] == "clarify":
                    suffix = "\n(continue your current research — this is clarification, not a new task)"
                notes.append(f"{prefix} {inj['text']}{suffix}")
            result_text = result_text + "\n\n" + "\n\n".join(notes)
            deferred_injections.clear()

        full_messages.append({"role": "assistant", "content": content})
        full_messages.append({"role": "user", "content": result_text})

        # Inject pending inflection nudge after tool result
        if inflection_nudge:
            nudge_count += 1
            full_messages.append({"role": "user", "content": inflection_nudge})

        # Dream auto-sim splice: fires when the tool just called was not
        # dream_submit / edit_revise and a submit-phase batch is staged.
        # Pass an explicit just_revised flag derived from the tool name we
        # actually dispatched this iteration — `full_messages` inference
        # fails on the prose-without-end path where the last tool_result is
        # stale (from a prior iteration).
        if after_iteration_hook is not None:
            _just_revised = tc["tool"] in {"dream_submit", "edit_revise"}
            hook_msg = await after_iteration_hook(full_messages, _just_revised)
            if hook_msg:
                full_messages.append({"role": "user", "content": hook_msg})

    # Exhausted iterations without a final answer — ask for a plain-text summary.
    # Also sweep up any injections that arrived mid-LLM-call (still in
    # session_state["pending"]) and any already-deferred notes — fold them into
    # the summary prompt so the final answer still addresses mid-flight notes.
    if session_state is not None:
        late_pending = session_state.get("pending") or []
        late_remaining: list[dict] = []
        for inj in late_pending:
            m = inj.get("mode")
            text = (inj.get("text") or "").strip()
            if text and m in ("not_urgent", "clarify", "immediate"):
                deferred_injections.append({"mode": m if m != "immediate" else "not_urgent", "text": text})
            else:
                late_remaining.append(inj)
        session_state["pending"] = late_remaining
    full_messages.append({"role": "assistant", "content": content})
    summary_prompt = "Summarize what happened and give your final answer in plain text."
    if deferred_injections:
        notes = []
        for inj in deferred_injections:
            prefix = "[user_note]" if inj["mode"] == "not_urgent" else "[user_clarification]"
            notes.append(f"{prefix} {inj['text']}")
        deferred_injections.clear()
        summary_prompt = "\n\n".join(notes) + "\n\n" + summary_prompt
    full_messages.append({"role": "user", "content": summary_prompt})
    resp    = await _llm_call(full_messages, cfg, temperature, role_cfg=role_cfg)
    content = _content(resp)
    if END_MARKER in content:
        content = content.replace(END_MARKER, "").rstrip()
    return content, full_messages, tool_traces
