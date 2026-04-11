"""
Supervisor / Worker agent orchestration.

Public functions:
  run_agent_loop(body, session_id)  — main chat completions handler
  run_soul_update()                  — called by APScheduler at 5 AM
  run_config_agent(body)             — guided config UI (POST /config/agent)

Tool calls:
  Workers detect tool calls by returning raw JSON with a "tool" key.
  The loop executes the tool via mcp_client.call_tool(), injects the result
  as a user message, and continues until a final (non-JSON) answer is given
  or max_tool_iterations is reached.
"""

import json
import os
from pathlib import Path

import httpx

from app.config_loader import get_config, get_agents_config
from app.mcp_client import call_tool, _extract_tool_call
from app.prompt_generator import generate, cleanup_generated
from app.session_logger import SessionLogger

LLAMA_URL = os.environ.get("LLAMA_URL", "http://host.docker.internal:8080")
WORKSPACE = Path(os.environ.get("WORKSPACE_DIR", "/workspace"))


# ── LLM call ──────────────────────────────────────────────────────────────────

async def _llm_call(messages: list[dict], cfg: dict) -> dict:
    """Single call to llama.cpp via its OpenAI-compatible endpoint."""
    async with httpx.AsyncClient(timeout=180) as client:
        r = await client.post(
            f"{LLAMA_URL}/v1/chat/completions",
            json={
                "model":       cfg["llm"]["model"],
                "messages":    messages,
                "temperature": cfg["llm"]["temperature"],
                "max_tokens":  cfg["llm"]["max_tokens"],
                "top_p":       cfg["llm"]["top_p"],
            },
        )
        r.raise_for_status()
    return r.json()


def _content(llm_response: dict) -> str:
    """Extract the assistant message content from a chat completion response."""
    try:
        return llm_response["choices"][0]["message"]["content"]
    except (KeyError, IndexError):
        return str(llm_response)


def _usage(llm_response: dict) -> dict:
    return llm_response.get("usage", {})


def _format_response(content: str, session_id: str) -> dict:
    """Wrap a final answer in OpenAI chat completion format."""
    return {
        "id":      f"mab-{session_id}",
        "object":  "chat.completion",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
        "session_id": session_id,
    }


# ── Worker (with inner tool loop) ─────────────────────────────────────────────

async def _run_worker(
    messages:      list[dict],
    system_prompt: str,
    allowed_tools: list[str],
    max_iterations: int,
    cfg:            dict,
) -> tuple[str, list[dict]]:
    """
    Run the worker agent with its inner tool-call loop.

    Returns (final_answer, updated_messages_list).
    """
    full_messages = [{"role": "system", "content": system_prompt}] + messages

    for _ in range(max_iterations):
        resp    = await _llm_call(full_messages, cfg)
        content = _content(resp)
        tc      = _extract_tool_call(content)

        if tc is None:
            # No tool call — this is the final answer
            return content, full_messages

        # Execute the tool
        tool_result = await call_tool(tc["tool"], tc.get("params", {}), allowed_tools)

        # Append tool call + result as messages and continue
        full_messages.append({"role": "assistant", "content": content})
        full_messages.append({
            "role":    "user",
            "content": f"[tool_result: {tc['tool']}]\n{json.dumps(tool_result, ensure_ascii=False)}",
        })

    # Exhausted iterations — last content is best effort
    return content, full_messages


# ── Supervisor ────────────────────────────────────────────────────────────────

async def _run_supervisor(
    worker_response:    str,
    original_messages:  list[dict],
    system_prompt:      str,
    cfg:                dict,
    include_history:    bool,
) -> dict:
    """
    Grade the worker response.
    Returns {"pass": bool, "score": float, "feedback": str, "alternative": str}.
    Falls back to pass=True on JSON parse errors to avoid infinite error loops.
    """
    context_messages = original_messages if include_history else []
    messages = (
        [{"role": "system", "content": system_prompt}]
        + context_messages
        + [{"role": "assistant", "content": worker_response}]
        + [{"role": "user", "content": "Grade the above response. Respond ONLY with JSON."}]
    )

    try:
        resp    = await _llm_call(messages, cfg)
        raw     = _content(resp).strip()

        # Strip markdown fence if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        result = json.loads(raw)
        assert all(k in result for k in ("pass", "score", "feedback", "alternative"))
        result["score"] = float(result["score"])
        return result
    except Exception:
        return {"pass": True, "score": 0.5, "feedback": "supervisor parse error — treating as pass", "alternative": ""}


# ── Main agent loop ────────────────────────────────────────────────────────────

async def run_agent_loop(body: dict, session_id: str) -> dict:
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
    messages   = body.get("messages", [])

    w_cfg  = agents_cfg.get("worker", {})
    s_cfg  = agents_cfg.get("supervisor", {})
    w_tools     = w_cfg.get("allowed_tools", [])
    w_max_iter  = w_cfg.get("max_tool_iterations", 10)
    threshold   = cfg["agent"]["supervisor_pass_threshold"]
    max_retries = cfg["agent"]["max_retries"]
    sup_enabled = cfg["agent"]["supervisor_enabled"]

    best_response = ""
    best_score    = -1.0

    total_attempts = 1 + max_retries if sup_enabled else 1

    for attempt in range(total_attempts):
        # Generate worker prompt for this specific attempt
        worker_prompt, worker_agent_id = generate(
            role="worker",
            allowed_tools=w_tools,
            session_id=session_id,
            attempt=attempt,
        )

        try:
            worker_response, _ = await _run_worker(
                messages, worker_prompt, w_tools, w_max_iter, cfg
            )
        except httpx.HTTPError as e:
            worker_response = f"[LLM error: {e}]"

        if cfg["logging"]["log_supervisor_turns"]:
            logger.log_turn(attempt, "worker", messages, worker_response)

        if not sup_enabled:
            cleanup_generated(session_id)
            return _format_response(worker_response, session_id)

        # Generate supervisor prompt
        sup_prompt, sup_agent_id = generate(
            role="supervisor",
            allowed_tools=s_cfg.get("allowed_tools", []),
            session_id=session_id,
            attempt=attempt,
        )

        supervisor_result = await _run_supervisor(
            worker_response,
            messages,
            sup_prompt,
            cfg,
            s_cfg.get("include_conversation_history", True),
        )

        if cfg["logging"]["log_supervisor_turns"]:
            logger.log_turn(attempt, "supervisor", messages, worker_response, supervisor_result)

        score = supervisor_result.get("score", 0.0)
        if score > best_score:
            best_score    = score
            best_response = worker_response

        if supervisor_result.get("pass", False):
            cleanup_generated(session_id)
            return _format_response(worker_response, session_id)

        # Prepare retry: inject supervisor feedback as user message
        feedback = supervisor_result.get("feedback", "")
        messages = messages + [
            {"role": "assistant", "content": worker_response},
            {"role": "user",      "content": (
                f"[supervisor_feedback] Score: {score:.2f}. {feedback} "
                "Please revise your response."
            )},
        ]

    cleanup_generated(session_id)
    return _format_response(best_response, session_id)


# ── Soul update (runs at 5 AM via APScheduler) ─────────────────────────────────

async def run_soul_update() -> None:
    """
    Reads workspace context, asks the model to rewrite SOUL.md.
    Post-write: enforces soul.max_chars as a hard character limit.
    """
    cfg        = get_config()
    agents_cfg = get_agents_config()
    soul_cfg   = agents_cfg.get("soul_updater", {})

    soul_tools   = soul_cfg.get("allowed_tools", ["file_read", "file_write"])
    soul_max_it  = soul_cfg.get("max_tool_iterations", 8)
    soul_session = "soul_update"

    prompt, _ = generate(
        role="soul_updater",
        allowed_tools=soul_tools,
        session_id=soul_session,
        attempt=0,
    )

    messages = [{"role": "user", "content": "Update SOUL.md now."}]

    try:
        response, _ = await _run_worker(
            messages, prompt, soul_tools, soul_max_it, cfg
        )
    except Exception as e:
        response = f"[soul update error: {e}]"

    # Enforce hard char limit on SOUL.md regardless of what the model wrote
    soul_path = WORKSPACE / "SOUL.md"
    if soul_path.exists():
        text = soul_path.read_text(encoding="utf-8")
        max_c = cfg["soul"]["max_chars"]
        if len(text) > max_c:
            soul_path.write_text(text[:max_c], encoding="utf-8")

    cleanup_generated(soul_session)


# ── Config agent (POST /config/agent) ─────────────────────────────────────────

async def run_config_agent(body: dict) -> dict:
    """
    Guided config UI agent. Walks the user through configuration questions.
    Only read_config and write_config tools are allowed.
    """
    cfg        = get_config()
    agents_cfg = get_agents_config()
    ca_cfg     = agents_cfg.get("config_agent", {})
    ca_tools   = ca_cfg.get("allowed_tools", ["read_config", "write_config"])

    session_id = body.get("session_id", "config_agent")
    messages   = body.get("messages", [])

    prompt, _ = generate(
        role="config_agent",
        allowed_tools=ca_tools,
        session_id=session_id,
        attempt=0,
    )

    try:
        response, _ = await _run_worker(messages, prompt, ca_tools, 6, cfg)
    except Exception as e:
        response = f"[config agent error: {e}]"

    cleanup_generated(session_id)
    return _format_response(response, session_id)
