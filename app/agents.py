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

import asyncio
import json
import os
import time
import uuid
from pathlib import Path

import anthropic as _anthropic_sdk
import httpx

from app.config_loader import get_config, get_agents_config
from app.mcp_client import call_tool, _extract_tool_call, strip_json_fences
from app.prompt_generator import generate, cleanup_generated
from app.session_logger import SessionLogger, get_session
from app.session_state import SessionState, TurnAccumulator, log_tool_error

LLAMA_URL = os.environ.get("LLAMA_URL", "http://host.docker.internal:8080")
WORKSPACE = Path(os.environ.get("WORKSPACE_DIR", "/workspace"))

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
        from app.session_state import SessionState
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

_llm_client = httpx.AsyncClient(timeout=180)

# Anthropic client cache: (token_used, client_instance)
# Re-created whenever the active token changes (e.g. after OAuth refresh).
_anthropic_client: tuple[str, _anthropic_sdk.AsyncAnthropic] | None = None

# Mounted read-only from the host's ~/.claude/.credentials.json
_CLAUDE_CREDS = Path("/app/.claude_credentials.json")


def _read_oauth_token() -> str | None:
    """
    Read the Claude Code OAuth access token from the mounted credentials file.
    Returns None if the file is missing, unreadable, or the token has expired.
    """
    if not _CLAUDE_CREDS.exists():
        return None
    try:
        creds  = json.loads(_CLAUDE_CREDS.read_text())
        oauth  = creds.get("claudeAiOauth", {})
        token  = oauth.get("accessToken", "")
        # expiresAt is milliseconds; skip if expiring within 5 minutes
        if token and oauth.get("expiresAt", 0) > time.time() * 1000 + 300_000:
            return token
    except Exception:
        pass
    return None


def _get_anthropic_client() -> _anthropic_sdk.AsyncAnthropic:
    global _anthropic_client

    oauth_token = _read_oauth_token()
    if oauth_token:
        # Re-create if this is the first call or the token was refreshed
        if _anthropic_client is None or _anthropic_client[0] != oauth_token:
            _anthropic_client = (oauth_token, _anthropic_sdk.AsyncAnthropic(auth_token=oauth_token))
        return _anthropic_client[1]

    # Fall back to ANTHROPIC_API_KEY env var
    if _anthropic_client is None:
        _anthropic_client = ("", _anthropic_sdk.AsyncAnthropic())
    return _anthropic_client[1]


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


# ── LLM call ──────────────────────────────────────────────────────────────────

async def _llm_call_local(messages: list[dict], llm: dict, temperature: float,
                           url: str | None = None, request_logprobs: bool = False) -> dict:
    """llama.cpp / Ollama path via OpenAI-compatible endpoint."""
    endpoint = url or LLAMA_URL
    payload: dict = {
        "model":       llm["model"],
        "messages":    messages,
        "temperature": temperature,
        "max_tokens":  llm["max_tokens"],
        "top_p":       llm["top_p"],
        "top_k":       llm.get("top_k", 40),
    }
    payload["chat_template_kwargs"] = {"enable_thinking": bool(llm.get("enable_thinking", False))}
    if request_logprobs:
        payload["logprobs"] = True
        payload["top_logprobs"] = llm.get("top_logprobs", 5)
    r = await _llm_client.post(f"{endpoint}/v1/chat/completions", json=payload)
    r.raise_for_status()
    return r.json()


# Hosted OpenAI-compatible providers. `default_base` is used when the config
# doesn't override via `url`/`base_url`; `max_output_tokens` caps the model's
# output limit so our 100k+ llama default doesn't trip a 400 on OpenAI.
_OPENAI_COMPAT = {
    "openai": {
        "api_key_env":       "OPENAI_API_KEY",
        "default_base":      "https://api.openai.com/v1",
        "max_output_tokens": 16384,
    },
    "glm": {
        "api_key_env":       "GLM_API_KEY",
        "default_base":      "https://api.z.ai/api/paas/v4",
        "max_output_tokens": 8192,
    },
}


async def _llm_call_openai_compat(messages: list[dict], llm: dict, temperature: float,
                                   provider: str, request_logprobs: bool = False) -> dict:
    """OpenAI-compatible chat-completions path (OpenAI, GLM/Z.ai, etc.)."""
    spec = _OPENAI_COMPAT[provider]
    api_key = os.environ.get(spec["api_key_env"], "").strip()
    if not api_key:
        raise RuntimeError(f"{spec['api_key_env']} not set — cannot call {provider}")

    base = llm.get("url") or llm.get("base_url") or spec["default_base"]
    payload: dict = {
        "model":       llm["model"],
        "messages":    messages,
        "temperature": temperature,
        "max_tokens":  min(int(llm["max_tokens"]), spec["max_output_tokens"]),
        "top_p":       llm["top_p"],
    }
    if request_logprobs:
        payload["logprobs"] = True
        payload["top_logprobs"] = llm.get("top_logprobs", 5)

    r = await _llm_client.post(
        f"{base.rstrip('/')}/chat/completions",
        json=payload,
        headers={"Authorization": f"Bearer {api_key}"},
    )
    r.raise_for_status()
    return r.json()


async def _llm_call_anthropic(messages: list[dict], llm: dict, temperature: float) -> dict:
    """Anthropic SDK path — returns an OpenAI-compatible dict."""
    client = _get_anthropic_client()

    # Anthropic requires system as a separate param, not a message role
    system = " ".join(m["content"] for m in messages if m["role"] == "system")
    conv   = [m for m in messages if m["role"] != "system"]

    thinking = llm.get("enable_thinking", False)
    # Anthropic output token caps: 8192 normally, 16000 with extended thinking
    max_tok = min(llm["max_tokens"], 16000 if thinking else 8192)

    kwargs: dict = {
        "model":      llm["model"],
        "messages":   conv,
        "max_tokens": max_tok,
        "temperature": temperature,
    }
    if system:
        kwargs["system"] = system
    if thinking:
        budget = llm.get("thinking_budget_tokens", 8000)
        kwargs["thinking"] = {"type": "enabled", "budget_tokens": budget}
        kwargs.pop("temperature", None)  # temperature is rejected when thinking is enabled

    resp = await client.messages.create(**kwargs)

    # Normalise to OpenAI-compatible shape so the rest of agents.py is unchanged
    text = next((b.text for b in resp.content if b.type == "text"), "")
    return {
        "choices": [{"message": {"role": "assistant", "content": text}}],
        "usage": {
            "input_tokens":  resp.usage.input_tokens,
            "output_tokens": resp.usage.output_tokens,
        },
    }


def _extract_logprobs(resp: dict) -> list[dict]:
    """Extract per-token logprob data from an LLM response, if present."""
    try:
        return resp["choices"][0].get("logprobs", {}).get("content", [])
    except (KeyError, IndexError, AttributeError):
        return []


async def _llm_call(messages: list[dict], cfg: dict, temperature: float | None = None,
                     role_cfg: dict | None = None, request_logprobs: bool = False) -> dict:
    """Dispatch to the correct provider based on config.

    If role_cfg contains a 'model' key, the named model from cfg['models'] is
    merged over the default llm block — allowing per-role model selection.
    """
    llm = dict(cfg["llm"])  # copy; don't mutate the live config
    if role_cfg and (model_name := role_cfg.get("model")):
        override = cfg.get("models", {}).get(model_name, {})
        if override:
            llm.update(override)

    temp = temperature if temperature is not None else llm["temperature"]
    provider = llm.get("provider", "local")
    if provider == "anthropic":
        return await _llm_call_anthropic(messages, llm, temp)
    if provider in _OPENAI_COMPAT:
        return await _llm_call_openai_compat(messages, llm, temp, provider,
                                              request_logprobs=request_logprobs)
    return await _llm_call_local(messages, llm, temp, url=llm.get("url"),
                                  request_logprobs=request_logprobs)


# ── Mode helpers ──────────────────────────────────────────────────────────────

def _mode_temperature(cfg: dict, mode: str) -> float:
    """plan = base - delta, build = base, converse = base + delta."""
    base  = cfg["llm"]["temperature"]
    delta = cfg["agent"].get("mode", {}).get("temperature_delta", 0.3)
    offset = {"plan": -delta, "build": 0.0, "converse": delta}.get(mode, 0.0)
    return max(0.0, min(2.0, base + offset))


def _mode_tools(cfg: dict, mode: str, base_tools: list[str]) -> list[str]:
    """Remove mode-specific excluded tools from the base allowed list."""
    excluded = set(cfg["agent"].get("mode", {}).get(mode, {}).get("excluded_tools", []))
    return [t for t in base_tools if t not in excluded]


_MODE_SHORT = {
    "plan": "**Mode: PLAN** — read + analyse only; produce a file-anchored plan, no writes/execution.",
    "build": "**Mode: BUILD** — full tool access; execute end-to-end and verify.",
    "converse": "**Mode: CONVERSE** — conversational; answer directly and terminate with `<|end|>`.",
}

_RESEARCH_RULE = (
    "**Research first**: if the request mentions external information "
    "('from the internet', 'popular', 'latest', 'find examples of', 'research'), "
    "call `web_search` before acting. Do not fabricate candidates."
)


def _mode_context_string(mode: str, cfg: dict | None = None, role_cfg: dict | None = None) -> str:
    """Render the {{AGENT_MODE}} block for a worker prompt.

    When `cfg` is supplied, the block is config-driven: it lists the tools actually
    excluded in this mode and includes a research-first rule. The long form can be
    collapsed to one line by setting `prompts.describe_mode_in_system_prompt: false`,
    or per-model via `models.<name>.describe_mode_in_system_prompt` — useful for
    models (e.g. Claude) that already understand plan/build/converse semantics.
    """
    short = _MODE_SHORT.get(mode, "")
    if cfg is None:
        return short

    toggle = cfg.get("prompts", {}).get("describe_mode_in_system_prompt", True)
    if role_cfg and (model_name := role_cfg.get("model")):
        model_override = cfg.get("models", {}).get(model_name, {}) or {}
        if "describe_mode_in_system_prompt" in model_override:
            toggle = bool(model_override["describe_mode_in_system_prompt"])
    if not toggle:
        return short

    excluded = (cfg.get("agent", {}).get("mode", {}).get(mode, {}) or {}).get("excluded_tools", []) or []
    excluded_list = ", ".join(f"`{t}`" for t in excluded) if excluded else "_(none)_"

    sections = {
        "plan": (
            "**Mode: PLAN** — research-first; produce a file-anchored plan, no writes or execution.\n\n"
            f"**Excluded tools in this mode**: {excluded_list}.\n\n"
            "- Your final answer MUST name specific file paths (with extensions like `.py`, `.ts`, `.yaml`, `.md`) and the edits to make.\n"
            "- Do NOT try to substitute an excluded tool with a similar one — e.g. `file_edit` cannot replace `file_write` because `file_edit` only modifies existing files and will fail on new paths.\n"
            "- If the task requires excluded tools (writing, executing, committing), produce the plan and ask the user to switch with `/mode build`.\n"
            f"- {_RESEARCH_RULE}"
        ),
        "build": (
            "**Mode: BUILD** — full tool access; execute the task end-to-end and verify results.\n\n"
            f"**Excluded tools in this mode**: {excluded_list}.\n\n"
            f"- {_RESEARCH_RULE}\n"
            "- Verify results (read the file you wrote, run the test, check the response) before claiming success."
        ),
        "converse": (
            "**Mode: CONVERSE** — conversational; answer directly. For simple greetings or short questions your single reply IS the final answer — end it with `<|end|>`.\n\n"
            f"**Excluded tools in this mode**: {excluded_list}.\n\n"
            "- Do not emit status-style hedges before the answer.\n"
            f"- {_RESEARCH_RULE}"
        ),
    }
    return sections.get(mode, short)


def _content(llm_response: dict) -> str:
    """Extract the assistant message content from a chat completion response."""
    try:
        return llm_response["choices"][0]["message"]["content"]
    except (KeyError, IndexError):
        return str(llm_response)


def _format_response(content: str, session_id: str) -> dict:
    """Wrap a final answer in OpenAI chat completion format."""
    return {
        "id":      f"phoebe-{session_id}",
        "object":  "chat.completion",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
        "session_id": session_id,
    }


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


# ── Supervisor context helpers ────────────────────────────────────────────────

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


# ── Worker (with inner tool loop) ─────────────────────────────────────────────

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
                                      extra_auto_allow_paths=extra_auto_allow_paths)
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
            result_text = (
                f"[tool_result: {tc['tool']}] OK\n"
                + json.dumps(tool_result, ensure_ascii=False, indent=2)
            )
            trace = {"call_id": call_id, "tool": tc["tool"], "duration_s": round(duration, 2),
                     "lines": result_text.count("\n"), "error": None, "params_preview": params_preview}
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


# ── Supervisor ────────────────────────────────────────────────────────────────

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


# ── Main agent loop ────────────────────────────────────────────────────────────

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
    if body.get("channel_id") is not None:
        state.set("channel_id", body.get("channel_id"))
    if body.get("user_id") is not None:
        state.set("user_id", body.get("user_id"))
    if body.get("_source_trigger") is not None and state.get("source_trigger") == {"type": "user", "ref": None}:
        # Only overwrite the default; never clobber a previously-set trigger.
        state.set("source_trigger", body.get("_source_trigger"))
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
                    "{{AGENT_MODE}}": _mode_context_string(mode, cfg=cfg, role_cfg=w_cfg),
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
        # Attach any queued mid-flight messages so the bot can replay them as a new turn.
        if session_state is not None and result_holder is not None:
            queued = [p for p in (session_state.get("pending") or []) if p.get("mode") == "queue"]
            if queued:
                result_holder["queued_injections"] = [q.get("text", "") for q in queued]
        # Flush the turn accumulator + latest verdict to the persistent state file.
        try:
            state.flush_turn(turn_acc, verdict=last_verdict)
        except Exception as e:
            print(f"[session_state] flush_turn failed for {session_id}: {e}", flush=True)
        if trace_queue is not None:
            trace_queue.put_nowait({"event": "done", "data": result_holder or {"error": "agent loop failed"}})
        cleanup_generated(session_id)


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

    try:
        _st = SessionState.load_or_create(soul_session)
        _st.set("agent_role", "soul_updater")
        _st.set("source_trigger", {"type": "cron", "ref": "soul_updater"})
        _st.save()
    except Exception:
        pass

    prompt, _ = generate(
        role="soul_updater",
        allowed_tools=soul_tools,
        session_id=soul_session,
        attempt=0,
    )

    messages = [{"role": "user", "content": "Update SOUL.md now."}]

    try:
        try:
            await _run_worker(messages, prompt, soul_tools, soul_max_it, cfg, role_cfg=soul_cfg,
                              session_id=soul_session)  # traces discarded
        except Exception:
            pass  # best-effort soul update; errors are non-fatal

        # Enforce hard char limit on SOUL.md regardless of what the model wrote
        soul_path = WORKSPACE / "SOUL.md"
        if soul_path.exists():
            text = soul_path.read_text(encoding="utf-8")
            max_c = cfg["soul"]["max_chars"]
            if len(text) > max_c:
                soul_path.write_text(text[:max_c], encoding="utf-8")
    finally:
        cleanup_generated(soul_session)


# ── Config agent (POST /config/agent) ─────────────────────────────────────────

async def run_config_agent(body: dict) -> dict:
    """Guided config UI — delegates to run_agent_role with role=config_agent."""
    session_id = body.pop("session_id", "config_agent")
    return await run_agent_role("config_agent", body, session_id)


# ── Generic role runner ────────────────────────────────────────────────────────

async def run_agent_role(role: str, body: dict, session_id: str) -> dict:
    """
    Run any agent role from agents.yaml directly.
    Used by POST /v1/agents/{role} and run_config_agent.
    Includes session continuity: prior turns are reconstructed from "final" logs.
    """
    cfg        = get_config()
    agents_cfg = get_agents_config()
    role_cfg   = agents_cfg.get(role, {})

    if not role_cfg:
        return {"error": f"Unknown role: {role!r}. Check agents.yaml."}

    # Mode applies to any role that supports it (primarily worker; others use base temp)
    mode_cfg       = cfg.get("agent", {}).get("mode", {})
    mode           = body.get("mode", mode_cfg.get("default", "converse"))
    base_tools     = role_cfg.get("allowed_tools", [])
    allowed_tools  = _mode_tools(cfg, mode, base_tools)
    max_iter       = role_cfg.get("max_tool_iterations", 10)
    temperature    = _mode_temperature(cfg, mode)
    approved_tools = body.get("approved_tools", [])
    logger         = SessionLogger(session_id)
    raw_input     = body.get("messages", [])

    # Plan context for build mode
    plan_context     = body.get("plan_context", "")
    privileged_paths = body.get("privileged_paths", [])

    # Stamp agent_role + source_trigger onto persistent state (first-turn only).
    try:
        _rst = SessionState.load_or_create(session_id)
        _rst.set("agent_role", role)
        if body.get("_source_trigger") is not None \
           and _rst.get("source_trigger") == {"type": "user", "ref": None}:
            _rst.set("source_trigger", body.get("_source_trigger"))
        _rst.save()
    except Exception:
        pass

    messages = _rebuild_session_context(session_id, raw_input, cfg)

    prompt, _ = generate(
        role=role,
        allowed_tools=allowed_tools,
        session_id=session_id,
        attempt=0,
        agent_mode=mode,
        extra={
            "{{AGENT_MODE}}": _mode_context_string(mode, cfg=cfg, role_cfg=role_cfg),
            "{{PLAN_CONTEXT}}": plan_context,
        },
    )

    verbose_tools = cfg["logging"].get("verbose_tools", False)
    tool_traces: list[dict] = []

    try:
        try:
            response, _, tool_traces = await _run_worker(
                messages, prompt, allowed_tools, max_iter, cfg, temperature, mode, approved_tools,
                role_cfg=role_cfg,
                session_id=session_id,
                extra_auto_allow_paths=privileged_paths or None,
            )
        except Exception as e:
            response = f"[{role} error: {e or type(e).__name__}]"
        logger.log_turn(0, "final", raw_input, response)
        await _auto_store_memory(raw_input, response, session_id, mode, allowed_tools)
        result = _format_response(response, session_id)
        if verbose_tools and tool_traces:
            result["tool_trace"] = tool_traces
        return result
    finally:
        cleanup_generated(session_id)
