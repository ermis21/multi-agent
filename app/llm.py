"""
LLM provider I/O — OpenAI-compatible, llama.cpp, and Anthropic SDK paths.

`_llm_call` dispatches to the right provider based on config. Workers and
supervisors both go through it so per-role `model` overrides work uniformly.
"""

import json
import os
import time
from pathlib import Path

import anthropic as _anthropic_sdk
import httpx

LLAMA_URL = os.environ.get("LLAMA_URL", "http://host.docker.internal:8080")

_llm_client = httpx.AsyncClient(timeout=180)

# Anthropic client cache: (token_used, client_instance)
# Re-created whenever the active token changes (e.g. after OAuth refresh).
_anthropic_client: tuple[str, _anthropic_sdk.AsyncAnthropic] | None = None

# Mounted read-only from the host's ~/.claude/.credentials.json
_CLAUDE_CREDS = Path("/app/.claude_credentials.json")


def _oauth_token_status() -> tuple[str | None, str]:
    """Return `(token_or_None, status_message)` for the mounted OAuth creds.

    `status_message` is a short human-readable explanation when the token is
    unusable — "no creds file", "malformed", "expired 49m ago". Callers use it
    to produce actionable errors instead of the SDK's generic auth-missing.
    """
    if not _CLAUDE_CREDS.exists():
        return None, "no creds file"
    try:
        creds = json.loads(_CLAUDE_CREDS.read_text())
    except Exception as e:
        return None, f"creds malformed: {e}"
    oauth = creds.get("claudeAiOauth", {})
    token = oauth.get("accessToken", "")
    if not token:
        return None, "creds missing accessToken"
    expires_ms = oauth.get("expiresAt", 0)
    now_ms = time.time() * 1000
    if expires_ms <= now_ms + 300_000:
        if expires_ms <= now_ms:
            age_min = int((now_ms - expires_ms) / 60_000)
            return None, f"oauth token expired {age_min}m ago (run `claude` on host to refresh)"
        return None, "oauth token expires within 5 minutes"
    return token, "ok"


def _read_oauth_token() -> str | None:
    """Back-compat shim; use `_oauth_token_status` for the reason."""
    token, _ = _oauth_token_status()
    return token


def _get_anthropic_client() -> _anthropic_sdk.AsyncAnthropic:
    global _anthropic_client

    oauth_token, oauth_status = _oauth_token_status()
    if oauth_token:
        # Re-create if this is the first call or the token was refreshed
        if _anthropic_client is None or _anthropic_client[0] != oauth_token:
            _anthropic_client = (oauth_token, _anthropic_sdk.AsyncAnthropic(auth_token=oauth_token))
        return _anthropic_client[1]

    # Fall back to ANTHROPIC_API_KEY env var. If both are missing we surface a
    # clear reason so the caller doesn't see the SDK's cryptic
    # "Expected either api_key or auth_token" error.
    env_key = os.environ.get("ANTHROPIC_API_KEY")
    if not env_key:
        raise RuntimeError(
            f"No Anthropic credentials available: {oauth_status}; "
            f"ANTHROPIC_API_KEY is also unset. Set the env var or refresh OAuth."
        )
    if _anthropic_client is None or _anthropic_client[0] != env_key:
        _anthropic_client = (env_key, _anthropic_sdk.AsyncAnthropic(api_key=env_key))
    return _anthropic_client[1]


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

    # Only honor `url` (an explicit per-alias override) — NOT `base_url`, which
    # is the global llama fallback bleeding through from cfg.llm. Previously
    # this shipped calls to llama-api-manager whenever a user picked an
    # OpenAI model that didn't match a cfg.models alias (e.g. `gpt-5-mini`
    # raw), because cfg.llm.base_url leaked in via the copy in `_llm_call`.
    base = llm.get("url") or spec["default_base"]
    model_name = str(llm["model"])
    # OpenAI reasoning models (gpt-5-*, o1-*, o3-*) reject `max_tokens` and
    # custom sampling params; they require `max_completion_tokens` and
    # default temperature. Detect by name prefix.
    is_reasoning = any(model_name.startswith(p) for p in ("gpt-5", "o1-", "o3-", "o4-"))
    payload: dict = {
        "model":    model_name,
        "messages": messages,
    }
    # New-API models use `max_completion_tokens`; older chat-completions
    # models accept it too, so we can use it unconditionally.
    payload["max_completion_tokens"] = min(int(llm["max_tokens"]), spec["max_output_tokens"])
    if not is_reasoning:
        payload["temperature"] = temperature
        payload["top_p"] = llm["top_p"]
    if request_logprobs and not is_reasoning:
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

    # Anthropic temperature range is [0, 1]. The shared `_mode_temperature`
    # can produce >1 (converse-mode adds +delta to a base that's already 1.0
    # for Claude configs) — clamp here instead of blowing up with
    # "temperature: range: 0..1" at request time.
    clamped_temp = max(0.0, min(1.0, temperature))

    kwargs: dict = {
        "model":      llm["model"],
        "messages":   conv,
        "max_tokens": max_tok,
        "temperature": clamped_temp,
    }
    if system:
        kwargs["system"] = system
    if thinking:
        budget = llm.get("thinking_budget_tokens", 8000)
        kwargs["thinking"] = {"type": "enabled", "budget_tokens": budget}
        kwargs.pop("temperature", None)  # temperature is rejected when thinking is enabled

    # Retry transient overloads (529) and rate-limit bursts (429) with
    # exponential backoff. Anthropic returns these during load spikes even
    # for accounts in good standing; they're idempotent to retry. Anything
    # else (auth, 400, usage-limit) bubbles up immediately.
    import asyncio as _asyncio
    last_err: Exception | None = None
    for attempt in range(4):  # 0, 1, 2, 3 — up to 3 retries
        try:
            resp = await client.messages.create(**kwargs)
            break
        except _anthropic_sdk.APIStatusError as e:
            last_err = e
            if getattr(e, "status_code", None) not in (429, 529):
                raise
            if attempt == 3:
                raise
            wait = 2 ** attempt  # 1s, 2s, 4s
            print(
                f"[llm] Anthropic {e.status_code} on attempt {attempt + 1}; "
                f"retrying in {wait}s",
                flush=True,
            )
            await _asyncio.sleep(wait)
    else:
        # Shouldn't reach here (the loop either breaks or re-raises), but
        # guard against the impossible so a future refactor can't silently
        # produce an unbound `resp`.
        raise last_err or RuntimeError("anthropic retry loop exited unexpectedly")

    # Normalise to OpenAI-compatible shape so the rest of the pipeline is unchanged
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


def _content(llm_response: dict) -> str:
    """Extract the assistant message content from a chat completion response."""
    try:
        return llm_response["choices"][0]["message"]["content"]
    except (KeyError, IndexError):
        return str(llm_response)
