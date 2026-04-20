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
