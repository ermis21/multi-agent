"""
Token counting / tokenization with llama.cpp primary + tiktoken fallback.

The compressor needs fast, bounded token counts to enforce per-section budgets.
We call llama.cpp's /tokenize endpoint first (matches the model actually doing
inference), fall back to tiktoken cl100k_base, and finally a char/4 heuristic.

Synchronous API on purpose: the compressor is invoked from a pure synchronous
code path (prompt_generator.generate). We use httpx's sync client with a short
timeout; a slow llama.cpp never blocks prompt assembly — the heuristic wins.
"""

from __future__ import annotations

import hashlib
import logging
import os
import threading
from enum import Enum
from functools import lru_cache

import httpx

logger = logging.getLogger("phoebe.tokenizer")

LLAMA_URL = os.environ.get("LLAMA_URL", "http://host.docker.internal:8080")

_TOKENIZE_TIMEOUT_S = 2.0
_sync_client: httpx.Client | None = None
_sync_client_lock = threading.Lock()


def _client() -> httpx.Client:
    global _sync_client
    if _sync_client is None:
        with _sync_client_lock:
            if _sync_client is None:
                _sync_client = httpx.Client(timeout=_TOKENIZE_TIMEOUT_S)
    return _sync_client


class ElisionStrategy(str, Enum):
    HEAD = "head"
    TAIL = "tail"
    HEAD_TAIL = "head_tail"
    MIDDLE = "middle"


_llama_reachable: bool | None = None
_tiktoken_enc = None


def _try_llama_tokenize(text: str) -> list[int] | None:
    global _llama_reachable
    if _llama_reachable is False:
        return None
    try:
        r = _client().post(f"{LLAMA_URL}/tokenize", json={"content": text})
        if r.status_code != 200:
            return None
        data = r.json()
        tokens = data.get("tokens")
        if isinstance(tokens, list):
            _llama_reachable = True
            return tokens
    except Exception as e:
        if _llama_reachable is None:
            logger.info("llama.cpp /tokenize unreachable (%s); falling back", e)
        _llama_reachable = False
    return None


def _try_llama_detokenize(ids: list[int]) -> str | None:
    if _llama_reachable is False or not ids:
        return None
    try:
        r = _client().post(f"{LLAMA_URL}/detokenize", json={"tokens": ids})
        if r.status_code != 200:
            return None
        data = r.json()
        content = data.get("content")
        if isinstance(content, str):
            return content
    except Exception:
        return None
    return None


def _tiktoken():
    global _tiktoken_enc
    if _tiktoken_enc is not None:
        return _tiktoken_enc
    try:
        import tiktoken
        _tiktoken_enc = tiktoken.get_encoding("cl100k_base")
    except Exception as e:
        logger.info("tiktoken unavailable (%s); using char heuristic", e)
        _tiktoken_enc = False
    return _tiktoken_enc


def tokenize(text: str, model: str | None = None) -> list[int]:
    """Return token ids. Falls back through llama → tiktoken → char heuristic."""
    if not text:
        return []
    ids = _try_llama_tokenize(text)
    if ids is not None:
        return ids
    enc = _tiktoken()
    if enc:
        return enc.encode(text)
    # Heuristic: pretend each 4 chars = 1 token. Stable ids irrelevant.
    return list(range(max(1, len(text) // 4)))


def detokenize(ids: list[int], model: str | None = None) -> str:
    """Reverse of tokenize() — best-effort via the same backend."""
    if not ids:
        return ""
    s = _try_llama_detokenize(ids)
    if s is not None:
        return s
    enc = _tiktoken()
    if enc:
        try:
            return enc.decode(ids)
        except Exception:
            pass
    # Heuristic can't reverse; return a placeholder
    return ""


def _hash_key(text: str) -> str:
    return hashlib.blake2b(text.encode("utf-8", "ignore"), digest_size=12).hexdigest()


@lru_cache(maxsize=2048)
def _count_cached(key: str, length: int, model: str | None) -> int:
    # key+length pin a specific text; the real text is passed separately because
    # lru_cache keys can't hold multi-MB strings efficiently. We still re-tokenize
    # on miss; the cache just short-circuits repeat calls for identical blobs.
    return -1  # sentinel — never used directly; see count()


_count_memo: dict[tuple[str, int, str | None], int] = {}


def count(text: str, model: str | None = None) -> int:
    """Count tokens in *text*. Cheap repeat calls; bounded cache."""
    if not text:
        return 0
    key = (_hash_key(text), len(text), model)
    cached = _count_memo.get(key)
    if cached is not None:
        return cached
    n = len(tokenize(text, model=model))
    # Bound the memo: drop oldest entries past 2048
    if len(_count_memo) > 2048:
        # Cheap trim: delete first 512 inserted items
        for k in list(_count_memo.keys())[:512]:
            _count_memo.pop(k, None)
    _count_memo[key] = n
    return n


def truncate(text: str, max_tokens: int,
             strategy: ElisionStrategy = ElisionStrategy.HEAD_TAIL,
             model: str | None = None) -> str:
    """
    Truncate *text* to at most *max_tokens* using the given elision strategy.

    HEAD_TAIL keeps head + tail halves with an ellipsis marker in the middle.
    HEAD keeps the leading max_tokens; TAIL keeps the trailing max_tokens.
    MIDDLE keeps the middle slice (rarely useful). Returns the original text
    when already under budget.
    """
    if max_tokens <= 0:
        return ""
    ids = tokenize(text, model=model)
    n = len(ids)
    if n <= max_tokens:
        return text

    if strategy == ElisionStrategy.HEAD:
        return detokenize(ids[:max_tokens], model=model) or text[: max_tokens * 4]
    if strategy == ElisionStrategy.TAIL:
        return detokenize(ids[-max_tokens:], model=model) or text[-max_tokens * 4:]
    if strategy == ElisionStrategy.MIDDLE:
        start = (n - max_tokens) // 2
        return detokenize(ids[start:start + max_tokens], model=model) or text

    # HEAD_TAIL
    half = max_tokens // 2
    head_ids = ids[:half]
    tail_ids = ids[-(max_tokens - half):]
    elided = n - len(head_ids) - len(tail_ids)
    head_text = detokenize(head_ids, model=model)
    tail_text = detokenize(tail_ids, model=model)
    if not head_text or not tail_text:
        # Heuristic fallback: approximate with character slicing
        approx_chars = max_tokens * 4
        h = approx_chars // 2
        return f"{text[:h]}\n…[≈{elided} tok elided]…\n{text[-h:]}"
    return f"{head_text}\n…[≈{elided} tok elided]…\n{tail_text}"


def backend_status() -> dict:
    """Report which backend is live — used by diagnostic_check."""
    # Probe once if unknown
    if _llama_reachable is None:
        _try_llama_tokenize("ok")
    return {
        "llama_reachable": bool(_llama_reachable),
        "tiktoken_available": bool(_tiktoken()),
        "llama_url": LLAMA_URL,
    }
