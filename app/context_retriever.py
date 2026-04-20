"""
Session-scoped RAG retrieval for Zone B of the Gemma prompt (PR 3).

Two responsibilities:

  1. **Chunker** (`chunk_by_sentence`) — splits a message body into overlapping,
     code-fence-safe chunks sized by token count. Pure function, no I/O.
     Called by the fire-and-forget indexer (`app.agents._index_final_turn`) on
     every committed `final` turn.

  2. **Retriever** (`query`) — thin client to the sandbox's `memory_search`
     tool, hitting the `session_chunks` collection with a `session_id` filter.
     Over-fetches k*2, re-ranks by (score-bucket, recency) so within a
     5% score band the more recent turn wins — keeps long conversations from
     drowning in ancient slightly-more-relevant hits.

Session scoping is mandatory: every query sets `where={"session_id": sid}`.
Cross-session retrieval is explicitly opt-in via `cross_session=True`.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any

import httpx

from app.tokenizer import count as count_tokens

logger = logging.getLogger("phoebe.retriever")

SANDBOX_URL = os.environ.get("SANDBOX_URL", "http://phoebe-sandbox:9000")
_RETRIEVER_TIMEOUT_S = 3.0

# Chunk sizing — 200-token chunks with 20-token overlap ride inside the
# sliding window even on a 1024-window Gemma variant; much smaller than that
# and the embedder loses signal; much larger and we waste Zone-B budget.
_DEFAULT_CHUNK_TOKENS = 200
_DEFAULT_OVERLAP_TOKENS = 20

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_CODE_FENCE = re.compile(r"```")


# ── Chunker ──────────────────────────────────────────────────────────────────

def _split_respecting_fences(text: str) -> list[str]:
    """Sentence-split but don't cut inside a ``` fenced block.

    We walk segments between code fences: prose segments get the regex split;
    code segments are emitted whole so the chunker can grow a chunk to close
    an open fence rather than stranding ``` on a chunk boundary.
    """
    segments: list[str] = []
    pos = 0
    in_code = False
    for match in _CODE_FENCE.finditer(text):
        chunk = text[pos:match.start()]
        if chunk:
            if in_code:
                # Preserve code block atomically.
                segments.append(chunk)
            else:
                segments.extend(s for s in _SENTENCE_SPLIT.split(chunk) if s)
        segments.append("```")
        in_code = not in_code
        pos = match.end()
    tail = text[pos:]
    if tail:
        if in_code:
            segments.append(tail)
        else:
            segments.extend(s for s in _SENTENCE_SPLIT.split(tail) if s)
    return segments


def chunk_by_sentence(text: str,
                      target_tokens: int = _DEFAULT_CHUNK_TOKENS,
                      overlap_tokens: int = _DEFAULT_OVERLAP_TOKENS,
                      model: str | None = None) -> list[str]:
    """Produce overlapping, code-fence-safe chunks.

    Empty input → []. Chunks shorter than _MIN_CHUNK_TOKENS are dropped.
    Overlap is approximated at character level — we keep the last
    `overlap_tokens * 4` characters as the next chunk's prefix. Exact
    token-level overlap isn't worth the round-trips; indexing is
    best-effort.
    """
    if not text or not text.strip():
        return []

    segments = _split_respecting_fences(text)
    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0
    in_code = False

    for seg in segments:
        if seg == "```":
            current.append(seg)
            in_code = not in_code
            continue

        seg_tokens = count_tokens(seg, model=model) or max(1, len(seg) // 4)

        # If inside a code fence, always accumulate until we close —
        # atomic emission of code blocks matters more than budget.
        if in_code:
            current.append(seg)
            current_tokens += seg_tokens
            continue

        # Single sentence blows the budget alone — emit it as its own chunk.
        if seg_tokens >= target_tokens and not current:
            chunks.append(seg)
            continue

        if current_tokens + seg_tokens > target_tokens and current:
            chunks.append(" ".join(current).strip())
            # Seed next chunk with tail overlap.
            overlap_chars = max(0, overlap_tokens * 4)
            tail = chunks[-1][-overlap_chars:] if overlap_chars else ""
            current = [tail] if tail else []
            current_tokens = count_tokens(tail, model=model) if tail else 0

        current.append(seg)
        current_tokens += seg_tokens

    if current:
        last = " ".join(current).strip()
        if last:
            chunks.append(last)

    return [c for c in chunks if c.strip()]


# ── Retriever ────────────────────────────────────────────────────────────────

@dataclass
class RetrievedChunk:
    content: str
    score: float
    turn_no: int = -1
    kind: str = "unknown"
    tool: str | None = None
    ts: str = ""
    metadata: dict = field(default_factory=dict)


def _sandbox_memory_search(query_text: str, n: int, where: dict) -> list[dict]:
    payload = {
        "method": "memory_search",
        "params": {
            "query": query_text,
            "n": n,
            "collection": "session_chunks",
            "where": where,
        },
    }
    try:
        with httpx.Client(timeout=_RETRIEVER_TIMEOUT_S) as client:
            r = client.post(f"{SANDBOX_URL}/mcp", json=payload)
            r.raise_for_status()
            body = r.json()
    except Exception as e:
        logger.warning("session_chunks query failed: %s", e)
        return []
    if body.get("error"):
        logger.warning("session_chunks query error: %s", body["error"])
        return []
    result = body.get("result") or {}
    if result.get("error"):
        logger.warning("session_chunks query error: %s", result["error"])
    return result.get("results") or []


def _rerank_score_bucket_recency(hits: list[RetrievedChunk],
                                 band: float = 0.05) -> list[RetrievedChunk]:
    """Within a `band` score window, prefer more recent turns.

    Discretizing scores into fixed buckets has boundary issues (0.80 and 0.78
    are "close" but straddle a 0.05-bucket edge). Instead: sort by score
    descending, then sweep adjacent pairs and swap when they're within-band
    but reverse-ordered by turn_no. Repeat until stable. O(n²) worst case but
    n ≤ k*2 ≈ 16 in practice.
    """
    ordered = sorted(hits, key=lambda h: -h.score)
    changed = True
    while changed:
        changed = False
        for i in range(len(ordered) - 1):
            a, b = ordered[i], ordered[i + 1]
            if abs(a.score - b.score) < band and b.turn_no > a.turn_no:
                ordered[i], ordered[i + 1] = b, a
                changed = True
    return ordered


def query(session_id: str,
          user_msg: str,
          pending_tool_target: str | None = None,
          k: int = 8,
          *,
          cross_session: bool = False) -> list[RetrievedChunk]:
    """Retrieve the top-k most-relevant chunks for the upcoming decode step.

    `pending_tool_target` is concatenated to the query when the worker has
    just emitted a tool call whose result hasn't landed yet — makes retrieval
    aware of what the model is *about* to do.

    Returns [] on any error — retrieval is strictly best-effort.
    """
    if not user_msg or not user_msg.strip():
        return []

    query_text = user_msg.strip()
    if pending_tool_target:
        query_text = f"{query_text} {pending_tool_target}".strip()

    where: dict[str, Any] = {} if cross_session else {"session_id": session_id}
    # Over-fetch, then re-rank — cheaper than asking Chroma for an ordered window.
    raw = _sandbox_memory_search(query_text, n=max(k * 2, k), where=where)

    hits: list[RetrievedChunk] = []
    for h in raw:
        meta = h.get("metadata") or {}
        hits.append(RetrievedChunk(
            content=str(h.get("content") or ""),
            score=float(h.get("score") or 0.0),
            turn_no=int(meta.get("turn_no") or -1),
            kind=str(meta.get("kind") or "unknown"),
            tool=(str(meta["tool"]) if meta.get("tool") else None),
            ts=str(meta.get("ts") or ""),
            metadata=meta,
        ))

    ranked = _rerank_score_bucket_recency(hits)
    return ranked[:k]


def format_zone_b(hits: list[RetrievedChunk], budget_tokens: int,
                  model: str | None = None) -> str:
    """Render retrieved chunks into the Zone-B block, token-bounded.

    Greedy fill — append chunks until the budget is exhausted. Each chunk
    gets a one-line provenance header so the model knows *where* each snippet
    came from; the header itself is budgeted.
    """
    if not hits or budget_tokens <= 0:
        return ""
    lines: list[str] = []
    used = 0
    for h in hits:
        header = (f"[retrieved turn={h.turn_no} kind={h.kind}"
                  f"{' tool=' + h.tool if h.tool else ''}"
                  f" score={h.score:.2f}]")
        block = f"{header}\n{h.content}"
        cost = count_tokens(block, model=model) or max(1, len(block) // 4)
        if used + cost > budget_tokens:
            break
        lines.append(block)
        used += cost
    return "\n\n".join(lines)
