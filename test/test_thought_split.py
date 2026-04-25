"""Tests for `_thought()` and `_strip_inline_thought()` in app/llm.py.

Three sources of model "thinking" content are normalized into a single
`reasoning_content` field on the response:
  1. llama.cpp / OpenAI o-series: native `message.reasoning_content` field.
  2. Anthropic: `block.type == "thinking"` blocks (lifted by `_llm_call_anthropic`).
  3. Inline regex fallback: `<|channel>thought<channel|>...`, `<|think|>...<|/think|>`,
     and `<think>...</think>` formats embedded in `message.content`.

`_thought()` returns the extracted string or None. `_strip_inline_thought()`
removes any inline thought blocks from a content string (used to clean up
final-answer text before showing to the user).
"""

from __future__ import annotations

import pytest

from app.llm import _strip_inline_thought, _thought


def _resp(content: str = "", reasoning: str | None = None) -> dict:
    """Build a minimal OpenAI-shaped response with optional reasoning_content."""
    message: dict = {"role": "assistant", "content": content}
    if reasoning is not None:
        message["reasoning_content"] = reasoning
    return {"choices": [{"message": message}]}


def test_reasoning_content_field_wins(tmp_path):
    """When `reasoning_content` is set, it takes precedence over inline content."""
    resp = _resp(content="final answer", reasoning="step-by-step reasoning")
    assert _thought(resp) == "step-by-step reasoning"


def test_no_thought_returns_none():
    """Plain content with no reasoning field and no inline blocks → None."""
    assert _thought(_resp(content="just an answer")) is None


def test_inline_channel_thought_block():
    """Gemma 4 raw chat-template format: <|channel>thought<channel|>..."""
    content = (
        "<|channel>thought<channel|>"
        "Let me think about this carefully.\n"
        "The user wants X.\n"
        "<channel|>The answer is X."
    )
    assert _thought(_resp(content=content)) == (
        "Let me think about this carefully.\n"
        "The user wants X."
    )


def test_inline_think_block_pipe_form():
    """<|think|>...<|/think|> form (some llama.cpp builds, Qwen)."""
    content = "<|think|>brief reasoning<|/think|>final"
    assert _thought(_resp(content=content)) == "brief reasoning"


def test_inline_think_block_html_form():
    """<think>...</think> form (DeepSeek, R1-distilled)."""
    content = "<think>quick thought</think>visible answer"
    assert _thought(_resp(content=content)) == "quick thought"


def test_strip_inline_thought_removes_all_forms():
    content = (
        "<|channel>thought<channel|>A<channel|>"
        "<|think|>B<|/think|>"
        "<think>C</think>"
        "visible"
    )
    stripped = _strip_inline_thought(content)
    assert "thought" not in stripped.lower()
    assert "<|channel>" not in stripped
    assert "<|think|>" not in stripped
    assert "<think>" not in stripped
    assert "visible" in stripped


def test_strip_preserves_content_without_thought():
    content = "no thought here, just answer"
    assert _strip_inline_thought(content) == content


def test_empty_reasoning_content_falls_through_to_inline():
    """Empty `reasoning_content` shouldn't shadow an inline block."""
    resp = _resp(
        content="<think>actual thought</think>answer",
        reasoning="   ",  # whitespace only
    )
    assert _thought(resp) == "actual thought"


def test_malformed_response_returns_none():
    """Defensive: missing choices/message shouldn't raise."""
    assert _thought({}) is None
    assert _thought({"choices": []}) is None
    assert _thought({"choices": [{}]}) is None


def test_anthropic_thinking_block_lifted_to_reasoning_content():
    """Verify the contract the modified `_llm_call_anthropic` upholds:
    if a thinking block is present in the response, it ends up in
    `message.reasoning_content` so `_thought()` finds it uniformly."""
    # Simulate the normalized output of `_llm_call_anthropic` for a thinking-on call
    resp = _resp(
        content="The answer is 42.",
        reasoning="The user asked for the meaning of life. Standard reference.",
    )
    assert _thought(resp).startswith("The user asked")


def test_thought_strips_surrounding_whitespace():
    resp = _resp(reasoning="\n\n  some thought  \n\n")
    assert _thought(resp) == "some thought"
