"""Sanitizers used at trust boundaries.

`_strip_injection_tokens` neutralises prompt-injection grammar that an
attacker might plant in attacker-controllable content (web pages,
third-party Discord/Notion bodies). It strips:
  - `<|tool_call|>` / `<tool_call>` openers/closers (any case)
  - `[user_note]` / `[user_interjection]` / `[user_clarification]` markers

The sandbox keeps a paste-copy of this regex+function in `sandbox/mcp_server.py`
because sandbox/ does not import from app/. `test/test_safety_parity.py`
asserts the two definitions match.
"""

from __future__ import annotations

import re

INJECTION_TOKEN_RE = re.compile(
    r"(<\|tool_call[^>]*\|?>|"
    r"\[user_(?:note|interjection|clarification)\][^\n]*)",
    re.IGNORECASE,
)


def strip_injection_tokens(text: str | None) -> str | None:
    """Remove tokens an attacker might smuggle through web/discord/notion."""
    if not text:
        return text
    return INJECTION_TOKEN_RE.sub("[redacted-injection-token]", text)
