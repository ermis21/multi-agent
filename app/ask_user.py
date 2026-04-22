"""Inline multi-choice question gate.

Parallels `mcp_client.resolve_approval`: the worker calls `ask_user_question`,
which POSTs a question to the Discord bot, parks an `asyncio.Event` keyed by
`question_id`, and awaits it. The user's button click in Discord round-trips to
`POST /v1/question_response`, which calls `resolve_question` to set the event.
"""

from __future__ import annotations

import asyncio
import os
from uuid import uuid4

import httpx

DISCORD_URL = os.environ.get("DISCORD_URL", "http://phoebe-discord:4000")
_QUESTION_TIMEOUT_S = 660  # parity with approval timeout

_http = httpx.AsyncClient(timeout=10)

# question_id → {event, answer, answer_text}
_pending_questions: dict[str, dict] = {}


def resolve_question(question_id: str, answer: str, answer_text: str) -> bool:
    """Unblock a waiting `ask_user_question` call. Returns False if id unknown."""
    state = _pending_questions.get(question_id)
    if not state:
        return False
    state["answer"] = answer
    state["answer_text"] = answer_text
    state["event"].set()
    return True


async def ask_user_question(
    question: str,
    options: list[str],
    context: str = "",
    session_id: str = "",
) -> dict:
    """Post a multiple-choice question to Discord and wait for the user to click."""
    if not question or not options:
        return {"error": "ask_user requires 'question' and non-empty 'options'."}
    if not 2 <= len(options) <= 5:
        return {"error": "ask_user requires 2-5 options."}
    if not session_id:
        return {"error": "ask_user requires a session_id bound to a Discord channel."}

    question_id = uuid4().hex
    event = asyncio.Event()
    _pending_questions[question_id] = {
        "event": event,
        "answer": "",
        "answer_text": "",
    }
    try:
        try:
            r = await _http.post(
                f"{DISCORD_URL}/discord/ask_question",
                json={
                    "question": question,
                    "options": options,
                    "question_id": question_id,
                    "session_id": session_id,
                },
            )
        except Exception as e:
            return {"error": f"Discord unreachable: {e}"}
        if r.status_code != 200 or not r.json().get("ok"):
            detail = r.json().get("error") if r.status_code == 200 else r.text
            return {"error": f"Discord refused question: {detail}"}

        try:
            await asyncio.wait_for(event.wait(), timeout=_QUESTION_TIMEOUT_S)
        except asyncio.TimeoutError:
            return {"error": "User did not answer within timeout."}

        state = _pending_questions[question_id]
        return {
            "answer": state["answer_text"] or state["answer"],
            "letter": state["answer"],
        }
    finally:
        _pending_questions.pop(question_id, None)
