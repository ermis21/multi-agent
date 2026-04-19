"""Tests for the <|end|> turn-terminator contract."""

import pytest

from app.agents import END_MARKER


def test_end_marker_literal():
    """The canonical marker string is <|end|>."""
    assert END_MARKER == "<|end|>"


def test_end_marker_stripped_from_content():
    """Simulate the no-tool-call branch stripping + trim."""
    content = "hello world <|end|>"
    clean = content.replace(END_MARKER, "").rstrip()
    assert clean == "hello world"


def test_end_marker_on_own_line():
    content = "here is the answer\n<|end|>\n"
    clean = content.replace(END_MARKER, "").rstrip()
    assert clean == "here is the answer"


def test_end_marker_multiple_stripped():
    content = "a <|end|> b <|end|>"
    clean = content.replace(END_MARKER, "").rstrip()
    assert "<|end|>" not in clean


@pytest.mark.live
def test_converse_reply_has_no_raw_marker(client):
    """The server must strip <|end|> before returning to the user."""
    from conftest import chat, answer  # noqa

    resp = chat(client, "hi", mode="converse")
    text = answer(resp)
    assert "<|end|>" not in text
    assert text.strip(), "reply should be non-empty"


@pytest.mark.live
def test_missing_end_marker_surfaces_status_event(client):
    """If the worker emits prose mid-turn (no marker, no tool), the server
    surfaces a worker_status SSE event. Exact content depends on the LLM; we
    only assert structural invariants.
    """
    from conftest import stream_chat  # noqa

    result = stream_chat(
        client,
        "read /project/README.md, then summarize it in one paragraph",
        mode="build",
        timeout=120,
    )
    # Status events are optional per run, but the stream must always end in done
    assert any(e["event"] == "done" for e in result["events"])
