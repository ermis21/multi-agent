"""Pure-unit tests for discord/utils.split_message fence-awareness.

Covers UX audit item 7: code-fence-unaware split leaves an unterminated fence
in msg N and a dangling opener in msg N+1 — Discord then renders the opener
as literal backticks.

discord/utils.py isn't on sys.path inside phoebe-api (the discord bot lives
in a sibling container), so we exec the source into an isolated namespace
with a patched MAX_MSG_LEN — pure-unit, no import side effects.
"""

from __future__ import annotations

import os
import types
from pathlib import Path


def _fresh_utils(max_len: int):
    """Exec discord/utils.py with a stubbed env and return the resulting module.

    The discord source lives in a sibling container; when run from
    phoebe-discord we see it at /app/utils.py, and from a host-side pytest
    at <repo>/discord/utils.py. Either works.
    """
    candidates = [
        Path("/app/utils.py"),                                         # phoebe-discord
        Path("/project/discord/utils.py"),                             # phoebe-api (ro mount)
        Path(__file__).resolve().parent.parent / "discord" / "utils.py",  # host
    ]
    src = next((p for p in candidates if p.exists()), None)
    if src is None:
        import pytest  # noqa: PLC0415
        pytest.skip(f"discord/utils.py not reachable from this container")
    os.environ["DISCORD_MAX_MESSAGE_LENGTH"] = str(max_len)
    mod = types.ModuleType("_utils_fresh")
    exec(src.read_text(), mod.__dict__)
    return mod


def test_short_text_one_chunk():
    utils = _fresh_utils(100)
    assert utils.split_message("hello") == ["hello"]


def test_long_plain_text_splits_at_newline():
    utils = _fresh_utils(50)
    text = "\n".join(f"line {i} is here" for i in range(20))
    chunks = utils.split_message(text)
    assert all(len(c) <= 50 for c in chunks), [len(c) for c in chunks]
    # No chunk should end mid-line when a newline was available.
    for c in chunks[:-1]:
        assert not c.endswith(" "), c


def test_fence_carry_preserves_tag():
    utils = _fresh_utils(80)
    body = "\n".join(f"print({i})" for i in range(40))
    text = f"```python\n{body}\n```"
    chunks = utils.split_message(text)
    # Every chunk must have balanced fences on its own.
    for c in chunks:
        assert c.count("```") % 2 == 0, f"unbalanced fence in chunk: {c!r}"
    # Continuation chunks re-open with the original tag.
    for c in chunks[1:]:
        assert c.startswith("```python"), f"tag lost: {c[:30]!r}"


def test_fence_tagless_carries_blank():
    utils = _fresh_utils(60)
    body = "\n".join(str(i) * 10 for i in range(20))
    text = f"```\n{body}\n```"
    chunks = utils.split_message(text)
    for c in chunks:
        assert c.count("```") % 2 == 0, c
    # Tagless fences re-open with bare ``` (no language).
    for c in chunks[1:]:
        assert c.startswith("```\n") or c.startswith("```"), c[:20]


def test_prose_before_and_after_fence():
    utils = _fresh_utils(100)
    text = (
        "Intro paragraph here.\n"
        + "```python\n"
        + "\n".join(f"x = {i}" for i in range(30))
        + "\n```\n"
        + "Outro paragraph after."
    )
    chunks = utils.split_message(text)
    # Fence parity per chunk.
    for c in chunks:
        assert c.count("```") % 2 == 0, c
    # Outro must survive to the final chunk.
    assert "Outro" in chunks[-1]


def test_empty_string_produces_empty_list():
    utils = _fresh_utils(100)
    assert utils.split_message("") == []


def test_fence_at_exact_boundary_not_double_opened():
    """Chunk boundary that happens to land right on ``` closer shouldn't
    leave a stub chunk that is just ```<tag>\n```."""
    utils = _fresh_utils(60)
    body = "\n".join("y = 2" for _ in range(10))
    text = f"```python\n{body}\n```"
    chunks = utils.split_message(text)
    # No chunk should be just the scaffolding (opener + closer, nothing in
    # between).
    for c in chunks:
        stripped = c.strip()
        assert stripped not in {"```python\n```", "```python", "```"}, c
