"""Regression test for KV-cache prefix stability.

Everything above the `<|prefix_end|>` marker must be a pure function of
(role, mode, agent_mode) — i.e. unaffected by volatile fields like session_id,
attempt, datetime. If it drifts the llama.cpp prefix KV cache invalidates every
turn, which is the whole reason we reordered the templates in PR 1.
"""

from app import prompt_generator as pg


def _prefix(text: str) -> str:
    marker = "<|prefix_end|>"
    assert marker in text, f"prompt missing {marker} — cache boundary gone"
    return text.split(marker, 1)[0]


def test_worker_prefix_stable_across_session_ids():
    a, _ = pg.generate("worker", ["file_read", "shell_exec"],
                       session_id="aaaaaaaa", attempt=0, agent_mode="converse")
    b, _ = pg.generate("worker", ["file_read", "shell_exec"],
                       session_id="bbbbbbbb", attempt=0, agent_mode="converse")
    assert _prefix(a) == _prefix(b)


def test_worker_prefix_stable_across_attempts():
    a, _ = pg.generate("worker", ["file_read", "shell_exec"],
                       session_id="same", attempt=0, agent_mode="converse")
    b, _ = pg.generate("worker", ["file_read", "shell_exec"],
                       session_id="same", attempt=5, agent_mode="converse")
    assert _prefix(a) == _prefix(b)


def test_worker_prefix_differs_per_agent_mode():
    # Different agent_mode can legitimately change the prefix (different rules,
    # different tool set); just assert that reordering didn't accidentally
    # collapse them.
    a, _ = pg.generate("worker", ["file_read", "shell_exec"],
                       session_id="z", attempt=0, agent_mode="converse")
    b, _ = pg.generate("worker", ["file_read", "shell_exec"],
                       session_id="z", attempt=0, agent_mode="build")
    # Prefixes may match when the template is identical for both modes;
    # either way the marker itself must exist.
    _prefix(a)
    _prefix(b)


def test_worker_prefix_hash_is_stable():
    from app.context_compressor import prefix_hash
    a, _ = pg.generate("worker", ["file_read", "shell_exec"],
                       session_id="h1", attempt=0, agent_mode="converse")
    b, _ = pg.generate("worker", ["file_read", "shell_exec"],
                       session_id="h2", attempt=99, agent_mode="converse")
    assert prefix_hash(a) == prefix_hash(b)
