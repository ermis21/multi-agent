"""Unit tests for app.dream.review_bus — per-edit review gate.

Covered:
  - register → request_decisions → resolve round-trip
  - request_decisions emits dream_finalize_review onto the queue
  - resolve on a non-active sid returns False
  - resolve twice: second call is a no-op (future already done)
  - timeout with no resolution returns {} + emits dream_finalize_review_timeout
  - unregister on an awaiter unblocks it with {} (drop-all)

And one integration test against dream_tools.dream_finalize:
  - when review_bus is armed, the user's decisions override the dreamer's
    keep/drop args (partial-keep).
"""

from __future__ import annotations

import asyncio
import importlib

import pytest

from app.dream import review_bus


@pytest.fixture(autouse=True)
def _clear_bus():
    # Isolate each test from the module-global dict.
    review_bus._active.clear()  # type: ignore[attr-defined]
    yield
    review_bus._active.clear()  # type: ignore[attr-defined]


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro) if False else asyncio.run(coro)


# ── Core round-trip ───────────────────────────────────────────────────────────

def test_register_is_active_and_unregister():
    q: asyncio.Queue = asyncio.Queue()
    assert review_bus.is_active("sid-1") is False
    review_bus.register("sid-1", q)
    assert review_bus.is_active("sid-1") is True
    assert "sid-1" in review_bus.active_sids()
    review_bus.unregister("sid-1")
    assert review_bus.is_active("sid-1") is False


def test_request_decisions_resolve_round_trip():
    async def scenario():
        q: asyncio.Queue = asyncio.Queue()
        review_bus.register("sid-2", q)
        summary = {"dreamer_sid": "sid-2", "edits": [{"phrase_id": "p1"}, {"phrase_id": "p2"}]}

        async def resolver():
            # The event must land on the queue before we resolve.
            evt = await asyncio.wait_for(q.get(), timeout=2.0)
            assert evt["event"] == "dream_finalize_review"
            assert evt["data"] == summary
            assert review_bus.resolve("sid-2", {"p1": "keep", "p2": "drop"}) is True

        decisions_task = asyncio.create_task(
            review_bus.request_decisions("sid-2", summary, timeout_s=5.0)
        )
        await resolver()
        decisions = await decisions_task
        assert decisions == {"p1": "keep", "p2": "drop"}
        review_bus.unregister("sid-2")

    _run(scenario())


def test_resolve_returns_false_when_not_active():
    assert review_bus.resolve("unknown-sid", {"p1": "keep"}) is False


def test_resolve_twice_second_is_noop():
    async def scenario():
        q: asyncio.Queue = asyncio.Queue()
        review_bus.register("sid-3", q)
        summary = {"dreamer_sid": "sid-3", "edits": []}

        async def resolver():
            await asyncio.wait_for(q.get(), timeout=2.0)
            assert review_bus.resolve("sid-3", {"p1": "keep"}) is True
            # Second call: future already resolved → False
            assert review_bus.resolve("sid-3", {"p1": "drop"}) is False

        task = asyncio.create_task(review_bus.request_decisions("sid-3", summary, timeout_s=5.0))
        await resolver()
        decisions = await task
        assert decisions == {"p1": "keep"}
        review_bus.unregister("sid-3")

    _run(scenario())


def test_timeout_emits_timeout_event_and_returns_empty():
    async def scenario():
        q: asyncio.Queue = asyncio.Queue()
        review_bus.register("sid-4", q)
        summary = {"dreamer_sid": "sid-4", "edits": []}
        # Short timeout; never resolve.
        decisions = await review_bus.request_decisions("sid-4", summary, timeout_s=0.1)
        assert decisions == {}
        # Drain queue: review event then timeout event.
        events: list[str] = []
        while not q.empty():
            events.append(q.get_nowait()["event"])
        assert "dream_finalize_review" in events
        assert "dream_finalize_review_timeout" in events
        review_bus.unregister("sid-4")

    _run(scenario())


def test_unregister_while_awaiting_unblocks_with_empty():
    async def scenario():
        q: asyncio.Queue = asyncio.Queue()
        review_bus.register("sid-5", q)
        summary = {"dreamer_sid": "sid-5", "edits": []}
        task = asyncio.create_task(review_bus.request_decisions("sid-5", summary, timeout_s=5.0))
        # Let the emit happen.
        await asyncio.wait_for(q.get(), timeout=2.0)
        review_bus.unregister("sid-5")
        decisions = await asyncio.wait_for(task, timeout=2.0)
        assert decisions == {}

    _run(scenario())


# ── Integration with dream_finalize ───────────────────────────────────────────

def test_dream_finalize_applies_user_decisions_over_dreamer_args(tmp_path, monkeypatch):
    """With review_bus armed, user's keep/drop beats the dreamer's args.

    Dreamer says keep=[p1, p2]; user says keep=[p1], drop=[p2] → only p1 lands.
    """
    from app.dream import dream_state, dream_tools, narrator, phrase_store

    state = tmp_path / "state"
    prompts = tmp_path / "prompts"
    (state / "dream" / "runs").mkdir(parents=True)
    prompts.mkdir()

    monkeypatch.setattr(phrase_store, "STATE_DIR", state)
    monkeypatch.setattr(phrase_store, "DREAM_ROOT", state / "dream")
    monkeypatch.setattr(phrase_store, "INDEX_DIR", state / "dream" / "phrase_index")
    monkeypatch.setattr(phrase_store, "HISTORY_DIR", state / "dream" / "phrase_history")
    monkeypatch.setattr(phrase_store, "PROMPTS_DIR", prompts)
    monkeypatch.setattr(dream_state, "DREAM_RUNS_ROOT", state / "dream" / "runs")

    async def fake_llm(prompt: str, cfg: dict) -> str:
        return "narrator-stub"
    narrator._set_llm_call(fake_llm)

    try:
        old = "# R\n\n## S1\n\nalpha-old.\n\n## S2\n\nbravo-old.\n"
        prompt_path = prompts / "worker_full.md"
        prompt_path.write_text(old, encoding="utf-8")

        new = "# R\n\n## S1\n\nalpha-new.\n\n## S2\n\nbravo-new.\n"
        cfg = {"dream": {"loop_guard": {
            "similarity_backend": "fuzzy",
            "similarity_threshold": 0.85,
            "max_history": 8,
            "period_detection_window": 6,
        }}}

        async def run_submit():
            return await dream_tools.dream_submit(
                path="worker_full", new_full_text=new, rationale="r",
                conversation_sid="c-review", session_id="sess-review", cfg=cfg,
            )
        submit_result = asyncio.run(run_submit())
        pids = [e["phrase_id"] for e in submit_result["edits"]]
        assert len(pids) >= 2, f"need at least 2 phrase edits for this test, got {pids!r}"

        async def run_review_scenario():
            # Arm the bus for this dreamer session.
            q: asyncio.Queue = asyncio.Queue()
            review_bus.register("sess-review", q)

            async def reviewer():
                # Wait for the review event, then resolve keeping only pids[0].
                evt = await asyncio.wait_for(q.get(), timeout=5.0)
                assert evt["event"] == "dream_finalize_review"
                review_bus.resolve(
                    "sess-review",
                    {pids[0]: "keep", **{pid: "drop" for pid in pids[1:]}},
                )

            finalize_task = asyncio.create_task(dream_tools.dream_finalize(
                # Dreamer wanted them all kept — user's verdict must override.
                keep=pids, drop=[], conversation_sid="c-review",
                session_id="sess-review", cfg=cfg,
            ))
            await reviewer()
            return await finalize_task

        final = asyncio.run(run_review_scenario())
        assert "error" not in final, final
        committed_pids = {c["phrase_id"] for c in final["committed"]}
        dropped_pids = {d["phrase_id"] for d in final["dropped"]}
        assert committed_pids == {pids[0]}
        assert dropped_pids == set(pids[1:])

        body = prompt_path.read_text()
        # First edit landed; second reverted.
        assert "alpha-new." in body
        assert "bravo-new." not in body
        assert "bravo-old." in body

        # Pending batch cleaned up.
        assert dream_state.has_pending_batch("c-review") is False
    finally:
        narrator._set_llm_call(narrator._default_llm_call)
