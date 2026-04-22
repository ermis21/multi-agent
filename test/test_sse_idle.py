"""Unit test for the SSE generator's sliding idle timeout.

Regression guard: the previous absolute-deadline implementation (660 s wall
clock) killed the stream at the same moment approval waits timed out, producing
`[stream ended unexpectedly]` on the client. The generator now resets on each
event yielded, so a trickle of heartbeats keeps the stream alive indefinitely.
"""

from __future__ import annotations

import asyncio

from app.main import _sse_generator


def _run(coro):
    return asyncio.run(coro)


async def _drain(queue, idle_timeout):
    events = []
    async for chunk in _sse_generator(queue, idle_timeout=idle_timeout):
        events.append(chunk)
    return events


def test_idle_generator_survives_slow_heartbeats():
    """Events arriving below the idle bound keep the stream alive past the old 660s limit."""
    async def _run_case():
        queue: asyncio.Queue = asyncio.Queue()

        async def _feed():
            for i in range(5):
                await asyncio.sleep(0.2)
                queue.put_nowait({"event": "approval_waiting", "data": {"elapsed_s": i}})
            queue.put_nowait({"event": "done", "data": {}})

        feeder = asyncio.create_task(_feed())
        events = await _drain(queue, idle_timeout=0.5)
        await feeder
        return events

    events = _run(_run_case())
    assert sum(1 for e in events if "approval_waiting" in e) == 5
    assert any("event: done" in e for e in events)
    assert not any("stream idle timeout" in e for e in events)


def test_idle_generator_times_out_on_silence():
    """With no events at all, the idle timeout fires and emits an error."""
    async def _run_case():
        queue: asyncio.Queue = asyncio.Queue()
        return await _drain(queue, idle_timeout=0.15)

    events = _run(_run_case())
    assert any("stream idle timeout" in e for e in events)
