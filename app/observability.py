"""OpenInference instrumentation (B6).

Wires OpenTelemetry traces to a self-hosted Phoenix instance running at
http://phoebe-phoenix:6006. Spans cover the four hot code paths:
  - agent workflows  (run_agent_loop, run_agent_role, run_soul_update, ...)
  - turn-level       (_run_worker, _run_supervisor)
  - tool dispatch    (call_tool)
  - LLM round-trips  (_llm_call_local, _llm_call_anthropic — Anthropic via
    openinference-instrumentation-anthropic auto-instrument; local llama.cpp
    via httpx auto-instrument or manual spans on _llm_call_local)

Gated by env var `PHOEBE_OBSERVABILITY_ENABLED=1` so a crash in OTel setup
never breaks the api startup. When disabled, `get_tracer()` returns a no-op
proxy and `traced(...)` decorators pass through cleanly.
"""

from __future__ import annotations

import os
from contextlib import contextmanager, nullcontext
from functools import wraps
from typing import Any, Callable, Iterator

_ENABLED = os.environ.get("PHOEBE_OBSERVABILITY_ENABLED", "").strip() == "1"
_PHOENIX_OTLP_ENDPOINT = os.environ.get(
    "PHOENIX_OTLP_ENDPOINT",
    "http://phoebe-phoenix:6006/v1/traces",
)
_SERVICE_NAME = os.environ.get("PHOEBE_SERVICE_NAME", "phoebe-api")

_initialized = False


def setup_observability() -> bool:
    """Initialize the global TracerProvider + Anthropic auto-instrument.

    Idempotent. Returns True iff observability is now active. Logs but does
    NOT raise on any failure — the api keeps running with no instrumentation.
    Call once from main.py lifespan startup.
    """
    global _initialized
    if not _ENABLED:
        return False
    if _initialized:
        return True
    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        resource = Resource.create({"service.name": _SERVICE_NAME})
        provider = TracerProvider(resource=resource)
        provider.add_span_processor(
            BatchSpanProcessor(OTLPSpanExporter(endpoint=_PHOENIX_OTLP_ENDPOINT))
        )
        trace.set_tracer_provider(provider)

        # Anthropic auto-instrument — wraps the SDK client so each call
        # appears as a span with model name, token counts, prompt/completion.
        try:
            from openinference.instrumentation.anthropic import AnthropicInstrumentor
            AnthropicInstrumentor().instrument()
        except Exception as e:
            print(f"[observability] anthropic instrument failed: {e}", flush=True)

        # httpx auto-instrument — covers _llm_call_local (llama-api-manager),
        # call_tool → sandbox/mcp, discord gateway calls, and the manager's
        # internal probes. Each call becomes an HTTP-client span attached to
        # whichever workflow span is current.
        try:
            from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
            HTTPXClientInstrumentor().instrument()
        except Exception as e:
            print(f"[observability] httpx instrument failed: {e}", flush=True)

        _initialized = True
        print(
            f"[observability] OpenInference active → {_PHOENIX_OTLP_ENDPOINT} "
            f"(service={_SERVICE_NAME})",
            flush=True,
        )
        return True
    except Exception as e:
        print(f"[observability] setup failed (instrumentation disabled): {e}", flush=True)
        return False


def get_tracer(name: str = "phoebe"):
    """Return a tracer if instrumentation is up; else a no-op proxy whose
    `start_as_current_span` is a context manager doing nothing."""
    if not _initialized:
        return _NoopTracer()
    try:
        from opentelemetry import trace
        return trace.get_tracer(name)
    except Exception:
        return _NoopTracer()


class _NoopTracer:
    @contextmanager
    def start_as_current_span(self, *_args, **_kwargs) -> Iterator[Any]:
        yield None


def traced(span_name: str, **default_attrs: Any):
    """Decorator wrapping an async function in a span. Cheap when disabled
    (single attribute lookup + nullcontext)."""
    def deco(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            tracer = get_tracer()
            ctx = tracer.start_as_current_span(span_name)
            with ctx as span:
                if span is not None and default_attrs:
                    try:
                        for k, v in default_attrs.items():
                            span.set_attribute(k, v)
                    except Exception:
                        pass
                return await func(*args, **kwargs)
        return wrapper
    return deco


def annotate(span, **attrs: Any) -> None:
    """Helper to set attributes on a current span when one exists. Safe
    when span is None (instrumentation disabled)."""
    if span is None:
        return
    try:
        for k, v in attrs.items():
            span.set_attribute(k, v)
    except Exception:
        pass
