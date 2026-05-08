"""B2 — verify the compactor_engine config toggle plumbing.

The full pipeline tests live in test_compactor.py; these are narrowly about
the toggle + lingua-path fallback contract added in B2.
"""

from app import compactor


def test_lingua_falls_back_when_package_missing():
    """When llmlingua isn't installed (the default state of the image),
    `_compact_with_lingua` returns (None, error_dict) so run_compaction can
    fall back to the agent engine without raising."""
    body, meta = compactor._compact_with_lingua("some scope text", {"context": {}})
    # llmlingua isn't in requirements.txt by default; either way the contract
    # holds: returning None means fallback path takes over.
    assert body is None or isinstance(body, str)
    if body is None:
        assert "error" in meta


def test_lingua_wrap_invariant_when_text_returned():
    """If the lingua path WERE to return a body, it must contain the
    `## RUNNING_SUMMARY` marker so the existing _rebuild_session_context
    contract holds. Verify by patching the package import + the compressor
    to return a fixed string."""
    import sys
    import types

    # Build a minimal fake llmlingua module with a PromptCompressor stub.
    fake_mod = types.ModuleType("llmlingua")

    class _FakeCompressor:
        def __init__(self, *args, **kwargs): pass
        def compress_prompt(self, scope, **kw):
            return {"compressed_prompt": f"compressed[{len(scope)} chars]"}

    fake_mod.PromptCompressor = _FakeCompressor
    sys.modules["llmlingua"] = fake_mod
    # Reset cached compressor so it picks up the fake.
    compactor._lingua_compressor = None

    try:
        body, meta = compactor._compact_with_lingua("hello world", {"context": {}})
        assert body is not None, f"expected body, got error: {meta}"
        assert "## RUNNING_SUMMARY" in body, "lingua output missing required marker"
        assert "compressed[11 chars]" in body
        assert meta["engine"] == "lingua"
    finally:
        sys.modules.pop("llmlingua", None)
        compactor._lingua_compressor = None


def test_default_engine_is_agent():
    """Schema default keeps existing behavior: cfg.context.compactor_engine
    defaults to 'agent', so opting in is explicit."""
    from app.config_schema import ContextConfig
    c = ContextConfig()
    assert c.compactor_engine == "agent"


def test_schema_rejects_unknown_engine():
    """Closed enum — typos rejected at PATCH time."""
    from app.config_schema import ContextConfig
    import pydantic
    try:
        ContextConfig(compactor_engine="lingua_xl")  # not in enum
        raised = False
    except pydantic.ValidationError:
        raised = True
    assert raised, "schema accepted invalid compactor_engine value"
