"""Unit tests for the MAD debate engine (B1).

Covers parser correctness + the run_debate adapter's engine-routing.
LLM-driven scenarios are covered by the live debate scenarios in
discord/e2e_scenarios.py — out of unit-test scope.
"""

from app import debate_mad


def test_moderator_parse_a_wins():
    out = debate_mad._parse_moderator(
        "DECISION: a_wins\nREASON: clear convergence\nCONFIDENCE: 0.91"
    )
    assert out == {"decision": "a_wins", "reason": "clear convergence", "confidence": 0.91}


def test_moderator_parse_continue_default_on_garbage():
    # Unparseable input must not crash; defaults to "continue" so the loop
    # keeps going rather than calling a phantom winner.
    out = debate_mad._parse_moderator("garbage in")
    assert out == {"decision": "continue", "reason": "", "confidence": 0.5}


def test_moderator_parse_invalid_decision_falls_back_to_continue():
    out = debate_mad._parse_moderator("DECISION: nuclear_war\nREASON: ?\nCONFIDENCE: 0.5")
    assert out["decision"] == "continue"


def test_judge_parse_clean_winner_b():
    out = debate_mad._parse_judge(
        "WINNER: B\nREASON: stronger evidence chain\nCONFIDENCE: 0.8\nKEY_POINT: scalability"
    )
    assert out == {
        "winner": "B", "reason": "stronger evidence chain",
        "confidence": 0.8, "key_point": "scalability",
    }


def test_judge_parse_invalid_winner_drops_field():
    out = debate_mad._parse_judge("WINNER: maybe\nREASON: hmm\nCONFIDENCE: 0.4")
    assert out["winner"] == ""  # invalid value → field stays empty


def test_build_result_matches_legacy_shape():
    state = {
        "transcript": ["[A opening] hello", "[B opening] world"],
        "messages_total": 2,
        "position_a": "yes",
        "position_b": "no",
        "concluded": True,
        "ended_by": "moderator_a_wins",
        "concessions": [],
        "rounds": 2,
    }
    verdict = {"winner": "A", "reason": "r", "confidence": 0.9, "key_point": "k", "source": "moderator"}
    result = debate_mad._build_result("debate_mad_xyz", state, verdict)
    # Required fields the run_debate adapter must produce in both engines:
    for k in ("debate_id", "status", "messages_total", "position_a", "position_b",
              "transcript", "last_a", "last_b", "concessions", "ended_by",
              "judge_winner", "judge_reason", "judge_confidence", "judge_key_point"):
        assert k in result, f"missing field {k!r} (legacy shape contract violated)"
    assert result["judge_winner"] == "A"
    assert result["engine"] == "mad"
    assert result["verdict_source"] == "moderator"


def test_last_advocate_text_finds_latest():
    transcript = [
        "[A opening] alpha",
        "[B opening] beta",
        "[A round 2] gamma",
        "[B round 2] delta",
        "[moderator round 2] decision=continue",
    ]
    assert debate_mad._last_advocate_text(transcript, "A") == "gamma"
    assert debate_mad._last_advocate_text(transcript, "B") == "delta"


def test_run_debate_routes_by_config(monkeypatch):
    """The adapter in app/debate.py must dispatch to the MAD engine when
    cfg.debate.engine='mad'. We monkey-patch the engine entry point to
    capture the call without firing an LLM."""
    from app import debate as debate_mod

    captured: dict = {}

    async def fake_mad(**kwargs):
        captured.update(kwargs)
        return {"debate_id": "fake", "status": "concluded", "engine": "mad"}

    # Stub the import inside run_debate. The adapter does
    # `from app.debate_mad import run_debate_mad`, so patch that symbol.
    monkeypatch.setattr(debate_mad, "run_debate_mad", fake_mad)
    # Force config to claim engine="mad"
    monkeypatch.setattr(debate_mod, "get_config", lambda: {"debate": {"engine": "mad"}})

    import asyncio
    out = asyncio.run(debate_mod.run_debate(
        question="q", context="c", position_a="A", position_b="B",
        session_id="s", debate_id="", max_exchanges=0,
    ))
    assert captured["question"] == "q"
    assert captured["position_a"] == "A"
    assert out["engine"] == "mad"


def test_run_debate_default_engine_is_current(monkeypatch):
    """When engine is unset or 'current', the legacy path runs (we just check
    it does NOT call into debate_mad). Sentinel via recording the import."""
    from app import debate as debate_mod

    called: list[bool] = []

    async def trip(**kwargs):
        called.append(True)
        return {}

    monkeypatch.setattr(debate_mad, "run_debate_mad", trip)
    monkeypatch.setattr(debate_mod, "get_config", lambda: {"debate": {}})

    # The legacy path needs _llm_call which would fail at runtime — patch it
    # so the call short-circuits cleanly. We only care that MAD wasn't taken.
    async def fake_llm(*a, **kw):
        return {"choices": [{"message": {"content": "POINT: x\nEVIDENCE: y"}}]}

    from app import agents as agents_mod
    monkeypatch.setattr(agents_mod, "_llm_call", fake_llm)

    import asyncio
    try:
        asyncio.run(debate_mod.run_debate(
            question="q", context="c", position_a="A", position_b="B",
            session_id="s", debate_id="", max_exchanges=2,
        ))
    except Exception:
        # Legacy engine has a richer state machine that may fail without
        # full plumbing — we don't care; the assertion below is the test.
        pass
    assert not called, "MAD engine fired when engine config was 'current'"
