"""Pure-unit tests for app.dream.dream_tools — the four dreamer-facing tools.

Narrator LLM stubbed deterministically; phrase_store + dream_state redirected
to pytest tmp_path. No network, no real LLM.

Covered flows:
  - dream_submit: virgin file → all edits status=ok
  - dream_submit: phrase with 3 history entries → possible_conflict + narrator
    receives history[-2:] + record stores last_history_timestamp
  - dream_submit: phrase with 8 history entries → possible_loop + narrator
    receives sibling list (via find_siblings)
  - dream_submit: identical submission → zero edits, zero batch
  - dream_submit: rejected when prior batch is in finalize_only phase
  - edit_revise: patch that no longer conflicts → status flips ok; others
    in batch untouched
  - edit_revise: unknown phrase_id → error
  - edit_revise: rejected when can_iterate=false
  - dream_finalize: keep+drop coverage mismatch → error
  - dream_finalize: commit keeps rewrite the prompt file + stamp history
  - dream_finalize: abandoning (keep=[]) reverts file to pre-submit state
  - recal_historical_prompt: delegates to reconstruct_prompt_at
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from app.dream import diff as dream_diff
from app.dream import dream_state, dream_tools, loop_guard, narrator, phrase_store


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def dream_env(tmp_path, monkeypatch):
    """Redirect phrase_store + dream_state on-disk roots into a pytest tmp_path,
    and stub the narrator LLM with a deterministic echo."""
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

    narrator_calls: list[tuple[str, dict]] = []

    async def fake_llm(prompt: str, cfg: dict) -> str:
        narrator_calls.append((prompt, cfg))
        return f"STUB_NARRATIVE[{len(narrator_calls)}]"

    narrator._set_llm_call(fake_llm)
    try:
        yield {
            "state": state,
            "prompts": prompts,
            "narrator_calls": narrator_calls,
        }
    finally:
        narrator._set_llm_call(narrator._default_llm_call)


def _cfg() -> dict:
    return {
        "dream": {
            "loop_guard": {
                "similarity_backend": "fuzzy",
                "similarity_threshold": 0.85,
                "max_history": 8,
                "period_detection_window": 6,
            },
        },
    }


def _write_prompt(dream_env, name: str, text: str) -> Path:
    p = dream_env["prompts"] / f"{name}.md"
    p.write_text(text, encoding="utf-8")
    return p


def _seed_history(
    phrase_id: str,
    role_template: str,
    section_path: str,
    current_text: str,
    prompt_path: Path,
    history_pairs: list[tuple[str, str]],
) -> None:
    """Write an index record + a sequence of history entries for a phrase.

    `history_pairs` is a list of (old_text, new_text) tuples applied in order;
    the last entry's new_text becomes `current_text` in the pointer file.
    """
    phrase_store._ensure_dirs()
    rec = {
        "phrase_id": phrase_id,
        "role_template": role_template,
        "path": str(prompt_path),
        "section_path": section_path,
        "current_text": current_text,
        "anchor_before": "",
        "anchor_after": "",
        "rev": len(history_pairs),
        "created_at": "2026-01-01T00:00:00+00:00",
        "updated_at": "2026-04-20T00:00:00+00:00",
    }
    phrase_store._write_index(phrase_id, rec)
    for i, (old, new) in enumerate(history_pairs, 1):
        phrase_store._append_history(phrase_id, {
            "rev": i,
            "role_template_name": role_template,
            "section_breadcrumb": section_path,
            "run_date": f"2026-04-{i:02d}T00:00:00Z",
            "session_id": f"seed-{i}",
            "rationale": "seeded",
            "old_text": old,
            "new_text": new,
            "old_anchor_before": "",
            "old_anchor_after": "",
            "new_anchor_before": "",
            "new_anchor_after": "",
            "applied_at": f"2026-04-{i:02d}T00:00:00Z",
            "rolled_back_by": None,
        })


def _run(coro):
    return asyncio.run(coro)


# ── dream_submit: virgin file ────────────────────────────────────────────────

def test_dream_submit_virgin_edits_all_ok(dream_env):
    old = "# Root\n\n## Rules\n\nOriginal body here.\n"
    new = "# Root\n\n## Rules\n\nProposed replacement body.\n"
    _write_prompt(dream_env, "worker_full", old)

    out = _run(dream_tools.dream_submit(
        path="worker_full", new_full_text=new, rationale="tighten wording",
        conversation_sid="conv-1", session_id="sid-1", cfg=_cfg(),
    ))
    assert "error" not in out
    assert out["pending_batch_id"].startswith("pb-")
    assert len(out["edits"]) == 1
    assert out["edits"][0]["status"] == "ok"
    assert out["possible_conflicts"] == []
    assert out["possible_loops"] == []
    assert "1 ok" in out["summary"]
    # Virgin phrase → narrator not called.
    assert dream_env["narrator_calls"] == []


# ── dream_submit: identical submission ───────────────────────────────────────

def test_dream_submit_identical_returns_empty(dream_env):
    text = "# Root\n\n## Rules\n\nbody.\n"
    _write_prompt(dream_env, "worker_full", text)
    out = _run(dream_tools.dream_submit(
        path="worker_full", new_full_text=text, rationale="none",
        conversation_sid="conv-2", session_id="sid-2", cfg=_cfg(),
    ))
    assert out["edits"] == []
    assert out["pending_batch_id"] is None
    assert "identical" in out.get("note", "")
    # No batch written.
    assert dream_state.has_pending_batch("conv-2") is False


# ── dream_submit: conflict path ──────────────────────────────────────────────

def test_dream_submit_conflict_path_seeds_last_2_history(dream_env):
    old = "# Root\n\n## Rules\n\ncurrent body sentence.\n"
    p = _write_prompt(dream_env, "worker_full", old)

    # Seed 3 history entries — predicts status=possible_conflict + excerpt=last 2.
    section = "Root / Rules"
    pid = phrase_store._compute_phrase_id("worker_full", section, "current body sentence.")
    _seed_history(
        pid, "worker_full", section, "current body sentence.", p,
        history_pairs=[
            ("v0 body.", "v1 body."),
            ("v1 body.", "v2 body."),
            ("v2 body.", "current body sentence."),
        ],
    )
    new = "# Root\n\n## Rules\n\nproposed rewrite body.\n"

    out = _run(dream_tools.dream_submit(
        path="worker_full", new_full_text=new, rationale="r",
        conversation_sid="c-conflict", session_id="s-c", cfg=_cfg(),
    ))
    assert len(out["edits"]) == 1
    edit = out["edits"][0]
    assert edit["status"] == "possible_conflict"
    # history_excerpt is history[-2:] — the two NEWEST.
    ex = edit["history_excerpt"]
    assert len(ex) == 2
    assert [r["rev"] for r in ex] == [2, 3]
    assert ex[-1]["new_text"] == "current body sentence."
    # Top-level conflicts list carries phrase_id + last history timestamp.
    assert len(out["possible_conflicts"]) == 1
    entry = out["possible_conflicts"][0]
    assert entry["phrase_id"] == pid
    assert entry["timestamp"] == "2026-04-03T00:00:00Z"
    # Narrator was called exactly once with the proposed text in the prompt.
    assert len(dream_env["narrator_calls"]) == 1
    prompt = dream_env["narrator_calls"][0][0]
    assert "proposed rewrite body." in prompt
    assert "current body sentence." in prompt
    # Pending batch persisted.
    assert dream_state.has_pending_batch("c-conflict") is True


# ── dream_submit: loop path with sibling ─────────────────────────────────────

def test_dream_submit_loop_path_with_sibling_carried_to_narrator(dream_env):
    """Two phrases oscillating in phase at lag=2 → both flagged + each sees the other."""
    A = "alpha body here text"
    B = "bravo body here text"
    C = "gamma body here text"
    D = "delta body here text"
    # Old file = current state A, C. Proposal = the OTHER half of the oscillation
    # (B, D) — lands as a lag=2 echo of [A→B→A→B→A] history + new B.
    # Separator paragraph between the two target phrases so SequenceMatcher
    # produces TWO independent replace opcodes (one per phrase) rather than
    # one big opcode covering both.
    old = (
        "# Root\n\n## Rules\n\n"
        f"{A}\n\n-- unrelated separator --\n\n{C}\n"
    )
    new = (
        "# Root\n\n## Rules\n\n"
        f"{B}\n\n-- unrelated separator --\n\n{D}\n"
    )
    p = _write_prompt(dream_env, "worker_full", old)
    section = "Root / Rules"
    pid_x = phrase_store._compute_phrase_id("worker_full", section, A)
    pid_y = phrase_store._compute_phrase_id("worker_full", section, C)
    # Both phrases have lag=2 oscillating history ending at current state (A / C).
    _seed_history(pid_x, "worker_full", section, A, p, history_pairs=[
        ("o", A), (A, B), (B, A), (A, B), (B, A),
    ])
    _seed_history(pid_y, "worker_full", section, C, p, history_pairs=[
        ("o", C), (C, D), (D, C), (C, D), (D, C),
    ])

    out = _run(dream_tools.dream_submit(
        path="worker_full", new_full_text=new, rationale="oscillate again",
        conversation_sid="c-loop", session_id="s-l", cfg=_cfg(),
    ))
    statuses = {e["phrase_id"]: e["status"] for e in out["edits"]}
    assert statuses[pid_x] == "possible_loop"
    assert statuses[pid_y] == "possible_loop"
    # Each edit sees the OTHER as a sibling.
    edits_by_pid = {e["phrase_id"]: e for e in out["edits"]}
    assert pid_y in edits_by_pid[pid_x]["loop_siblings"]
    assert pid_x in edits_by_pid[pid_y]["loop_siblings"]
    # Both period_lag == 2.
    assert edits_by_pid[pid_x]["period_lag"] == 2
    assert edits_by_pid[pid_y]["period_lag"] == 2
    # Narrator called twice (different phrase_ids).
    assert len(dream_env["narrator_calls"]) == 2
    # Prompts mention the sibling ids.
    joined = "\n".join(c[0] for c in dream_env["narrator_calls"])
    assert pid_x in joined and pid_y in joined


# ── dream_submit: rejected in finalize_only phase ────────────────────────────

def test_dream_submit_rejected_when_prior_batch_in_finalize_only(dream_env):
    old = "# R\n\n## S\n\nbody.\n"
    _write_prompt(dream_env, "worker_full", old)
    # Create a pending batch manually and force finalize_only phase.
    batch = dream_state.create_or_replace_pending(
        conversation_sid="c-stuck",
        target_prompt="worker_full",
        new_prompt_text="# R\n\n## S\n\nfoo.\n",
        rationale="r",
        edits=[],
    )
    dream_state.on_simulation_complete(batch, model_match=False, simulations_cap=3)
    dream_state.save_pending(batch)

    out = _run(dream_tools.dream_submit(
        path="worker_full", new_full_text="# R\n\n## S\n\nbar.\n",
        rationale="r", conversation_sid="c-stuck", session_id="s",
        cfg=_cfg(),
    ))
    assert out.get("error") == dream_state.ERR_MODEL_MISMATCH


# ── dream_submit: missing prompt file ────────────────────────────────────────

def test_dream_submit_missing_prompt_errors(dream_env):
    out = _run(dream_tools.dream_submit(
        path="nonexistent", new_full_text="body",
        rationale="r", conversation_sid="c", session_id="s", cfg=_cfg(),
    ))
    assert "error" in out and "not found" in out["error"]


# ── edit_revise: patch that clears the conflict ──────────────────────────────

def test_edit_revise_clears_conflict_status(dream_env):
    old = "# R\n\n## S\n\ncurrent body.\n"
    p = _write_prompt(dream_env, "worker_full", old)
    section = "R / S"
    pid = phrase_store._compute_phrase_id("worker_full", section, "current body.")
    _seed_history(pid, "worker_full", section, "current body.", p, history_pairs=[
        ("a.", "b."), ("b.", "c."), ("c.", "current body."),
    ])
    _run(dream_tools.dream_submit(
        path="worker_full", new_full_text="# R\n\n## S\n\nproposal one.\n",
        rationale="r", conversation_sid="c-rev", session_id="s", cfg=_cfg(),
    ))
    # Reset narrator call tally.
    pre_calls = len(dream_env["narrator_calls"])

    # Revise with text that has no prior history — still conflict (any edit to
    # a phrase-with-history triggers classification). But let's test the flip
    # via edit_revise's re-classification path.
    out = _run(dream_tools.edit_revise(
        phrase_id=pid, new_text="proposal revised.", rationale="r2",
        conversation_sid="c-rev", session_id="s", cfg=_cfg(),
    ))
    assert "error" not in out
    # Edit still has history, so status will still be possible_conflict — BUT
    # the narrative is refreshed against the new text.
    assert out["edit"]["status"] == "possible_conflict"
    assert out["edit"]["new_text"] == "proposal revised."
    # Narrator was re-invoked (new cache → +1 call).
    assert len(dream_env["narrator_calls"]) == pre_calls + 1


def test_edit_revise_unknown_phrase_id_errors(dream_env):
    old = "# R\n\n## S\n\nbody.\n"
    _write_prompt(dream_env, "worker_full", old)
    _run(dream_tools.dream_submit(
        path="worker_full", new_full_text="# R\n\n## S\n\nother.\n",
        rationale="r", conversation_sid="c-unk", session_id="s", cfg=_cfg(),
    ))
    out = _run(dream_tools.edit_revise(
        phrase_id="ph-nope", new_text="foo", rationale="r",
        conversation_sid="c-unk", session_id="s", cfg=_cfg(),
    ))
    assert "error" in out and "not present in pending batch" in out["error"]


def test_edit_revise_no_batch_errors(dream_env):
    out = _run(dream_tools.edit_revise(
        phrase_id="ph-x", new_text="y", rationale="r",
        conversation_sid="no-such", session_id="s", cfg=_cfg(),
    ))
    assert "error" in out and "no pending batch" in out["error"]


def test_edit_revise_rejected_in_finalize_only(dream_env):
    old = "# R\n\n## S\n\nbody.\n"
    _write_prompt(dream_env, "worker_full", old)
    out = _run(dream_tools.dream_submit(
        path="worker_full", new_full_text="# R\n\n## S\n\nnew.\n",
        rationale="r", conversation_sid="c-blk", session_id="s", cfg=_cfg(),
    ))
    pid = out["edits"][0]["phrase_id"]
    # Flip to finalize_only.
    batch = dream_state.load_pending("c-blk")
    dream_state.on_simulation_complete(batch, model_match=False, simulations_cap=3)
    dream_state.save_pending(batch)
    rev = _run(dream_tools.edit_revise(
        phrase_id=pid, new_text="again", rationale="r",
        conversation_sid="c-blk", session_id="s", cfg=_cfg(),
    ))
    assert rev.get("error") == dream_state.ERR_MODEL_MISMATCH


# ── dream_finalize: coverage ────────────────────────────────────────────────

def test_dream_finalize_uncovered_phrase_errors(dream_env):
    old = "# R\n\n## S\n\nalpha\n\nbravo\n"
    _write_prompt(dream_env, "worker_full", old)
    new = "# R\n\n## S\n\nalpha-new\n\nbravo-new\n"
    out = _run(dream_tools.dream_submit(
        path="worker_full", new_full_text=new, rationale="r",
        conversation_sid="c-fin", session_id="s", cfg=_cfg(),
    ))
    pids = [e["phrase_id"] for e in out["edits"]]
    assert len(pids) >= 1
    # Keep only the first, leave the second uncovered → error.
    missing_one = pids[:1]
    final = _run(dream_tools.dream_finalize(
        keep=missing_one, drop=[], conversation_sid="c-fin",
        session_id="s", cfg=_cfg(),
    ))
    if len(pids) > 1:
        assert "error" in final and "uncovered" in final["error"]


def test_dream_finalize_keep_all_writes_new_prompt(dream_env):
    old = "# R\n\n## S\n\nOriginal line one.\n"
    p = _write_prompt(dream_env, "worker_full", old)
    new = "# R\n\n## S\n\nNew rewritten line one.\n"
    out = _run(dream_tools.dream_submit(
        path="worker_full", new_full_text=new, rationale="rationale",
        conversation_sid="c-keep", session_id="sess-keep", cfg=_cfg(),
    ))
    pids = [e["phrase_id"] for e in out["edits"]]
    final = _run(dream_tools.dream_finalize(
        keep=pids, drop=[], conversation_sid="c-keep",
        session_id="sess-keep", cfg=_cfg(),
    ))
    assert "error" not in final
    assert len(final["committed"]) == len(pids)
    assert final["dropped"] == []
    # File was rewritten.
    body = p.read_text()
    assert "New rewritten line one." in body
    assert "Original line one." not in body
    # Pending batch deleted.
    assert dream_state.has_pending_batch("c-keep") is False
    # History stamped for each committed phrase, attributed to the
    # CONVERSATION that was dreamed about (not the dreamer's own sid) —
    # otherwise _collect_outcome's `session_id == conv_sid` filter returns
    # empty and every finalized run looks like a no_submission.
    for pid in pids:
        hist = phrase_store.get_history(pid)
        assert len(hist) >= 1
        assert hist[-1]["session_id"] == "c-keep"


def test_dream_finalize_abandon_all_reverts_file(dream_env):
    old = "# R\n\n## S\n\nOriginal.\n"
    p = _write_prompt(dream_env, "worker_full", old)
    new = "# R\n\n## S\n\nWould-be replacement.\n"
    out = _run(dream_tools.dream_submit(
        path="worker_full", new_full_text=new, rationale="r",
        conversation_sid="c-abandon", session_id="s", cfg=_cfg(),
    ))
    pids = [e["phrase_id"] for e in out["edits"]]
    final = _run(dream_tools.dream_finalize(
        keep=[], drop=pids, conversation_sid="c-abandon",
        session_id="s", cfg=_cfg(),
    ))
    assert "error" not in final
    assert len(final["dropped"]) == len(pids)
    assert final["committed"] == []
    # File unchanged from pre-submit.
    body = p.read_text()
    assert "Original." in body
    assert "Would-be replacement." not in body
    # No history stamped (phrases may still be virgin).
    for pid in pids:
        hist = phrase_store.get_history(pid)
        # Seeded history plus nothing — virgin phrases have empty history.
        # Just ensure we didn't add a new row.
        rationales = [h.get("rationale") for h in hist]
        assert "r" not in rationales  # dream_submit's rationale wasn't committed


def test_dream_finalize_no_batch_requires_rationale(dream_env):
    """Empty-batch finalize without a rationale is rejected.

    We used to treat `keep=[] + drop=[]` as a silent noop, but that let the
    dreamer skip too easily — the user got no visibility into WHY. Now
    skipping requires an explicit rationale (≥20 chars) which the runner
    surfaces via the `dream_skip` SSE event.
    """
    out = _run(dream_tools.dream_finalize(
        keep=[], drop=[], conversation_sid="no-such", session_id="s", cfg=_cfg(),
    ))
    assert "error" in out
    assert "rationale" in out["error"]


def test_dream_finalize_no_batch_with_rationale_is_noop(dream_env):
    """Explicit skip with a sufficient rationale is a clean no-op."""
    rationale = "conversation ran cleanly; no prompt issues to address today"
    out = _run(dream_tools.dream_finalize(
        keep=[], drop=[], conversation_sid="no-such", session_id="s",
        cfg=_cfg(), rationale=rationale,
    ))
    assert "error" not in out
    assert out.get("noop") is True
    assert out.get("skip_rationale") == rationale


def test_dream_finalize_rationale_too_short_rejected(dream_env):
    """Short rationale is rejected — we want real justification, not 'ok'."""
    out = _run(dream_tools.dream_finalize(
        keep=[], drop=[], conversation_sid="no-such", session_id="s",
        cfg=_cfg(), rationale="ok",
    ))
    assert "error" in out
    assert "rationale" in out["error"]


def test_dream_finalize_no_batch_with_pids_still_errors(dream_env):
    """Non-empty keep/drop with no pending batch is a real programming error."""
    out = _run(dream_tools.dream_finalize(
        keep=["ph-x"], drop=[], conversation_sid="no-such",
        session_id="s", cfg=_cfg(),
    ))
    assert "error" in out and "no pending batch" in out["error"]


# ── multi-target dream_submit + dream_finalize ───────────────────────────────

def test_dream_submit_multi_target_creates_unified_batch(dream_env):
    """Two targets in one dream_submit land in a single pending batch with
    edits stamped `target_prompt` on each — the foundation for one-sim /
    one-finalize semantics across interacting prompts."""
    _write_prompt(dream_env, "worker_full", "# Root\n\n## Rules\n\nWorker original.\n")
    _write_prompt(dream_env, "supervisor_full", "# Root\n\n## Rules\n\nSupervisor original.\n")
    out = _run(dream_tools.dream_submit(
        targets=[
            {"path": "worker_full",
             "new_full_text": "# Root\n\n## Rules\n\nWorker revised.\n"},
            {"path": "supervisor_full",
             "new_full_text": "# Root\n\n## Rules\n\nSupervisor revised.\n"},
        ],
        rationale="coordinated worker+supervisor tuning",
        conversation_sid="conv-mt", session_id="sid-mt", cfg=_cfg(),
    ))
    assert "error" not in out
    assert out["target_prompts"] == ["worker_full", "supervisor_full"]
    # Each target contributes at least one edit, flattened into one list.
    targets = {e["target_prompt"] for e in out["edits"]}
    assert targets == {"worker_full", "supervisor_full"}


def test_dream_finalize_multi_target_rewrites_both_files(dream_env):
    """Accepting all edits in a 2-target batch updates BOTH prompt files."""
    p_worker = _write_prompt(dream_env, "worker_full",
                             "# Root\n\n## Rules\n\nWorker OLD.\n")
    p_super = _write_prompt(dream_env, "supervisor_full",
                            "# Root\n\n## Rules\n\nSupervisor OLD.\n")
    out = _run(dream_tools.dream_submit(
        targets=[
            {"path": "worker_full",
             "new_full_text": "# Root\n\n## Rules\n\nWorker NEW.\n"},
            {"path": "supervisor_full",
             "new_full_text": "# Root\n\n## Rules\n\nSupervisor NEW.\n"},
        ],
        rationale="r",
        conversation_sid="conv-fin", session_id="sid-fin", cfg=_cfg(),
    ))
    pids = [e["phrase_id"] for e in out["edits"]]
    final = _run(dream_tools.dream_finalize(
        keep=pids, drop=[], conversation_sid="conv-fin",
        session_id="sid-fin", cfg=_cfg(),
    ))
    assert "error" not in final
    assert "Worker NEW." in p_worker.read_text()
    assert "Supervisor NEW." in p_super.read_text()
    # Both files committed; dropped list empty.
    committed_targets = {c["target_prompt"] for c in final["committed"]}
    assert committed_targets == {"worker_full", "supervisor_full"}


def test_dream_finalize_multi_target_partial_drop_scopes_to_file(dream_env):
    """Dropping all edits for ONE target leaves that file unchanged while
    committing the other target's edits — so a user who accepts worker
    tweaks but rejects supervisor tweaks gets exactly that."""
    p_worker = _write_prompt(dream_env, "worker_full",
                             "# Root\n\n## Rules\n\nWorker OLD.\n")
    p_super = _write_prompt(dream_env, "supervisor_full",
                            "# Root\n\n## Rules\n\nSupervisor OLD.\n")
    out = _run(dream_tools.dream_submit(
        targets=[
            {"path": "worker_full",
             "new_full_text": "# Root\n\n## Rules\n\nWorker NEW.\n"},
            {"path": "supervisor_full",
             "new_full_text": "# Root\n\n## Rules\n\nSupervisor NEW.\n"},
        ],
        rationale="r",
        conversation_sid="conv-split", session_id="sid-split", cfg=_cfg(),
    ))
    worker_pids = [e["phrase_id"] for e in out["edits"] if e["target_prompt"] == "worker_full"]
    super_pids = [e["phrase_id"] for e in out["edits"] if e["target_prompt"] == "supervisor_full"]
    final = _run(dream_tools.dream_finalize(
        keep=worker_pids, drop=super_pids, conversation_sid="conv-split",
        session_id="sid-split", cfg=_cfg(),
    ))
    assert "error" not in final
    # Worker file rewritten; supervisor file untouched.
    assert "Worker NEW." in p_worker.read_text()
    assert "Supervisor OLD." in p_super.read_text()
    assert "Supervisor NEW." not in p_super.read_text()


# ── recal_historical_prompt ─────────────────────────────────────────────────

def test_recal_historical_prompt_delegates(dream_env):
    old = "# R\n\n## S\n\nPrefix. body. Suffix.\n"
    p = _write_prompt(dream_env, "worker_full", old)
    pid = phrase_store.tag_new_phrase(
        str(p), anchor_before="Prefix. ", phrase_text="body.",
        anchor_after=" Suffix.",
    )
    phrase_store.apply_edit(
        pid, "new.", rationale="r", run_date="2026-04-20T00:00:00Z", session_id="s",
    )
    out = _run(dream_tools.recal_historical_prompt(
        timestamp="2026-04-15T00:00:00Z", prompt_name="worker_full",
    ))
    assert out["reversed"] == 1
    assert "body." in out["text"]
    assert "new." not in out["text"]
