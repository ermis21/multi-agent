"""Pure-unit tests for app.dream.phrase_store — no stack, no network.

Covers every public surface of the phrase store:
  - id determinism + section-path disambiguation of repeated subheaders
  - tag_new_phrase idempotence + anchor-sandwich validation
  - apply_edit round-trip + stale-anchor failure
  - rollback_last restores file + pops history
  - get_history_excerpt on 0/1/5-entry phrases
  - two independent phrase edits in the same file (anchor uniqueness)
  - phrase_locate_by_text unique-match / unknown-match paths
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.dream import phrase_store


@pytest.fixture
def dream_dirs(tmp_path, monkeypatch):
    """Redirect phrase_store's on-disk roots into a pytest tmp_path."""
    state = tmp_path / "state"
    prompts = tmp_path / "prompts"
    (state / "dream").mkdir(parents=True)
    prompts.mkdir()
    monkeypatch.setattr(phrase_store, "STATE_DIR", state)
    monkeypatch.setattr(phrase_store, "DREAM_ROOT", state / "dream")
    monkeypatch.setattr(phrase_store, "INDEX_DIR", state / "dream" / "phrase_index")
    monkeypatch.setattr(phrase_store, "HISTORY_DIR", state / "dream" / "phrase_history")
    monkeypatch.setattr(phrase_store, "PROMPTS_DIR", prompts)
    return {"state": state, "prompts": prompts}


def _write_prompt(dream_dirs, name: str, text: str) -> Path:
    p = dream_dirs["prompts"] / f"{name}.md"
    p.write_text(text, encoding="utf-8")
    return p


# ── section_path_for_offset ──────────────────────────────────────────────────

def test_section_path_nested_headers():
    text = (
        "# Root\n\nintro\n\n"
        "## Behavioral Rules\n\nrules intro\n\n"
        "### Hard rules\n\nalpha\n\n"
        "## Tools\n\n"
        "### Hard rules\n\nbeta\n\n"
    )
    alpha_off = text.index("alpha")
    beta_off = text.index("beta")
    assert phrase_store.section_path_for_offset(text, alpha_off) == \
        "Root / Behavioral Rules / Hard rules"
    assert phrase_store.section_path_for_offset(text, beta_off) == \
        "Root / Tools / Hard rules"


def test_section_path_preamble_is_empty():
    text = "some intro line\n\n# Only header\n\nbody\n"
    off = text.index("some intro")
    assert phrase_store.section_path_for_offset(text, off) == ""


# ── tag_new_phrase / _compute_phrase_id ──────────────────────────────────────

def test_tag_new_phrase_is_deterministic(dream_dirs):
    text = "# Root\n\n## Hard rules\n\nPrefix. The phrase to tag. Suffix.\n"
    p = _write_prompt(dream_dirs, "worker_full", text)
    pid1 = phrase_store.tag_new_phrase(
        str(p), anchor_before="Prefix. ", phrase_text="The phrase to tag.",
        anchor_after=" Suffix.",
    )
    pid2 = phrase_store.tag_new_phrase(
        str(p), anchor_before="Prefix. ", phrase_text="The phrase to tag.",
        anchor_after=" Suffix.",
    )
    # Same inputs → same id (idempotent).
    assert pid1 == pid2
    # Id shape.
    assert pid1.startswith("ph-") and len(pid1) == 3 + 10


def test_tag_new_phrase_content_change_changes_id(dream_dirs):
    text_a = "# Root\n\n## Rules\n\nalpha body here.\n"
    text_b = "# Root\n\n## Rules\n\nbravo body here.\n"
    pa = _write_prompt(dream_dirs, "a", text_a)
    pb = _write_prompt(dream_dirs, "b", text_b)
    pid_a = phrase_store.tag_new_phrase(
        str(pa), anchor_before="## Rules\n\n", phrase_text="alpha body here.",
        anchor_after="\n",
    )
    pid_b = phrase_store.tag_new_phrase(
        str(pb), anchor_before="## Rules\n\n", phrase_text="bravo body here.",
        anchor_after="\n",
    )
    assert pid_a != pid_b


def test_tag_new_phrase_repeated_subheaders_get_distinct_ids(dream_dirs):
    """Two `## Hard rules` under different parents must NOT collide."""
    text = (
        "# Root\n\n"
        "## Behavioral Rules\n\n### Hard rules\n\nShared sentence body.\n\n"
        "## Tools\n\n### Hard rules\n\nShared sentence body.\n"
    )
    p = _write_prompt(dream_dirs, "twohards", text)
    # We can't tag both with the same anchor-sandwich because the text appears
    # twice — but the section-path arg to _compute_phrase_id IS what disambiguates.
    pid_behavioral = phrase_store._compute_phrase_id(
        "twohards", "Root / Behavioral Rules / Hard rules", "Shared sentence body.",
    )
    pid_tools = phrase_store._compute_phrase_id(
        "twohards", "Root / Tools / Hard rules", "Shared sentence body.",
    )
    assert pid_behavioral != pid_tools


def test_tag_new_phrase_missing_sandwich_raises(dream_dirs):
    text = "# Root\n\nbody\n"
    p = _write_prompt(dream_dirs, "x", text)
    with pytest.raises(phrase_store.LocateFailure):
        phrase_store.tag_new_phrase(
            str(p), anchor_before="BOGUS", phrase_text="nothing",
            anchor_after="MORE_BOGUS",
        )


# ── apply_edit round-trip + stale-anchor failure ─────────────────────────────

def test_apply_edit_roundtrip(dream_dirs):
    text = "# Root\n\n## Rules\n\nPrefix. Edit me please. Suffix.\n"
    p = _write_prompt(dream_dirs, "worker_full", text)
    pid = phrase_store.tag_new_phrase(
        str(p), anchor_before="Prefix. ", phrase_text="Edit me please.",
        anchor_after=" Suffix.",
    )
    rev = phrase_store.apply_edit(
        pid, "Edited content now.",
        rationale="test edit", run_date="2026-04-21T04:00:00Z", session_id="sid-1",
    )
    assert rev == 1
    # File was rewritten.
    assert "Edited content now." in p.read_text()
    assert "Edit me please." not in p.read_text()
    # Pointer reflects new state; locate returns new text.
    located = phrase_store.locate_phrase(pid)
    assert located.current_text == "Edited content now."
    # History grew by one.
    hist = phrase_store.get_history(pid)
    assert len(hist) == 1 and hist[0]["new_text"] == "Edited content now."
    assert hist[0]["old_text"] == "Edit me please."
    assert hist[0]["session_id"] == "sid-1"


def test_apply_edit_stale_anchor_raises(dream_dirs):
    text = "# Root\n\n## Rules\n\nPrefix. Target. Suffix.\n"
    p = _write_prompt(dream_dirs, "wf", text)
    pid = phrase_store.tag_new_phrase(
        str(p), anchor_before="Prefix. ", phrase_text="Target.",
        anchor_after=" Suffix.",
    )
    # Out-of-band edit removes the phrase.
    p.write_text("# Root\n\n## Rules\n\nunrelated body\n", encoding="utf-8")
    with pytest.raises(phrase_store.EditConflict):
        phrase_store.apply_edit(
            pid, "does not matter",
            rationale="r", run_date="2026-04-21", session_id="s",
        )
    # History untouched.
    assert phrase_store.get_history(pid) == []


def test_apply_edit_ambiguous_current_text_raises(dream_dirs):
    text = "# Root\n\n## Rules\n\nPrefix. UNIQ. Suffix.\n"
    p = _write_prompt(dream_dirs, "wf", text)
    pid = phrase_store.tag_new_phrase(
        str(p), anchor_before="Prefix. ", phrase_text="UNIQ.",
        anchor_after=" Suffix.",
    )
    # Now there are two occurrences of the current_text → ambiguity.
    p.write_text(text + "\nAlso UNIQ. somewhere else.\n", encoding="utf-8")
    with pytest.raises(phrase_store.EditConflict):
        phrase_store.apply_edit(
            pid, "replacement",
            rationale="r", run_date="d", session_id="s",
        )


# ── rollback_last ────────────────────────────────────────────────────────────

def test_rollback_last_restores_file_and_pops_history(dream_dirs):
    text = "# Root\n\n## Rules\n\nPrefix. Original. Suffix.\n"
    p = _write_prompt(dream_dirs, "wf", text)
    pid = phrase_store.tag_new_phrase(
        str(p), anchor_before="Prefix. ", phrase_text="Original.",
        anchor_after=" Suffix.",
    )
    phrase_store.apply_edit(
        pid, "Changed.",
        rationale="x", run_date="2026-04-21", session_id="s",
    )
    assert "Changed." in p.read_text()
    popped = phrase_store.rollback_last(pid)
    assert popped is not None and popped["new_text"] == "Changed."
    assert "Original." in p.read_text()
    assert "Changed." not in p.read_text()
    assert phrase_store.get_history(pid) == []
    # Index rev walks back.
    rec = phrase_store._read_index(pid)
    assert rec["rev"] == 0 and rec["current_text"] == "Original."


def test_rollback_last_on_virgin_phrase_returns_none(dream_dirs):
    text = "# Root\n\n## Rules\n\nPrefix. Virgin. Suffix.\n"
    p = _write_prompt(dream_dirs, "wf", text)
    pid = phrase_store.tag_new_phrase(
        str(p), anchor_before="Prefix. ", phrase_text="Virgin.",
        anchor_after=" Suffix.",
    )
    assert phrase_store.rollback_last(pid) is None


# ── get_history_excerpt ──────────────────────────────────────────────────────

def test_get_history_excerpt_0_1_5_entries(dream_dirs):
    text = "# Root\n\n## R\n\nPrefix. start. Suffix.\n"
    p = _write_prompt(dream_dirs, "wf", text)
    pid = phrase_store.tag_new_phrase(
        str(p), anchor_before="Prefix. ", phrase_text="start.",
        anchor_after=" Suffix.",
    )
    # 0-entry.
    assert phrase_store.get_history_excerpt(pid, k=3) == []
    # 1-entry.
    phrase_store.apply_edit(pid, "v1.",
                            rationale="r", run_date="d1", session_id="s")
    ex1 = phrase_store.get_history_excerpt(pid, k=3)
    assert len(ex1) == 1 and ex1[0]["new_text"] == "v1."
    # Grow to 5 entries (v2..v5).
    for i in range(2, 6):
        phrase_store.apply_edit(
            pid, f"v{i}.",
            rationale="r", run_date=f"d{i}", session_id="s",
        )
    all_hist = phrase_store.get_history(pid)
    assert len(all_hist) == 5
    # k=3 returns the newest three.
    ex3 = phrase_store.get_history_excerpt(pid, k=3)
    assert [e["new_text"] for e in ex3] == ["v3.", "v4.", "v5."]
    # k=0 → [].
    assert phrase_store.get_history_excerpt(pid, k=0) == []


# ── two independent phrases in the same file ────────────────────────────────

def test_two_independent_edits_same_file(dream_dirs):
    text = (
        "# Root\n\n## R\n\n"
        "Pre1. Phrase A here. Post1.\n\n"
        "Pre2. Phrase B here. Post2.\n"
    )
    p = _write_prompt(dream_dirs, "wf", text)
    pid_a = phrase_store.tag_new_phrase(
        str(p), anchor_before="Pre1. ", phrase_text="Phrase A here.",
        anchor_after=" Post1.",
    )
    pid_b = phrase_store.tag_new_phrase(
        str(p), anchor_before="Pre2. ", phrase_text="Phrase B here.",
        anchor_after=" Post2.",
    )
    assert pid_a != pid_b
    phrase_store.apply_edit(
        pid_a, "Rewritten A.",
        rationale="r", run_date="d", session_id="s",
    )
    phrase_store.apply_edit(
        pid_b, "Rewritten B.",
        rationale="r", run_date="d", session_id="s",
    )
    final = p.read_text()
    assert "Rewritten A." in final
    assert "Rewritten B." in final
    assert "Phrase A here." not in final
    assert "Phrase B here." not in final


# ── phrase_locate_by_text ────────────────────────────────────────────────────

def test_phrase_locate_by_text_hit_and_miss(dream_dirs):
    text = "# Root\n\n## R\n\nPrefix. findable body. Suffix.\n"
    p = _write_prompt(dream_dirs, "wf", text)
    pid = phrase_store.tag_new_phrase(
        str(p), anchor_before="Prefix. ", phrase_text="findable body.",
        anchor_after=" Suffix.",
    )
    hit = phrase_store.phrase_locate_by_text(str(p), "findable body.")
    assert hit.get("phrase_id") == pid
    assert hit.get("rev") == 0
    miss = phrase_store.phrase_locate_by_text(str(p), "never written here")
    assert miss == {"unknown": True}


def test_resolve_prompt_path_accepts_short_name_and_full_path(dream_dirs, monkeypatch):
    # Short name under PROMPTS_DIR.
    _write_prompt(dream_dirs, "worker_full", "# x\n\nbody\n")
    resolved_short = phrase_store._resolve_prompt_path("worker_full")
    assert resolved_short == (dream_dirs["prompts"] / "worker_full.md").resolve()
    # Absolute path round-trips.
    abs_p = dream_dirs["prompts"] / "worker_full.md"
    assert phrase_store._resolve_prompt_path(str(abs_p)) == abs_p.resolve()


# ── apply_edit history schema carries role + breadcrumb ──────────────────────

def test_apply_edit_history_row_carries_role_and_breadcrumb(dream_dirs):
    text = "# Root\n\n## Rules\n\nPrefix. Target. Suffix.\n"
    p = _write_prompt(dream_dirs, "worker_full", text)
    pid = phrase_store.tag_new_phrase(
        str(p), anchor_before="Prefix. ", phrase_text="Target.",
        anchor_after=" Suffix.",
    )
    phrase_store.apply_edit(
        pid, "Rewritten.",
        rationale="r", run_date="2026-04-21T04:00:00Z", session_id="sid-1",
    )
    hist = phrase_store.get_history(pid)
    assert len(hist) == 1
    row = hist[0]
    # New schema fields (used by reconstruct_prompt_at + dream narrative).
    assert row["role_template_name"] == "worker_full"
    assert row["section_breadcrumb"] == "Root / Rules"


# ── tag_virgin_insert ────────────────────────────────────────────────────────

def test_tag_virgin_insert_deterministic_and_matches_compute_id(dream_dirs):
    """Virgin-insert id must match diff.py's insert-case derivation exactly."""
    _write_prompt(dream_dirs, "worker_full", "# Root\n\n## Rules\n\nbody\n")
    section = "Root / Rules"
    new_text = "Freshly inserted paragraph body."
    pid = phrase_store.tag_virgin_insert(
        "worker_full", section, new_text,
        anchor_before="before anchor ", anchor_after=" after anchor",
    )
    expected = phrase_store._compute_phrase_id("worker_full", section, new_text)
    assert pid == expected
    # Pointer file exists, rev=0, current_text matches.
    rec = phrase_store._read_index(pid)
    assert rec["rev"] == 0
    assert rec["current_text"] == new_text
    assert rec["section_path"] == section
    assert rec["role_template"] == "worker_full"
    # History untouched at tag time.
    assert phrase_store.get_history(pid) == []
    # Calling a second time is idempotent.
    pid2 = phrase_store.tag_virgin_insert(
        "worker_full", section, new_text,
        anchor_before="different before ", anchor_after=" different after",
    )
    assert pid2 == pid


# ── append_history_for_insert ────────────────────────────────────────────────

def test_append_history_for_insert_stamps_rev1_without_touching_file(dream_dirs):
    file_before = "# Root\n\n## Rules\n\nstatic body\n"
    p = _write_prompt(dream_dirs, "worker_full", file_before)
    pid = phrase_store.tag_virgin_insert(
        "worker_full", "Root / Rules", "Freshly inserted body.",
        anchor_before="## Rules\n\n", anchor_after="\n",
    )
    rev = phrase_store.append_history_for_insert(
        pid, "Freshly inserted body.",
        rationale="new rule", run_date="2026-04-21T04:00:00Z", session_id="sid-x",
    )
    assert rev == 1
    # File not rewritten by this helper — caller owns that.
    assert p.read_text() == file_before
    hist = phrase_store.get_history(pid)
    assert len(hist) == 1
    row = hist[0]
    assert row["rev"] == 1
    assert row["old_text"] == ""
    assert row["new_text"] == "Freshly inserted body."
    assert row["role_template_name"] == "worker_full"
    assert row["section_breadcrumb"] == "Root / Rules"
    assert row["session_id"] == "sid-x"
    # Index rev advanced.
    rec = phrase_store._read_index(pid)
    assert rec["rev"] == 1


# ── reconstruct_prompt_at ────────────────────────────────────────────────────

def _seed_edit(dream_dirs, prompt_name: str, phrase_text: str, new_text: str,
               run_date: str, anchor_before: str = "Prefix. ",
               anchor_after: str = " Suffix.") -> str:
    """Tag + apply_edit; returns the phrase_id."""
    p = dream_dirs["prompts"] / f"{prompt_name}.md"
    pid = phrase_store.tag_new_phrase(
        str(p), anchor_before=anchor_before, phrase_text=phrase_text,
        anchor_after=anchor_after,
    )
    phrase_store.apply_edit(
        pid, new_text, rationale="r", run_date=run_date, session_id="s",
    )
    return pid


def test_reconstruct_prompt_at_reverses_only_post_timestamp_edits(dream_dirs):
    """3 edits before `ts`, 2 after → reconstruction reverses only the 2 after."""
    # Seed a file with 5 independent phrases we'll edit in sequence.
    text = (
        "# Root\n\n## Rules\n\n"
        "Prefix. A-body. Suffix.\n\n"
        "Prefix. B-body. Suffix.\n\n"
        "Prefix. C-body. Suffix.\n\n"
        "Prefix. D-body. Suffix.\n\n"
        "Prefix. E-body. Suffix.\n"
    )
    p = _write_prompt(dream_dirs, "worker_full", text)

    # Three edits dated BEFORE the timestamp.
    _seed_edit(dream_dirs, "worker_full", "A-body.", "A-v2.", "2026-04-10T00:00:00Z")
    _seed_edit(dream_dirs, "worker_full", "B-body.", "B-v2.", "2026-04-11T00:00:00Z")
    _seed_edit(dream_dirs, "worker_full", "C-body.", "C-v2.", "2026-04-12T00:00:00Z")

    ts = "2026-04-15T00:00:00Z"  # target timestamp

    # Two edits dated AFTER the timestamp — these should be reverse-applied.
    _seed_edit(dream_dirs, "worker_full", "D-body.", "D-v2.", "2026-04-18T00:00:00Z")
    _seed_edit(dream_dirs, "worker_full", "E-body.", "E-v2.", "2026-04-20T00:00:00Z")

    # Sanity: live file contains all five v2 bodies.
    live = p.read_text()
    for v in ("A-v2.", "B-v2.", "C-v2.", "D-v2.", "E-v2."):
        assert v in live

    out = phrase_store.reconstruct_prompt_at(ts, "worker_full")
    assert out["reversed"] == 2
    buf = out["text"]
    # Post-ts edits reversed: D and E revert to v1.
    assert "D-body." in buf and "D-v2." not in buf
    assert "E-body." in buf and "E-v2." not in buf
    # Pre-ts edits preserved: A/B/C still at v2.
    assert "A-v2." in buf and "A-body." not in buf
    assert "B-v2." in buf and "B-body." not in buf
    assert "C-v2." in buf and "C-body." not in buf


def test_reconstruct_prompt_at_warns_on_drift(dream_dirs):
    """new_text missing from buffer (non-dreamer edit) → warning, not raise."""
    text = "# Root\n\n## Rules\n\nPrefix. original body. Suffix.\n"
    p = _write_prompt(dream_dirs, "worker_full", text)
    _seed_edit(dream_dirs, "worker_full", "original body.", "rewritten body.",
               "2026-04-20T00:00:00Z")
    # Someone edits the file out-of-band, destroying the new_text anchor.
    p.write_text(
        "# Root\n\n## Rules\n\nPrefix. completely different content now. Suffix.\n",
        encoding="utf-8",
    )
    out = phrase_store.reconstruct_prompt_at("2026-04-15T00:00:00Z", "worker_full")
    assert out["reversed"] == 0
    assert len(out["warnings"]) == 1
    assert "not uniquely present" in out["warnings"][0]


def test_reconstruct_prompt_at_filters_by_prompt_name(dream_dirs):
    """Edits to other prompt files must NOT be reverse-applied."""
    text_a = "# Root\n\n## R\n\nPrefix. A-body. Suffix.\n"
    text_b = "# Root\n\n## R\n\nPrefix. B-body. Suffix.\n"
    pa = _write_prompt(dream_dirs, "worker_full", text_a)
    pb = _write_prompt(dream_dirs, "supervisor_full", text_b)
    _seed_edit(dream_dirs, "worker_full", "A-body.", "A-v2.", "2026-04-20T00:00:00Z")
    _seed_edit(dream_dirs, "supervisor_full", "B-body.", "B-v2.", "2026-04-20T00:00:00Z")

    out = phrase_store.reconstruct_prompt_at(
        "2026-04-15T00:00:00Z", "worker_full",
    )
    # Only the worker_full edit reversed.
    assert out["reversed"] == 1
    assert "A-body." in out["text"]
    # supervisor_full's live file is still at v2, untouched.
    assert "B-v2." in pb.read_text()


def test_reconstruct_prompt_at_empty_when_no_edits_after(dream_dirs):
    text = "# Root\n\n## R\n\nPrefix. body. Suffix.\n"
    _write_prompt(dream_dirs, "worker_full", text)
    _seed_edit(dream_dirs, "worker_full", "body.", "bv2.", "2026-04-10T00:00:00Z")
    out = phrase_store.reconstruct_prompt_at("2026-04-20T00:00:00Z", "worker_full")
    assert out["reversed"] == 0
    assert out["warnings"] == []
    # Returned text equals current live file (nothing reversed).
    assert "bv2." in out["text"]


def test_reconstruct_prompt_at_missing_prompt_returns_warning(dream_dirs):
    out = phrase_store.reconstruct_prompt_at("2026-04-20T00:00:00Z", "nonexistent")
    assert out["reversed"] == 0
    assert out["text"] == ""
    assert any("prompt file not found" in w for w in out["warnings"])
