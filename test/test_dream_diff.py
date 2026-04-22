"""Pure-unit tests for app.dream.diff — paragraph splitter + opcode→Edit mapper.

Covers:
  - split_paragraphs: blank-line splitting + header-as-own-paragraph
  - compute_edits: identical texts → no edits; single paragraph replaced →
    one `replace`; inserted paragraph → one `insert` with fresh phrase_id;
    deleted paragraph → one `delete`
  - phrase_id derivation: replace/delete anchored to OLD section path; insert
    anchored to NEW section path
  - rebuild_with_decisions: keep → new paragraphs; drop → old paragraphs
"""

from __future__ import annotations

import pytest

from app.dream import diff as dd
from app.dream import phrase_store


# ── split_paragraphs ─────────────────────────────────────────────────────────

def test_split_blank_separated():
    text = "para one.\n\npara two body.\n\npara three.\n"
    paras = dd.split_paragraphs(text)
    assert [p.normalized for p in paras] == ["para one.", "para two body.", "para three."]


def test_split_headers_always_own_paragraph():
    text = (
        "# Root\n\n"
        "intro line.\n\n"
        "## Sub\nbody right under header.\n"
    )
    paras = dd.split_paragraphs(text)
    assert paras[0].normalized == "# Root"
    assert paras[1].normalized == "intro line."
    assert paras[2].normalized == "## Sub"
    # A header followed immediately by a non-blank line still separates the body.
    assert paras[3].normalized == "body right under header."


def test_split_ignores_empty_blocks():
    text = "\n\n\nsolo.\n\n\n"
    paras = dd.split_paragraphs(text)
    assert len(paras) == 1 and paras[0].normalized == "solo."


# ── compute_edits ────────────────────────────────────────────────────────────

def test_identical_texts_no_edits():
    text = "# Root\n\nalpha\n\nbravo\n"
    assert dd.compute_edits(text, text, "worker_full") == []


def test_single_paragraph_replace():
    old = "# Root\n\n## Rules\n\nOriginal rule.\n"
    new = "# Root\n\n## Rules\n\nRewritten rule.\n"
    edits = dd.compute_edits(old, new, "worker_full")
    assert len(edits) == 1
    assert edits[0].kind == "replace"
    assert edits[0].old_text.strip() == "Original rule."
    assert edits[0].new_text.strip() == "Rewritten rule."
    assert edits[0].section_path == "Root / Rules"
    # Deterministic phrase_id from OLD text + OLD section path.
    expected = phrase_store._compute_phrase_id(
        "worker_full", "Root / Rules", "Original rule.",
    )
    assert edits[0].phrase_id == expected


def test_paragraph_insert_gets_fresh_id_from_new_buffer():
    old = "# Root\n\n## Rules\n\nFirst body.\n"
    new = "# Root\n\n## Rules\n\nFirst body.\n\nNewly added paragraph.\n"
    edits = dd.compute_edits(old, new, "worker_full")
    assert len(edits) == 1
    assert edits[0].kind == "insert"
    assert edits[0].old_text == ""
    assert edits[0].new_text.strip() == "Newly added paragraph."
    # Fresh id seeded from NEW section path + NEW paragraph text.
    expected = phrase_store._compute_phrase_id(
        "worker_full", "Root / Rules", "Newly added paragraph.",
    )
    assert edits[0].phrase_id == expected


def test_paragraph_delete():
    old = "# Root\n\n## Rules\n\nKeep this.\n\nDrop this one.\n"
    new = "# Root\n\n## Rules\n\nKeep this.\n"
    edits = dd.compute_edits(old, new, "worker_full")
    assert len(edits) == 1
    assert edits[0].kind == "delete"
    assert edits[0].old_text.strip() == "Drop this one."
    assert edits[0].new_text == ""


def test_two_disjoint_edits_two_opcodes():
    old = (
        "# Root\n\n"
        "## Alpha\n\nAlpha body.\n\n"
        "## Beta\n\nBeta body.\n"
    )
    new = (
        "# Root\n\n"
        "## Alpha\n\nAlpha REWRITTEN.\n\n"
        "## Beta\n\nBeta REWRITTEN.\n"
    )
    edits = dd.compute_edits(old, new, "worker_full")
    assert len(edits) == 2
    assert {e.kind for e in edits} == {"replace"}
    # Different section paths → different phrase_ids.
    assert edits[0].section_path == "Root / Alpha"
    assert edits[1].section_path == "Root / Beta"
    assert edits[0].phrase_id != edits[1].phrase_id


# ── rebuild_with_decisions ───────────────────────────────────────────────────

def test_rebuild_keep_all_matches_new():
    old = "# Root\n\nalpha.\n\nbravo.\n"
    new = "# Root\n\nalpha rewritten.\n\nbravo rewritten.\n"
    edits = dd.compute_edits(old, new, "wf")
    decisions = {e.phrase_id: "keep" for e in edits}
    out = dd.rebuild_with_decisions(old, new, edits, decisions)
    assert "alpha rewritten." in out
    assert "bravo rewritten." in out
    assert "alpha." not in out.replace("alpha rewritten.", "")


def test_rebuild_drop_all_matches_old():
    old = "# Root\n\nalpha.\n\nbravo.\n"
    new = "# Root\n\nalpha rewritten.\n\nbravo rewritten.\n"
    edits = dd.compute_edits(old, new, "wf")
    decisions = {e.phrase_id: "drop" for e in edits}
    out = dd.rebuild_with_decisions(old, new, edits, decisions)
    assert "alpha." in out and "bravo." in out
    assert "rewritten" not in out


def test_rebuild_mixed_decisions():
    # Two disjoint section bodies → two separate replace opcodes, so we can
    # keep one and drop the other.
    old = (
        "# Root\n\n"
        "## Alpha\n\nalpha body.\n\n"
        "## Beta\n\nbravo body.\n"
    )
    new = (
        "# Root\n\n"
        "## Alpha\n\nalpha rewritten.\n\n"
        "## Beta\n\nbravo rewritten.\n"
    )
    edits = dd.compute_edits(old, new, "wf")
    assert len(edits) == 2
    # Keep alpha's rewrite, drop bravo's.
    alpha_edit = next(e for e in edits if e.section_path == "Root / Alpha")
    beta_edit = next(e for e in edits if e.section_path == "Root / Beta")
    decisions = {alpha_edit.phrase_id: "keep", beta_edit.phrase_id: "drop"}
    out = dd.rebuild_with_decisions(old, new, edits, decisions)
    assert "alpha rewritten." in out
    assert "bravo body." in out and "bravo rewritten." not in out


def test_rebuild_dropped_insert_is_omitted():
    old = "# Root\n\nalpha.\n"
    new = "# Root\n\nalpha.\n\ninserted body.\n"
    edits = dd.compute_edits(old, new, "wf")
    assert len(edits) == 1 and edits[0].kind == "insert"
    decisions = {edits[0].phrase_id: "drop"}
    out = dd.rebuild_with_decisions(old, new, edits, decisions)
    assert "inserted body." not in out
    # Kept path inserts the paragraph.
    out_keep = dd.rebuild_with_decisions(old, new, edits, {edits[0].phrase_id: "keep"})
    assert "inserted body." in out_keep


def test_rebuild_dropped_delete_preserves_old():
    old = "# Root\n\nalpha.\n\ndelete candidate.\n"
    new = "# Root\n\nalpha.\n"
    edits = dd.compute_edits(old, new, "wf")
    assert len(edits) == 1 and edits[0].kind == "delete"
    decisions = {edits[0].phrase_id: "drop"}
    out = dd.rebuild_with_decisions(old, new, edits, decisions)
    assert "delete candidate." in out
