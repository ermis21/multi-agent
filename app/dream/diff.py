"""Paragraph-level diff for prompt-file dreams.

The dreamer submits a fully rewritten prompt. We diff old↔new at paragraph
granularity so that each stable intent ("rewrite the Hard rules block") maps
onto one addressable edit. Paragraphs are blank-line separated, and every
markdown header line (`^#{1,6} `) is treated as its own paragraph so that a
renamed section and its body show up as distinct edits.

Each non-`equal` opcode → exactly one `Edit`, even when the opcode spans
several paragraphs; that keeps the staged batch flat and gives the dreamer
one knob (`keep`/`drop`) per intent rather than per line.

Phrase-id derivation:
  * `replace` / `delete`  → id of the first OLD paragraph (role_template,
    section_breadcrumb at that paragraph in the OLD buffer, normalized text).
  * pure `insert`         → FRESH id of the first NEW paragraph (role_template,
    section_breadcrumb at the insertion point in the NEW buffer, normalized
    text of the first inserted paragraph). Inserted paragraphs start virgin;
    they pick up history only once `dream_finalize` keeps them.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass
from typing import Literal

from app.dream import phrase_store

_HEADER_RE = re.compile(r"^#{1,6} ")
EditKind = Literal["replace", "insert", "delete"]


@dataclass
class Paragraph:
    """A paragraph block with its character offset in the source buffer."""
    text: str
    start: int  # byte/char offset of first char in source
    end: int    # exclusive end offset

    @property
    def normalized(self) -> str:
        return self.text.strip()


@dataclass
class Edit:
    kind: EditKind
    phrase_id: str
    section_path: str
    old_text: str      # "" for pure insert
    new_text: str      # "" for pure delete
    old_start: int     # char offset in old buffer (0 for insert-at-top)
    new_start: int     # char offset in new buffer (0 for delete-at-top)
    opcode_index: int  # stable ordering key for commit replay


# ── Paragraph splitter ───────────────────────────────────────────────────────

def split_paragraphs(text: str) -> list[Paragraph]:
    """Split `text` on blank lines; each header line is its own paragraph.

    Empty paragraphs are dropped. Leading/trailing whitespace inside a block
    is preserved in `text` (so reassembly round-trips), but the `normalized`
    property returns `.strip()`ed content for diffing.
    """
    if not text:
        return []
    lines = text.splitlines(keepends=True)
    blocks: list[Paragraph] = []
    buf_lines: list[str] = []
    buf_start = 0
    pos = 0

    def flush(end: int) -> None:
        joined = "".join(buf_lines)
        if joined.strip():
            blocks.append(Paragraph(text=joined, start=buf_start, end=end))
        buf_lines.clear()

    for line in lines:
        stripped = line.strip()
        is_header = bool(_HEADER_RE.match(line))
        is_blank = stripped == ""
        if is_header:
            # Flush whatever we were accumulating, then emit the header alone.
            flush(pos)
            blocks.append(Paragraph(text=line, start=pos, end=pos + len(line)))
            buf_start = pos + len(line)
        elif is_blank:
            flush(pos)
            pos += len(line)
            buf_start = pos
            continue
        else:
            if not buf_lines:
                buf_start = pos
            buf_lines.append(line)
        pos += len(line)

    flush(pos)
    return blocks


# ── Opcode → Edit mapper ─────────────────────────────────────────────────────

def compute_edits(
    old_text: str,
    new_text: str,
    role_template_name: str,
) -> list[Edit]:
    """Return the ordered list of `Edit` objects describing old→new.

    One `Edit` per non-`equal` opcode. Exact-identical inputs → empty list.
    """
    old_paras = split_paragraphs(old_text)
    new_paras = split_paragraphs(new_text)
    old_tokens = [p.normalized for p in old_paras]
    new_tokens = [p.normalized for p in new_paras]

    sm = difflib.SequenceMatcher(a=old_tokens, b=new_tokens, autojunk=False)
    edits: list[Edit] = []
    for op_idx, (tag, i1, i2, j1, j2) in enumerate(sm.get_opcodes()):
        if tag == "equal":
            continue
        if tag == "replace":
            old_slice = "\n\n".join(p.text.rstrip("\n") for p in old_paras[i1:i2])
            new_slice = "\n\n".join(p.text.rstrip("\n") for p in new_paras[j1:j2])
            section = phrase_store.section_path_for_offset(old_text, old_paras[i1].start)
            pid = phrase_store._compute_phrase_id(
                role_template_name, section, old_paras[i1].normalized,
            )
            edits.append(Edit(
                kind="replace", phrase_id=pid, section_path=section,
                old_text=old_slice, new_text=new_slice,
                old_start=old_paras[i1].start, new_start=new_paras[j1].start,
                opcode_index=op_idx,
            ))
        elif tag == "delete":
            old_slice = "\n\n".join(p.text.rstrip("\n") for p in old_paras[i1:i2])
            section = phrase_store.section_path_for_offset(old_text, old_paras[i1].start)
            pid = phrase_store._compute_phrase_id(
                role_template_name, section, old_paras[i1].normalized,
            )
            # `new_start` anchors to where the deletion takes effect in the new buffer:
            # the offset of new_paras[j1] if j1 < len(new_paras), else end-of-new.
            new_anchor = new_paras[j1].start if j1 < len(new_paras) else len(new_text)
            edits.append(Edit(
                kind="delete", phrase_id=pid, section_path=section,
                old_text=old_slice, new_text="",
                old_start=old_paras[i1].start, new_start=new_anchor,
                opcode_index=op_idx,
            ))
        elif tag == "insert":
            new_slice = "\n\n".join(p.text.rstrip("\n") for p in new_paras[j1:j2])
            section = phrase_store.section_path_for_offset(new_text, new_paras[j1].start)
            pid = phrase_store._compute_phrase_id(
                role_template_name, section, new_paras[j1].normalized,
            )
            old_anchor = old_paras[i1].start if i1 < len(old_paras) else len(old_text)
            edits.append(Edit(
                kind="insert", phrase_id=pid, section_path=section,
                old_text="", new_text=new_slice,
                old_start=old_anchor, new_start=new_paras[j1].start,
                opcode_index=op_idx,
            ))
    return edits


# ── Paragraph reassembly for selective commit ───────────────────────────────

def rebuild_with_decisions(
    old_text: str,
    new_text: str,
    edits: list[Edit],
    decisions: dict[str, str],  # phrase_id → "keep" | "drop"
) -> str:
    """Re-assemble the prompt honoring per-edit keep/drop decisions.

    For each opcode: keep → emit the new range; drop → emit the old range.
    `equal` ranges come through unchanged. Joins paragraphs with one blank
    line between them; any trailing newline in the source is preserved.
    """
    old_paras = split_paragraphs(old_text)
    new_paras = split_paragraphs(new_text)
    old_tokens = [p.normalized for p in old_paras]
    new_tokens = [p.normalized for p in new_paras]
    sm = difflib.SequenceMatcher(a=old_tokens, b=new_tokens, autojunk=False)

    # opcode_index → phrase_id for quick decision lookup.
    op_to_pid = {e.opcode_index: e.phrase_id for e in edits}

    out_paras: list[str] = []
    for op_idx, (tag, i1, i2, j1, j2) in enumerate(sm.get_opcodes()):
        if tag == "equal":
            out_paras.extend(p.text for p in old_paras[i1:i2])
            continue
        decision = decisions.get(op_to_pid.get(op_idx, ""), "drop")
        if tag == "replace":
            if decision == "keep":
                out_paras.extend(p.text for p in new_paras[j1:j2])
            else:
                out_paras.extend(p.text for p in old_paras[i1:i2])
        elif tag == "delete":
            if decision == "keep":
                pass  # drop the range
            else:
                out_paras.extend(p.text for p in old_paras[i1:i2])
        elif tag == "insert":
            if decision == "keep":
                out_paras.extend(p.text for p in new_paras[j1:j2])
            # drop: emit nothing

    out = _join_paragraphs(out_paras)
    if old_text.endswith("\n") and not out.endswith("\n"):
        out += "\n"
    return out


def _join_paragraphs(paragraphs: list[str]) -> str:
    """Rejoin paragraph blocks with blank-line separators.

    Each paragraph usually already ends with `\\n`; we normalize to exactly
    one trailing newline then insert a blank line between consecutive blocks.
    """
    if not paragraphs:
        return ""
    normalized = [p if p.endswith("\n") else p + "\n" for p in paragraphs]
    return "\n".join(p.rstrip("\n") for p in normalized) + "\n" * (
        1 if normalized[-1].endswith("\n") else 0
    )
