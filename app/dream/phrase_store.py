"""Phrase-level provenance for dreamer prompt edits.

Stores a stable `phrase_id` for each piece of prompt text the dreamer has ever
touched, and an append-only history of edits keyed on that id. The id survives
subsequent re-edits because it is seeded from the FIRST version of the phrase;
the pointer file at `state/dream/phrase_index/{id}.json` then tracks the
current text + anchors so we can re-locate the phrase after file drift.

Why section-path-from-root in the id: two `## Hard rules` subheaders under
different `#` parents would collide if we hashed only the nearest header, so
we include the full breadcrumb (`"Behavioral Rules / Hard rules"` vs
`"Tools / Hard rules"`).

Directory layout (all under `$STATE_DIR/dream/`):
  phrase_index/{phrase_id}.json    — pointer file (current text + anchors + rev)
  phrase_history/{phrase_id}.jsonl — append-only edit log

Anchoring: we keep ~80 chars of context on each side of the phrase
(`anchor_before` / `anchor_after`) so `locate_phrase` can re-find the phrase
after unrelated edits shift line numbers. Matching is:
  1. Exact substring of `current_text` (must be unique in the file).
  2. Anchor-sandwich match (anchor_before + anything + anchor_after).
  3. Else LocateFailure — surfaces as an orphaned phrase in diagnostics.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

STATE_DIR = Path(os.environ.get("STATE_DIR", "/state"))
DREAM_ROOT = STATE_DIR / "dream"
INDEX_DIR = DREAM_ROOT / "phrase_index"
HISTORY_DIR = DREAM_ROOT / "phrase_history"

PROMPTS_DIR = Path(os.environ.get("PROMPTS_DIR", "/config/prompts"))

ANCHOR_CHARS = 80
MAX_PHRASE_NORMALIZED = 200

_HEADER_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_WS_RE = re.compile(r"\s+")


class PhraseStoreError(RuntimeError):
    """Base for phrase-store failures."""


class LocateFailure(PhraseStoreError):
    """Cannot re-find a phrase in its original file."""


class EditConflict(PhraseStoreError):
    """The anchor no longer matches the on-disk file."""


# ── ID + section-path helpers ────────────────────────────────────────────────

def _normalize_phrase(text: str) -> str:
    return _WS_RE.sub(" ", text).strip()[:MAX_PHRASE_NORMALIZED]


def _compute_phrase_id(role_template_name: str, section_path: str, phrase_text: str) -> str:
    key = f"{role_template_name}|{section_path}|{_normalize_phrase(phrase_text)}"
    return "ph-" + hashlib.sha1(key.encode("utf-8")).hexdigest()[:10]


def _role_template_name(path: str | Path) -> str:
    """Basename minus `.md`. `config/prompts/worker_full.md` → `worker_full`."""
    return Path(path).stem


def section_path_for_offset(file_text: str, char_offset: int) -> str:
    """Walk the markdown header stack and return the breadcrumb for `char_offset`.

    Uses a simple stack: header of depth d pops all deeper entries off the stack.
    Returns an empty string if the offset sits above any header (document preamble).
    """
    stack: list[tuple[int, str]] = []
    pos = 0
    for line in file_text.splitlines(keepends=True):
        if pos > char_offset:
            break
        m = _HEADER_RE.match(line)
        if m:
            depth = len(m.group(1))
            title = m.group(2).strip()
            while stack and stack[-1][0] >= depth:
                stack.pop()
            stack.append((depth, title))
        pos += len(line)
    return " / ".join(title for _, title in stack)


# ── Low-level file I/O ───────────────────────────────────────────────────────

def _ensure_dirs() -> None:
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)


def _index_path(phrase_id: str) -> Path:
    return INDEX_DIR / f"{phrase_id}.json"


def _history_path(phrase_id: str) -> Path:
    return HISTORY_DIR / f"{phrase_id}.jsonl"


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise


def _read_index(phrase_id: str) -> dict:
    p = _index_path(phrase_id)
    if not p.exists():
        raise LocateFailure(f"phrase_id {phrase_id!r} not found in phrase_index")
    return json.loads(p.read_text(encoding="utf-8"))


def _write_index(phrase_id: str, record: dict) -> None:
    _ensure_dirs()
    _atomic_write_text(_index_path(phrase_id), json.dumps(record, ensure_ascii=False, indent=2))


def _append_history(phrase_id: str, entry: dict) -> None:
    _ensure_dirs()
    line = json.dumps(entry, ensure_ascii=False) + "\n"
    with _history_path(phrase_id).open("a", encoding="utf-8") as f:
        f.write(line)


def _pop_last_history(phrase_id: str) -> dict | None:
    """Remove and return the last line of the history file (or None if empty)."""
    p = _history_path(phrase_id)
    if not p.exists():
        return None
    lines = p.read_text(encoding="utf-8").splitlines()
    if not lines:
        return None
    last = lines.pop()
    _atomic_write_text(p, ("\n".join(lines) + "\n") if lines else "")
    try:
        return json.loads(last)
    except json.JSONDecodeError:
        return None


# ── Prompt-file path resolution ──────────────────────────────────────────────

def _resolve_prompt_path(rel_or_abs: str | Path) -> Path:
    """Accept either an absolute path, or a repo-style `config/prompts/foo.md`,
    or a bare template name (`foo` or `foo.md`) relative to PROMPTS_DIR.
    """
    p = Path(rel_or_abs)
    if p.is_absolute():
        return p
    s = str(rel_or_abs)
    if s.startswith("config/prompts/"):
        return (PROMPTS_DIR.parent.parent / s).resolve()
    if s.startswith("prompts/"):
        return (PROMPTS_DIR.parent / s).resolve()
    stem = p.name if p.suffix == ".md" else f"{p.name}.md"
    return (PROMPTS_DIR / stem).resolve()


# ── Public API ───────────────────────────────────────────────────────────────

@dataclass
class LocatedPhrase:
    phrase_id: str
    path: str
    current_text: str
    anchor_before: str
    anchor_after: str
    section_path: str
    line: int


def tag_new_phrase(
    path: str | Path,
    anchor_before: str,
    phrase_text: str,
    anchor_after: str,
) -> str:
    """Register a phrase for provenance tracking.

    If the combination of (role_template_name, section_path, normalized_phrase)
    already has an index entry, returns the existing id (idempotent). Otherwise
    writes a new pointer file. Does NOT mutate the prompt file.
    """
    prompt_path = _resolve_prompt_path(path)
    file_text = prompt_path.read_text(encoding="utf-8")
    sandwich = f"{anchor_before}{phrase_text}{anchor_after}"
    idx = file_text.find(sandwich)
    if idx < 0:
        raise LocateFailure(
            f"could not find anchored phrase in {prompt_path} — "
            "anchors + phrase must appear contiguously in the file"
        )
    phrase_offset = idx + len(anchor_before)
    section = section_path_for_offset(file_text, phrase_offset)
    role_name = _role_template_name(prompt_path)
    phrase_id = _compute_phrase_id(role_name, section, phrase_text)

    ip = _index_path(phrase_id)
    if ip.exists():
        return phrase_id

    record = {
        "phrase_id": phrase_id,
        "role_template": role_name,
        "path": str(prompt_path),
        "section_path": section,
        "current_text": phrase_text,
        "anchor_before": anchor_before[-ANCHOR_CHARS:],
        "anchor_after": anchor_after[:ANCHOR_CHARS],
        "rev": 0,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_index(phrase_id, record)
    return phrase_id


def locate_phrase(phrase_id: str) -> LocatedPhrase:
    """Re-find a phrase in its live file. Uses stored text first, then anchors."""
    rec = _read_index(phrase_id)
    prompt_path = Path(rec["path"])
    file_text = prompt_path.read_text(encoding="utf-8")
    current = rec["current_text"]

    count = file_text.count(current)
    if count == 1:
        idx = file_text.index(current)
    else:
        idx = _anchor_match(file_text, rec["anchor_before"], rec["anchor_after"])
        if idx < 0:
            raise LocateFailure(
                f"phrase_id {phrase_id!r}: text not found in {prompt_path} "
                f"(occurrences={count}, anchor match failed)"
            )
        current = file_text[idx : idx + _anchor_span_len(file_text, idx, rec["anchor_before"], rec["anchor_after"])]

    line = file_text.count("\n", 0, idx) + 1
    return LocatedPhrase(
        phrase_id=phrase_id,
        path=str(prompt_path),
        current_text=current,
        anchor_before=rec["anchor_before"],
        anchor_after=rec["anchor_after"],
        section_path=rec.get("section_path", ""),
        line=line,
    )


def _anchor_match(file_text: str, anchor_before: str, anchor_after: str) -> int:
    """Locate `anchor_before…anchor_after` and return offset immediately after anchor_before."""
    if not anchor_before or not anchor_after:
        return -1
    start = 0
    while True:
        i = file_text.find(anchor_before, start)
        if i < 0:
            return -1
        phrase_start = i + len(anchor_before)
        j = file_text.find(anchor_after, phrase_start)
        if j < 0:
            return -1
        return phrase_start


def _anchor_span_len(file_text: str, phrase_start: int, anchor_before: str, anchor_after: str) -> int:
    j = file_text.find(anchor_after, phrase_start)
    return max(0, j - phrase_start)


def apply_edit(
    phrase_id: str,
    new_text: str,
    *,
    rationale: str,
    run_date: str,
    session_id: str,
) -> int:
    """Rewrite the on-disk phrase. Returns the new rev number.

    Fails if the pointer's `current_text` no longer matches exactly once in the
    file (caller must resolve the conflict before retrying).
    """
    rec = _read_index(phrase_id)
    prompt_path = Path(rec["path"])
    file_text = prompt_path.read_text(encoding="utf-8")
    current = rec["current_text"]

    occ = file_text.count(current)
    if occ == 0:
        raise EditConflict(
            f"phrase_id {phrase_id!r}: current_text no longer present in {prompt_path}"
        )
    if occ > 1:
        raise EditConflict(
            f"phrase_id {phrase_id!r}: current_text matches {occ} sites in {prompt_path} — "
            "anchors have drifted; run locate_phrase + reset pointer before editing"
        )

    new_file_text = file_text.replace(current, new_text, 1)
    _atomic_write_text(prompt_path, new_file_text)

    idx = new_file_text.index(new_text)
    new_anchor_before = new_file_text[max(0, idx - ANCHOR_CHARS) : idx]
    new_anchor_after = new_file_text[idx + len(new_text) : idx + len(new_text) + ANCHOR_CHARS]

    old_current = rec["current_text"]
    old_anchor_before = rec["anchor_before"]
    old_anchor_after = rec["anchor_after"]
    new_rev = int(rec.get("rev", 0)) + 1

    rec["current_text"] = new_text
    rec["anchor_before"] = new_anchor_before
    rec["anchor_after"] = new_anchor_after
    rec["rev"] = new_rev
    rec["updated_at"] = datetime.now(timezone.utc).isoformat()
    _write_index(phrase_id, rec)

    _append_history(phrase_id, {
        "rev": new_rev,
        "role_template_name": rec.get("role_template", ""),
        "section_breadcrumb": rec.get("section_path", ""),
        "run_date": run_date,
        "session_id": session_id,
        "rationale": rationale,
        "old_text": old_current,
        "new_text": new_text,
        "old_anchor_before": old_anchor_before,
        "old_anchor_after": old_anchor_after,
        "new_anchor_before": new_anchor_before,
        "new_anchor_after": new_anchor_after,
        "applied_at": datetime.now(timezone.utc).isoformat(),
        "rolled_back_by": None,
    })
    return new_rev


def tag_virgin_insert(
    path: str | Path,
    section_path: str,
    new_text: str,
    anchor_before: str,
    anchor_after: str,
) -> str:
    """Register a freshly-inserted phrase that has never appeared in the file.

    Derives the phrase_id from the NEW section path + NEW text (matching diff.py's
    insert-case id derivation), writes a virgin pointer, and returns the id.
    Caller is responsible for having already written the paragraph into the live
    file before / alongside this call; the pointer just records where it lives.
    """
    prompt_path = _resolve_prompt_path(path)
    role_name = _role_template_name(prompt_path)
    phrase_id = _compute_phrase_id(role_name, section_path, new_text)
    ip = _index_path(phrase_id)
    if ip.exists():
        return phrase_id
    record = {
        "phrase_id": phrase_id,
        "role_template": role_name,
        "path": str(prompt_path),
        "section_path": section_path,
        "current_text": new_text,
        "anchor_before": anchor_before[-ANCHOR_CHARS:],
        "anchor_after": anchor_after[:ANCHOR_CHARS],
        "rev": 0,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_index(phrase_id, record)
    return phrase_id


def append_history_for_insert(
    phrase_id: str,
    new_text: str,
    *,
    rationale: str,
    run_date: str,
    session_id: str,
) -> int:
    """Record a virgin-insert edit as rev 1 without rewriting the file.

    Used by `dream_finalize` when committing a kept `insert` edit: the file has
    already been rewritten via `rebuild_with_decisions`; we just need to stamp
    the history so future loop-guard passes can see this edit.
    """
    rec = _read_index(phrase_id)
    new_rev = int(rec.get("rev", 0)) + 1
    rec["rev"] = new_rev
    rec["updated_at"] = datetime.now(timezone.utc).isoformat()
    _write_index(phrase_id, rec)
    _append_history(phrase_id, {
        "rev": new_rev,
        "role_template_name": rec.get("role_template", ""),
        "section_breadcrumb": rec.get("section_path", ""),
        "run_date": run_date,
        "session_id": session_id,
        "rationale": rationale,
        "old_text": "",
        "new_text": new_text,
        "old_anchor_before": "",
        "old_anchor_after": "",
        "new_anchor_before": rec.get("anchor_before", ""),
        "new_anchor_after": rec.get("anchor_after", ""),
        "applied_at": datetime.now(timezone.utc).isoformat(),
        "rolled_back_by": None,
    })
    return new_rev


def record_committed_edit(
    phrase_id: str,
    *,
    old_text: str,
    new_text: str,
    rationale: str,
    run_date: str,
    session_id: str,
    role_template: str | None = None,
    section_path: str | None = None,
    path: str | Path | None = None,
) -> int:
    """Stamp history + update pointer for a dreamer-committed edit.

    Distinct from `apply_edit` in that the caller has already rewritten the
    prompt file (typically via `diff.rebuild_with_decisions` in
    `dream_finalize`). This helper only persists provenance.

    `new_text == ""` marks a pure-delete commit: the phrase is gone; pointer's
    `current_text` is cleared and anchors blanked. Future `locate_phrase`
    calls on this id will fail cleanly.

    If the pointer does not yet exist and `role_template`, `section_path`, and
    `path` are all supplied, a fresh pointer is bootstrapped at rev 0 before
    the edit is recorded — used by `dream_finalize` when committing the first
    edit for a phrase that was never previously tagged.
    """
    try:
        rec = _read_index(phrase_id)
    except LocateFailure:
        if role_template is None or section_path is None or path is None:
            raise
        prompt_path = _resolve_prompt_path(path)
        rec = {
            "phrase_id": phrase_id,
            "role_template": role_template,
            "path": str(prompt_path),
            "section_path": section_path,
            "current_text": old_text,
            "anchor_before": "",
            "anchor_after": "",
            "rev": 0,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        _write_index(phrase_id, rec)
    new_rev = int(rec.get("rev", 0)) + 1

    # Refresh anchors against the live file where possible.
    prompt_path = Path(rec["path"])
    new_anchor_before = rec.get("anchor_before", "")
    new_anchor_after = rec.get("anchor_after", "")
    if new_text and prompt_path.exists():
        file_text = prompt_path.read_text(encoding="utf-8")
        if file_text.count(new_text) == 1:
            idx = file_text.index(new_text)
            new_anchor_before = file_text[max(0, idx - ANCHOR_CHARS) : idx]
            new_anchor_after = file_text[
                idx + len(new_text) : idx + len(new_text) + ANCHOR_CHARS
            ]

    rec["current_text"] = new_text
    rec["anchor_before"] = new_anchor_before
    rec["anchor_after"] = new_anchor_after
    rec["rev"] = new_rev
    rec["updated_at"] = datetime.now(timezone.utc).isoformat()
    _write_index(phrase_id, rec)

    _append_history(phrase_id, {
        "rev": new_rev,
        "role_template_name": rec.get("role_template", ""),
        "section_breadcrumb": rec.get("section_path", ""),
        "run_date": run_date,
        "session_id": session_id,
        "rationale": rationale,
        "old_text": old_text,
        "new_text": new_text,
        "old_anchor_before": "",
        "old_anchor_after": "",
        "new_anchor_before": new_anchor_before,
        "new_anchor_after": new_anchor_after,
        "applied_at": datetime.now(timezone.utc).isoformat(),
        "rolled_back_by": None,
    })
    return new_rev


def reconstruct_prompt_at(timestamp: str, prompt_name: str) -> dict:
    """Return the prompt file as it existed at `timestamp` by reverse-applying
    every `phrase_history` entry for `prompt_name` whose `run_date > timestamp`.

    Returns `{"text": <reconstructed>, "warnings": [<str>, ...], "reversed": N}`.
    A warning is added for each entry whose `new_text` can't be found in the
    working buffer (a non-dreamer edit drifted the file); those entries are
    skipped rather than raising, so the reconstruction is best-effort.
    """
    prompt_path = _resolve_prompt_path(prompt_name)
    if not prompt_path.exists():
        return {"text": "", "warnings": [f"prompt file not found: {prompt_path}"], "reversed": 0}
    buf = prompt_path.read_text(encoding="utf-8")

    entries: list[tuple[str, dict, str]] = []  # (run_date, entry, phrase_id)
    _ensure_dirs()
    for hist_file in HISTORY_DIR.glob("*.jsonl"):
        phrase_id = hist_file.stem
        for line in hist_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            row_prompt = row.get("role_template_name") or ""
            if not row_prompt:
                # Legacy rows lack role_template_name — fall back to cross-ref via index.
                try:
                    idx_rec = _read_index(phrase_id)
                    row_prompt = idx_rec.get("role_template", "")
                except LocateFailure:
                    continue
            if row_prompt != prompt_name:
                continue
            rd = row.get("run_date") or ""
            if rd and rd > timestamp:
                entries.append((rd, row, phrase_id))

    # Reverse-apply newest first so each reversal lands in the right buffer state.
    entries.sort(key=lambda t: t[0], reverse=True)
    warnings: list[str] = []
    reversed_count = 0
    for run_date, row, phrase_id in entries:
        new_text = row.get("new_text", "")
        old_text = row.get("old_text", "")
        if not new_text:
            # Pure-delete reversal: re-insert old_text. Best-effort via anchors.
            warnings.append(
                f"{phrase_id}@{run_date}: pure-delete reversal skipped "
                "(re-insert requires anchor heuristics, not implemented)"
            )
            continue
        if buf.count(new_text) != 1:
            warnings.append(
                f"{phrase_id}@{run_date}: new_text not uniquely present in buffer — skipped"
            )
            continue
        buf = buf.replace(new_text, old_text, 1)
        reversed_count += 1

    return {"text": buf, "warnings": warnings, "reversed": reversed_count}


def get_history(phrase_id: str) -> list[dict]:
    p = _history_path(phrase_id)
    if not p.exists():
        return []
    out: list[dict] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def get_history_excerpt(phrase_id: str, k: int = 3) -> list[dict]:
    """Last `k` history entries — what the dreamer sees when we surface conflict."""
    hist = get_history(phrase_id)
    if k <= 0:
        return []
    return hist[-k:]


def rollback_last(phrase_id: str) -> dict | None:
    """Undo the most recent edit. Returns the popped history entry, or None if nothing to undo."""
    rec = _read_index(phrase_id)
    prompt_path = Path(rec["path"])
    file_text = prompt_path.read_text(encoding="utf-8")

    hist = get_history(phrase_id)
    if not hist:
        return None
    last = hist[-1]

    new_text = last["new_text"]
    old_text = last["old_text"]
    occ = file_text.count(new_text)
    if occ != 1:
        raise EditConflict(
            f"rollback_last({phrase_id!r}): new_text from last edit has {occ} occurrences — "
            "cannot safely restore"
        )
    restored = file_text.replace(new_text, old_text, 1)
    _atomic_write_text(prompt_path, restored)

    rec["current_text"] = old_text
    rec["anchor_before"] = last["old_anchor_before"]
    rec["anchor_after"] = last["old_anchor_after"]
    rec["rev"] = int(rec.get("rev", 1)) - 1
    rec["updated_at"] = datetime.now(timezone.utc).isoformat()
    _write_index(phrase_id, rec)

    popped = _pop_last_history(phrase_id)
    return popped


def phrase_locate_by_text(path: str | Path, search_text: str) -> dict[str, Any]:
    """Tool-facing: 'has this phrase text been tagged anywhere?' lookup.

    Scans the index for a matching (role_template, normalized_phrase). Returns
    `{phrase_id, current_text, rev}` if found, else `{unknown: True}`.
    """
    _ensure_dirs()
    role = _role_template_name(_resolve_prompt_path(path))
    needle = _normalize_phrase(search_text)
    for idx_file in INDEX_DIR.glob("*.json"):
        try:
            rec = json.loads(idx_file.read_text(encoding="utf-8"))
        except Exception:
            continue
        if rec.get("role_template") != role:
            continue
        if _normalize_phrase(rec.get("current_text", "")) == needle:
            return {
                "phrase_id": rec["phrase_id"],
                "current_text": rec["current_text"],
                "rev": rec.get("rev", 0),
                "section_path": rec.get("section_path", ""),
            }
    return {"unknown": True}
