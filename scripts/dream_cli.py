#!/usr/bin/env python3
"""Interactive CLI for manually-triggered dream runs.

Streams SSE from `POST /internal/dream-run` and renders events live. When a
`dream_finalize_review` event arrives, pauses the stream, prints per-edit
diffs, collects the user's keep/drop choices, and POSTs them back to
`/v1/dream/review_response` before continuing.

Zero extra dependencies — stdlib only (urllib, json, difflib).

Invoked by `make dream-run`; see Makefile for env var conventions.
"""

from __future__ import annotations

import argparse
import curses
import difflib
import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Callable, Iterator


# ── HTTP helpers ─────────────────────────────────────────────────────────────

def _post_json(url: str, payload: dict) -> urllib.request.addinfourl:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    return urllib.request.urlopen(req, timeout=660)


def _get_json(url: str) -> dict:
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _post_json_body(url: str, payload: dict) -> dict:
    """POST a JSON body and decode the JSON response (non-streaming)."""
    resp = _post_json(url, payload)
    return json.loads(resp.read().decode("utf-8"))


def _post_json_result(url: str, payload: dict) -> dict:
    try:
        resp = _post_json(url, payload)
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        print(f"[dream-cli] POST {url} → HTTP {e.code}: {body}", file=sys.stderr)
        return {"ok": False, "status": e.code, "body": body}
    try:
        return json.loads(resp.read().decode("utf-8"))
    except Exception:
        return {"ok": True}


# ── SSE parser ───────────────────────────────────────────────────────────────

def _iter_sse(resp) -> Iterator[tuple[str, dict]]:
    """Yield (event, data_dict) pairs from a chunked SSE response."""
    event = "message"
    data_lines: list[str] = []
    for raw in resp:
        line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
        if line == "":
            if data_lines:
                blob = "\n".join(data_lines)
                try:
                    data = json.loads(blob)
                except Exception:
                    data = {"_raw": blob}
                yield event, data
            event = "message"
            data_lines = []
            continue
        if line.startswith("event:"):
            event = line[len("event:"):].strip()
        elif line.startswith("data:"):
            data_lines.append(line[len("data:"):].lstrip())


# ── Rendering ────────────────────────────────────────────────────────────────

_BAR = "─" * 72

# Last `dream_conversation_start` event — consumed by `_tui_skip_ack` so the
# skip modal can show which conversation was skipped (channel, index, etc.)
# rather than just the raw rationale.
_LAST_CONV_CONTEXT: dict = {}


def _short(text: str, n: int = 120) -> str:
    text = (text or "").replace("\n", " ").strip()
    return text if len(text) <= n else text[: n - 1] + "…"


def _render_event(event: str, data: dict) -> None:
    if event == "dream_run_start":
        window = data.get("window") or {}
        if window:
            scope = f"last {window.get('hours', '?')}h  (until {(window.get('end') or '')[:19]})"
        else:
            scope = f"UTC day {data.get('date', '?')}"
        print(f"\n{_BAR}\ndream run: {scope}  "
              f"candidates={data.get('candidates', 0)}  "
              f"skipped={data.get('skipped', 0)}  "
              f"dreamer={data.get('dreamer_model', '?')}  "
              f"review={'on' if data.get('review_required') else 'off'}\n{_BAR}")
    elif event == "dream_conversation_start":
        _LAST_CONV_CONTEXT.clear()
        _LAST_CONV_CONTEXT.update(data)
        ch = data.get("channel_id")
        ch_str = f"  channel={ch}" if ch else ""
        worker_model = data.get("worker_model") or "?"
        mode = data.get("mode") or "?"
        # `prompt_files` lists the prompt files the dreamer can touch for
        # this conversation — lets the user see upfront whether this is a
        # single-target pass or a multi-agent coordinated edit.
        pf = data.get("prompt_files") or []
        pf_str = ""
        if pf:
            pf_str = "  prompts: " + ", ".join(
                f"{p.get('file', '?')}" for p in pf if p.get("file")
            )
        print(
            f"\n▸ conv {data.get('idx', '?')}/{data.get('total', '?')}  "
            f"role={data.get('role', '?')}  mode={mode}  "
            f"worker_model={worker_model}{ch_str}{pf_str}\n"
            f"  sid={data.get('sid', '?')}"
        )
    elif event == "dream_conversation_end":
        status = data.get("status", "?")
        committed = len(data.get("committed") or [])
        flagged = len(data.get("flagged") or [])
        err = data.get("error")
        if err:
            print(f"  ✗ {status}  {err}")
        else:
            print(f"  ✓ {status}  committed={committed}  flagged={flagged}")
    elif event == "dream_meta_start":
        print(f"\n▸ meta-dreamer  flagged_total={data.get('flagged_total', 0)}  "
              f"top_k={data.get('top_k', 0)}")
    elif event == "dream_meta_briefing":
        head = _short(data.get("head", ""), 300)
        print(f"  briefing: {head}")
    elif event == "dream_meta_end":
        print(f"  meta: {data.get('status', '?')}")
    elif event == "dream_run_end":
        print(f"\n{_BAR}\ndone: seen={data.get('seen', 0)}  "
              f"completed={data.get('completed', 0)}  "
              f"interrupted={data.get('interrupted', False)}\n{_BAR}")
    elif event == "worker_status":
        text = _short(data.get("text", ""), 200)
        if text:
            print(f"    · {text}")
    elif event == "tool_started":
        preview = data.get("params_preview") or ""
        print(f"    ⏳ {data.get('tool', '?')}  {_short(preview, 100)}")
    elif event == "tool_trace":
        tool = data.get("tool", "?")
        err = data.get("error")
        dur = data.get("duration_s")
        lines = data.get("lines")
        if err:
            print(f"    ✗ {tool}  ({dur}s)  {_short(str(err), 120)}")
        else:
            tail = f"({dur}s, {lines} lines)" if lines is not None else f"({dur}s)"
            print(f"    ✓ {tool}  {tail}")
    elif event == "injection":
        print(f"    ↩ inject[{data.get('mode', '?')}]: {_short(data.get('text', ''), 100)}")
    elif event == "dream_finalize_review_timeout":
        print(f"\n⚠ finalize review timed out for {data.get('dreamer_sid')!r} — dropping all edits.")
    elif event == "dream_skip":
        rationale = data.get("rationale") or "(no reason given)"
        # Print a one-line trace first (persists after TUI dismissal), then
        # pop a full-screen modal so the user has to acknowledge — otherwise
        # the skip flashes by inside a larger stream.
        print(f"\n  ⊘ dreamer chose to skip. rationale: {rationale}")
        if sys.stdin.isatty():
            _tui_skip_ack(rationale, context=dict(_LAST_CONV_CONTEXT))
    elif event == "error":
        print(f"\n✗ error: {data.get('error', data)}", file=sys.stderr)
    elif event == "done":
        pass  # caller handles stream exit
    else:
        print(f"    · [{event}] {_short(json.dumps(data), 120)}")


# ── Review flow (curses TUI) ─────────────────────────────────────────────────

def _wrap(text: str, width: int) -> list[str]:
    """Minimal word-wrap that preserves explicit \\n breaks."""
    out: list[str] = []
    for paragraph in (text or "").split("\n"):
        if not paragraph:
            out.append("")
            continue
        line = ""
        for word in paragraph.split(" "):
            if not line:
                line = word
                continue
            if len(line) + 1 + len(word) <= width:
                line += " " + word
            else:
                out.append(line)
                line = word
        if line:
            out.append(line)
    return out


def _diff_lines(old_text: str, new_text: str) -> list[tuple[str, str]]:
    """Return (line, attr_key) pairs — attr_key is 'minus'/'plus'/'hunk'/'norm'."""
    raw = list(difflib.unified_diff(
        (old_text or "").splitlines(),
        (new_text or "").splitlines(),
        lineterm="",
        fromfile="before",
        tofile="after",
        n=2,
    ))
    if not raw:
        return [("(no textual diff — metadata-only edit)", "norm")]
    out: list[tuple[str, str]] = []
    for line in raw:
        if line.startswith("+++") or line.startswith("---"):
            out.append((line, "hunk"))
        elif line.startswith("@@"):
            out.append((line, "hunk"))
        elif line.startswith("+"):
            out.append((line, "plus"))
        elif line.startswith("-"):
            out.append((line, "minus"))
        else:
            out.append((line, "norm"))
    return out


def _wrap_diff_line(line: str, attr_key: str, width: int) -> list[tuple[str, str]]:
    """Wrap a diff line to fit in `width` columns, preserving the leading
    prefix (+/-/space) on continuation lines so the color gutter reads
    consistently top-to-bottom. Uses a soft-wrap marker (`↳`) on the
    continuation prefix so wrapped chunks are visually distinct from real
    prefixed lines in the original diff.

    Hunk/header lines (attr_key='hunk') are just hard-clipped — wrapping a
    file-header or `@@` marker across lines would look worse than truncating.
    """
    if width <= 2:
        return [(line[:width], attr_key)]
    if attr_key == "hunk":
        return [(line[:width - 1], attr_key)]
    # Diff lines: prefix is the first char (+/-/ ).
    if not line:
        return [("", attr_key)]
    prefix = line[0] if line[0] in ("+", "-", " ") else ""
    body = line[len(prefix):]
    if prefix == "":
        # No diff prefix — wrap as plain text.
        chunks = [body[i:i + (width - 1)] for i in range(0, len(body), width - 1)] or [""]
        return [(c, attr_key) for c in chunks]
    avail = max(1, width - 1 - len(prefix))
    first_prefix = prefix + " "
    cont_prefix = prefix + "↳"
    out: list[tuple[str, str]] = []
    if not body:
        return [(first_prefix.rstrip(), attr_key)]
    for i in range(0, len(body), avail):
        chunk = body[i:i + avail]
        p = first_prefix if i == 0 else cont_prefix
        out.append((p + chunk, attr_key))
    return out


def _tui_review_edits(data: dict) -> dict[str, str] | None:
    """Full-screen curses review TUI. Returns {phrase_id: keep|drop} or None
    if the user aborted (treated as drop-all by caller).

    Layout:
        header: target prompts (count), dreamer sid, batch rationale (wrapped)
        strip:  edit index + group-of-total, status, kind, phrase_id, current target
        body:   per-edit narrative + colored diff (scrollable with PgUp/PgDn)
        help:   key bindings (incl. A/R per-target shortcuts)

    Multi-target batches render the edits sorted by target_prompt so all
    edits for `worker_full.md` appear together, followed by all edits for
    `supervisor_full.md`, etc. `A`/`R` (uppercase) flip keep/drop for every
    edit in the current target's group.
    """
    raw_edits = data.get("edits") or []
    if not raw_edits:
        return {}

    # Sort stable-by-target so edits cluster by prompt file. Use the flat
    # list's original order (idx) as the tie-breaker so edits within a
    # target stay in the order the diff produced them.
    edits = sorted(
        enumerate(raw_edits),
        key=lambda ei: (ei[1].get("target_prompt") or "", ei[0]),
    )
    edits = [e for _, e in edits]

    target_prompts = data.get("target_prompts") or []
    if not target_prompts:
        t = data.get("target_prompt")
        target_prompts = [t] if t else []
    dreamer_sid = data.get("dreamer_sid") or "?"
    batch_rationale = (data.get("rationale") or "").strip() or "(no batch-level rationale)"

    # Precompute group boundaries: for each edit index, which target it
    # belongs to — used by the A/R per-group shortcut.
    edit_targets: list[str] = [e.get("target_prompt") or "?" for e in edits]
    # For the strip: "edit 3/7 in `worker_full.md` (group 1/2)" style text.
    unique_targets: list[str] = []
    for t in edit_targets:
        if t not in unique_targets:
            unique_targets.append(t)

    def _run(stdscr):
        curses.curs_set(0)
        stdscr.keypad(True)
        # Color pairs: keep terminal-friendly fallbacks when color init fails.
        try:
            curses.start_color()
            curses.use_default_colors()
            curses.init_pair(1, curses.COLOR_RED, -1)     # minus
            curses.init_pair(2, curses.COLOR_GREEN, -1)   # plus
            curses.init_pair(3, curses.COLOR_CYAN, -1)    # hunk / header
            curses.init_pair(4, curses.COLOR_YELLOW, -1)  # rationale
            attr = {
                "minus": curses.color_pair(1),
                "plus":  curses.color_pair(2),
                "hunk":  curses.color_pair(3) | curses.A_DIM,
                "norm":  0,
                "rationale": curses.color_pair(4),
            }
        except Exception:
            attr = {"minus": curses.A_DIM, "plus": curses.A_BOLD,
                    "hunk": curses.A_DIM, "norm": 0, "rationale": curses.A_BOLD}

        selected = {e["phrase_id"]: "keep" for e in edits}
        cursor = 0
        scroll = 0  # vertical scroll into body

        while True:
            stdscr.erase()
            h, w = stdscr.getmaxyx()
            edit = edits[cursor]
            pid = edit["phrase_id"]

            # Header: batch-level context — list the target prompts at a
            # glance so the user sees the cross-file scope, then the
            # dreamer's rationale (wrapped).
            if len(target_prompts) > 1:
                header_title = (
                    f"Review {len(edits)} edits across {len(target_prompts)} prompts "
                    f"({', '.join(target_prompts)})"
                )
            else:
                header_title = f"Review edits for `{target_prompts[0] if target_prompts else '?'}`"
            header_lines = [f"{header_title}  dreamer_sid=`{dreamer_sid[:40]}`"]
            rationale_wrapped = _wrap(f"rationale: {batch_rationale}", max(1, w - 2))[:4]
            header_lines.extend(rationale_wrapped)
            for i, line in enumerate(header_lines):
                _safe_addstr(stdscr, i, 0, line,
                             curses.A_BOLD if i == 0 else attr["rationale"])

            # Group header row — called out so it's clear which prompt
            # the current edit belongs to. Renders as a cyan separator
            # line above the per-edit status strip.
            current_target = edit_targets[cursor]
            group_total = sum(1 for t in edit_targets if t == current_target)
            group_idx_within = (
                sum(1 for j in range(cursor + 1) if edit_targets[j] == current_target)
            )
            group_row = len(header_lines) + 1
            _safe_addstr(
                stdscr, group_row, 0,
                f"── {current_target}  (edit {group_idx_within}/{group_total} in this prompt) ──",
                attr["hunk"] | curses.A_BOLD,
            )

            # Status strip
            strip_row = group_row + 1
            kept_count = sum(1 for v in selected.values() if v == "keep")
            strip = (
                f"[{cursor + 1}/{len(edits)} overall]  "
                f"{'● KEEP ' if selected[pid] == 'keep' else '○ DROP '}"
                f"kind={edit.get('kind', '?')}  status={edit.get('status', 'ok')}  "
                f"selected: {kept_count}/{len(edits)}  phrase_id={pid[:16]}"
            )
            _safe_addstr(stdscr, strip_row, 0, strip, curses.A_REVERSE)
            section = f"section: {edit.get('section_path') or '(root)'}"
            _safe_addstr(stdscr, strip_row + 1, 0, section, curses.A_DIM)

            # Body: narrative (if any) + diff, scrollable
            body_top = strip_row + 3
            help_rows = 2
            body_bottom = h - help_rows - 1
            body_height = max(1, body_bottom - body_top)

            # Body is a list of (text, attr_code) pairs. We pre-wrap diff
            # lines here so the scroll window works on visible rows, not raw
            # diff rows — otherwise long paragraphs clipped off the right
            # edge invisibly.
            body: list[tuple[str, int]] = []
            wrap_w = max(1, w - 1)
            narrative = (edit.get("narrative") or "").strip()
            if narrative:
                body.append(("── per-edit narrative ──", attr["hunk"]))
                for line in _wrap(narrative, wrap_w):
                    body.append((line, attr["rationale"]))
                body.append(("", 0))
            body.append(("── diff ──", attr["hunk"]))
            for raw_line, key in _diff_lines(edit.get("old_text", ""),
                                             edit.get("new_text", "")):
                for wrapped, wkey in _wrap_diff_line(raw_line, key, wrap_w):
                    body.append((wrapped, attr[wkey]))

            max_scroll = max(0, len(body) - body_height)
            scroll = min(scroll, max_scroll)
            for i in range(body_height):
                src_idx = scroll + i
                if src_idx >= len(body):
                    break
                line, key = body[src_idx]
                _safe_addstr(stdscr, body_top + i, 0, line, attr.get(key, 0) if isinstance(key, str) else key)

            # Help line at bottom — uppercase A/R apply to the current
            # target's group only; lowercase a/r apply to the full batch.
            help1 = (
                "←/→ edit  ↑/↓ scroll  space=keep/drop  "
                "a/r=all keep/drop  A/R=group keep/drop  Enter=submit  q=abort"
            )
            _safe_addstr(stdscr, h - 1, 0, help1, curses.A_REVERSE)
            if max_scroll > 0:
                _safe_addstr(stdscr, h - 2, 0,
                             f"scroll {scroll}/{max_scroll}   (PgUp/PgDn jump)", curses.A_DIM)

            stdscr.refresh()
            k = stdscr.getch()
            if k in (curses.KEY_LEFT, ord("h")):
                cursor = (cursor - 1) % len(edits)
                scroll = 0
            elif k in (curses.KEY_RIGHT, ord("l")):
                cursor = (cursor + 1) % len(edits)
                scroll = 0
            elif k in (curses.KEY_UP, ord("k")):
                scroll = max(0, scroll - 1)
            elif k in (curses.KEY_DOWN, ord("j")):
                scroll = min(max_scroll, scroll + 1)
            elif k == curses.KEY_PPAGE:
                scroll = max(0, scroll - body_height)
            elif k == curses.KEY_NPAGE:
                scroll = min(max_scroll, scroll + body_height)
            elif k == curses.KEY_HOME:
                scroll = 0
            elif k == curses.KEY_END:
                scroll = max_scroll
            elif k == ord(" "):
                selected[pid] = "drop" if selected[pid] == "keep" else "keep"
            elif k == ord("a"):
                selected = {e["phrase_id"]: "keep" for e in edits}
            elif k == ord("r"):
                selected = {e["phrase_id"]: "drop" for e in edits}
            elif k == ord("A"):
                # Keep all edits in the current target's group only.
                for j, e in enumerate(edits):
                    if edit_targets[j] == current_target:
                        selected[e["phrase_id"]] = "keep"
            elif k == ord("R"):
                # Drop all edits in the current target's group only.
                for j, e in enumerate(edits):
                    if edit_targets[j] == current_target:
                        selected[e["phrase_id"]] = "drop"
            elif k in (10, 13, curses.KEY_ENTER):
                return selected
            elif k in (27, ord("q")):
                return None

    try:
        return curses.wrapper(_run)
    except Exception as e:
        print(f"[dream-cli] review TUI unavailable ({e}); falling back to text prompt.",
              file=sys.stderr)
        print(f"\nbatch rationale: {batch_rationale}")
        for i, edit in enumerate(edits, start=1):
            print(f"\n─── edit {i}/{len(edits)}  kind={edit.get('kind')}  "
                  f"status={edit.get('status', 'ok')}  phrase_id={edit.get('phrase_id')}")
            nar = (edit.get("narrative") or "").strip()
            if nar:
                print(f"  narrative: {nar[:300]}")
            for line, _ in _diff_lines(edit.get("old_text", ""), edit.get("new_text", "")):
                print(f"  {line}")
        raw = input("\n[A]ccept all / [R]eject all / [q]uit > ").strip().lower()
        if raw in ("", "a", "accept"):
            return {e["phrase_id"]: "keep" for e in edits}
        if raw in ("r", "reject"):
            return {e["phrase_id"]: "drop" for e in edits}
        return None


def _tui_skip_ack(rationale: str, *, context: dict | None = None) -> None:
    """Modal showing the dreamer's skip rationale. Wait for Enter/q to
    dismiss. Purely informational — we can't force the dreamer to submit
    after the fact; this just makes the decision visible.

    `context` (shaped like a dream_conversation_start event) is used to show
    which conversation was skipped and what comes next, so the user doesn't
    dismiss the modal wondering what just happened.
    """
    ctx = context or {}
    idx = ctx.get("idx")
    total = ctx.get("total")
    title = "Dreamer chose to skip this conversation"
    sub_lines: list[str] = []
    channel_name = ctx.get("channel_name") or (
        f"channel {ctx.get('channel_id')}" if ctx.get("channel_id") else ""
    )
    if idx is not None and total is not None:
        sub_lines.append(
            f"conv {int(idx) + 1}/{total}"
            + (f"  #{channel_name.lstrip('#')}" if channel_name else "")
            + (f"  role={ctx.get('role', '?')}/{ctx.get('mode', '?')}" if ctx.get("role") else "")
        )
    if ctx.get("sid"):
        sub_lines.append(f"sid {ctx.get('sid')}")
    if idx is not None and total is not None:
        remaining = max(0, int(total) - int(idx) - 1)
        if remaining > 0:
            sub_lines.append(
                f"Enter acknowledges → run continues with {remaining} more conversation(s)."
            )
        else:
            sub_lines.append("Enter acknowledges → this was the last conversation.")

    def _run(stdscr):
        curses.curs_set(0)
        stdscr.keypad(True)
        try:
            curses.start_color()
            curses.use_default_colors()
            curses.init_pair(5, curses.COLOR_YELLOW, -1)
            curses.init_pair(6, curses.COLOR_CYAN, -1)
            yellow = curses.color_pair(5)
            cyan = curses.color_pair(6)
        except Exception:
            yellow = curses.A_BOLD
            cyan = curses.A_DIM

        scroll = 0
        while True:
            stdscr.erase()
            h, w = stdscr.getmaxyx()
            _safe_addstr(stdscr, 0, 0, title, curses.A_BOLD)
            row = 1
            for sub in sub_lines:
                _safe_addstr(stdscr, row, 0, sub, cyan)
                row += 1
            _safe_addstr(stdscr, row, 0, "─" * max(1, w - 1), curses.A_DIM)
            body_top = row + 2
            body = _wrap(rationale, max(1, w - 2))
            body_height = max(1, h - body_top - 2)
            max_scroll = max(0, len(body) - body_height)
            scroll = min(scroll, max_scroll)
            for i in range(body_height):
                src = scroll + i
                if src >= len(body):
                    break
                _safe_addstr(stdscr, body_top + i, 0, body[src], yellow)
            _safe_addstr(stdscr, h - 1, 0,
                         "↑/↓ scroll  Enter/q acknowledge", curses.A_REVERSE)
            stdscr.refresh()
            k = stdscr.getch()
            if k in (curses.KEY_UP, ord("k")):
                scroll = max(0, scroll - 1)
            elif k in (curses.KEY_DOWN, ord("j")):
                scroll = min(max_scroll, scroll + 1)
            elif k == curses.KEY_PPAGE:
                scroll = max(0, scroll - body_height)
            elif k == curses.KEY_NPAGE:
                scroll = min(max_scroll, scroll + body_height)
            elif k in (10, 13, curses.KEY_ENTER, 27, ord("q")):
                return

    try:
        curses.wrapper(_run)
    except Exception:
        # Fallback: just print it. The subtext line already printed earlier
        # remains the persistent record.
        print(f"\n  ⊘ dreamer chose to skip. rationale:\n  {rationale}")


def _handle_review(url: str, data: dict) -> None:
    sid = data.get("dreamer_sid")
    edits = data.get("edits") or []
    decisions = _tui_review_edits(data)
    if decisions is None:
        # Aborted — drop everything to be safe.
        print("Review aborted — all edits dropped.")
        decisions = {e["phrase_id"]: "drop" for e in edits}
    kept = sum(1 for v in decisions.values() if v == "keep")
    dropped = sum(1 for v in decisions.values() if v == "drop")
    print(f"→ POSTing decisions: keep={kept}  drop={dropped}")
    result = _post_json_result(
        f"{url}/v1/dream/review_response",
        {"dreamer_sid": sid, "decisions": decisions},
    )
    if result.get("ok"):
        print(f"✓ review submitted (n_keep={result.get('n_keep')}, n_drop={result.get('n_drop')})")
    else:
        print(f"⚠ review submission failed: {result}")


# ── Startup pickers (curses TUI) ─────────────────────────────────────────────

# Sentinel return from the TUI helpers meaning "user aborted; don't run".
_ABORT = object()


def _fmt_ts(ts: str) -> str:
    return (ts or "")[:19].replace("T", " ")


def _safe_addstr(stdscr, row: int, col: int, text: str, attr: int = 0) -> None:
    """addstr that clips to the screen width and swallows curses errors.

    curses raises on writes past the last cell of the last row even when
    innocuous; truncating here lets us compose arbitrary lines without
    guarding every call site.
    """
    try:
        _, w = stdscr.getmaxyx()
        if col >= w:
            return
        text = text[: max(0, w - col - 1)]
        stdscr.addstr(row, col, text, attr)
    except curses.error:
        pass


def _tui_single_select(
    title: str,
    items: list,
    render: Callable[[Any], str],
    *,
    default_idx: int = 0,
) -> int | None:
    """Arrow-navigation single-select. Returns chosen index or None on abort."""
    if not items:
        return None

    def _run(stdscr):
        cursor = max(0, min(default_idx, len(items) - 1))
        curses.curs_set(0)
        stdscr.keypad(True)
        while True:
            stdscr.erase()
            h, _w = stdscr.getmaxyx()
            _safe_addstr(stdscr, 0, 0, title, curses.A_BOLD)
            help_line = "↑/↓ or j/k  Enter=confirm  q=abort"
            _safe_addstr(stdscr, h - 1, 0, help_line, curses.A_REVERSE)

            visible = max(1, h - 3)
            scroll = max(0, cursor - (visible - 1))
            for i in range(visible):
                idx = scroll + i
                if idx >= len(items):
                    break
                row = 2 + i
                marker = "▶" if idx == cursor else " "
                line = f"{marker} {render(items[idx])}"
                attr = curses.A_REVERSE if idx == cursor else 0
                _safe_addstr(stdscr, row, 0, line, attr)

            stdscr.refresh()
            k = stdscr.getch()
            if k in (curses.KEY_UP, ord("k")):
                cursor = (cursor - 1) % len(items)
            elif k in (curses.KEY_DOWN, ord("j")):
                cursor = (cursor + 1) % len(items)
            elif k in (curses.KEY_HOME, ord("g")):
                cursor = 0
            elif k in (curses.KEY_END, ord("G")):
                cursor = len(items) - 1
            elif k in (10, 13, curses.KEY_ENTER):
                return cursor
            elif k in (27, ord("q")):
                return None

    try:
        return curses.wrapper(_run)
    except Exception as e:
        print(f"[dream-cli] TUI unavailable ({e}); falling back to numeric prompt.", file=sys.stderr)
        # Minimal fallback if curses can't start (no controlling tty, weird TERM).
        for i, it in enumerate(items, start=1):
            print(f"  {i:2d}. {render(it)}")
        raw = input(f"> pick [1-{len(items)}, Enter={default_idx + 1}]: ").strip()
        if raw == "":
            return default_idx
        if raw.isdigit() and 1 <= int(raw) <= len(items):
            return int(raw) - 1
        return None


def _tui_multi_select(
    title: str,
    items: list,
    render: Callable[[Any], str],
    *,
    default_all: bool = True,
    footer: Callable[[Any], str] | None = None,
) -> list[int] | None:
    """Arrow-navigation multi-select with space toggle. Returns indices."""
    if not items:
        return []

    def _run(stdscr):
        cursor = 0
        selected = set(range(len(items))) if default_all else set()
        curses.curs_set(0)
        stdscr.keypad(True)
        while True:
            stdscr.erase()
            h, _w = stdscr.getmaxyx()
            _safe_addstr(stdscr, 0, 0, title, curses.A_BOLD)
            help_line = "↑/↓  space=toggle  a=all  n=none  Enter=confirm  q=abort"
            _safe_addstr(stdscr, h - 1, 0, help_line, curses.A_REVERSE)
            status = f"{len(selected)} of {len(items)} selected"
            _safe_addstr(stdscr, h - 2, 0, status)
            if footer:
                _safe_addstr(stdscr, h - 3, 0, footer(items[cursor]), curses.A_DIM)

            header_rows = 2
            footer_rows = 3 if footer else 2
            visible = max(1, h - header_rows - footer_rows)
            scroll = max(0, cursor - (visible - 1))
            for i in range(visible):
                idx = scroll + i
                if idx >= len(items):
                    break
                row = header_rows + i
                chk = "[x]" if idx in selected else "[ ]"
                marker = "▶" if idx == cursor else " "
                line = f"{marker} {chk}  {render(items[idx])}"
                attr = curses.A_REVERSE if idx == cursor else 0
                _safe_addstr(stdscr, row, 0, line, attr)

            stdscr.refresh()
            k = stdscr.getch()
            if k in (curses.KEY_UP, ord("k")):
                cursor = (cursor - 1) % len(items)
            elif k in (curses.KEY_DOWN, ord("j")):
                cursor = (cursor + 1) % len(items)
            elif k in (curses.KEY_HOME, ord("g")):
                cursor = 0
            elif k in (curses.KEY_END, ord("G")):
                cursor = len(items) - 1
            elif k == ord(" "):
                if cursor in selected:
                    selected.discard(cursor)
                else:
                    selected.add(cursor)
            elif k == ord("a"):
                selected = set(range(len(items)))
            elif k == ord("n"):
                selected = set()
            elif k in (10, 13, curses.KEY_ENTER):
                return sorted(selected)
            elif k in (27, ord("q")):
                return None

    try:
        return curses.wrapper(_run)
    except Exception as e:
        print(f"[dream-cli] TUI unavailable ({e}); falling back to numeric prompt.", file=sys.stderr)
        for i, it in enumerate(items, start=1):
            print(f"  {i:2d}. {render(it)}")
        raw = input("> numbers to KEEP (comma-sep, Enter=all, 'none'): ").strip().lower()
        if raw == "":
            return list(range(len(items)))
        if raw == "none":
            return []
        picks: list[int] = []
        try:
            picks = [int(t.strip()) - 1 for t in raw.split(",") if t.strip()]
        except ValueError:
            return None
        if any(p < 0 or p >= len(items) for p in picks):
            return None
        return sorted(set(picks))


def _pick_model(url: str) -> str | None:
    """Single-select TUI for the dreamer model. Returns the chosen name, or
    None if the user aborted (caller should exit)."""
    try:
        data = _get_json(f"{url}/internal/dream-models")
    except Exception as e:
        print(f"[dream-cli] could not fetch model list: {e}", file=sys.stderr)
        return None
    options = data.get("options") or []
    default = data.get("default")
    if not options:
        return None
    default_idx = next(
        (i for i, o in enumerate(options) if o.get("name") == default), 0
    )

    def _render(opt: dict) -> str:
        label = opt.get("label") or opt.get("name") or "?"
        tag = "  (default)" if opt.get("name") == default else ""
        return f"{label}{tag}"

    picked = _tui_single_select(
        f"Select dreamer model ({len(options)} available)",
        options,
        _render,
        default_idx=default_idx,
    )
    if picked is None:
        return None
    return options[picked].get("name")


def _pick_candidates(
    url: str,
    *,
    date: str | None,
    window_hours: float | None,
) -> list[str] | None:
    """Multi-select TUI for which conversations to dream. Returns a list of
    sids. None = user aborted; [] = user deselected everything."""
    payload: dict = {}
    if date:
        payload["date"] = date
    elif window_hours is not None:
        payload["window_hours"] = window_hours
    try:
        data = _post_json_body(f"{url}/internal/dream-candidates", payload)
    except Exception as e:
        print(f"[dream-cli] could not preview candidates: {e}", file=sys.stderr)
        return None
    candidates = data.get("candidates") or []
    skipped = data.get("skipped") or []
    scope = data.get("scope") or {}
    scope_label = (
        f"last {scope.get('hours', '?')}h"
        if scope.get("mode") == "window"
        else f"date {scope.get('date', '?')}"
    )
    if not candidates:
        print(f"(no dreamable candidates in {scope_label}; {len(skipped)} filtered out)")
        return []

    def _row(c: dict) -> str:
        name = c.get("channel_name")
        if name:
            ch = f"#{name}"
        else:
            ch = "(no channel)"
        role = c.get("agent_role") or "?"
        mode = c.get("mode") or "?"
        turns = c.get("final_turn_count", "?")
        ts = _fmt_ts(c.get("last_final_ts", ""))
        return f"{ch:32s}  {role}/{mode:10s}  turns={turns:<3}  last={ts}"

    def _footer(c: dict) -> str:
        return f"sid: {c.get('session_id')}"

    title = (
        f"Select conversations to dream ({len(candidates)} available, "
        f"{scope_label}; {len(skipped)} filtered out)"
    )
    picked = _tui_multi_select(title, candidates, _row, footer=_footer)
    if picked is None:
        return None
    return [candidates[i]["session_id"] for i in picked]


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description="Manually trigger a verbose dream run.")
    ap.add_argument("--url", default="http://localhost:8090", help="phoebe-api base URL")
    ap.add_argument("--date", default=None,
                    help="YYYY-MM-DD UTC calendar day (default: rolling last 24h)")
    ap.add_argument("--window-hours", type=float, default=None,
                    help="rolling-window size in hours (default: 24; ignored when --date is set)")
    ap.add_argument("--meta", default="1", help="meta-dreamer on/off (1/0)")
    ap.add_argument("--review", default="1", help="per-edit review on/off (1/0)")
    ap.add_argument("--yes", "-y", action="store_true",
                    help="skip startup pickers (use server defaults for model and candidates)")
    ap.add_argument("--dreamer-model", default=None,
                    help="override dreamer model (bypasses the picker)")
    args = ap.parse_args()

    payload: dict[str, Any] = {
        "verbose": True,
        "meta_enabled": args.meta not in ("0", "false", "no", ""),
        "review": args.review not in ("0", "false", "no", ""),
    }
    if args.date:
        payload["date"] = args.date
    elif args.window_hours is not None:
        payload["window_hours"] = args.window_hours

    interactive = sys.stdin.isatty() and not args.yes

    if args.dreamer_model:
        payload["dreamer_model"] = args.dreamer_model
    elif interactive:
        chosen_model = _pick_model(args.url)
        if chosen_model is None:
            print("Aborted.")
            return 0
        payload["dreamer_model"] = chosen_model

    if interactive:
        chosen_sids = _pick_candidates(
            args.url,
            date=args.date,
            window_hours=args.window_hours,
        )
        if chosen_sids is None:
            print("Aborted.")
            return 0
        if not chosen_sids:
            print("Nothing selected — exiting.")
            return 0
        payload["conversation_sids"] = chosen_sids
        # Strip date/window — explicit sids override and we want a clean log.
        payload.pop("date", None)
        payload.pop("window_hours", None)

    try:
        resp = _post_json(f"{args.url}/internal/dream-run", payload)
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        print(f"[dream-cli] POST /internal/dream-run → HTTP {e.code}: {body}", file=sys.stderr)
        return 2
    except urllib.error.URLError as e:
        print(f"[dream-cli] connection to {args.url} failed: {e}", file=sys.stderr)
        return 2

    # Per-conversation errors are common (transient provider overloads,
    # rate limits, one dreamer call misfiring). They shouldn't fail the
    # whole run — other conversations may still complete. We tally them and
    # exit non-zero only if `done` never arrives (fatal stream failure).
    conv_errors = 0
    got_done = False
    for event, data in _iter_sse(resp):
        _render_event(event, data)
        if event == "dream_finalize_review":
            _handle_review(args.url, data)
        elif event == "done":
            got_done = True
            break
        elif event == "error":
            conv_errors += 1
    if not got_done:
        print("[dream-cli] stream ended without `done` event", file=sys.stderr)
        return 1
    if conv_errors:
        print(f"\n[dream-cli] {conv_errors} per-conversation error(s) during run.",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
