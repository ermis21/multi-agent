import os
import re

MAX_MSG_LEN = int(os.environ.get("DISCORD_MAX_MESSAGE_LENGTH", "1900"))

# If set, only these user IDs may interact with the bots.
# Parsed once at import time so every message check is an O(1) set lookup.
_raw = os.environ.get("DISCORD_ALLOWED_USER_IDS", "")
ALLOWED_USER_IDS: frozenset[int] = frozenset(
    int(uid) for uid in _raw.split(",") if uid.strip()
)


def is_allowed(user_id: int) -> bool:
    """Return True if this user is permitted to use the bot.

    When DISCORD_ALLOWED_USER_IDS is empty, everyone is allowed.
    When it is set, only listed IDs pass.
    """
    if not ALLOWED_USER_IDS:
        return True
    return user_id in ALLOWED_USER_IDS


_SEPARATOR_CELL_RE = re.compile(r"^\s*:?-{3,}:?\s*$")


def _is_pipe_row(line: str) -> bool:
    stripped = line.rstrip()
    return stripped.startswith("|") and stripped.endswith("|") and stripped.count("|") >= 2


def _parse_cells(line: str) -> list[str]:
    """Split a |...| row into cells, dropping the empty outer cells."""
    parts = line.rstrip().split("|")
    # Outer pipes produce leading/trailing empty strings — discard them.
    return [p.strip() for p in parts[1:-1]]


def _is_separator_row(line: str) -> bool:
    if not _is_pipe_row(line):
        return False
    cells = _parse_cells(line)
    return bool(cells) and all(_SEPARATOR_CELL_RE.match(c) for c in cells)


def _render_table(header: list[str], body: list[list[str]]) -> list[str]:
    """Return the rewritten lines for one detected table (without fence wrappers)."""
    ncols = max(len(header), max((len(r) for r in body), default=0))
    # Pad ragged rows.
    header = header + [""] * (ncols - len(header))
    body   = [r + [""] * (ncols - len(r)) for r in body]

    widths = [len(header[i]) for i in range(ncols)]
    for row in body:
        for i, cell in enumerate(row):
            if len(cell) > widths[i]:
                widths[i] = len(cell)

    def fmt(cells: list[str]) -> str:
        return " | ".join(cells[i].ljust(widths[i]) for i in range(ncols))

    divider = "-+-".join("-" * widths[i] for i in range(ncols))
    return [fmt(header), divider, *(fmt(r) for r in body)]


def transform_markdown_tables(text: str) -> str:
    """Rewrite GitHub-style markdown tables as fixed-width blocks in a code fence.

    Discord does not render markdown tables; this converts them to aligned
    monospace text wrapped in ``` so they render readably.

    Rules:
      - Content inside existing ``` fences is passed through unchanged.
      - A table is recognised only when a pipe-row is immediately followed
        by a separator row (``|---|---|``).
      - If the padded table is wider than MAX_MSG_LEN - 10, the original
        raw lines are emitted unchanged (fallback for pathological widths).
    """
    lines = text.split("\n")
    out: list[str] = []
    i = 0
    in_fence = False

    while i < len(lines):
        line = lines[i]

        # Toggle code-fence state and pass the line through verbatim.
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            out.append(line)
            i += 1
            continue

        if in_fence:
            out.append(line)
            i += 1
            continue

        # Detect start of a table: header row + separator row.
        if (_is_pipe_row(line)
                and i + 1 < len(lines)
                and _is_separator_row(lines[i + 1])):
            header = _parse_cells(line)
            body: list[list[str]] = []
            j = i + 2
            while j < len(lines) and _is_pipe_row(lines[j]):
                body.append(_parse_cells(lines[j]))
                j += 1

            rendered = _render_table(header, body)
            max_width = max(len(r) for r in rendered) if rendered else 0

            if max_width <= MAX_MSG_LEN - 10:
                out.append("```")
                out.extend(rendered)
                out.append("```")
            else:
                # Fallback: leave the raw markdown as-is.
                out.extend(lines[i:j])

            i = j
            continue

        out.append(line)
        i += 1

    return "\n".join(out)


_FENCE_OPEN_RE = re.compile(r"^```([^\n`]*)", re.MULTILINE)


def _active_fence_tag(text: str) -> str | None:
    """Return the tag of the fence that is still open at the end of `text`,
    or None if fences are balanced.

    The tag is whatever followed ``` on the opening line (e.g. 'python',
    'json', '') — we preserve it so the next chunk re-opens with the same
    syntax highlighting.
    """
    opens = list(_FENCE_OPEN_RE.finditer(text))
    if len(opens) % 2 == 0:
        return None
    return opens[-1].group(1)


def split_message(text: str) -> list[str]:
    """Split text into Discord-safe chunks, breaking at newlines where possible.

    Code-fence-aware: if a chunk ends with an unclosed ```-block, the chunk is
    closed with ``` and the next chunk is opened with ```<tag> so syntax
    highlighting survives the boundary. Prevents the "1950-char code block
    cuts mid-function" bug where msg N ends with an unterminated fence and
    msg N+1 renders the opener as literal backticks.
    """
    text = transform_markdown_tables(text)
    chunks: list[str] = []
    carry_fence_tag: str | None = None  # tag to re-open at start of next chunk

    while True:
        # Re-open the previous chunk's still-open fence before measuring length.
        prefix = f"```{carry_fence_tag}\n" if carry_fence_tag is not None else ""
        remaining = prefix + text
        if len(remaining) <= MAX_MSG_LEN:
            if remaining:
                chunks.append(remaining)
            break

        # Hunt for a newline-aligned split inside the budget.
        # We budget an extra 4 chars for a possible trailing "\n```" close.
        budget = MAX_MSG_LEN - 4
        split_at = remaining.rfind("\n", 0, budget)
        if split_at == -1:
            split_at = budget

        piece = remaining[:split_at]
        rest  = remaining[split_at:].lstrip("\n")

        # If the piece leaves a fence open, close it and remember the tag for
        # the next chunk. `_active_fence_tag` sees the prefix we prepended, so
        # it correctly counts the re-opened carry fence too.
        open_tag = _active_fence_tag(piece)
        if open_tag is not None:
            piece = piece.rstrip() + "\n```"
            carry_fence_tag = open_tag
        else:
            carry_fence_tag = None

        chunks.append(piece)

        # If the rest is only a closing fence we just opened, skip carrying —
        # the next iteration would emit ```<tag>\n```\n, which is noise.
        if carry_fence_tag is not None and rest.strip() == "```":
            chunks[-1] = chunks[-1].rstrip().removesuffix("```").rstrip() + "\n```"
            carry_fence_tag = None
            text = ""
            continue

        text = rest

    return [c for c in chunks if c]
