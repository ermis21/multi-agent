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


def split_message(text: str) -> list[str]:
    """Split text into Discord-safe chunks, breaking at newlines where possible."""
    text = transform_markdown_tables(text)
    chunks = []
    while len(text) > MAX_MSG_LEN:
        split_at = text.rfind("\n", 0, MAX_MSG_LEN)
        if split_at == -1:
            split_at = MAX_MSG_LEN
        chunks.append(text[:split_at])
        text = text[split_at:].lstrip()
    if text:
        chunks.append(text)
    return chunks
