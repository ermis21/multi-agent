"""Simulation overlay filesystem — helpers for sandbox handlers.

Under a dream-simulator replay, `mcp_client.call_tool` injects a `_simulate`
marker into tool params. Sandbox handlers route writes into an overlay tree
so real state is never mutated; subsequent reads/lists must see the overlay
content for the agent's behavior to remain consistent.

This module is pure helpers: no side effects at import, no global state. Each
function takes the marker's `overlay_root` explicitly. Handlers consult
`is_sim(params)` at entry and branch.

Overlay layout under `overlay_root = /cache/dream_sim_overlay/{sim_sid}/fs`:

    fs/
      workspace/...        mirrors writable roots
      config/...
      state/...
      cache/...
      .tombstones          one `prefix/relpath` per line

`project/` is read-only by design — no overlay twin; project reads always
pass through to the real mount.
"""

from __future__ import annotations

import os
from pathlib import Path


# ── Marker extraction ──────────────────────────────────────────────────────

def is_sim(params: dict) -> dict | None:
    """Return the `_simulate` marker dict if present, else None.

    Shape of the marker (set by app/mcp_client.py):
        {
          "sim_sid":          str,
          "overlay_root":     str,       # abs path
          "memory_collection": str,
        }
    """
    marker = params.get("_simulate") if isinstance(params, dict) else None
    if not marker or not isinstance(marker, dict):
        return None
    if not marker.get("overlay_root"):
        return None
    return marker


def overlay_root_of(marker: dict) -> Path:
    return Path(marker["overlay_root"])


# ── Path mapping ───────────────────────────────────────────────────────────

# Writable roots and their overlay prefix labels. Kept in one place so
# handlers and overlay stay in sync. `project` is intentionally absent —
# /project is read-only; the overlay cannot shadow it.
_WRITABLE_ROOT_NAMES = ("workspace", "config", "state", "cache")


def _roots_from_env() -> dict[str, Path]:
    """Return resolved paths for each writable root, matching mcp_server."""
    return {
        "workspace": Path(os.environ.get("WORKSPACE_DIR", "/workspace")).resolve(),
        "config":    Path(os.environ.get("CONFIG_DIR",    "/config")).resolve(),
        "state":     Path(os.environ.get("STATE_DIR",     "/state")).resolve(),
        "cache":     Path(os.environ.get("CACHE_DIR",     "/cache")).resolve(),
    }


def classify_root(real_path: Path) -> tuple[str | None, Path | None]:
    """Return (root_name, relpath) or (None, None) if path is outside any
    writable root. Used to map a resolved real path to an overlay twin."""
    roots = _roots_from_env()
    for name in _WRITABLE_ROOT_NAMES:
        root = roots[name]
        try:
            rel = real_path.resolve().relative_to(root)
            return name, rel
        except ValueError:
            continue
    return None, None


def overlay_path_for(real_path: Path, overlay_root: Path) -> Path | None:
    """Map a resolved real path to its overlay twin.

    Returns None when the real path is not under any writable root (e.g.
    /project reads). Callers should fall through to the real path in that case.
    """
    name, rel = classify_root(real_path)
    if name is None or rel is None:
        return None
    return (overlay_root / name / rel)


def ensure_parent(path: Path) -> None:
    """mkdir -p the parent of `path`. Idempotent."""
    path.parent.mkdir(parents=True, exist_ok=True)


# ── Tombstones ─────────────────────────────────────────────────────────────

def _tombstone_file(overlay_root: Path) -> Path:
    return overlay_root / ".tombstones"


def _tombstone_key(real_path: Path) -> str | None:
    name, rel = classify_root(real_path)
    if name is None:
        return None
    return f"{name}/{rel.as_posix()}"


def mark_deleted(overlay_root: Path, real_path: Path) -> None:
    """Record `real_path` as tombstoned within the overlay. Idempotent."""
    key = _tombstone_key(real_path)
    if key is None:
        return
    f = _tombstone_file(overlay_root)
    f.parent.mkdir(parents=True, exist_ok=True)
    existing = set()
    if f.exists():
        try:
            existing = set(f.read_text(encoding="utf-8").splitlines())
        except OSError:
            existing = set()
    if key not in existing:
        existing.add(key)
        f.write_text("\n".join(sorted(existing)) + "\n", encoding="utf-8")


def is_tombstoned(overlay_root: Path, real_path: Path) -> bool:
    """True iff `real_path` has been tombstoned in this overlay."""
    key = _tombstone_key(real_path)
    if key is None:
        return False
    f = _tombstone_file(overlay_root)
    if not f.exists():
        return False
    try:
        lines = set(f.read_text(encoding="utf-8").splitlines())
    except OSError:
        return False
    return key in lines


# ── Read / list resolution ─────────────────────────────────────────────────

def resolve_read_with_overlay(real_path: Path, overlay_root: Path) -> Path:
    """Return the path that should actually be read.

    - If the real path has been tombstoned within this overlay, raise
      FileNotFoundError (the caller surfaces it to the agent as a real miss).
    - If an overlay twin exists, return it.
    - Otherwise return the real path.
    """
    if is_tombstoned(overlay_root, real_path):
        raise FileNotFoundError(str(real_path))
    twin = overlay_path_for(real_path, overlay_root)
    if twin is not None and twin.exists():
        return twin
    return real_path


def list_merged(real_dir: Path, overlay_root: Path) -> list[str]:
    """Return a sorted list of entry names under `real_dir`, merging overlay
    entries and honouring tombstones.

    The return shape matches `os.listdir(real_dir)` (names only, no paths)
    so handlers can substitute it in directly.
    """
    seen: set[str] = set()
    tombstoned_names = _tombstoned_names_in(real_dir, overlay_root)
    if real_dir.exists():
        for name in os.listdir(real_dir):
            if name in tombstoned_names:
                continue
            seen.add(name)
    twin_dir = overlay_path_for(real_dir, overlay_root)
    if twin_dir is not None and twin_dir.exists():
        for name in os.listdir(twin_dir):
            if name in tombstoned_names:
                continue
            seen.add(name)
    return sorted(seen)


def _tombstoned_names_in(real_dir: Path, overlay_root: Path) -> set[str]:
    """Names within `real_dir` that have been tombstoned.

    Walks the tombstone set once; callers invoking this repeatedly in a tight
    loop should cache the result.
    """
    names: set[str] = set()
    f = _tombstone_file(overlay_root)
    if not f.exists():
        return names
    try:
        lines = f.read_text(encoding="utf-8").splitlines()
    except OSError:
        return names
    # Compute the prefix/relpath for real_dir to find children under it.
    # For a root-level dir (rel == Path(".")), as_posix() returns "." which
    # corrupts the prefix — strip it explicitly.
    name_rel = classify_root(real_dir)
    if name_rel[0] is None:
        return names
    rel_str = name_rel[1].as_posix()
    if rel_str == ".":
        rel_str = ""
    dir_key = f"{name_rel[0]}/{rel_str}".rstrip("/")
    prefix = f"{dir_key}/"
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if not line.startswith(prefix):
            continue
        tail = line[len(prefix):]
        if "/" in tail:
            # tombstone applies to a descendant, not a direct child
            continue
        names.add(tail)
    return names


# ── Write support ──────────────────────────────────────────────────────────

def prepare_write(real_path: Path, overlay_root: Path) -> Path | None:
    """Compute the overlay target for `real_path` and mkdir its parent.

    Returns the overlay target path, or None if the real path is not under
    any writable root (caller should surface that as an error like the
    real handler would).
    """
    twin = overlay_path_for(real_path, overlay_root)
    if twin is None:
        return None
    ensure_parent(twin)
    return twin
