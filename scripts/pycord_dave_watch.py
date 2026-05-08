#!/usr/bin/env python3
"""pycord DAVE landing watcher (local cron job).

Polls PyPI for the newest py-cord wheel (stable + pre-release), compares
against the last-checked version, and if there's a new one, greps the wheel
contents for any of the markers that would indicate DAVE/MLS/E2EE support
has shipped. Writes the result to STATE_FILE so the user (or a follow-up
agent) can act on it. Also posts to the worker bot's Discord channel via
the existing /discord/send HTTP gateway when something interesting changes.

Designed to run from cron — exits 0 even when nothing's new so cron doesn't
spam mail. All exceptions land in STATE_FILE under `last_error`.

Cron entry (every 3 days at 00:00 UTC):
  0 0 */3 * * /usr/bin/python3 /home/homelab/Phoebe/scripts/pycord_dave_watch.py >>/home/homelab/Phoebe/state/pycord_dave_watch.log 2>&1
"""

from __future__ import annotations

import io
import json
import os
import sys
import urllib.request
import urllib.error
import zipfile
from datetime import datetime, timezone
from pathlib import Path

STATE_FILE = Path(os.environ.get(
    "PYCORD_WATCH_STATE",
    "/home/homelab/Phoebe/state/pycord_dave_watch.json",
))

# Markers we look for in the wheel source. All case-insensitive.
DAVE_MARKERS = ("dave", "mls", "davey", "e2ee", "libdave")

# Notification: optional, off by default. Set to a channel ID and the bot
# gateway URL; the script will POST a message to /discord/send when DAVE
# lands. Both required to send.
NOTIFY_CHANNEL_ID = os.environ.get("PYCORD_WATCH_NOTIFY_CHANNEL", "")
NOTIFY_GATEWAY    = os.environ.get("PYCORD_WATCH_NOTIFY_GATEWAY", "http://localhost:4000")

PYPI_JSON = "https://pypi.org/pypi/py-cord/json"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_state() -> dict:
    if not STATE_FILE.exists():
        return {}
    try:
        return json.loads(STATE_FILE.read_text())
    except Exception:
        return {}


def save_state(d: dict) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE_FILE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(d, indent=2, ensure_ascii=False))
    tmp.replace(STATE_FILE)


def fetch_pypi_versions() -> tuple[str, list[str]]:
    """Return (latest_version_including_prereleases, all_versions)."""
    with urllib.request.urlopen(PYPI_JSON, timeout=30) as r:
        data = json.loads(r.read().decode("utf-8"))
    releases = data.get("releases") or {}
    # Filter to versions that have at least one wheel/sdist not yanked.
    valid = [
        v for v, files in releases.items()
        if files and not all(f.get("yanked") for f in files)
    ]
    # PyPI returns versions in arbitrary order; sort using packaging if
    # available, otherwise lexicographic which is wrong for some bumps.
    try:
        from packaging.version import Version
        valid.sort(key=Version)
    except ImportError:
        valid.sort()
    latest = valid[-1] if valid else ""
    return latest, valid


def download_and_grep(version: str) -> dict:
    """Download the wheel for `version` and return a dict with marker hits.
    Shape: {"hits": {marker: [path1, path2, ...]}, "wheel_size": int}
    """
    with urllib.request.urlopen(f"https://pypi.org/pypi/py-cord/{version}/json", timeout=30) as r:
        meta = json.loads(r.read().decode("utf-8"))
    files = meta.get("urls") or []
    wheel = next(
        (f for f in files if f.get("packagetype") == "bdist_wheel" and f.get("url", "").endswith(".whl")),
        None,
    )
    if not wheel:
        return {"hits": {}, "wheel_size": 0, "error": f"no wheel found for {version}"}
    with urllib.request.urlopen(wheel["url"], timeout=120) as r:
        data = r.read()
    hits: dict[str, list[str]] = {m: [] for m in DAVE_MARKERS}
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        for name in z.namelist():
            if not name.endswith(".py") and not name.endswith(".pyi"):
                continue
            try:
                src = z.read(name).decode("utf-8", errors="replace").lower()
            except Exception:
                continue
            for m in DAVE_MARKERS:
                if m in src:
                    hits[m].append(name)
    # Strip empty
    hits = {m: paths for m, paths in hits.items() if paths}
    return {"hits": hits, "wheel_size": len(data)}


def notify(text: str) -> None:
    """Best-effort POST to the worker bot's gateway. No-op when not configured."""
    if not NOTIFY_CHANNEL_ID or not NOTIFY_GATEWAY:
        return
    try:
        body = json.dumps({
            "channel_id": int(NOTIFY_CHANNEL_ID),
            "content":    text,
            "bot":        "worker",
        }).encode("utf-8")
        req = urllib.request.Request(
            f"{NOTIFY_GATEWAY}/discord/send",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as r:
            r.read()
    except Exception as e:
        print(f"[pycord_dave_watch] notify failed: {e}", flush=True)


def main() -> int:
    state = load_state()
    state["last_check_ts"] = now_iso()
    state.pop("last_error", None)

    try:
        latest, _ = fetch_pypi_versions()
    except Exception as e:
        state["last_error"] = f"pypi fetch failed: {e}"
        save_state(state)
        print(f"[pycord_dave_watch] {state['last_error']}", flush=True)
        return 0

    state["latest_pypi_version"] = latest
    last_seen = state.get("last_seen_version", "")
    print(f"[pycord_dave_watch] latest_pypi={latest!r} last_seen={last_seen!r}", flush=True)

    if latest and latest != last_seen:
        try:
            grep = download_and_grep(latest)
        except Exception as e:
            state["last_error"] = f"download/grep {latest} failed: {e}"
            save_state(state)
            print(f"[pycord_dave_watch] {state['last_error']}", flush=True)
            return 0
        state["last_seen_version"] = latest
        state["last_grep"] = grep
        if grep.get("hits"):
            summary = ", ".join(grep["hits"].keys())
            state["dave_landed"] = True
            state["dave_landed_version"] = latest
            state["dave_landed_at"] = now_iso()
            msg = (
                f"🚨 py-cord {latest} contains DAVE markers ({summary}). "
                f"Bump discord/pycord/requirements.txt and integrate. "
                f"Details: cat {STATE_FILE}"
            )
            print(f"[pycord_dave_watch] {msg}", flush=True)
            notify(msg)
        else:
            print(f"[pycord_dave_watch] new version {latest} but no DAVE markers yet", flush=True)
    else:
        print(f"[pycord_dave_watch] no new version", flush=True)

    save_state(state)
    return 0


if __name__ == "__main__":
    sys.exit(main())
