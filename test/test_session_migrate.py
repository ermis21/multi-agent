"""Tests for the flat-to-folder session migration."""

import json

from app.sessions.migrate import migrate_flat_to_folders


def _write(p, text=""):
    p.write_text(text, encoding="utf-8")


def test_migration_moves_all_sidecars(tmp_path):
    sid = "20260101_120000_abcdef"
    _write(tmp_path / f"{sid}.jsonl", '{"ok":true}\n')
    _write(tmp_path / f"{sid}.state.json", json.dumps({
        "session_id": sid, "history": {"full": f"sessions/{sid}.jsonl"}
    }))
    _write(tmp_path / f"{sid}.approvals.jsonl", '{"status":"approved"}\n')
    _write(tmp_path / f"{sid}.tool_errors.jsonl", '{"error":"x"}\n')
    _write(tmp_path / f"{sid}.active.jsonl", '{"role":"final"}\n')

    summary = migrate_flat_to_folders(tmp_path)

    assert summary["moved"] == 5
    assert summary["sessions"] == 1
    assert summary["skipped_existing"] == 0

    sdir = tmp_path / sid
    assert (sdir / "turns.jsonl").exists()
    assert (sdir / "state.json").exists()
    assert (sdir / "approvals.jsonl").exists()
    assert (sdir / "tool_errors.jsonl").exists()
    assert (sdir / "active.jsonl").exists()

    # All flat files gone
    assert not any(p.is_file() for p in tmp_path.iterdir())


def test_migration_rewrites_history_full_pointer(tmp_path):
    sid = "sid-rewrite"
    _write(tmp_path / f"{sid}.jsonl", "")
    _write(tmp_path / f"{sid}.state.json", json.dumps({
        "session_id": sid, "history": {"full": f"sessions/{sid}.jsonl"}
    }))

    migrate_flat_to_folders(tmp_path)

    data = json.loads((tmp_path / sid / "state.json").read_text())
    assert data["history"]["full"] == f"sessions/{sid}/turns.jsonl"


def test_migration_idempotent(tmp_path):
    sid = "sid-idem"
    _write(tmp_path / f"{sid}.jsonl", '{"a":1}\n')

    summary1 = migrate_flat_to_folders(tmp_path)
    assert summary1["moved"] == 1

    summary2 = migrate_flat_to_folders(tmp_path)
    assert summary2["moved"] == 0
    assert summary2["sessions"] == 0


def test_migration_skips_folders(tmp_path):
    """Already-migrated session folders must not be touched."""
    sid = "sid-already"
    sdir = tmp_path / sid
    sdir.mkdir()
    (sdir / "turns.jsonl").write_text('{"ok":true}\n')

    summary = migrate_flat_to_folders(tmp_path)

    assert summary["moved"] == 0
    assert (sdir / "turns.jsonl").read_text() == '{"ok":true}\n'


def test_migration_preserves_existing_file_when_flat_exists(tmp_path):
    """If a folder and a flat file both exist for the same sid, leave the flat file alone."""
    sid = "sid-conflict"
    sdir = tmp_path / sid
    sdir.mkdir()
    (sdir / "turns.jsonl").write_text("new")
    _write(tmp_path / f"{sid}.jsonl", "old")

    summary = migrate_flat_to_folders(tmp_path)

    # The new file is untouched
    assert (sdir / "turns.jsonl").read_text() == "new"
    # The flat file is left in place (skipped)
    assert (tmp_path / f"{sid}.jsonl").read_text() == "old"
    assert summary["skipped_existing"] == 1


def test_migration_ignores_non_session_files(tmp_path):
    """Hidden files, README, and unrelated content are not touched."""
    (tmp_path / ".gitkeep").write_text("")
    _write(tmp_path / "README.md", "docs")

    summary = migrate_flat_to_folders(tmp_path)

    assert summary["moved"] == 0
    assert (tmp_path / ".gitkeep").exists()
    assert (tmp_path / "README.md").exists()


def test_new_session_goes_directly_to_folder(tmp_path, monkeypatch):
    """A freshly-created SessionLogger writes to the new layout."""
    import app.sessions.logger as slog

    monkeypatch.setattr(slog, "SESSIONS_DIR", tmp_path)

    sid = "sid-fresh"
    logger = slog.SessionLogger(sid)
    logger.log_turn(
        attempt=0, role="final",
        messages=[{"role": "user", "content": "hi"}],
        response="hello",
    )

    assert (tmp_path / sid / "turns.jsonl").exists()
    assert not (tmp_path / f"{sid}.jsonl").exists()


def test_list_sessions_reads_new_layout(tmp_path, monkeypatch):
    import app.sessions.logger as slog

    monkeypatch.setattr(slog, "SESSIONS_DIR", tmp_path)

    (tmp_path / "sid-a").mkdir()
    (tmp_path / "sid-a" / "turns.jsonl").write_text('{"timestamp":"t1"}\n')
    (tmp_path / "sid-b").mkdir()
    (tmp_path / "sid-b" / "turns.jsonl").write_text('{"timestamp":"t2"}\n{"timestamp":"t3"}\n')

    out = slog.list_sessions()
    sids = {s["session_id"] for s in out}
    assert sids == {"sid-a", "sid-b"}
    by_sid = {s["session_id"]: s for s in out}
    assert by_sid["sid-a"]["turns"] == 1
    assert by_sid["sid-b"]["turns"] == 2
