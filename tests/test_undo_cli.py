"""Tests for ai_pdf_renamer.undo_cli — undo rename-log operations."""

from __future__ import annotations

import pytest

from ai_pdf_renamer.undo_cli import main, run_undo


def test_undo_no_log_file(tmp_path, capsys) -> None:
    """When the rename log file does not exist, main() prints an error and exits."""
    missing = tmp_path / "nonexistent.log"
    with pytest.raises(SystemExit) as exc:
        main(["--rename-log", str(missing)])
    assert exc.value.code == 1
    captured = capsys.readouterr()
    assert "not found" in captured.err


def test_undo_empty_log(tmp_path, capsys) -> None:
    """An empty log file should produce 'No entries' and no crash."""
    log = tmp_path / "rename.log"
    log.write_text("", encoding="utf-8")
    main(["--rename-log", str(log)])
    captured = capsys.readouterr()
    assert "No entries" in captured.err


def test_undo_single_entry(tmp_path, capsys) -> None:
    """A single log entry reverts the renamed file back to its original name."""
    old = tmp_path / "original.pdf"
    new = tmp_path / "renamed.pdf"
    new.write_text("content", encoding="utf-8")

    log = tmp_path / "rename.log"
    log.write_text(f"{old}\t{new}\n", encoding="utf-8")

    main(["--rename-log", str(log)])

    assert old.exists()
    assert not new.exists()
    captured = capsys.readouterr()
    assert "Reverted" in captured.out


def test_undo_multiple_entries_lifo(tmp_path, capsys) -> None:
    """Three entries are undone in LIFO order (last entry reverted first)."""
    # Simulate three sequential renames:
    #   a -> b, b -> c, c -> d
    # After all renames only 'd' exists on disk.
    # LIFO undo should revert d->c, then c->b, then b->a.
    a = tmp_path / "a.pdf"
    b = tmp_path / "b.pdf"
    c = tmp_path / "c.pdf"
    d = tmp_path / "d.pdf"

    # Only the final file exists on disk.
    d.write_text("content", encoding="utf-8")

    log = tmp_path / "rename.log"
    log.write_text(
        f"{a}\t{b}\n{b}\t{c}\n{c}\t{d}\n",
        encoding="utf-8",
    )

    main(["--rename-log", str(log)])

    assert a.exists()
    assert not b.exists()
    assert not c.exists()
    assert not d.exists()

    captured = capsys.readouterr()
    lines = [ln for ln in captured.out.splitlines() if "Reverted" in ln]
    assert len(lines) == 3
    # First revert printed should be d -> c (last log entry)
    assert "d.pdf" in lines[0] and "c.pdf" in lines[0]


def test_undo_missing_target(tmp_path, capsys) -> None:
    """When the renamed file no longer exists on disk, skip gracefully."""
    old = tmp_path / "original.pdf"
    new = tmp_path / "gone.pdf"
    # Do NOT create 'new' — it's missing.

    log = tmp_path / "rename.log"
    log.write_text(f"{old}\t{new}\n", encoding="utf-8")

    main(["--rename-log", str(log)])

    assert not old.exists()
    assert not new.exists()
    captured = capsys.readouterr()
    assert "Skip (new path missing)" in captured.err


def test_undo_dry_run(tmp_path, capsys) -> None:
    """With --dry-run no actual file operations happen."""
    old = tmp_path / "original.pdf"
    new = tmp_path / "renamed.pdf"
    new.write_text("content", encoding="utf-8")

    log = tmp_path / "rename.log"
    log.write_text(f"{old}\t{new}\n", encoding="utf-8")

    main(["--rename-log", str(log), "--dry-run"])

    # File should NOT have been moved.
    assert new.exists()
    assert not old.exists()
    captured = capsys.readouterr()
    assert "Would revert" in captured.out


def test_undo_source_already_exists(tmp_path, capsys) -> None:
    """When the original location already has a file, skip to avoid collision."""
    old = tmp_path / "original.pdf"
    new = tmp_path / "renamed.pdf"
    old.write_text("old content", encoding="utf-8")
    new.write_text("new content", encoding="utf-8")

    log = tmp_path / "rename.log"
    log.write_text(f"{old}\t{new}\n", encoding="utf-8")

    main(["--rename-log", str(log)])

    # Both files should still exist; nothing was overwritten.
    assert old.exists()
    assert new.exists()
    assert old.read_text(encoding="utf-8") == "old content"
    captured = capsys.readouterr()
    assert "Skip (old path already exists)" in captured.err


def test_undo_whitespace_only_lines_ignored(tmp_path, capsys) -> None:
    """Lines that are blank or whitespace-only are silently skipped."""
    old = tmp_path / "a.pdf"
    new = tmp_path / "b.pdf"
    new.write_text("data", encoding="utf-8")

    log = tmp_path / "rename.log"
    log.write_text(f"\n  \n{old}\t{new}\n\n", encoding="utf-8")

    main(["--rename-log", str(log)])

    assert old.exists()
    assert not new.exists()


def test_undo_malformed_line_skipped(tmp_path, capsys) -> None:
    """Lines without a tab separator are silently skipped."""
    old = tmp_path / "a.pdf"
    new = tmp_path / "b.pdf"
    new.write_text("data", encoding="utf-8")

    log = tmp_path / "rename.log"
    log.write_text(f"no-tab-here\n{old}\t{new}\n", encoding="utf-8")

    main(["--rename-log", str(log)])

    assert old.exists()
    assert not new.exists()


def test_undo_log_is_directory(tmp_path, capsys) -> None:
    """When --rename-log points to a directory, exit with error."""
    d = tmp_path / "adir"
    d.mkdir()
    with pytest.raises(SystemExit) as exc:
        main(["--rename-log", str(d)])
    assert exc.value.code == 1
    captured = capsys.readouterr()
    assert "not a file" in captured.err


def test_run_undo_on_directory_path(tmp_path, capsys) -> None:
    """run_undo() called directly with a directory path prints error."""
    d = tmp_path / "adir"
    d.mkdir()
    run_undo(d, dry_run=False)
    captured = capsys.readouterr()
    assert "not a file" in captured.err


def test_undo_rejects_cross_directory_entries(tmp_path, capsys) -> None:
    """Undo entries must stay within the same parent directory."""
    original_dir = tmp_path / "original"
    renamed_dir = tmp_path / "renamed"
    original_dir.mkdir()
    renamed_dir.mkdir()

    old = original_dir / "document.pdf"
    new = renamed_dir / "document-renamed.pdf"
    new.write_text("content", encoding="utf-8")

    log = tmp_path / "rename.log"
    log.write_text(f"{old}\t{new}\n", encoding="utf-8")

    main(["--rename-log", str(log)])

    assert not old.exists()
    assert new.exists()
    captured = capsys.readouterr()
    assert "cross-directory undo denied" in captured.err


def test_undo_rejects_symlink_escape_within_trusted_root(tmp_path, capsys) -> None:
    """Symlinked paths that resolve outside the trusted root are rejected."""
    outside = tmp_path.parent / f"{tmp_path.name}-outside"
    outside.mkdir(exist_ok=True)
    link = tmp_path / "linked"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("Symlinks are not available in this environment")

    old = tmp_path / "restored.pdf"
    escaped_target = link / "escaped.pdf"
    (outside / "escaped.pdf").write_text("content", encoding="utf-8")

    log = tmp_path / "rename.log"
    log.write_text(f"{old}\t{escaped_target}\n", encoding="utf-8")

    main(["--rename-log", str(log)])

    assert not old.exists()
    assert (outside / "escaped.pdf").exists()
    captured = capsys.readouterr()
    assert "path traversal detected" in captured.err
