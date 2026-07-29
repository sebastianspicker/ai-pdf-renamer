"""Tests for rename and filename-sanitization helpers."""

from __future__ import annotations

import errno
import os
import stat
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pytest

import folionym.rename_ops as rename_ops
from folionym.rename_ops import (
    MAX_RENAME_RETRIES,
    RenameApplyOptions,
    _copy_file_to_fd,
    _validate_path_within_parent,
    is_path_within,
    sanitize_filename_base,
    sanitize_filename_from_llm,
)
from tests.conftest import rename_pdf


def test_sanitize_filename_from_llm_empty() -> None:
    assert sanitize_filename_from_llm("") == "document"
    assert sanitize_filename_from_llm("   ") == "document"


def test_sanitize_filename_from_llm_unsafe_chars() -> None:
    assert sanitize_filename_from_llm('a/b\\c:d*e?f"g<h>i|j') == "a_b_c_d_e_f_g_h_i_j"


def test_sanitize_filename_from_llm_strip_pdf() -> None:
    assert sanitize_filename_from_llm("INVOICE_AMAZON.PDF") == "INVOICE_AMAZON"
    assert sanitize_filename_from_llm("doc.pdf") == "doc"


def test_sanitize_filename_from_llm_newlines_and_spaces() -> None:
    assert sanitize_filename_from_llm("a  b\nc\rd") == "a_b_c_d"


def test_sanitize_filename_from_llm_length() -> None:
    long_str = "a" * 150
    result = sanitize_filename_from_llm(long_str)
    assert len(result) == 120


def test_sanitize_filename_from_llm_strip_leading_trailing_dots_underscores() -> None:
    assert sanitize_filename_from_llm("._only_._") == "only"
    assert sanitize_filename_from_llm("._.") == "document"


def test_sanitize_filename_base_unchanged() -> None:
    assert sanitize_filename_base("INVOICE_AMAZON_2023") == "INVOICE_AMAZON_2023"


def test_sanitize_filename_base_empty() -> None:
    assert sanitize_filename_base("") == "unnamed"
    assert sanitize_filename_base("   ") == "unnamed"


def test_apply_single_rename_backup_collision_does_not_overwrite(tmp_path: Path) -> None:
    src = tmp_path / "doc.pdf"
    src.write_text("new-content", encoding="utf-8")

    backup_dir = tmp_path / "backups"
    backup_dir.mkdir()
    existing_backup = backup_dir / "doc.pdf"
    existing_backup.write_text("old-content", encoding="utf-8")

    success, target = rename_pdf(
        src,
        "renamed",
        backup_dir=backup_dir,
    )

    assert success is True
    assert target.exists()
    assert existing_backup.read_text(encoding="utf-8") == "old-content"
    assert (backup_dir / "doc_1.pdf").read_text(encoding="utf-8") == "new-content"


@pytest.mark.skipif(os.name == "nt", reason="POSIX permissions and symlinks required")
def test_backup_rejects_dangling_symlink_candidate_without_writing_through_it(tmp_path: Path) -> None:
    src = tmp_path / "doc.pdf"
    src.write_text("source-content", encoding="utf-8")
    backup_dir = tmp_path / "backups"
    backup_dir.mkdir()
    (backup_dir / "doc.pdf").write_text("existing-backup", encoding="utf-8")
    outside_target = tmp_path / "escaped.pdf"
    dangling_candidate = backup_dir / "doc_1.pdf"
    dangling_candidate.symlink_to(outside_target)

    ok, _ = rename_pdf(src, "renamed", backup_dir=backup_dir)

    assert ok is True
    assert not outside_target.exists()
    assert dangling_candidate.is_symlink()
    assert (backup_dir / "doc_2.pdf").read_text(encoding="utf-8") == "source-content"


@pytest.mark.skipif(os.name == "nt", reason="POSIX permissions and symlinks required")
def test_backup_target_swap_preserves_source_and_victim(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    src = tmp_path / "doc.pdf"
    src.write_text("source-content", encoding="utf-8")
    backup_dir = tmp_path / "backups"
    victim = tmp_path / "victim.txt"
    victim.write_text("unchanged", encoding="utf-8")
    backup_target = backup_dir / "doc.pdf"
    original_copy = _copy_file_to_fd

    def _copy_then_swap(source: Path, target_fd: int) -> os.stat_result:
        source_status = original_copy(source, target_fd)
        backup_target.unlink()
        backup_target.symlink_to(victim)
        return source_status

    monkeypatch.setattr(rename_ops, "_copy_file_to_fd", _copy_then_swap)

    with pytest.raises(OSError, match="Backup path changed"):
        rename_pdf(src, "renamed", backup_dir=backup_dir)

    assert src.read_text(encoding="utf-8") == "source-content"
    assert victim.read_text(encoding="utf-8") == "unchanged"
    assert backup_target.is_symlink()


@pytest.mark.skipif(os.name == "nt", reason="POSIX permissions required")
def test_backup_directory_and_file_are_owner_only(tmp_path: Path) -> None:
    src = tmp_path / "doc.pdf"
    src.write_text("source-content", encoding="utf-8")
    backup_dir = tmp_path / "backups"

    ok, _ = rename_pdf(src, "renamed", backup_dir=backup_dir)

    assert ok is True
    assert stat.S_IMODE(backup_dir.stat().st_mode) == 0o700
    assert stat.S_IMODE((backup_dir / "doc.pdf").stat().st_mode) == 0o600


def test_backup_is_created_once_when_all_rename_targets_collide(tmp_path: Path) -> None:
    src = tmp_path / "doc.pdf"
    src.write_text("source-content", encoding="utf-8")
    backup_dir = tmp_path / "backups"
    for counter in range(MAX_RENAME_RETRIES + 1):
        name = "renamed.pdf" if counter == 0 else f"renamed_{counter}.pdf"
        (tmp_path / name).write_text("occupied", encoding="utf-8")

    ok, _ = rename_pdf(src, "renamed", backup_dir=backup_dir)

    assert ok is False
    assert sorted(path.name for path in backup_dir.iterdir()) == ["doc.pdf"]
    assert (backup_dir / "doc.pdf").read_text(encoding="utf-8") == "source-content"


def test_backup_is_created_once_when_rename_succeeds_after_collision(tmp_path: Path) -> None:
    src = tmp_path / "doc.pdf"
    src.write_text("source-content", encoding="utf-8")
    backup_dir = tmp_path / "backups"
    (tmp_path / "renamed.pdf").write_text("occupied", encoding="utf-8")

    ok, target = rename_pdf(src, "renamed", backup_dir=backup_dir)

    assert ok is True
    assert target.name == "renamed_1.pdf"
    assert sorted(path.name for path in backup_dir.iterdir()) == ["doc.pdf"]


def test_exact_target_collision_does_not_select_a_suffix(tmp_path: Path) -> None:
    src = tmp_path / "doc.pdf"
    src.write_text("source-content", encoding="utf-8")
    occupied = tmp_path / "renamed.pdf"
    occupied.write_text("occupied", encoding="utf-8")

    ok, target = rename_ops.apply_single_rename(
        src,
        "renamed",
        RenameApplyOptions(dry_run=False, exact_target=True),
    )

    assert ok is False
    assert target == occupied
    assert src.read_text(encoding="utf-8") == "source-content"
    assert not (tmp_path / "renamed_1.pdf").exists()


def _write_concurrent_sources(tmp_path: Path, count: int = 5) -> tuple[list[Path], list[str]]:
    sources: list[Path] = []
    contents: list[str] = []
    for i in range(count):
        src = tmp_path / f"src_{i}.pdf"
        content = f"distinct-content-{i}"
        src.write_text(content, encoding="utf-8")
        sources.append(src)
        contents.append(content)
    return (sources, contents)


def _rename_to_report(src: Path) -> tuple[bool, Path]:
    return rename_pdf(src, "report")


def _run_concurrent_report_renames(sources: list[Path]) -> list[tuple[bool, Path]]:
    results: list[tuple[bool, Path]] = []
    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = {pool.submit(_rename_to_report, source): source for source in sources}
        for future in as_completed(futures):
            results.append(future.result())
    return results


def _assert_unique_rename_results(results: list[tuple[bool, Path]], contents: list[str]) -> None:
    assert all(ok for ok, _ in results)
    target_paths = [target for _, target in results]
    assert len(set(target_paths)) == len(target_paths)
    for _, target in results:
        assert target.exists(), f"Expected target {target} to exist"
    target_contents = {target: target.read_text(encoding="utf-8") for _, target in results}
    for content in contents:
        assert content in target_contents.values(), f"Content {content!r} missing from targets"


def test_concurrent_renames_same_target_no_overwrite(tmp_path: Path) -> None:
    """Five concurrent renames to the same base name must all succeed with unique targets."""
    sources, contents = _write_concurrent_sources(tmp_path)
    # Pre-create the target so every rename collides with it *and* with each other.
    existing_target = tmp_path / "report.pdf"
    existing_target.write_text("pre-existing", encoding="utf-8")

    _assert_unique_rename_results(_run_concurrent_report_renames(sources), contents)
    # The pre-existing file must not have been overwritten.
    assert existing_target.read_text(encoding="utf-8") == "pre-existing"


def test_rename_collision_suffix_increments(tmp_path: Path) -> None:
    """Collision suffixes increment: first collision gets _1, second gets _2."""
    # Create existing target.
    existing = tmp_path / "invoice.pdf"
    existing.write_text("original", encoding="utf-8")

    # First rename should land on invoice_1.pdf.
    src1 = tmp_path / "a.pdf"
    src1.write_text("first-rename", encoding="utf-8")
    ok1, target1 = rename_pdf(src1, "invoice")
    assert ok1 is True
    assert target1.name == "invoice_1.pdf"
    assert target1.read_text(encoding="utf-8") == "first-rename"

    # Second rename should land on invoice_2.pdf.
    src2 = tmp_path / "b.pdf"
    src2.write_text("second-rename", encoding="utf-8")
    ok2, target2 = rename_pdf(src2, "invoice")
    assert ok2 is True
    assert target2.name == "invoice_2.pdf"
    assert target2.read_text(encoding="utf-8") == "second-rename"

    # Original must remain untouched.
    assert existing.read_text(encoding="utf-8") == "original"


def test_rename_dry_run_no_filesystem_change(tmp_path: Path) -> None:
    """dry_run=True must not create, rename, or remove any files."""
    src = tmp_path / "original.pdf"
    src.write_text("keep-me", encoding="utf-8")

    files_before = sorted(tmp_path.iterdir())

    ok, _target = rename_pdf(
        src,
        "new_name",
        dry_run=True,
    )

    assert ok is True
    # Source must still exist with original content.
    assert src.exists()
    assert src.read_text(encoding="utf-8") == "keep-me"
    # No new files should appear.
    files_after = sorted(tmp_path.iterdir())
    assert files_before == files_after


def test_rename_dry_run_uses_same_collision_suffix_as_apply(tmp_path: Path) -> None:
    """A preview must not offer a target already occupied on disk."""
    src = tmp_path / "original.pdf"
    src.write_text("keep-me", encoding="utf-8")
    (tmp_path / "new_name.pdf").write_text("existing", encoding="utf-8")

    preview_ok, preview_target = rename_pdf(src, "new_name", dry_run=True)
    apply_ok, apply_target = rename_pdf(src, "new_name")

    assert preview_ok is True
    assert preview_target.name == "new_name_1.pdf"
    assert apply_ok is True
    assert apply_target == preview_target


def test_rename_plan_uses_same_collision_suffix_as_apply(tmp_path: Path) -> None:
    src = tmp_path / "original.pdf"
    src.write_text("keep-me", encoding="utf-8")
    (tmp_path / "new_name.pdf").write_text("existing", encoding="utf-8")
    plan_entries: list[dict[str, str]] = []

    preview_ok, preview_target = rename_pdf(
        src,
        "new_name",
        dry_run=True,
        plan_file_path=tmp_path / "plan.json",
        plan_entries=plan_entries,
    )
    apply_ok, apply_target = rename_pdf(src, "new_name")

    assert preview_ok is True
    assert preview_target.name == "new_name_1.pdf"
    assert plan_entries == [{"old": str(src), "new": str(preview_target)}]
    assert apply_ok is True
    assert apply_target == preview_target


def test_rename_dry_run_treats_dangling_symlink_as_collision(tmp_path: Path) -> None:
    src = tmp_path / "original.pdf"
    src.write_text("keep-me", encoding="utf-8")
    dangling_target = tmp_path / "new_name.pdf"
    try:
        dangling_target.symlink_to(tmp_path / "missing.pdf")
    except (
        NotImplementedError,
        OSError,
    ):
        pytest.skip("symlinks are unavailable on this platform")

    preview_ok, preview_target = rename_pdf(src, "new_name", dry_run=True)
    apply_ok, apply_target = rename_pdf(src, "new_name")

    assert preview_ok is True
    assert preview_target.name == "new_name_1.pdf"
    assert apply_ok is True
    assert apply_target == preview_target


def test_rename_callback_failure_does_not_mark_completed_rename_as_failed(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Post-rename failures are observable but cannot undo a completed rename."""
    src = tmp_path / "original.pdf"
    src.write_text("keep-me", encoding="utf-8")

    def failing_callback(*_args: object) -> None:
        raise RuntimeError("post-rename hook failed")

    with caplog.at_level("ERROR", logger="folionym.rename_ops"):
        ok, target = rename_pdf(src, "new_name", on_success=failing_callback)

    assert ok is True
    assert target.exists()
    assert not src.exists()
    assert "Rename completed but its post-rename callback failed" in caplog.text
    assert "post-rename hook failed" in caplog.text


def test_path_traversal_blocked(tmp_path: Path) -> None:
    """A base name containing '..' must be rejected before the rename occurs.

    On Python 3.12+, ``Path.with_name()`` itself raises ``ValueError`` for
    names containing path separators (``/``), which blocks traversal before
    ``_validate_path_within_parent`` is even reached.  On older Pythons the
    custom validator catches it.  Either way, a ``ValueError`` must be raised
    and no files should be created outside ``tmp_path``.
    """
    src = tmp_path / "legit.pdf"
    src.write_text("payload", encoding="utf-8")

    parent_dir = tmp_path.parent
    files_in_parent_before = set(parent_dir.iterdir())

    with pytest.raises(ValueError):
        rename_pdf(src, "../escape")

    # Source must still be intact.
    assert src.exists()
    assert src.read_text(encoding="utf-8") == "payload"

    # No file must have been created outside tmp_path.
    files_in_parent_after = set(parent_dir.iterdir())
    assert files_in_parent_before == files_in_parent_after


# ---------------------------------------------------------------------------
# Cross-filesystem (EXDEV) handling
# ---------------------------------------------------------------------------


def test_apply_single_rename_exdev_copy_unlink(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """When os.link raises EXDEV and os.rename also raises EXDEV, shutil.copy2 + unlink fallback is used."""
    src = tmp_path / "doc.pdf"
    src.write_text("cross-fs-content", encoding="utf-8")

    def _link_exdev(src: object, dst: object) -> None:
        raise OSError(errno.EXDEV, "Invalid cross-device link")

    def _rename_exdev(src: object, dst: object) -> None:
        raise OSError(errno.EXDEV, "Invalid cross-device link")

    monkeypatch.setattr(os, "link", _link_exdev)
    monkeypatch.setattr(os, "rename", _rename_exdev)

    ok, target = rename_pdf(src, "moved")

    assert ok is True
    assert target.name == "moved.pdf"
    assert target.read_text(encoding="utf-8") == "cross-fs-content"
    assert not src.exists(), "Source should be removed after cross-fs fallback"


def test_apply_single_rename_exdev_target_swap_preserves_source_and_victim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    src = tmp_path / "doc.pdf"
    src.write_text("cross-fs-content", encoding="utf-8")
    victim = tmp_path / "victim.txt"
    victim.write_text("unchanged", encoding="utf-8")
    target = tmp_path / "moved.pdf"

    def _link_exdev(source: object, destination: object) -> None:
        raise OSError(errno.EXDEV, "Invalid cross-device link")

    original_copy = _copy_file_to_fd

    def _copy_then_swap(source: Path, target_fd: int) -> os.stat_result:
        source_status = original_copy(source, target_fd)
        target.unlink()
        target.symlink_to(victim)
        return source_status

    monkeypatch.setattr(os, "link", _link_exdev)
    monkeypatch.setattr(rename_ops, "_copy_file_to_fd", _copy_then_swap)

    with pytest.raises(OSError, match="Target path changed"):
        rename_pdf(src, "moved")

    assert src.read_text(encoding="utf-8") == "cross-fs-content"
    assert victim.read_text(encoding="utf-8") == "unchanged"
    assert target.is_symlink()


def test_apply_single_rename_reservation_open_error_propagates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    src = tmp_path / "doc.pdf"
    src.write_text("content", encoding="utf-8")
    original_open = os.open

    def _link_exdev(source: object, destination: object) -> None:
        raise OSError(errno.EXDEV, "Invalid cross-device link")

    def _open_eacces(path: str | bytes | os.PathLike[str], flags: int, mode: int = 0o777, **kwargs: object) -> int:
        if os.fspath(path) == os.fspath(src):
            return original_open(path, flags, mode, **kwargs)
        raise PermissionError(errno.EACCES, "Permission denied", str(path))

    monkeypatch.setattr(os, "link", _link_exdev)
    monkeypatch.setattr(os, "open", _open_eacces)

    with pytest.raises(PermissionError):
        rename_pdf(src, "moved")

    assert src.exists()
    assert not (tmp_path / "moved.pdf").exists()


@pytest.mark.skipif(os.name == "nt", reason="Unix hard-link behavior")
def test_apply_single_rename_hard_link_unlink_failure_cleans_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    src = tmp_path / "doc.pdf"
    src.write_text("content", encoding="utf-8")
    target = tmp_path / "moved.pdf"
    original_unlink = Path.unlink

    def _unlink_source_only(self: Path, *args: object, **kwargs: object) -> None:
        if self == src:
            raise OSError(errno.EACCES, "Permission denied")
        original_unlink(self, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", _unlink_source_only)

    with pytest.raises(OSError, match="Permission denied"):
        rename_pdf(src, "moved")

    assert src.read_text(encoding="utf-8") == "content"
    assert not target.exists()


def test_apply_single_rename_exdev_copy_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """When fd-backed EXDEV copying fails, the error propagates and target is cleaned up."""
    src = tmp_path / "doc.pdf"
    src.write_text("content", encoding="utf-8")

    target_path = tmp_path / "moved.pdf"

    def _link_exdev(src: object, dst: object) -> None:
        raise OSError(errno.EXDEV, "Invalid cross-device link")

    def _copy_to_fd_fail(source: Path, target_fd: int) -> None:
        raise OSError(errno.EIO, "I/O error during copy")

    monkeypatch.setattr(os, "link", _link_exdev)
    monkeypatch.setattr(rename_ops, "_copy_file_to_fd", _copy_to_fd_fail)

    with pytest.raises(OSError, match="I/O error"):
        rename_pdf(src, "moved")

    # Target should have been cleaned up
    assert not target_path.exists(), "Target should be cleaned up after copy failure"
    # Source should still be intact
    assert src.exists()
    assert src.read_text(encoding="utf-8") == "content"


# ---------------------------------------------------------------------------
# ENAMETOOLONG handling
# ---------------------------------------------------------------------------


def test_apply_single_rename_enametoolong(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """When os.link raises ENAMETOOLONG, a helpful error message about shortening names is raised."""
    src = tmp_path / "doc.pdf"
    src.write_text("content", encoding="utf-8")

    def _link_enametoolong(src: object, dst: object) -> None:
        raise OSError(errno.ENAMETOOLONG, "File name too long")

    monkeypatch.setattr(os, "link", _link_enametoolong)

    with pytest.raises(OSError, match="Shorten"):
        rename_pdf(src, "a" * 300)

    # Source must still be intact.
    assert src.exists()


# ---------------------------------------------------------------------------
# Plan file edge cases
# ---------------------------------------------------------------------------


def test_apply_single_rename_plan_entries_none(tmp_path: Path) -> None:
    """plan_file_path set but plan_entries=None must not crash and should return True."""
    src = tmp_path / "doc.pdf"
    src.write_text("content", encoding="utf-8")

    ok, target = rename_pdf(
        src,
        "planned",
        plan_file_path=tmp_path / "plan.json",
        plan_entries=None,
    )

    assert ok is True
    assert target.name == "planned.pdf"
    # Source should still exist because plan mode does not rename
    assert src.exists()


# ---------------------------------------------------------------------------
# _validate_path_within_parent
# ---------------------------------------------------------------------------


def test_validate_path_within_parent_valid(tmp_path: Path) -> None:
    """A path inside the parent directory is accepted and returned resolved."""
    child = tmp_path / "subdir" / "file.pdf"
    result = _validate_path_within_parent(child, tmp_path)
    assert result == child.resolve()


def test_validate_path_within_parent_traversal(tmp_path: Path) -> None:
    """A path that resolves outside the parent directory raises ValueError."""
    # tmp_path / ".." / "escape.pdf" resolves to tmp_path.parent / "escape.pdf"
    evil_path = tmp_path / ".." / "escape.pdf"
    with pytest.raises(ValueError, match="Path traversal detected"):
        _validate_path_within_parent(evil_path, tmp_path)


# ---------------------------------------------------------------------------
# is_path_within: OSError branch
# ---------------------------------------------------------------------------


def test_is_path_within_oserror_returns_false(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """When path.resolve() raises OSError, is_path_within returns False gracefully."""
    evil = tmp_path / "file.pdf"

    original_resolve = Path.resolve

    def _bad_resolve(self: Path, *args: object, **kwargs: object) -> Path:
        if self == evil:
            raise OSError("resolve failed")
        return original_resolve(self)

    monkeypatch.setattr(Path, "resolve", _bad_resolve)
    assert is_path_within(evil, tmp_path) is False
