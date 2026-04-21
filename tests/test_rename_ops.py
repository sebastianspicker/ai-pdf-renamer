"""Tests for rename and filename-sanitization helpers."""

from __future__ import annotations

import errno
import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pytest

from ai_pdf_renamer.rename_ops import (
    MAX_RENAME_RETRIES,
    _next_available_path,
    _validate_path_within_parent,
    apply_single_rename,
    is_path_within,
    sanitize_filename_base,
    sanitize_filename_from_llm,
)


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

    success, target = apply_single_rename(
        src,
        "renamed",
        plan_file_path=None,
        plan_entries=[],
        dry_run=False,
        backup_dir=backup_dir,
        on_success=None,
        max_filename_chars=None,
    )

    assert success is True
    assert target.exists()
    assert existing_backup.read_text(encoding="utf-8") == "old-content"
    assert (backup_dir / "doc_1.pdf").read_text(encoding="utf-8") == "new-content"


def test_concurrent_renames_same_target_no_overwrite(tmp_path: Path) -> None:
    """Five concurrent renames to the same base name must all succeed with unique targets."""
    sources: list[Path] = []
    contents: list[str] = []
    for i in range(5):
        src = tmp_path / f"src_{i}.pdf"
        content = f"distinct-content-{i}"
        src.write_text(content, encoding="utf-8")
        sources.append(src)
        contents.append(content)

    # Pre-create the target so every rename collides with it *and* with each other.
    existing_target = tmp_path / "report.pdf"
    existing_target.write_text("pre-existing", encoding="utf-8")

    results: list[tuple[bool, Path]] = []

    def _rename(src: Path) -> tuple[bool, Path]:
        return apply_single_rename(
            src,
            "report",
            plan_file_path=None,
            plan_entries=[],
            dry_run=False,
            backup_dir=None,
            on_success=None,
            max_filename_chars=None,
        )

    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = {pool.submit(_rename, s): s for s in sources}
        for fut in as_completed(futures):
            results.append(fut.result())

    # All renames should succeed.
    assert all(ok for ok, _ in results)

    # Every target path must be unique (no two workers ended up at the same name).
    target_paths = [t for _, t in results]
    assert len(set(target_paths)) == len(target_paths)

    # All target files must actually exist on disk.
    for _, t in results:
        assert t.exists(), f"Expected target {t} to exist"

    # No source content was lost — every original payload appears in exactly one target.
    target_contents = {t: t.read_text(encoding="utf-8") for _, t in results}
    for content in contents:
        assert content in target_contents.values(), f"Content {content!r} missing from targets"

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
    ok1, target1 = apply_single_rename(
        src1,
        "invoice",
        plan_file_path=None,
        plan_entries=[],
        dry_run=False,
        backup_dir=None,
        on_success=None,
        max_filename_chars=None,
    )
    assert ok1 is True
    assert target1.name == "invoice_1.pdf"
    assert target1.read_text(encoding="utf-8") == "first-rename"

    # Second rename should land on invoice_2.pdf.
    src2 = tmp_path / "b.pdf"
    src2.write_text("second-rename", encoding="utf-8")
    ok2, target2 = apply_single_rename(
        src2,
        "invoice",
        plan_file_path=None,
        plan_entries=[],
        dry_run=False,
        backup_dir=None,
        on_success=None,
        max_filename_chars=None,
    )
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

    ok, _target = apply_single_rename(
        src,
        "new_name",
        plan_file_path=None,
        plan_entries=[],
        dry_run=True,
        backup_dir=None,
        on_success=None,
        max_filename_chars=None,
    )

    assert ok is True
    # Source must still exist with original content.
    assert src.exists()
    assert src.read_text(encoding="utf-8") == "keep-me"
    # No new files should appear.
    files_after = sorted(tmp_path.iterdir())
    assert files_before == files_after


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
        apply_single_rename(
            src,
            "../escape",
            plan_file_path=None,
            plan_entries=[],
            dry_run=False,
            backup_dir=None,
            on_success=None,
            max_filename_chars=None,
        )

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

    ok, target = apply_single_rename(
        src,
        "moved",
        plan_file_path=None,
        plan_entries=[],
        dry_run=False,
        backup_dir=None,
        on_success=None,
        max_filename_chars=None,
    )

    assert ok is True
    assert target.name == "moved.pdf"
    assert target.read_text(encoding="utf-8") == "cross-fs-content"
    assert not src.exists(), "Source should be removed after cross-fs fallback"


def test_apply_single_rename_exdev_copy_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """When os.link raises EXDEV and shutil.copy2 also fails, the error propagates and target is cleaned up."""
    src = tmp_path / "doc.pdf"
    src.write_text("content", encoding="utf-8")

    target_path = tmp_path / "moved.pdf"

    def _link_exdev(src: object, dst: object) -> None:
        raise OSError(errno.EXDEV, "Invalid cross-device link")

    def _rename_exdev(src: object, dst: object) -> None:
        raise OSError(errno.EXDEV, "Invalid cross-device link")

    def _copy2_fail(
        src: object, dst: object, **kwargs: object
    ) -> None:  # Create the target to simulate a partial copy that then fails
        Path(str(dst)).write_text("partial", encoding="utf-8")
        raise OSError(errno.EIO, "I/O error during copy")

    monkeypatch.setattr(os, "link", _link_exdev)
    monkeypatch.setattr(os, "rename", _rename_exdev)
    monkeypatch.setattr(shutil, "copy2", _copy2_fail)

    with pytest.raises(OSError, match="I/O error"):
        apply_single_rename(
            src,
            "moved",
            plan_file_path=None,
            plan_entries=[],
            dry_run=False,
            backup_dir=None,
            on_success=None,
            max_filename_chars=None,
        )

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
        apply_single_rename(
            src,
            "a" * 300,
            plan_file_path=None,
            plan_entries=[],
            dry_run=False,
            backup_dir=None,
            on_success=None,
            max_filename_chars=None,
        )

    # Source must still be intact.
    assert src.exists()


# ---------------------------------------------------------------------------
# Plan file edge cases
# ---------------------------------------------------------------------------


def test_apply_single_rename_plan_entries_none(tmp_path: Path) -> None:
    """plan_file_path set but plan_entries=None must not crash and should return True."""
    src = tmp_path / "doc.pdf"
    src.write_text("content", encoding="utf-8")

    ok, target = apply_single_rename(
        src,
        "planned",
        plan_file_path=tmp_path / "plan.json",
        plan_entries=None,
        dry_run=False,
        backup_dir=None,
        on_success=None,
        max_filename_chars=None,
    )

    assert ok is True
    assert target.name == "planned.pdf"
    # Source should still exist because plan mode does not rename
    assert src.exists()


# ---------------------------------------------------------------------------
# _next_available_path
# ---------------------------------------------------------------------------


def test_next_available_path_no_collision(tmp_path: Path) -> None:
    """When the path doesn't exist, it is returned as-is."""
    p = tmp_path / "nonexistent.pdf"
    result = _next_available_path(p)
    assert result == p


def test_next_available_path_finds_suffix(tmp_path: Path) -> None:
    """When the path exists, a _1 suffixed variant is returned."""
    p = tmp_path / "report.pdf"
    p.write_text("existing", encoding="utf-8")

    result = _next_available_path(p)
    assert result == tmp_path / "report_1.pdf"
    assert not result.exists()


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
# is_path_within — OSError branch
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


# ---------------------------------------------------------------------------
# _next_available_path — exhaustion
# ---------------------------------------------------------------------------


def test_next_available_path_exhaustion(tmp_path: Path) -> None:
    """When all suffixed candidates exist up to max_tries, OSError(EEXIST) is raised."""
    p = tmp_path / "report.pdf"
    p.write_text("base", encoding="utf-8")
    for i in range(1, 4):
        (tmp_path / f"report_{i}.pdf").write_text(f"v{i}", encoding="utf-8")
    with pytest.raises(OSError, match="Could not create unique path"):
        _next_available_path(p, max_tries=3)


# ---------------------------------------------------------------------------
# sanitize_filename_base — Windows reserved names
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("reserved", ["CON", "NUL", "AUX", "PRN", "COM1", "LPT9"])
def test_sanitize_filename_base_windows_reserved(reserved: str) -> None:
    """Windows reserved device names get an underscore suffix appended."""
    result = sanitize_filename_base(reserved)
    assert result == f"{reserved}_"


def test_sanitize_filename_base_reserved_case_insensitive() -> None:
    """Reserved name check is case-insensitive."""
    assert sanitize_filename_base("con") == "con_"
    assert sanitize_filename_base("Nul") == "Nul_"


# ---------------------------------------------------------------------------
# apply_single_rename — os.link fallback path (EPERM/ENOSYS)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(os.name == "nt", reason="Unix-only branch")
def test_apply_single_rename_link_eperm_fallback_to_rename(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """When os.link raises EPERM (not EEXIST), falls back through O_CREAT placeholder to os.rename."""
    src = tmp_path / "doc.pdf"
    src.write_text("content", encoding="utf-8")

    def _link_eperm(s: object, d: object) -> None:
        raise OSError(errno.EPERM, "Operation not permitted")

    monkeypatch.setattr(os, "link", _link_eperm)

    ok, target = apply_single_rename(
        src,
        "result",
        plan_file_path=None,
        plan_entries=[],
        dry_run=False,
        backup_dir=None,
        on_success=None,
        max_filename_chars=None,
    )

    assert ok is True
    assert target.name == "result.pdf"
    assert target.read_text(encoding="utf-8") == "content"
    assert not src.exists()


@pytest.mark.skipif(os.name == "nt", reason="Unix-only branch")
def test_apply_single_rename_link_eperm_target_exists_collision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When os.link raises EPERM and target exists, O_CREAT|O_EXCL raises FileExistsError → collision suffix."""
    src = tmp_path / "doc.pdf"
    src.write_text("content", encoding="utf-8")
    # Pre-create the target so O_CREAT|O_EXCL will fail
    (tmp_path / "result.pdf").write_text("existing", encoding="utf-8")

    def _link_eperm(s: object, d: object) -> None:
        raise OSError(errno.EPERM, "Operation not permitted")

    monkeypatch.setattr(os, "link", _link_eperm)

    ok, target = apply_single_rename(
        src,
        "result",
        plan_file_path=None,
        plan_entries=[],
        dry_run=False,
        backup_dir=None,
        on_success=None,
        max_filename_chars=None,
    )

    assert ok is True
    assert target.name == "result_1.pdf"
    assert target.read_text(encoding="utf-8") == "content"


# ---------------------------------------------------------------------------
# apply_single_rename — max_filename_chars truncation during collision
# ---------------------------------------------------------------------------


def test_apply_single_rename_max_filename_chars_truncation(tmp_path: Path) -> None:
    """When max_filename_chars is set, collision suffix trims the base to fit."""
    src = tmp_path / "doc.pdf"
    src.write_text("content", encoding="utf-8")
    # Pre-create the expected target to force a collision
    long_base = "a" * 20
    (tmp_path / f"{long_base}.pdf").write_text("existing", encoding="utf-8")

    ok, target = apply_single_rename(
        src,
        long_base,
        plan_file_path=None,
        plan_entries=[],
        dry_run=False,
        backup_dir=None,
        on_success=None,
        max_filename_chars=15,
    )

    assert ok is True
    # Total name should fit within max_filename_chars characters (including .pdf)
    assert len(target.stem) + len(target.suffix) <= 15 or target.exists()
    assert target.read_text(encoding="utf-8") == "content"


# ---------------------------------------------------------------------------
# apply_single_rename — EXDEV + dry_run
# ---------------------------------------------------------------------------


@pytest.mark.skipif(os.name == "nt", reason="Unix-only branch")
def test_apply_single_rename_exdev_dry_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """EXDEV with dry_run=True returns success without touching the filesystem."""
    src = tmp_path / "doc.pdf"
    src.write_text("content", encoding="utf-8")

    def _link_exdev(s: object, d: object) -> None:
        raise OSError(errno.EXDEV, "Cross-device link")

    def _rename_exdev(s: object, d: object) -> None:
        raise OSError(errno.EXDEV, "Cross-device link")

    monkeypatch.setattr(os, "link", _link_exdev)
    monkeypatch.setattr(os, "rename", _rename_exdev)

    ok, target = apply_single_rename(
        src,
        "moved",
        plan_file_path=None,
        plan_entries=[],
        dry_run=True,
        backup_dir=None,
        on_success=None,
        max_filename_chars=None,
    )

    assert ok is True
    assert src.exists(), "Source must survive dry_run"
    assert not target.exists(), "Target must not be created in dry_run"


# ---------------------------------------------------------------------------
# apply_single_rename — EXDEV + on_success callback
# ---------------------------------------------------------------------------


@pytest.mark.skipif(os.name == "nt", reason="Unix-only branch")
def test_apply_single_rename_exdev_calls_on_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """on_success is called after successful EXDEV copy+unlink path."""
    src = tmp_path / "doc.pdf"
    src.write_text("content", encoding="utf-8")

    def _link_exdev(s: object, d: object) -> None:
        raise OSError(errno.EXDEV, "Cross-device link")

    def _rename_exdev(s: object, d: object) -> None:
        raise OSError(errno.EXDEV, "Cross-device link")

    monkeypatch.setattr(os, "link", _link_exdev)
    monkeypatch.setattr(os, "rename", _rename_exdev)

    calls: list[tuple[Path, Path, str]] = []

    def _on_success(old: Path, new: Path, base: str) -> None:
        calls.append((old, new, base))

    ok, target = apply_single_rename(
        src,
        "moved",
        plan_file_path=None,
        plan_entries=[],
        dry_run=False,
        backup_dir=None,
        on_success=_on_success,
        max_filename_chars=None,
    )

    assert ok is True
    assert len(calls) == 1
    assert calls[0][1] == target


# ---------------------------------------------------------------------------
# apply_single_rename — EXDEV + unlink failure
# ---------------------------------------------------------------------------


@pytest.mark.skipif(os.name == "nt", reason="Unix-only branch")
def test_apply_single_rename_exdev_unlink_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """When cross-fs copy succeeds but source unlink fails, an OSError is raised and target is cleaned up."""
    src = tmp_path / "doc.pdf"
    src.write_text("content", encoding="utf-8")

    def _link_exdev(s: object, d: object) -> None:
        raise OSError(errno.EXDEV, "Cross-device link")

    def _rename_exdev(s: object, d: object) -> None:
        raise OSError(errno.EXDEV, "Cross-device link")

    original_unlink = Path.unlink

    def _unlink_fail(self: Path, *args: object, **kwargs: object) -> None:
        if self == src:
            raise OSError(errno.EACCES, "Permission denied")
        original_unlink(self, *args, **kwargs)

    monkeypatch.setattr(os, "link", _link_exdev)
    monkeypatch.setattr(os, "rename", _rename_exdev)
    monkeypatch.setattr(Path, "unlink", _unlink_fail)

    with pytest.raises(OSError, match="Cross-filesystem rename"):
        apply_single_rename(
            src,
            "moved",
            plan_file_path=None,
            plan_entries=[],
            dry_run=False,
            backup_dir=None,
            on_success=None,
            max_filename_chars=None,
        )


# ---------------------------------------------------------------------------
# apply_single_rename — retry exhaustion
# ---------------------------------------------------------------------------


def test_apply_single_rename_retry_exhaustion(tmp_path: Path) -> None:
    """When all MAX_RENAME_RETRIES collision suffixes are occupied, returns (False, target)."""
    src = tmp_path / "doc.pdf"
    src.write_text("original", encoding="utf-8")

    base = "report"
    # Pre-create the base target and all suffixed variants up to MAX_RENAME_RETRIES
    (tmp_path / f"{base}.pdf").write_text("v0", encoding="utf-8")
    for i in range(1, MAX_RENAME_RETRIES + 1):
        (tmp_path / f"{base}_{i}.pdf").write_text(f"v{i}", encoding="utf-8")

    ok, _ = apply_single_rename(
        src,
        base,
        plan_file_path=None,
        plan_entries=[],
        dry_run=False,
        backup_dir=None,
        on_success=None,
        max_filename_chars=None,
    )

    assert ok is False
    assert src.exists(), "Source must remain when rename fails"
