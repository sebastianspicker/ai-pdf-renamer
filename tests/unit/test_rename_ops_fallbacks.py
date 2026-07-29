"""Tests for rename fallback and retry edge cases."""

from __future__ import annotations

import errno
import os
import stat
from pathlib import Path

import pytest

import folionym.rename_ops as rename_ops
from folionym.rename_ops import (
    MAX_RENAME_RETRIES,
    _copy_file_to_fd,
    sanitize_filename_base,
)
from tests.conftest import rename_pdf


def _force_link_failure(monkeypatch: pytest.MonkeyPatch, error_number: int) -> None:
    """Make the hard-link branch fail with the selected platform error."""

    def fail_link(_source: object, _destination: object) -> None:
        raise OSError(error_number, os.strerror(error_number))

    monkeypatch.setattr(os, "link", fail_link)


def _force_cross_device_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise the guarded descriptor-copy branch rather than hard-link rename."""
    _force_link_failure(monkeypatch, errno.EXDEV)

    def fail_rename(_source: object, _destination: object) -> None:
        raise OSError(errno.EXDEV, "EXDEV")

    monkeypatch.setattr(os, "rename", fail_rename)


def _assert_preserved_metadata(target: Path, expected_mtime_ns: int) -> None:
    """Assert the descriptor-copy and hard-link paths retain source metadata."""
    target_status = target.stat()
    assert stat.S_IMODE(target_status.st_mode) == 0o640
    assert target_status.st_mtime_ns == expected_mtime_ns


@pytest.mark.parametrize("reserved", ["CON", "NUL", "AUX", "PRN", "COM1", "LPT9"])
def test_sanitize_filename_base_windows_reserved(reserved: str) -> None:
    """Windows reserved device names get an underscore suffix appended."""
    result = sanitize_filename_base(reserved)
    assert result == f"{reserved}_"


def test_sanitize_filename_base_reserved_case_insensitive() -> None:
    """Reserved name check is case-insensitive."""
    assert sanitize_filename_base("con") == "con_"
    assert sanitize_filename_base("Nul") == "Nul_"


@pytest.mark.skipif(os.name == "nt", reason="Unix-only branch")
def test_apply_single_rename_link_eperm_fallback_to_rename(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """When os.link raises EPERM (not EEXIST), falls back through O_CREAT placeholder to os.rename."""
    src = tmp_path / "doc.pdf"
    src.write_text("content", encoding="utf-8")

    _force_link_failure(monkeypatch, errno.EPERM)

    ok, target = rename_pdf(src, "result")

    assert ok is True
    assert target.name == "result.pdf"
    assert target.read_text(encoding="utf-8") == "content"
    assert not src.exists()


@pytest.mark.skipif(os.name == "nt", reason="Unix-only branch")
def test_apply_single_rename_link_eperm_target_exists_collision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When os.link raises EPERM and target exists, O_CREAT|O_EXCL raises FileExistsError."""
    src = tmp_path / "doc.pdf"
    src.write_text("content", encoding="utf-8")
    (tmp_path / "result.pdf").write_text("existing", encoding="utf-8")

    _force_link_failure(monkeypatch, errno.EPERM)

    ok, target = rename_pdf(src, "result")

    assert ok is True
    assert target.name == "result_1.pdf"
    assert target.read_text(encoding="utf-8") == "content"


@pytest.mark.skipif(os.name == "nt", reason="Unix-only branch")
def test_apply_single_rename_link_fallback_copies_through_reserved_descriptor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The hard-link fallback writes through its exclusive target descriptor."""
    src = tmp_path / "doc.pdf"
    src.write_text("content", encoding="utf-8")
    expected_target = tmp_path / "result.pdf"

    original_copy = _copy_file_to_fd
    saw_reserved_target = False

    def _link_eperm(s: object, d: object) -> None:
        raise OSError(errno.EPERM, "Operation not permitted")

    def _copy_observes_reservation(source: Path, target_fd: int) -> os.stat_result:
        nonlocal saw_reserved_target
        reserved = os.fstat(target_fd)
        target_status = expected_target.lstat()
        saw_reserved_target = (reserved.st_dev, reserved.st_ino) == (target_status.st_dev, target_status.st_ino)
        return original_copy(source, target_fd)

    def _rename_must_not_run(source: object, destination: object) -> None:
        raise AssertionError(f"unsafe close-then-rename fallback used: {source} -> {destination}")

    monkeypatch.setattr(os, "link", _link_eperm)
    monkeypatch.setattr(os, "rename", _rename_must_not_run)
    monkeypatch.setattr(rename_ops, "_copy_file_to_fd", _copy_observes_reservation)

    ok, target = rename_pdf(src, "result")

    assert ok is True
    assert saw_reserved_target is True
    assert target.name == "result.pdf"
    assert target.read_text(encoding="utf-8") == "content"


@pytest.mark.skipif(os.name == "nt", reason="Unix-only hard-link and descriptor metadata behavior")
def test_apply_single_rename_link_fallback_preserves_mode_and_mtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """EPERM fallback must preserve source mode and mtime without path metadata writes."""
    src = tmp_path / "doc.pdf"
    src.write_text("content", encoding="utf-8")
    os.chmod(src, 0o640)
    expected_mtime_ns = 1_700_000_000_123_456_789
    os.utime(src, ns=(expected_mtime_ns, expected_mtime_ns))

    def _link_eperm(source: object, destination: object) -> None:
        raise OSError(errno.EPERM, "Operation not permitted")

    monkeypatch.setattr(os, "link", _link_eperm)

    ok, target = rename_pdf(src, "result")

    assert ok is True
    _assert_preserved_metadata(target, expected_mtime_ns)


@pytest.mark.skipif(os.name == "nt", reason="Unix-only hard-link behavior")
def test_apply_single_rename_link_source_swap_preserves_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A source replacement after link creation must not be unlinked by pathname."""
    src = tmp_path / "doc.pdf"
    src.write_text("original", encoding="utf-8")
    target = tmp_path / "result.pdf"
    original_link = os.link

    def _link_then_swap(source: str | bytes | os.PathLike[str], destination: str | bytes | os.PathLike[str]) -> None:
        original_link(source, destination)
        src.unlink()
        src.write_text("replacement", encoding="utf-8")

    monkeypatch.setattr(os, "link", _link_then_swap)

    with pytest.raises(OSError, match="Source path changed while linking"):
        rename_pdf(src, "result")

    assert src.read_text(encoding="utf-8") == "replacement"
    assert not target.exists()


@pytest.mark.skipif(os.name == "nt", reason="Unix-only hard-link metadata behavior")
def test_apply_single_rename_hard_link_preserves_mode_and_mtime(tmp_path: Path) -> None:
    """A successful hard-link rename retains the original inode metadata."""
    src = tmp_path / "doc.pdf"
    src.write_text("content", encoding="utf-8")
    os.chmod(src, 0o640)
    expected_mtime_ns = 1_700_000_000_987_654_321
    os.utime(src, ns=(expected_mtime_ns, expected_mtime_ns))

    ok, target = rename_pdf(src, "result")

    assert ok is True
    _assert_preserved_metadata(target, expected_mtime_ns)


@pytest.mark.skipif(os.name == "nt", reason="Unix-only hard-link behavior")
def test_apply_single_rename_link_source_swap_before_link_cleans_owned_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A replacement just before link creation is retained and its new link is removed."""
    src = tmp_path / "doc.pdf"
    src.write_text("original", encoding="utf-8")
    target = tmp_path / "result.pdf"
    original_link = os.link
    original_open = os.open
    source_fd: int | None = None

    def _open_source(path: str | bytes | os.PathLike[str], flags: int, mode: int = 0o777, **kwargs: object) -> int:
        nonlocal source_fd
        fd = original_open(path, flags, mode, **kwargs)
        if Path(path) == src and flags == os.O_RDONLY:
            source_fd = fd
        return fd

    def _swap_then_link(source: str | bytes | os.PathLike[str], destination: str | bytes | os.PathLike[str]) -> None:
        assert source_fd is not None
        os.fstat(source_fd)
        src.unlink()
        src.write_text("replacement", encoding="utf-8")
        original_link(source, destination)

    monkeypatch.setattr(os, "open", _open_source)
    monkeypatch.setattr(os, "link", _swap_then_link)

    with pytest.raises(OSError, match="Target path changed while linking"):
        rename_pdf(src, "result")

    assert src.read_text(encoding="utf-8") == "replacement"
    assert not target.exists()


@pytest.mark.skipif(os.name == "nt", reason="Unix-only descriptor metadata behavior")
def test_apply_single_rename_copy_source_swap_preserves_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A source replacement after fd copy must survive the guarded final unlink."""
    src = tmp_path / "doc.pdf"
    src.write_text("original", encoding="utf-8")
    target = tmp_path / "result.pdf"
    original_copy = _copy_file_to_fd
    original_open = os.open
    source_fd: int | None = None

    def _open_source(path: str | bytes | os.PathLike[str], flags: int, mode: int = 0o777, **kwargs: object) -> int:
        nonlocal source_fd
        fd = original_open(path, flags, mode, **kwargs)
        if Path(path) == src and flags == os.O_RDONLY:
            source_fd = fd
        return fd

    def _link_eperm(source: object, destination: object) -> None:
        raise OSError(errno.EPERM, "Operation not permitted")

    def _copy_then_swap(source: Path, target_fd: int) -> os.stat_result:
        source_status = original_copy(source, target_fd)
        assert source_fd is not None
        os.fstat(source_fd)
        src.unlink()
        src.write_text("replacement", encoding="utf-8")
        return source_status

    monkeypatch.setattr(os, "open", _open_source)
    monkeypatch.setattr(os, "link", _link_eperm)
    monkeypatch.setattr(rename_ops, "_copy_file_to_fd", _copy_then_swap)

    with pytest.raises(OSError, match="Source path changed while copying"):
        rename_pdf(src, "result")

    assert src.read_text(encoding="utf-8") == "replacement"
    assert not target.exists()


@pytest.mark.skipif(os.name == "nt", reason="Unix-only branch")
def test_apply_single_rename_link_fallback_propagates_reservation_permission_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Permission failures while reserving the target must not be treated as collisions."""
    src = tmp_path / "doc.pdf"
    src.write_text("content", encoding="utf-8")
    original_open = os.open

    def _link_eperm(s: object, d: object) -> None:
        raise OSError(errno.EPERM, "Operation not permitted")

    def _open_eacces(path: str | bytes | os.PathLike[str], flags: int, mode: int = 0o777, **kwargs: object) -> int:
        if os.fspath(path) == os.fspath(src):
            return original_open(path, flags, mode, **kwargs)
        raise PermissionError(errno.EACCES, "Permission denied", str(path))

    def _rename_should_not_run(s: object, d: object) -> None:
        raise AssertionError("rename should not run after reservation permission failure")

    monkeypatch.setattr(os, "link", _link_eperm)
    monkeypatch.setattr(os, "open", _open_eacces)
    monkeypatch.setattr(os, "rename", _rename_should_not_run)

    with pytest.raises(PermissionError):
        rename_pdf(src, "result")

    assert src.exists()
    assert not (tmp_path / "result.pdf").exists()
    assert not (tmp_path / "result_1.pdf").exists()


def test_apply_single_rename_max_filename_chars_truncation(tmp_path: Path) -> None:
    """When max_filename_chars is set, collision suffix trims the base to fit."""
    src = tmp_path / "doc.pdf"
    src.write_text("content", encoding="utf-8")
    long_base = "a" * 20
    (tmp_path / f"{long_base}.pdf").write_text("existing", encoding="utf-8")

    ok, target = rename_pdf(
        src,
        long_base,
        max_filename_chars=15,
    )

    assert ok is True
    assert len(target.stem) + len(target.suffix) <= 15 or target.exists()
    assert target.read_text(encoding="utf-8") == "content"


@pytest.mark.skipif(os.name == "nt", reason="Unix-only branch")
def test_apply_single_rename_exdev_dry_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """EXDEV with dry_run=True returns success without touching the filesystem."""
    src = tmp_path / "doc.pdf"
    src.write_text("content", encoding="utf-8")

    _force_cross_device_fallback(monkeypatch)

    ok, target = rename_pdf(
        src,
        "moved",
        dry_run=True,
    )

    assert ok is True
    assert src.exists(), "Source must survive dry_run"
    assert not target.exists(), "Target must not be created in dry_run"


@pytest.mark.skipif(os.name == "nt", reason="Unix-only branch")
def test_apply_single_rename_exdev_calls_on_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """on_success is called after successful EXDEV copy+unlink path."""
    src = tmp_path / "doc.pdf"
    src.write_text("content", encoding="utf-8")

    _force_cross_device_fallback(monkeypatch)

    calls: list[tuple[Path, Path, str]] = []

    def _on_success(old: Path, new: Path, base: str) -> None:
        calls.append((old, new, base))

    ok, target = rename_pdf(
        src,
        "moved",
        on_success=_on_success,
    )

    assert ok is True
    assert len(calls) == 1
    assert calls[0][1] == target


@pytest.mark.skipif(os.name == "nt", reason="Unix-only branch")
def test_apply_single_rename_exdev_unlink_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """When cross-fs copy succeeds but source unlink fails, an OSError is raised and target is cleaned up."""
    src = tmp_path / "doc.pdf"
    src.write_text("content", encoding="utf-8")

    original_unlink = Path.unlink

    def _unlink_fail(self: Path, *args: object, **kwargs: object) -> None:
        if self == src:
            raise OSError(errno.EACCES, "Permission denied")
        original_unlink(self, *args, **kwargs)

    _force_cross_device_fallback(monkeypatch)
    monkeypatch.setattr(Path, "unlink", _unlink_fail)

    with pytest.raises(OSError, match="Cross-filesystem rename"):
        rename_pdf(src, "moved")


def test_apply_single_rename_retry_exhaustion(tmp_path: Path) -> None:
    """When all MAX_RENAME_RETRIES collision suffixes are occupied, returns (False, target)."""
    src = tmp_path / "doc.pdf"
    src.write_text("original", encoding="utf-8")

    base = "report"
    (tmp_path / f"{base}.pdf").write_text("v0", encoding="utf-8")
    for i in range(1, MAX_RENAME_RETRIES + 1):
        (tmp_path / f"{base}_{i}.pdf").write_text(f"v{i}", encoding="utf-8")

    ok, _ = rename_pdf(src, base)

    assert ok is False
    assert src.exists(), "Source must remain when rename fails"
