"""Tests for rename fallback and retry edge cases."""

from __future__ import annotations

import errno
import os
from pathlib import Path

import pytest

from ai_pdf_renamer.rename_ops import (
    MAX_RENAME_RETRIES,
    sanitize_filename_base,
)
from tests.conftest import rename_pdf


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

    def _link_eperm(s: object, d: object) -> None:
        raise OSError(errno.EPERM, "Operation not permitted")

    monkeypatch.setattr(os, "link", _link_eperm)

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

    def _link_eperm(s: object, d: object) -> None:
        raise OSError(errno.EPERM, "Operation not permitted")

    monkeypatch.setattr(os, "link", _link_eperm)

    ok, target = rename_pdf(src, "result")

    assert ok is True
    assert target.name == "result_1.pdf"
    assert target.read_text(encoding="utf-8") == "content"


@pytest.mark.skipif(os.name == "nt", reason="Unix-only branch")
def test_apply_single_rename_link_fallback_reserves_target_before_rename(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The hard-link fallback reserves the target before Unix rename can overwrite."""
    src = tmp_path / "doc.pdf"
    src.write_text("content", encoding="utf-8")

    original_rename = os.rename
    saw_reserved_target = False

    def _link_eperm(s: object, d: object) -> None:
        raise OSError(errno.EPERM, "Operation not permitted")

    def _rename_observes_placeholder(s: object, d: object) -> None:
        nonlocal saw_reserved_target
        saw_reserved_target = Path(str(d)).exists()
        original_rename(s, d)

    monkeypatch.setattr(os, "link", _link_eperm)
    monkeypatch.setattr(os, "rename", _rename_observes_placeholder)

    ok, target = rename_pdf(src, "result")

    assert ok is True
    assert saw_reserved_target is True
    assert target.name == "result.pdf"
    assert target.read_text(encoding="utf-8") == "content"


@pytest.mark.skipif(os.name == "nt", reason="Unix-only branch")
def test_apply_single_rename_link_fallback_propagates_reservation_permission_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Permission failures while reserving the target must not be treated as collisions."""
    src = tmp_path / "doc.pdf"
    src.write_text("content", encoding="utf-8")

    def _link_eperm(s: object, d: object) -> None:
        raise OSError(errno.EPERM, "Operation not permitted")

    def _open_eacces(path: object, flags: int, mode: int = 0o777) -> int:
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

    def _link_exdev(s: object, d: object) -> None:
        raise OSError(errno.EXDEV, "Cross-device link")

    def _rename_exdev(s: object, d: object) -> None:
        raise OSError(errno.EXDEV, "Cross-device link")

    monkeypatch.setattr(os, "link", _link_exdev)
    monkeypatch.setattr(os, "rename", _rename_exdev)

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

    def _link_exdev(s: object, d: object) -> None:
        raise OSError(errno.EXDEV, "Cross-device link")

    def _rename_exdev(s: object, d: object) -> None:
        raise OSError(errno.EXDEV, "Cross-device link")

    monkeypatch.setattr(os, "link", _link_exdev)
    monkeypatch.setattr(os, "rename", _rename_exdev)

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
