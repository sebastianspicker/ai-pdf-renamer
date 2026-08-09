"""Private backup safety and collision behavior."""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

import folionym.rename_ops.backups as rename_backups
from folionym.rename_ops import MAX_RENAME_RETRIES
from folionym.rename_ops.filesystem import _copy_file_to_fd
from tests.conftest import rename_pdf


def test_apply_single_rename_backup_collision_does_not_overwrite(tmp_path: Path) -> None:
    source, backup = tmp_path / "doc.pdf", tmp_path / "backups"
    source.write_text("new", encoding="utf-8")
    backup.mkdir()
    (backup / "doc.pdf").write_text("old", encoding="utf-8")
    assert rename_pdf(source, "renamed", backup_dir=backup)[0]
    assert (backup / "doc.pdf").read_text() == "old"
    assert (backup / "doc_1.pdf").read_text() == "new"


@pytest.mark.skipif(os.name == "nt", reason="POSIX permissions and symlinks required")
def test_backup_rejects_dangling_symlink_candidate_without_writing_through_it(tmp_path: Path) -> None:
    source, backup, outside = tmp_path / "doc.pdf", tmp_path / "backups", tmp_path / "outside.pdf"
    source.write_text("source", encoding="utf-8")
    backup.mkdir()
    (backup / "doc.pdf").write_text("old")
    (backup / "doc_1.pdf").symlink_to(outside)
    assert rename_pdf(source, "renamed", backup_dir=backup)[0]
    assert not outside.exists() and (backup / "doc_2.pdf").read_text() == "source"


@pytest.mark.skipif(os.name == "nt", reason="POSIX permissions and symlinks required")
def test_backup_target_swap_preserves_source_and_victim(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source, backup, victim = tmp_path / "doc.pdf", tmp_path / "backups", tmp_path / "victim.txt"
    target = backup / "doc.pdf"
    source.write_text("source")
    victim.write_text("victim")
    original = _copy_file_to_fd

    def copy_then_swap(path: Path, fd: int) -> os.stat_result:
        result = original(path, fd)
        target.unlink()
        target.symlink_to(victim)
        return result

    monkeypatch.setattr(rename_backups, "_copy_file_to_fd", copy_then_swap)
    with pytest.raises(OSError, match="Backup path changed"):
        rename_pdf(source, "renamed", backup_dir=backup)
    assert source.read_text() == "source" and victim.read_text() == "victim" and target.is_symlink()


@pytest.mark.skipif(os.name == "nt", reason="POSIX permissions required")
def test_backup_directory_and_file_are_owner_only(tmp_path: Path) -> None:
    source, backup = tmp_path / "doc.pdf", tmp_path / "backups"
    source.write_text("source")
    assert rename_pdf(source, "renamed", backup_dir=backup)[0]
    assert stat.S_IMODE(backup.stat().st_mode) == 0o700
    assert stat.S_IMODE((backup / "doc.pdf").stat().st_mode) == 0o600


def test_backup_is_created_once_when_all_rename_targets_collide(tmp_path: Path) -> None:
    source, backup = tmp_path / "doc.pdf", tmp_path / "backups"
    source.write_text("source")
    for counter in range(MAX_RENAME_RETRIES + 1):
        (tmp_path / ("renamed.pdf" if counter == 0 else f"renamed_{counter}.pdf")).write_text("occupied")
    assert not rename_pdf(source, "renamed", backup_dir=backup)[0]
    assert (backup / "doc.pdf").read_text() == "source"


def test_backup_is_created_once_when_rename_succeeds_after_collision(tmp_path: Path) -> None:
    source, backup = tmp_path / "doc.pdf", tmp_path / "backups"
    source.write_text("source")
    (tmp_path / "renamed.pdf").write_text("occupied")
    assert rename_pdf(source, "renamed", backup_dir=backup)[1].name == "renamed_1.pdf"
    assert [path.name for path in backup.iterdir()] == ["doc.pdf"]
