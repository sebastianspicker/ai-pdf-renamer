"""Verify atomic owner-only writes and symlink-safe private append behavior."""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

import folionym.private_io as private_io
from folionym.renamer_hooks import _write_rename_log_entry


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


@pytest.mark.skipif(os.name != "posix", reason="POSIX permission bits are not portable")
def test_atomic_private_write_replaces_with_owner_only_file(tmp_path: Path) -> None:
    target = tmp_path / "metadata.json"
    target.write_text("old", encoding="utf-8")
    target.chmod(0o644)

    private_io.atomic_write_private_text(target, '{"summary":"private"}')

    assert target.read_text(encoding="utf-8") == '{"summary":"private"}'
    assert _mode(target) == 0o600


def test_atomic_private_write_preserves_existing_target_and_cleans_temp_on_replace_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "summary.json"
    target.write_text("old", encoding="utf-8")

    def fail_replace(source: Path, destination: Path) -> None:
        raise OSError("simulated replace failure")

    monkeypatch.setattr(private_io.os, "replace", fail_replace)

    with pytest.raises(OSError, match="simulated replace failure"):
        private_io.atomic_write_private_text(target, "new")

    assert target.read_text(encoding="utf-8") == "old"
    assert list(tmp_path.glob(".summary.json.*")) == []


@pytest.mark.skipif(os.name != "posix", reason="POSIX permission bits are not portable")
def test_private_append_restricts_existing_log_before_appending(tmp_path: Path) -> None:
    log_path = tmp_path / "renames.log"
    log_path.write_text("old\n", encoding="utf-8")
    log_path.chmod(0o644)

    with private_io.open_private_append_text(log_path) as handle:
        handle.write("new\n")

    assert log_path.read_text(encoding="utf-8") == "old\nnew\n"
    assert _mode(log_path) == 0o600


@pytest.mark.skipif(os.name != "posix", reason="symbolic-link behavior is platform-specific")
def test_private_append_rejects_symlink_target(tmp_path: Path) -> None:
    victim = tmp_path / "victim.txt"
    victim.write_text("unchanged\n", encoding="utf-8")
    log_path = tmp_path / "renames.log"
    log_path.symlink_to(victim)

    with pytest.raises(OSError):
        private_io.open_private_append_text(log_path)

    assert victim.read_text(encoding="utf-8") == "unchanged\n"


@pytest.mark.skipif(os.name != "posix", reason="symbolic-link behavior is platform-specific")
def test_private_append_rejects_symlink_without_o_nofollow(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    victim = tmp_path / "victim.txt"
    victim.write_text("unchanged\n", encoding="utf-8")
    log_path = tmp_path / "renames.log"
    log_path.symlink_to(victim)
    monkeypatch.delattr(private_io.os, "O_NOFOLLOW", raising=False)

    with pytest.raises(OSError):
        private_io.open_private_append_text(log_path)

    assert victim.read_text(encoding="utf-8") == "unchanged\n"


@pytest.mark.skipif(os.name != "posix", reason="POSIX permission bits are not portable")
def test_rename_log_entry_uses_private_append(tmp_path: Path) -> None:
    log_path = tmp_path / "renames.log"
    log_path.write_text("old\n", encoding="utf-8")
    log_path.chmod(0o644)

    _write_rename_log_entry(log_path, tmp_path / "old.pdf", tmp_path / "new.pdf")

    assert log_path.read_text(encoding="utf-8").endswith("old.pdf\t" + str(tmp_path / "new.pdf") + "\n")
    assert _mode(log_path) == 0o600
