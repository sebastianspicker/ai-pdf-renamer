"""Private, collision-safe backup creation for rename operations."""

from __future__ import annotations

import errno
import os
import stat
from pathlib import Path

from .filesystem import (
    _cleanup_reserved_directory_entry,
    _cleanup_reserved_target,
    _copy_file_to_fd,
    _directory_entry_matches_identity,
    _open_exclusive_target,
    _path_matches_identity,
)
from .naming import _validate_path_within_parent

_OWNER_ONLY_DIRECTORY_MODE = stat.S_IRWXU
_OWNER_ONLY_FILE_MODE = stat.S_IRUSR | stat.S_IWUSR


def _write_backup(file_path: Path, backup_dir: Path | str | None) -> None:
    """Create a unique private backup before a non-preview rename."""
    if not backup_dir:
        return
    backup_root = Path(backup_dir)
    backup_path = backup_root / file_path.name
    _validate_path_within_parent(backup_path, backup_root)
    backup_root.mkdir(parents=True, exist_ok=True, mode=_OWNER_ONLY_DIRECTORY_MODE)
    if os.name == "nt":
        _write_backup_by_path(file_path, backup_path)
        return
    directory_fd = _open_private_backup_directory(backup_root)
    try:
        _write_backup_with_directory_fd(file_path, backup_path, directory_fd)
    finally:
        os.close(directory_fd)


def _open_private_backup_directory(backup_root: Path) -> int:
    """Open a private backup directory without following a final symlink."""
    directory_flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        directory_flags |= os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        directory_flags |= os.O_NOFOLLOW
    directory_fd = os.open(backup_root, directory_flags)
    try:
        os.fchmod(directory_fd, _OWNER_ONLY_DIRECTORY_MODE)
    except OSError:
        os.close(directory_fd)
        raise
    return directory_fd


def _write_backup_with_directory_fd(file_path: Path, backup_path: Path, directory_fd: int) -> None:
    """Reserve a unique backup name relative to an opened directory."""
    for counter in range(10_001):
        candidate_name = file_path.name if counter == 0 else f"{file_path.stem}_{counter}{file_path.suffix}"
        try:
            fd = _open_exclusive_target(candidate_name, mode=_OWNER_ONLY_FILE_MODE, dir_fd=directory_fd)
        except FileExistsError:
            continue
        _write_reserved_backup(file_path, backup_path, candidate_name, directory_fd, fd)
        return
    raise OSError(errno.EEXIST, f"Could not create unique path for backup after 10000 attempts: {backup_path}")


def _write_reserved_backup(file_path: Path, backup_path: Path, candidate_name: str, directory_fd: int, fd: int) -> None:
    """Copy through a reservation and remove only that unchanged reservation on failure."""
    identity: os.stat_result | None = None
    try:
        identity = os.fstat(fd)
        _copy_file_to_fd(file_path, fd)
        os.fchmod(fd, _OWNER_ONLY_FILE_MODE)
        if not _directory_entry_matches_identity(candidate_name, directory_fd, identity):
            raise OSError(errno.EBUSY, f"Backup path changed while writing: {backup_path}")
    except OSError:
        if identity is not None:
            _cleanup_reserved_directory_entry(candidate_name, directory_fd, identity)
        raise
    finally:
        os.close(fd)


def _write_backup_by_path(file_path: Path, backup_path: Path) -> None:
    """Create and validate a unique private Windows backup by pathname."""
    for counter in range(10_001):
        candidate = _backup_candidate(file_path, backup_path, counter)
        try:
            fd = _open_exclusive_target(candidate, mode=_OWNER_ONLY_FILE_MODE)
        except FileExistsError:
            continue
        _copy_reserved_backup_by_path(file_path, candidate, fd)
        return
    raise OSError(errno.EEXIST, f"Could not create unique path for backup after 10000 attempts: {backup_path}")


def _backup_candidate(file_path: Path, backup_path: Path, counter: int) -> Path:
    """Return the unsuffixed or numbered backup candidate for an attempt."""
    if counter == 0:
        return backup_path
    return backup_path.with_name(f"{file_path.stem}_{counter}{file_path.suffix}")


def _copy_reserved_backup_by_path(file_path: Path, candidate: Path, fd: int) -> None:
    """Populate and validate a pathname-reserved backup, cleaning it on failure."""
    identity: os.stat_result | None = None
    try:
        identity = os.fstat(fd)
        _copy_file_to_fd(file_path, fd)
        if not _path_matches_identity(candidate, identity):
            raise OSError(errno.EBUSY, f"Backup path changed while writing: {candidate}")
    except OSError:
        if identity is not None:
            _cleanup_reserved_target(candidate, identity)
        raise
    finally:
        os.close(fd)
