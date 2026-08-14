"""Atomic, no-overwrite filesystem primitives for PDF renames."""

from __future__ import annotations

import contextlib
import errno
import os
import shutil
import stat
from collections.abc import Callable
from pathlib import Path

_OWNER_ONLY_FILE_MODE = stat.S_IRUSR | stat.S_IWUSR


def _open_exclusive_target(path: Path | str, *, mode: int = _OWNER_ONLY_FILE_MODE, dir_fd: int | None = None) -> int:
    """Create a write-only target without following a pre-existing link."""
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    kwargs: dict[str, int] = {} if dir_fd is None else {"dir_fd": dir_fd}
    return os.open(path, flags, mode, **kwargs)


def _copy_file_to_fd(file_path: Path, target_fd: int) -> os.stat_result:
    """Copy content and portable metadata through an already-reserved descriptor."""
    with file_path.open("rb") as source, os.fdopen(os.dup(target_fd), "wb") as target_file:
        source_status = os.fstat(source.fileno())
        shutil.copyfileobj(source, target_file)
    if os.name != "nt":
        os.fchmod(target_fd, stat.S_IMODE(source_status.st_mode))
        os.utime(target_fd, ns=(source_status.st_atime_ns, source_status.st_mtime_ns))
    return source_status


def _directory_entry_matches_identity(name: str, directory_fd: int, identity: os.stat_result) -> bool:
    """Return whether a directory entry still names a reserved non-symlink inode."""
    try:
        path_status = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    except OSError:
        return False
    return _identity_matches(path_status, identity)


def _path_matches_identity(path: Path, identity: os.stat_result) -> bool:
    """Return whether a pathname still names a reserved non-symlink inode."""
    try:
        path_status = path.lstat()
    except OSError:
        return False
    return _identity_matches(path_status, identity)


def _identity_matches(path_status: os.stat_result, identity: os.stat_result) -> bool:
    """Compare device and inode identity while rejecting symbolic links."""
    return (
        not stat.S_ISLNK(path_status.st_mode)
        and path_status.st_dev == identity.st_dev
        and path_status.st_ino == identity.st_ino
    )


def _cleanup_reserved_directory_entry(name: str, directory_fd: int, identity: os.stat_result) -> None:
    """Best-effort unlink only when the directory entry identity remains ours."""
    if _directory_entry_matches_identity(name, directory_fd, identity):
        with contextlib.suppress(OSError):
            os.unlink(name, dir_fd=directory_fd)


def _cleanup_reserved_target(target: Path, identity: os.stat_result) -> None:
    """Best-effort unlink only when the target identity remains ours."""
    if _path_matches_identity(target, identity):
        with contextlib.suppress(OSError):
            target.unlink()


def _rename_without_overwrite(file_path: Path, target: Path) -> None:
    """Rename through Windows rename, a hard link, or an exclusive-copy fallback."""
    if os.name == "nt":
        os.rename(file_path, target)
        return
    source_fd = os.open(file_path, os.O_RDONLY)
    try:
        source_identity = os.fstat(source_fd)
        linked_identity = _try_hard_link_without_overwrite(file_path, target, source_identity)
        if linked_identity is None:
            _copy_to_reserved_target_then_unlink(file_path, target, source_fd)
            return
        _validate_link_and_unlink_source(file_path, target, linked_identity)
    finally:
        os.close(source_fd)


def _try_hard_link_without_overwrite(
    file_path: Path, target: Path, source_identity: os.stat_result
) -> os.stat_result | None:
    """Try a hard-link rename, retaining collision semantics on every platform."""
    try:
        os.link(file_path, target)
    except (AttributeError, OSError) as exc:
        if isinstance(exc, OSError) and exc.errno == errno.EEXIST:
            raise FileExistsError from exc
        return None
    return source_identity


def _validate_link_and_unlink_source(file_path: Path, target: Path, source_identity: os.stat_result) -> None:
    """Validate both entries before removing a hard-link source pathname."""
    if not _path_matches_identity(target, source_identity):
        _cleanup_link_created_from_replaced_source(file_path, target)
        raise OSError(errno.EBUSY, f"Target path changed while linking: {target}")
    if not _path_matches_identity(file_path, source_identity):
        _cleanup_reserved_target(target, source_identity)
        raise OSError(errno.EBUSY, f"Source path changed while linking: {file_path}")
    try:
        file_path.unlink()
    except OSError:
        _cleanup_reserved_target(target, source_identity)
        raise


def _cleanup_link_created_from_replaced_source(file_path: Path, target: Path) -> None:
    """Remove a mismatched link only when it still aliases the current source."""
    try:
        target_identity = target.lstat()
    except OSError:
        return
    if _path_matches_identity(file_path, target_identity):
        _cleanup_reserved_target(target, target_identity)


def _copy_to_reserved_target_then_unlink(file_path: Path, target: Path, source_fd: int | None = None) -> None:
    """Copy into an exclusive target while retaining the original source inode."""
    owns_source_fd = source_fd is None
    if source_fd is None:
        source_fd = os.open(file_path, os.O_RDONLY)

    def release_windows_source() -> None:
        """Close the owned source descriptor before Windows removes its path."""
        nonlocal source_fd
        if source_fd is not None:
            os.close(source_fd)
            source_fd = None

    try:
        _copy_with_pinned_source(
            file_path,
            target,
            source_fd,
            before_unlink=release_windows_source if owns_source_fd and os.name == "nt" else None,
        )
    finally:
        if owns_source_fd and source_fd is not None:
            os.close(source_fd)


def _copy_with_pinned_source(
    file_path: Path, target: Path, source_fd: int, *, before_unlink: Callable[[], None] | None
) -> None:
    """Reserve, copy, validate, and unlink while the source inode is pinned."""
    held_identity = os.fstat(source_fd)
    target_fd = _reserve_copy_target(target)
    try:
        target_identity = os.fstat(target_fd)
        _copy_and_validate_reserved_target(file_path, target, target_fd, held_identity, target_identity)
        _run_before_unlink(before_unlink, target, target_identity)
        _unlink_copied_source(file_path, target, target_identity)
    finally:
        os.close(target_fd)


def _reserve_copy_target(target: Path) -> int:
    """Reserve a no-overwrite copy target and normalize collision errors."""
    try:
        return _open_exclusive_target(target)
    except OSError as open_exc:
        if open_exc.errno == errno.EEXIST:
            raise FileExistsError(f"Target already exists: {target}") from open_exc
        raise


def _copy_and_validate_reserved_target(
    file_path: Path,
    target: Path,
    target_fd: int,
    source_identity: os.stat_result,
    target_identity: os.stat_result,
) -> None:
    """Copy only while the source and the exclusive target paths remain pinned."""
    try:
        _copy_file_to_fd(file_path, target_fd)
    except OSError:
        _cleanup_reserved_target(target, target_identity)
        raise
    if not _path_matches_identity(target, target_identity):
        raise OSError(errno.EBUSY, f"Target path changed while copying: {target}")
    if not _path_matches_identity(file_path, source_identity):
        _cleanup_reserved_target(target, target_identity)
        raise OSError(errno.EBUSY, f"Source path changed while copying: {file_path}")


def _run_before_unlink(before_unlink: Callable[[], None] | None, target: Path, target_identity: os.stat_result) -> None:
    """Run an optional pre-unlink hook and clean the owned target if it fails."""
    if before_unlink is None:
        return
    try:
        before_unlink()
    except OSError:
        _cleanup_reserved_target(target, target_identity)
        raise


def _unlink_copied_source(file_path: Path, target: Path, target_identity: os.stat_result) -> None:
    """Unlink the validated source and clean the reservation if that fails."""
    try:
        file_path.unlink()
    except OSError as unlink_err:
        _cleanup_reserved_target(target, target_identity)
        raise OSError(
            f"Cross-filesystem rename: copied to {target}, could not remove source {file_path}: {unlink_err}"
        ) from unlink_err
