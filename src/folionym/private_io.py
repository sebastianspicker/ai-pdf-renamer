"""Private local text-output helpers for atomic replacement and safe append."""

from __future__ import annotations

import contextlib
import errno
import os
import stat
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import TextIO

_OWNER_ONLY_FILE_MODE = 0o600


def _restrict_fd(fd: int) -> None:
    """Restrict a newly opened file before any document data is written."""
    if os.name == "posix":
        os.fchmod(fd, _OWNER_ONLY_FILE_MODE)


def atomic_write_private_text(
    path: str | Path,
    text: str,
    *,
    encoding: str = "utf-8",
    before_write: Callable[[Path], None] | None = None,
) -> None:
    """Atomically replace ``path`` with owner-only text written in its directory.

    A failed write leaves an existing target untouched and removes the temporary
    file. ``before_write`` is for callers that need to validate permissions on
    the temporary path as part of their existing error contract.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    temp_path = Path(temp_name)
    replaced = False
    try:
        _restrict_fd(fd)
        if before_write is not None:
            before_write(temp_path)
        with os.fdopen(fd, "w", encoding=encoding) as handle:
            fd = -1
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, target)
        replaced = True
    finally:
        if fd >= 0:
            with contextlib.suppress(OSError):
                os.close(fd)
        if not replaced:
            with contextlib.suppress(OSError):
                temp_path.unlink()


def open_private_append_text(path: str | Path, *, encoding: str = "utf-8") -> TextIO:
    """Open an owner-only append stream without relying on the process umask."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NONBLOCK"):
        flags |= os.O_NONBLOCK
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(target, flags, _OWNER_ONLY_FILE_MODE)
    handle: TextIO | None = None
    try:
        _validate_private_append_target(fd, target)
        _restrict_fd(fd)
        handle = os.fdopen(fd, "a", encoding=encoding)
        return handle
    finally:
        if handle is None:
            with contextlib.suppress(OSError):
                os.close(fd)


def _validate_private_append_target(fd: int, target: Path) -> None:
    """Reject links, non-files, and path swaps before changing permissions."""
    opened = os.fstat(fd)
    current = os.stat(target, follow_symlinks=False)
    if stat.S_ISLNK(current.st_mode):
        raise OSError(errno.ELOOP, "refusing to append through a symbolic link", target)
    if not stat.S_ISREG(opened.st_mode):
        raise OSError(errno.EINVAL, "private append target is not a regular file", target)
    if (opened.st_dev, opened.st_ino) != (current.st_dev, current.st_ino):
        raise OSError(errno.ESTALE, "private append target changed while opening", target)
