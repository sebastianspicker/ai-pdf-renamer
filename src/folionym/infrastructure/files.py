"""Filesystem safety checks shared across application boundaries."""

from __future__ import annotations

import errno
from pathlib import Path


def reject_source_symlink(path: Path) -> None:
    """Reject a visible source symlink before extraction or mutation."""
    if path.is_symlink():
        raise OSError(errno.ELOOP, "Source PDF symbolic links are not supported", path)
