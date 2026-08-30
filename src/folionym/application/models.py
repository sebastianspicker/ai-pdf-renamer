"""Typed application values shared by batch and reviewed rename workflows."""

from __future__ import annotations

import stat
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path

from ..settings import RenamerConfig


class PreviewStatus(StrEnum):
    """Operator-facing state for a generated or reviewed rename proposal."""

    READY = "ready"
    REVIEW = "review"
    SKIPPED = "skipped"
    FAILED = "failed"


class ApplyStatus(StrEnum):
    """Outcome of one reviewed-plan apply item."""

    RENAMED = "renamed"
    SKIPPED = "skipped"
    UNCHANGED = "unchanged"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ApplyPolicy(StrEnum):
    """Collision policy for applying a proposal."""

    UNIQUE_AVAILABLE = "unique_available"
    EXACT_REVIEWED = "exact_reviewed"


@dataclass(frozen=True)
class Proposal:
    """A typed extraction and naming outcome before any filesystem mutation."""

    source: Path
    proposed_base: str | None
    metadata: dict[str, object] | None
    error: BaseException | None = None

    @property
    def status(self) -> PreviewStatus:
        """Classify success, an empty proposal, and recoverable failure."""
        if self.error is not None:
            return PreviewStatus.FAILED
        if self.proposed_base is None:
            return PreviewStatus.SKIPPED
        return PreviewStatus.READY


@dataclass(frozen=True)
class FileFingerprint:
    """Filesystem identity used to reject stale reviewed plans."""

    device: int
    inode: int
    size: int
    modified_ns: int

    @classmethod
    def capture(cls, path: Path) -> FileFingerprint:
        """Capture a regular, non-symlink file identity."""
        current = path.stat(follow_symlinks=False)
        if not stat.S_ISREG(current.st_mode):
            raise OSError(f"Source is not a regular file: {path}")
        return cls(current.st_dev, current.st_ino, current.st_size, current.st_mtime_ns)

    def matches(self, path: Path) -> bool:
        """Return whether ``path`` still identifies the reviewed file bytes."""
        try:
            return self == self.capture(path)
        except OSError:
            return False


@dataclass(frozen=True)
class PreviewItem:
    """One structured source-to-proposal result for a reviewed plan."""

    id: str
    source: Path
    proposed_base: str | None
    metadata: dict[str, object]
    status: PreviewStatus
    included: bool
    fingerprint: FileFingerprint | None
    reason: str | None = None

    @property
    def proposed_name(self) -> str | None:
        """Return the complete proposed filename when a proposal exists."""
        return None if self.proposed_base is None else self.proposed_base + self.source.suffix


@dataclass(frozen=True)
class PreviewPlan:
    """Immutable reviewed-plan input retained by the local process."""

    id: str
    source: Path
    source_kind: str
    config: RenamerConfig
    items: tuple[PreviewItem, ...]
    created_at: datetime
    revision: int = 1


@dataclass(frozen=True)
class ApplyItemResult:
    """One reviewed-plan apply outcome."""

    item_id: str
    source_name: str
    target_name: str | None
    status: ApplyStatus
    reason: str | None = None


@dataclass(frozen=True)
class ApplyReport:
    """Complete result of applying a selected subset of a preview plan."""

    id: str
    plan_id: str
    source: Path
    started_at: datetime
    completed_at: datetime
    items: tuple[ApplyItemResult, ...]

    def count(self, status: ApplyStatus) -> int:
        """Count outcomes with ``status``."""
        return sum(item.status == status for item in self.items)
