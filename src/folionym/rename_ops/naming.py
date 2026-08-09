"""Filename policy and value objects used by the rename workflow."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

# Path separators and control characters (including NUL) must not appear in names.
FILENAME_UNSAFE_RE = re.compile(r"[\x00-\x1f\x7f/\\:*?\"<>|]")
FILENAME_RESERVED_WIN = frozenset(
    {"CON", "PRN", "AUX", "NUL"} | {f"COM{i}" for i in range(1, 10)} | {f"LPT{i}" for i in range(1, 10)}
)
MAX_LLM_FILENAME_LEN = 120
MAX_RENAME_RETRIES = 20


@dataclass(frozen=True)
class RenameApplyOptions:
    """Options controlling backup, plan, dry-run, naming limits, and post-rename work."""

    plan_file_path: Path | str | None = None
    plan_entries: list[dict[str, str]] | None = None
    dry_run: bool = False
    backup_dir: Path | str | None = None
    on_success: Callable[[Path, Path, str], None] | None = None
    max_filename_chars: int | None = None
    exact_target: bool = False


@dataclass(frozen=True)
class RenameAttemptState:
    """Current collision candidate and suffix counter for a rename attempt."""

    current_base: str
    target: Path
    counter: int


@dataclass(frozen=True)
class RenameRetryContext:
    """Source, base name, and suffix retained across collision retries."""

    file_path: Path
    base: str
    suffix: str


def is_path_within(path: Path, root: Path) -> bool:
    """Return whether a resolved path is equal to or below a resolved root."""
    try:
        resolved = path.resolve()
        root_resolved = root.resolve()
        return resolved == root_resolved or resolved.is_relative_to(root_resolved)
    except OSError, ValueError:
        return False


def _validate_path_within_parent(path: Path, parent: Path) -> Path:
    """Ensure a resolved path stays under its expected parent."""
    resolved = path.resolve()
    parent_resolved = parent.resolve()
    if not is_path_within(path, parent):
        raise ValueError(f"Path traversal detected: {path} resolves to {resolved}, which is outside {parent_resolved}")
    return resolved


def sanitize_filename_base(name: str) -> str:
    """Remove unsafe characters, empty names, and Windows device names."""
    if not name or not name.strip():
        return "unnamed"
    safe = FILENAME_UNSAFE_RE.sub("", name.strip()).strip() or "unnamed"
    return f"{safe}_" if safe.upper() in FILENAME_RESERVED_WIN else safe


def sanitize_filename_from_llm(raw: str) -> str:
    """Sanitize LLM or vision output for use as a filename component."""
    if not raw or not isinstance(raw, str):
        return "document"
    cleaned = raw.strip()
    for character in '/\\:*?"<>|':
        cleaned = cleaned.replace(character, "_")
    cleaned = "_".join(" ".join(cleaned.replace("\n", " ").replace("\r", " ").split()).split(" "))
    if cleaned.lower().endswith(".pdf"):
        cleaned = cleaned[:-4]
    cleaned = cleaned.strip("._") or "document"
    return cleaned[:MAX_LLM_FILENAME_LEN]
