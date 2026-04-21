"""
Rename and filename-sanitization helpers.

Extracted from renamer to keep collision/sanitize logic in one place.
"""

from __future__ import annotations

import contextlib
import errno
import logging
import os
import re
import shutil
from collections.abc import Callable
from pathlib import Path

logger = logging.getLogger(__name__)

# Path separators and control characters (incl. NUL) that must not appear in filenames.
# This pattern is consolidated here and shared across modules.
FILENAME_UNSAFE_RE = re.compile(r"[\x00-\x1f\x7f/\\:*?\"<>|]")

# Reserved names on Windows (case-insensitive). Avoid using as base name to prevent EINVAL on rename.
FILENAME_RESERVED_WIN = frozenset(
    {"CON", "PRN", "AUX", "NUL"} | {f"COM{i}" for i in range(1, 10)} | {f"LPT{i}" for i in range(1, 10)}
)

# Max retries for rename when target exists. After this, fail with clear message.
MAX_RENAME_RETRIES = 20

# Max length for sanitized LLM/vision filename output.
MAX_LLM_FILENAME_LEN = 120


def is_path_within(path: Path, root: Path) -> bool:
    """Return True if resolved path is equal to or a descendant of root. Safe against symlink traversal."""
    try:
        resolved = path.resolve()
        root_resolved = root.resolve()
        return resolved == root_resolved or resolved.is_relative_to(root_resolved)
    except (OSError, ValueError):
        return False


def _validate_path_within_parent(path: Path, parent: Path) -> Path:
    """Ensure resolved path is within parent directory. Raises ValueError on traversal."""
    resolved = path.resolve()
    parent_resolved = parent.resolve()
    if not is_path_within(path, parent):
        raise ValueError(f"Path traversal detected: {path} resolves to {resolved}, which is outside {parent_resolved}")
    return resolved


def _next_available_path(path: Path, *, max_tries: int = 10_000) -> Path:
    """Return first non-existing sibling path using _N suffix before extension."""
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    for i in range(1, max_tries + 1):
        candidate = path.with_name(f"{stem}_{i}{suffix}")
        if not candidate.exists():
            return candidate
    raise OSError(
        errno.EEXIST,
        f"Could not create unique path for backup after {max_tries} attempts: {path}",
    )


def sanitize_filename_base(name: str) -> str:
    """Remove path separators and control chars; ensure non-empty; avoid Windows reserved names."""
    if not name or not name.strip():
        return "unnamed"
    safe = FILENAME_UNSAFE_RE.sub("", name.strip())
    safe = safe.strip() or "unnamed"
    if safe.upper() in FILENAME_RESERVED_WIN:
        return f"{safe}_"
    return safe


def sanitize_filename_from_llm(raw: str) -> str:
    """
    Sanitize raw LLM or vision API output for use as filename or content.
    Strips invalid chars, newlines, .pdf extension; collapses spaces to underscores; max 120 chars.
    Use this for vision/simple-naming output before using as content or filename part.
    sanitize_filename_base() remains for the final base name before rename.
    """
    if not raw or not isinstance(raw, str):
        return "document"
    s = raw.strip()
    for char in '/\\:*?"<>|':
        s = s.replace(char, "_")
    s = s.replace("\n", " ").replace("\r", " ")
    s = " ".join(s.split())
    s = s.replace(" ", "_")
    if s.lower().endswith(".pdf"):
        s = s[:-4]
    s = s.strip("._") or "document"
    return s[:MAX_LLM_FILENAME_LEN] if len(s) > MAX_LLM_FILENAME_LEN else s


def apply_single_rename(
    file_path: Path,
    base: str,
    *,
    plan_file_path: Path | str | None,
    plan_entries: list[dict[str, str]] | None,
    dry_run: bool,
    backup_dir: Path | str | None,
    on_success: Callable[[Path, Path, str], None] | None = None,
    max_filename_chars: int | None = None,
) -> tuple[bool, Path]:
    """
    Apply rename for one file: collision loop, backup, optional plan.
    Returns (success, final_target).

    Uses rename as the existence check to avoid TOCTOU. On FileExistsError, tries next suffix.
    EXDEV (cross-fs): best-effort copy+unlink.

    The retry loop handles collisions atomically: attempt the rename first, then increment
    the collision suffix on FileExistsError. This avoids the TOCTOU race that exists()
    pre-checks introduce. For reasonable collision counts this is safe even under concurrency.
    """
    suffix = file_path.suffix
    current_base = base
    target = file_path.with_name(base + suffix)
    _validate_path_within_parent(target, file_path.parent)
    counter = 0

    for _attempt in range(MAX_RENAME_RETRIES):
        try:
            if plan_file_path:
                if plan_entries is not None:
                    plan_entries.append({"old": str(file_path), "new": str(target)})
                logger.info("Plan: %s -> %s", file_path.name, target.name)
                return (True, target)

            if not dry_run:
                if backup_dir:
                    backup_path = Path(backup_dir) / file_path.name
                    _validate_path_within_parent(backup_path, Path(backup_dir))
                    backup_path.parent.mkdir(parents=True, exist_ok=True)
                    backup_path = _next_available_path(backup_path)
                    shutil.copy2(file_path, backup_path)

                # Cross-platform atomic rename attempt.
                # On Windows, os.rename fails if target exists.
                # On Unix, os.rename overwrites. We try os.link as a safer alternative (fails if exists).
                if os.name != "nt":
                    try:
                        os.link(file_path, target)
                        file_path.unlink()
                    except (AttributeError, OSError) as e:
                        # Fallback to os.rename if link is not supported or cross-FS (EXDEV).
                        if isinstance(e, OSError) and e.errno == errno.EEXIST:
                            raise FileExistsError from e
                        # os.link unsupported (e.g. EPERM on some filesystems).
                        # os.rename is atomic and will overwrite on Unix, so guard with
                        # an existence check first. A narrow TOCTOU window remains, but
                        # the old O_CREAT/unlink/rename sequence had the same window and
                        # created a visible placeholder file as a side-effect.
                        if target.exists():
                            raise FileExistsError(f"Target already exists: {target}")
                        os.rename(file_path, target)
                else:
                    os.rename(file_path, target)

                if on_success is not None:
                    on_success(file_path, target, current_base)

            return (True, target)

        except (FileExistsError, OSError) as e:
            # Catch FileExistsError (direct or from link/rename on some platforms)
            # or OSError with EEXIST/EACCES (Windows rename often raises EACCES for existing targets).
            is_exists = isinstance(e, FileExistsError) or (
                isinstance(e, OSError) and e.errno in (errno.EEXIST, getattr(errno, "EACCES", None))
            )

            if is_exists:
                counter += 1
                suffix_str = f"_{counter}"
                if max_filename_chars and max_filename_chars > len(suffix_str + suffix):
                    effective_base = base[: max_filename_chars - len(suffix_str + suffix)]
                else:
                    effective_base = base
                current_base = f"{effective_base}{suffix_str}"
                candidate_name = current_base + suffix
                # P2: Proactive filesystem name length check (255 bytes is common limit)
                if len(candidate_name.encode("utf-8")) > 255:
                    # Truncate base to fit within filesystem limit
                    max_base_bytes = 255 - len((suffix_str + suffix).encode("utf-8"))
                    truncated = effective_base.encode("utf-8")[:max_base_bytes].decode("utf-8", errors="ignore")
                    current_base = f"{truncated}{suffix_str}"
                target = file_path.with_name(current_base + suffix)
                continue

            # Handle length limit errors specifically.
            if getattr(errno, "ENAMETOOLONG", None) is not None and e.errno == errno.ENAMETOOLONG:
                raise OSError(
                    e.errno,
                    f"Filename too long for filesystem: {target.name!r}. "
                    "Shorten project/version or content-derived parts.",
                ) from e

            # Handle cross-filesystem rename (only if link/rename both failed with EXDEV).
            if e.errno == errno.EXDEV:
                if dry_run:
                    return (True, target)
                # Atomic existence check via O_CREAT|O_EXCL placeholder (closes TOCTOU window).
                try:
                    fd = os.open(str(target), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                    os.close(fd)
                except FileExistsError:
                    raise FileExistsError(f"Target already exists: {target}")
                except OSError:
                    # O_EXCL not supported on this FS; fall back to best-effort check
                    if target.exists():
                        raise FileExistsError(f"Target already exists: {target}")
                # Copy over the placeholder, then remove source
                try:
                    shutil.copy2(file_path, target)
                except OSError as copy_err:
                    with contextlib.suppress(OSError):
                        target.unlink()
                    raise copy_err
                try:
                    file_path.unlink()
                except OSError as unlink_err:
                    with contextlib.suppress(OSError):
                        target.unlink()
                    raise OSError(
                        f"Cross-filesystem rename: copied to {target}, "
                        f"could not remove source {file_path}: {unlink_err}"
                    ) from unlink_err
                if on_success is not None:
                    on_success(file_path, target, current_base)
                return (True, target)
            raise

    logger.error(
        "Rename failed after %d attempts (target already exists): %s -> %s",
        MAX_RENAME_RETRIES,
        file_path,
        target,
    )
    return (False, target)
