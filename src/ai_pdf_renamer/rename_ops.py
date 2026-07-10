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
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any

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


@dataclass(frozen=True)
class RenameApplyOptions:
    plan_file_path: Path | str | None = None
    plan_entries: list[dict[str, str]] | None = None
    dry_run: bool = False
    backup_dir: Path | str | None = None
    on_success: Callable[[Path, Path, str], None] | None = None
    max_filename_chars: int | None = None


@dataclass(frozen=True)
class RenameAttemptState:
    current_base: str
    target: Path
    counter: int


@dataclass(frozen=True)
class RenameRetryContext:
    file_path: Path
    base: str
    suffix: str


def _merge_apply_options(
    options: RenameApplyOptions | None,
    overrides: dict[str, Any],
) -> RenameApplyOptions:
    base = RenameApplyOptions() if options is None else options
    if not overrides:
        return base
    allowed = {field.name for field in fields(base)}
    unknown = set(overrides) - allowed
    if unknown:
        joined = ", ".join(sorted(unknown))
        raise TypeError(f"Unknown option(s): {joined}")
    return replace(base, **overrides)


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


def _record_rename_plan(
    file_path: Path,
    target: Path,
    *,
    plan_file_path: Path | str | None,
    plan_entries: list[dict[str, str]] | None,
) -> bool:
    if not plan_file_path:
        return False
    if plan_entries is not None:
        plan_entries.append({"old": str(file_path), "new": str(target)})
    logger.info("Plan: %s -> %s", file_path.name, target.name)
    return True


def _write_backup(file_path: Path, backup_dir: Path | str | None) -> None:
    if not backup_dir:
        return
    backup_root = Path(backup_dir)
    backup_path = backup_root / file_path.name
    _validate_path_within_parent(backup_path, backup_root)
    backup_path.parent.mkdir(parents=True, exist_ok=True)
    backup_path = _next_available_path(backup_path)
    shutil.copy2(file_path, backup_path)


def _rename_without_overwrite(file_path: Path, target: Path) -> None:
    if os.name == "nt":
        os.rename(file_path, target)
        return
    try:
        os.link(file_path, target)
        file_path.unlink()
    except (AttributeError, OSError) as exc:
        if isinstance(exc, OSError) and exc.errno == errno.EEXIST:
            raise FileExistsError from exc
        _reserve_target_then_rename(file_path, target)


def _reserve_target_then_rename(file_path: Path, target: Path) -> None:
    try:
        fd = os.open(str(target), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
    except OSError as open_exc:
        if open_exc.errno == errno.EEXIST:
            raise FileExistsError(f"Target already exists: {target}") from open_exc
        raise
    try:
        os.rename(file_path, target)
    except OSError:
        with contextlib.suppress(OSError):
            target.unlink()
        raise


def _is_target_exists_error(exc: BaseException) -> bool:
    return isinstance(exc, FileExistsError) or (
        isinstance(exc, OSError)
        and (exc.errno == errno.EEXIST or (os.name == "nt" and exc.errno == getattr(errno, "EACCES", None)))
    )


def _collision_target(
    file_path: Path,
    base: str,
    suffix: str,
    counter: int,
    max_filename_chars: int | None,
) -> tuple[str, Path]:
    suffix_str = f"_{counter}"
    if max_filename_chars and max_filename_chars > len(suffix_str + suffix):
        effective_base = base[: max_filename_chars - len(suffix_str + suffix)]
    else:
        effective_base = base
    current_base = f"{effective_base}{suffix_str}"
    candidate_name = current_base + suffix
    if len(candidate_name.encode("utf-8")) > 255:
        max_base_bytes = 255 - len((suffix_str + suffix).encode("utf-8"))
        truncated = effective_base.encode("utf-8")[:max_base_bytes].decode("utf-8", errors="ignore")
        current_base = f"{truncated}{suffix_str}"
    return (current_base, file_path.with_name(current_base + suffix))


def _handle_cross_filesystem_rename(
    file_path: Path,
    target: Path,
    *,
    dry_run: bool,
    on_success: Callable[[Path, Path, str], None] | None,
    current_base: str,
) -> tuple[bool, Path]:
    if dry_run:
        return (True, target)
    try:
        fd = os.open(str(target), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
    except FileExistsError as exc:
        raise FileExistsError(f"Target already exists: {target}") from exc
    except OSError as exc:
        if target.exists():
            raise FileExistsError(f"Target already exists: {target}") from exc
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
            f"Cross-filesystem rename: copied to {target}, could not remove source {file_path}: {unlink_err}"
        ) from unlink_err
    if on_success is not None:
        on_success(file_path, target, current_base)
    return (True, target)


def apply_single_rename(
    file_path: Path,
    base: str,
    options: RenameApplyOptions | None = None,
    **overrides: Any,
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
    options = _merge_apply_options(options, overrides)
    suffix = file_path.suffix
    state = RenameAttemptState(base, file_path.with_name(base + suffix), 0)
    retry_context = RenameRetryContext(file_path, base, suffix)
    _validate_path_within_parent(state.target, file_path.parent)

    for _attempt in range(MAX_RENAME_RETRIES):
        try:
            return _apply_rename_attempt(file_path, state, options)
        except (FileExistsError, OSError) as e:
            handled = _handle_rename_attempt_error(retry_context, state, options, e)
            if isinstance(handled, RenameAttemptState):
                state = handled
                continue
            return handled

    logger.error(
        "Rename failed after %d attempts (target already exists): %s -> %s",
        MAX_RENAME_RETRIES,
        file_path,
        state.target,
    )
    return (False, state.target)


def _apply_rename_attempt(
    file_path: Path,
    state: RenameAttemptState,
    options: RenameApplyOptions,
) -> tuple[bool, Path]:
    if _record_rename_plan(
        file_path,
        state.target,
        plan_file_path=options.plan_file_path,
        plan_entries=options.plan_entries,
    ):
        return (True, state.target)

    if not options.dry_run:
        _write_backup(file_path, options.backup_dir)
        _rename_without_overwrite(file_path, state.target)
        if options.on_success is not None:
            options.on_success(file_path, state.target, state.current_base)

    return (True, state.target)


def _handle_rename_attempt_error(
    retry_context: RenameRetryContext,
    state: RenameAttemptState,
    options: RenameApplyOptions,
    error: FileExistsError | OSError,
) -> RenameAttemptState | tuple[bool, Path]:
    if _is_target_exists_error(error):
        counter = state.counter + 1
        current_base, target = _collision_target(
            retry_context.file_path,
            retry_context.base,
            retry_context.suffix,
            counter,
            options.max_filename_chars,
        )
        return RenameAttemptState(current_base, target, counter)
    if _is_filename_too_long_error(error):
        raise OSError(
            error.errno,
            f"Filename too long for filesystem: {state.target.name!r}. "
            "Shorten project/version or content-derived parts.",
        ) from error
    if error.errno == errno.EXDEV:
        return _handle_cross_filesystem_rename(
            retry_context.file_path,
            state.target,
            dry_run=options.dry_run,
            on_success=options.on_success,
            current_base=state.current_base,
        )
    raise error


def _is_filename_too_long_error(error: OSError) -> bool:
    return getattr(errno, "ENAMETOOLONG", None) is not None and error.errno == errno.ENAMETOOLONG
