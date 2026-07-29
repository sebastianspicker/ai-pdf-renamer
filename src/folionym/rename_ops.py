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
import stat
from collections.abc import Callable
from dataclasses import dataclass
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

_OWNER_ONLY_DIRECTORY_MODE = stat.S_IRWXU
_OWNER_ONLY_FILE_MODE = stat.S_IRUSR | stat.S_IWUSR


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
    """Return True if resolved path is equal to or a descendant of root. Safe against symlink traversal."""
    # fmt: off
    try:
        resolved = path.resolve()
        root_resolved = root.resolve()
        return resolved == root_resolved or resolved.is_relative_to(root_resolved)
    except (OSError, ValueError):
        return False
    # fmt: on


def _validate_path_within_parent(path: Path, parent: Path) -> Path:
    """Ensure resolved path is within parent directory. Raises ValueError on traversal."""
    resolved = path.resolve()
    parent_resolved = parent.resolve()
    if not is_path_within(path, parent):
        raise ValueError(f"Path traversal detected: {path} resolves to {resolved}, which is outside {parent_resolved}")
    return resolved


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
    """Record a plan entry when plan mode is enabled and report that no rename is needed."""
    if not plan_file_path:
        return False
    if plan_entries is not None:
        plan_entries.append({"old": str(file_path), "new": str(target)})
    logger.info("Plan: %s -> %s", file_path.name, target.name)
    return True


def _write_backup(file_path: Path, backup_dir: Path | str | None) -> None:
    """Create a unique private backup before renaming when configured."""
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
    """Open the backup directory with O_NOFOLLOW when available and restrict it to mode 0700."""
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
    """Reserve a unique backup name relative to the open directory and write the copy."""
    for counter in range(10_001):
        candidate_name = file_path.name if counter == 0 else f"{file_path.stem}_{counter}{file_path.suffix}"
        try:
            fd = _open_exclusive_target(candidate_name, mode=_OWNER_ONLY_FILE_MODE, dir_fd=directory_fd)
        except FileExistsError:
            continue
        _write_reserved_backup(file_path, backup_path, candidate_name, directory_fd, fd)
        return
    raise OSError(
        errno.EEXIST,
        f"Could not create unique path for backup after 10000 attempts: {backup_path}",
    )


def _write_reserved_backup(
    file_path: Path,
    backup_path: Path,
    candidate_name: str,
    directory_fd: int,
    fd: int,
) -> None:
    """Copy into a reserved backup descriptor and remove only the unchanged reservation on failure."""
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
    """Create a unique private backup by pathname on Windows and validate its identity."""
    for counter in range(10_001):
        candidate = (
            backup_path if counter == 0 else backup_path.with_name(f"{file_path.stem}_{counter}{file_path.suffix}")
        )
        try:
            fd = _open_exclusive_target(candidate, mode=_OWNER_ONLY_FILE_MODE)
        except FileExistsError:
            continue
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
        return
    raise OSError(errno.EEXIST, f"Could not create unique path for backup after 10000 attempts: {backup_path}")


def _open_exclusive_target(
    path: Path | str,
    *,
    mode: int = _OWNER_ONLY_FILE_MODE,
    dir_fd: int | None = None,
) -> int:
    """Create a new write-only target exclusively, adding O_NOFOLLOW when supported."""
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    kwargs: dict[str, int] = {} if dir_fd is None else {"dir_fd": dir_fd}
    return os.open(path, flags, mode, **kwargs)


def _copy_file_to_fd(file_path: Path, target_fd: int) -> os.stat_result:
    """Copy content and basic metadata through an already-reserved target descriptor.

    ACLs and extended attributes deliberately remain out of scope: the stdlib has
    no portable descriptor-only API for them, while pathname-based metadata
    copying would reintroduce the replacement race this helper avoids.
    """
    with file_path.open("rb") as source, os.fdopen(os.dup(target_fd), "wb") as target_file:
        source_status = os.fstat(source.fileno())
        shutil.copyfileobj(source, target_file)
    if os.name != "nt":
        os.fchmod(target_fd, stat.S_IMODE(source_status.st_mode))
        os.utime(target_fd, ns=(source_status.st_atime_ns, source_status.st_mtime_ns))
    return source_status


def _directory_entry_matches_identity(name: str, directory_fd: int, identity: os.stat_result) -> bool:
    """Return whether a directory entry still names the reserved non-symlink inode."""
    try:
        path_status = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    except OSError:
        return False
    return (
        not stat.S_ISLNK(path_status.st_mode)
        and path_status.st_dev == identity.st_dev
        and path_status.st_ino == identity.st_ino
    )


def _cleanup_reserved_directory_entry(name: str, directory_fd: int, identity: os.stat_result) -> None:
    """Best-effort unlink the reserved entry only while its identity still matches."""
    if _directory_entry_matches_identity(name, directory_fd, identity):
        with contextlib.suppress(OSError):
            os.unlink(name, dir_fd=directory_fd)


def _path_matches_identity(path: Path, identity: os.stat_result) -> bool:
    """Return whether a path still names the reserved non-symlink inode."""
    try:
        path_status = path.lstat()
    except OSError:
        return False
    return (
        not stat.S_ISLNK(path_status.st_mode)
        and path_status.st_dev == identity.st_dev
        and path_status.st_ino == identity.st_ino
    )


def _cleanup_reserved_target(target: Path, identity: os.stat_result) -> None:
    """Remove only the path entry still referring to the reserved file."""
    if _path_matches_identity(target, identity):
        with contextlib.suppress(OSError):
            target.unlink()


def _rename_without_overwrite(file_path: Path, target: Path) -> None:
    """Rename without overwrite using Windows rename, a POSIX hard link, or reserved-copy fallback."""
    if os.name == "nt":
        os.rename(file_path, target)
        return

    # Keep the original inode alive until the source pathname has been checked
    # and removed. Without this descriptor, a just-unlinked source can have its
    # inode reused immediately, making a replacement look identical by dev/ino.
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
    """Try a no-overwrite hard link; return the source identity on success or None for non-collision failures."""
    try:
        os.link(file_path, target)
    except (AttributeError, OSError) as exc:
        if isinstance(exc, OSError) and exc.errno == errno.EEXIST:
            raise FileExistsError from exc
        return None
    return source_identity


def _validate_link_and_unlink_source(file_path: Path, target: Path, source_identity: os.stat_result) -> None:
    """Validate both entries before removing the source of a hard-link rename."""
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
    """Remove a mismatched target only when it still links the current source."""
    try:
        target_identity = target.lstat()
    except OSError:
        return
    if _path_matches_identity(file_path, target_identity):
        _cleanup_reserved_target(target, target_identity)


def _copy_to_reserved_target_then_unlink(file_path: Path, target: Path, source_fd: int | None = None) -> None:
    """Copy into an exclusive target, validate both identities, then unlink the source."""
    owns_source_fd = source_fd is None
    if source_fd is None:
        source_fd = os.open(file_path, os.O_RDONLY)

    def _release_windows_source() -> None:
        """Release the Windows source handle immediately before deleting its path."""
        nonlocal source_fd
        if source_fd is None:
            return
        os.close(source_fd)
        source_fd = None

    try:
        _copy_with_pinned_source(
            file_path,
            target,
            source_fd,
            before_unlink=_release_windows_source if owns_source_fd and os.name == "nt" else None,
        )
    finally:
        if owns_source_fd and source_fd is not None:
            os.close(source_fd)


def _copy_with_pinned_source(
    file_path: Path,
    target: Path,
    source_fd: int,
    *,
    before_unlink: Callable[[], None] | None,
) -> None:
    """Reserve the target and copy while the original source inode remains pinned."""
    held_identity = os.fstat(source_fd)
    fd = _reserve_copy_target(target)
    try:
        target_identity = os.fstat(fd)
        _copy_and_validate_reserved_target(file_path, target, fd, held_identity, target_identity)
        if before_unlink is not None:
            try:
                before_unlink()
            except OSError:
                _cleanup_reserved_target(target, target_identity)
                raise
        _unlink_copied_source(file_path, target, target_identity)
    finally:
        os.close(fd)


def _reserve_copy_target(target: Path) -> int:
    """Reserve a no-overwrite target and preserve collision semantics."""
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
    """Copy and validate both path identities while the source remains pinned."""
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


def _unlink_copied_source(file_path: Path, target: Path, target_identity: os.stat_result) -> None:
    """Remove the validated source and clean up the reserved target on failure."""
    try:
        file_path.unlink()
    except OSError as unlink_err:
        _cleanup_reserved_target(target, target_identity)
        raise OSError(
            f"Cross-filesystem rename: copied to {target}, could not remove source {file_path}: {unlink_err}"
        ) from unlink_err


def _is_target_exists_error(exc: BaseException) -> bool:
    """Return whether an exception represents a target collision on this platform."""
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
    """Build the next collision-suffixed target within character and byte limits."""
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
    """Complete a cross-filesystem rename through reserved copy-and-unlink."""
    if dry_run:
        return (True, target)
    _copy_to_reserved_target_then_unlink(file_path, target)
    _notify_rename_success(on_success, file_path, target, current_base)
    return (True, target)


def apply_single_rename(
    file_path: Path,
    base: str,
    options: RenameApplyOptions,
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
    state = RenameAttemptState(base, file_path.with_name(base + suffix), 0)
    retry_context = RenameRetryContext(file_path, base, suffix)
    _validate_path_within_parent(state.target, file_path.parent)

    if not options.dry_run and not options.plan_file_path:
        _write_backup(file_path, options.backup_dir)

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
    """Record, simulate, or attempt one no-overwrite rename candidate."""
    if (options.dry_run or options.plan_file_path) and os.path.lexists(state.target):
        # A dangling symlink still occupies the directory entry even though
        # Path.exists() follows it and reports false.
        raise FileExistsError(f"Target already exists: {state.target}")

    if _record_rename_plan(
        file_path,
        state.target,
        plan_file_path=options.plan_file_path,
        plan_entries=options.plan_entries,
    ):
        return (True, state.target)

    if not options.dry_run:
        _rename_without_overwrite(file_path, state.target)
        _notify_rename_success(options.on_success, file_path, state.target, state.current_base)

    return (True, state.target)


def _notify_rename_success(
    on_success: Callable[[Path, Path, str], None] | None,
    file_path: Path,
    target: Path,
    current_base: str,
) -> None:
    """Run post-rename work without changing the completed rename outcome."""
    if on_success is None:
        return
    try:
        on_success(file_path, target, current_base)
    except Exception:  # pylint: disable=broad-exception-caught
        logger.exception("Rename completed but its post-rename callback failed: %s -> %s", file_path, target)


def _handle_rename_attempt_error(
    retry_context: RenameRetryContext,
    state: RenameAttemptState,
    options: RenameApplyOptions,
    error: FileExistsError | OSError,
) -> RenameAttemptState | tuple[bool, Path]:
    """Advance after a collision, translate filename-length errors, or handle EXDEV."""
    if _is_target_exists_error(error):
        if options.exact_target:
            return (False, state.target)
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
    """Return whether an OSError is ENAMETOOLONG on this platform."""
    return getattr(errno, "ENAMETOOLONG", None) is not None and error.errno == errno.ENAMETOOLONG
