"""File collection and path filtering helpers for renamer."""

from __future__ import annotations

import errno
import fnmatch
import logging
import re
from dataclasses import dataclass
from pathlib import Path

from .rename_ops import is_path_within
from .rules import ProcessingRules, should_skip_file_by_rules

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PdfCollectionOptions:
    """Options controlling PDF discovery and post-discovery filtering."""

    recursive: bool = False
    max_depth: int = 0
    include_patterns: list[str] | None = None
    exclude_patterns: list[str] | None = None
    skip_if_already_named: bool = False
    files_override: list[Path] | None = None
    rules: ProcessingRules | None = None


def matches_patterns(name: str, include: list[str] | None, exclude: list[str] | None) -> bool:
    """True if basename matches include (if set) and does not match any exclude."""
    # Operators write file globs by eye; matching should not depend on PDF name casing.
    name_lower = name.lower()
    if include is not None and include and not any(fnmatch.fnmatchcase(name_lower, p.lower()) for p in include):
        return False
    return not (exclude is not None and exclude and any(fnmatch.fnmatchcase(name_lower, p.lower()) for p in exclude))


def _is_safe_path(path: Path, _root: Path) -> bool:
    """Return whether the candidate's final path component is not a symlink."""
    return not path.is_symlink()


def reject_source_symlink(path: Path) -> None:
    """Reject a currently visible source symlink; callers must still handle later replacement races."""
    if path.is_symlink():
        raise OSError(errno.ELOOP, "Source PDF symbolic links are not supported", path)


def _collect_override_candidates(directory: Path, files_override: list[Path]) -> list[Path]:
    """Select regular PDF overrides inside the directory and warn about rejected paths."""
    candidates = []
    rejected = []
    for p in files_override:
        if not p.is_file() or p.suffix.lower() != ".pdf":
            continue
        if _is_safe_path(p, directory) and is_path_within(p, directory):
            candidates.append(p)
        else:
            rejected.append(p)
    if rejected:
        logger.warning(
            "Ignoring files_override PDFs outside selected directory or symbolic links in %s: %s",
            directory,
            ", ".join(str(p) for p in rejected),
        )
    return candidates


def _is_within_max_depth(path: Path, root: Path, max_depth: int) -> bool:
    """Return whether the candidate's directory depth is within the configured limit."""
    if max_depth <= 0:
        return True
    try:
        rel = path.relative_to(root)
    except ValueError:
        return False
    # rel.parts includes the filename; depth counts directories only.
    return max(0, len(rel.parts) - 1) <= max_depth


def _is_visible_pdf(path: Path) -> bool:
    """Return whether the path currently resolves to a visible regular PDF."""
    return path.is_file() and path.suffix.lower() == ".pdf" and not path.name.startswith(".")


def _collect_recursive_candidates(directory: Path, max_depth: int) -> list[Path]:
    """Discover visible non-symlink PDFs recursively within the maximum depth."""
    candidates = []
    for p in directory.rglob("*"):
        if not _is_visible_pdf(p):
            continue
        # Renaming a symlink entry safely without overwrite is not portable.
        if not _is_safe_path(p, directory):
            continue
        if not _is_within_max_depth(p, directory, max_depth):
            continue
        candidates.append(p)
    return candidates


def _collect_direct_candidates(directory: Path) -> list[Path]:
    """Discover visible non-symlink PDFs directly below the directory."""
    return [p for p in directory.iterdir() if _is_visible_pdf(p) and _is_safe_path(p, directory)]


def _collect_candidates(directory: Path, options: PdfCollectionOptions) -> list[Path]:
    """Choose explicit, recursive, or direct candidate discovery from the options."""
    if options.files_override is not None:
        return _collect_override_candidates(directory, options.files_override)
    if options.recursive:
        return _collect_recursive_candidates(directory, options.max_depth)
    return _collect_direct_candidates(directory)


def _filter_by_patterns(candidates: list[Path], options: PdfCollectionOptions) -> list[Path]:
    """Apply include and exclude globs without reordering candidates."""
    return [
        path for path in candidates if matches_patterns(path.name, options.include_patterns, options.exclude_patterns)
    ]


def _filter_by_rules(candidates: list[Path], rules: ProcessingRules | None) -> list[Path]:
    """Exclude candidates matched by processing rules without reordering them."""
    if rules is None:
        return candidates
    return [path for path in candidates if not should_skip_file_by_rules(rules, path.name)]


def _filter_already_named(candidates: list[Path], skip_if_already_named: bool) -> list[Path]:
    """Optionally exclude files already matching the YYYYMMDD-*.pdf convention."""
    if not skip_if_already_named:
        return candidates
    already_named = re.compile(r"^\d{8}-.+\.[pP][dD][fF]$")
    return [path for path in candidates if not already_named.match(path.name)]


def _filter_candidates(candidates: list[Path], options: PdfCollectionOptions) -> list[Path]:
    """Apply pattern, rules, and already-named filters in sequence."""
    out = _filter_by_patterns(candidates, options)
    out = _filter_by_rules(out, options.rules)
    return _filter_already_named(out, options.skip_if_already_named)


def collect_pdf_files(
    directory: Path,
    options: PdfCollectionOptions | None = None,
) -> list[Path]:
    """Collect PDFs from directory (or files_override). Rules skip_files_by_pattern filters out matches."""
    resolved_options = options or PdfCollectionOptions()
    return _filter_candidates(_collect_candidates(directory, resolved_options), resolved_options)
