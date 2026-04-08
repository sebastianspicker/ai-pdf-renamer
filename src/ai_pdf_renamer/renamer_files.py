"""File collection and path filtering helpers for renamer."""

from __future__ import annotations

import fnmatch
import re
from pathlib import Path

from .rename_ops import is_path_within
from .rules import ProcessingRules, should_skip_file_by_rules


def matches_patterns(name: str, include: list[str] | None, exclude: list[str] | None) -> bool:
    """True if basename matches include (if set) and does not match any exclude."""
    # P2: Use case-insensitive matching with lowered inputs
    name_lower = name.lower()
    if include is not None and include and not any(fnmatch.fnmatchcase(name_lower, p.lower()) for p in include):
        return False
    return not (exclude is not None and exclude and any(fnmatch.fnmatchcase(name_lower, p.lower()) for p in exclude))


def _is_safe_path(path: Path, root: Path) -> bool:
    """Check that path is not a symlink pointing outside the root directory."""
    if not path.is_symlink():
        return True
    return is_path_within(path, root)


def collect_pdf_files(
    directory: Path,
    *,
    recursive: bool = False,
    max_depth: int = 0,
    include_patterns: list[str] | None = None,
    exclude_patterns: list[str] | None = None,
    skip_if_already_named: bool = False,
    files_override: list[Path] | None = None,
    rules: ProcessingRules | None = None,
) -> list[Path]:
    """Collect PDFs from directory (or files_override). Rules skip_files_by_pattern filters out matches."""
    if files_override is not None:
        candidates = [p for p in files_override if p.is_file() and p.suffix.lower() == ".pdf"]
    elif recursive:
        candidates = []
        for p in directory.rglob("*.pdf"):
            if not p.is_file() or p.name.startswith("."):
                continue
            # P2: Skip symlinks pointing outside the directory tree
            if not _is_safe_path(p, directory):
                continue
            if max_depth > 0:
                try:
                    rel = p.relative_to(directory)
                    # rel.parts includes the filename; depth counts directories only.
                    depth = max(0, len(rel.parts) - 1)
                    if depth > max_depth:
                        continue
                except ValueError:
                    continue
            candidates.append(p)
    else:
        candidates = [
            p
            for p in directory.iterdir()
            if p.is_file() and p.suffix.lower() == ".pdf" and not p.name.startswith(".") and _is_safe_path(p, directory)
        ]

    out = [p for p in candidates if matches_patterns(p.name, include_patterns, exclude_patterns)]
    if rules is not None:
        out = [p for p in out if not should_skip_file_by_rules(rules, p.name)]
    if skip_if_already_named:
        already_named = re.compile(r"^\d{8}-.+\.[pP][dD][fF]$")
        out = [p for p in out if not already_named.match(p.name)]
    return out
