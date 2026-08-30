"""Directory, rules, and PDF discovery helpers for the batch renamer."""

from __future__ import annotations

import fnmatch
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from ..naming.rules import ProcessingRules, load_processing_rules, should_skip_file_by_rules
from ..rename_ops import is_path_within
from ..settings import RenamerConfig

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


def _matches_patterns(name: str, include: list[str] | None, exclude: list[str] | None) -> bool:
    name_lower = name.lower()
    if include and not any(fnmatch.fnmatchcase(name_lower, pattern.lower()) for pattern in include):
        return False
    return not (exclude and any(fnmatch.fnmatchcase(name_lower, pattern.lower()) for pattern in exclude))


def _is_visible_pdf(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() == ".pdf" and not path.name.startswith(".") and not path.is_symlink()


def _is_within_max_depth(path: Path, root: Path, max_depth: int) -> bool:
    if max_depth <= 0:
        return True
    try:
        return max(0, len(path.relative_to(root).parts) - 1) <= max_depth
    except ValueError:
        return False


def _collect_override_candidates(directory: Path, files_override: list[Path]) -> list[Path]:
    candidates = [
        path
        for path in files_override
        if path.is_file()
        and path.suffix.lower() == ".pdf"
        and not path.is_symlink()
        and is_path_within(path, directory)
    ]
    if len(candidates) != len([path for path in files_override if path.suffix.lower() == ".pdf"]):
        logger.warning("Ignoring files_override PDFs outside selected directory or symbolic links in %s", directory)
    return candidates


def collect_pdf_files(directory: Path, options: PdfCollectionOptions | None = None) -> list[Path]:
    """Collect configured visible PDFs, rejecting source symlinks and applying filters."""
    opts = options or PdfCollectionOptions()
    if opts.files_override is not None:
        candidates = _collect_override_candidates(directory, opts.files_override)
    elif opts.recursive:
        candidates = [
            path
            for path in directory.rglob("*")
            if _is_visible_pdf(path) and _is_within_max_depth(path, directory, opts.max_depth)
        ]
    else:
        candidates = [path for path in directory.iterdir() if _is_visible_pdf(path)]
    candidates = [
        path for path in candidates if _matches_patterns(path.name, opts.include_patterns, opts.exclude_patterns)
    ]
    if opts.rules is not None:
        candidates = [path for path in candidates if not should_skip_file_by_rules(opts.rules, path.name)]
    if opts.skip_if_already_named:
        already_named = re.compile(r"^\d{8}-.+\.[pP][dD][fF]$")
        candidates = [path for path in candidates if not already_named.match(path.name)]
    return candidates


def _resolve_rename_directory(directory: str | Path, *, files_override: list[Path] | None) -> Path:
    """Validate and resolve the selected path, permitting a non-directory only with file overrides."""
    dir_str = str(directory).strip()
    if not dir_str:
        raise ValueError("Directory path must be non-empty. Use --dir or provide when prompted.")
    path = Path(directory)
    if not path.exists():
        raise FileNotFoundError(f"Directory does not exist: {path}")
    if files_override is None and not path.is_dir():
        raise NotADirectoryError(f"Not a directory: {path}")
    return path.resolve()


def _load_effective_rules(config: RenamerConfig, rules_override: ProcessingRules | None) -> ProcessingRules | None:
    """Use the supplied rules override or load the configured rules file."""
    if rules_override is not None:
        return rules_override
    rules_file = config.output.paths.rules_file
    return load_processing_rules(rules_file, raise_on_error=bool(rules_file))


def _mtime_key(path: Path) -> float:
    """Return modification time for sorting, using zero when stat fails."""
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def _collect_sorted_pdf_files_with(
    path: Path,
    config: RenamerConfig,
    *,
    files_override: list[Path] | None,
    rules: ProcessingRules | None,
    collect_pdf_files_fn: Callable[..., list[Path]],
) -> list[Path]:
    """Collect configured PDF candidates and sort them by newest modification time."""
    files = _collect_pdf_files_with_config(
        path,
        config,
        files_override=files_override,
        rules=rules,
        collect_pdf_files_fn=collect_pdf_files_fn,
    )
    files.sort(key=_mtime_key, reverse=True)
    return files


def _collect_pdf_files_with_config(
    path: Path,
    config: RenamerConfig,
    *,
    files_override: list[Path] | None,
    rules: ProcessingRules | None,
    collect_pdf_files_fn: Callable[..., list[Path]],
) -> list[Path]:
    """Translate traversal configuration into collection options and invoke the collector."""
    traversal = config.output.traversal
    return collect_pdf_files_fn(
        path,
        PdfCollectionOptions(
            recursive=traversal.recursive,
            max_depth=traversal.max_depth,
            include_patterns=traversal.include_patterns,
            exclude_patterns=traversal.exclude_patterns,
            skip_if_already_named=config.output.mode.skip_if_already_named,
            files_override=files_override,
            rules=rules,
        ),
    )
