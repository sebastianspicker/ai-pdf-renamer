"""Directory, rules, and PDF discovery helpers for the batch renamer."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from .config import RenamerConfig
from .renamer_files import PdfCollectionOptions
from .rules import ProcessingRules, load_processing_rules


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
