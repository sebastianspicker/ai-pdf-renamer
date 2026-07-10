"""Directory, rules, and PDF discovery helpers for the batch renamer."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from .config import RenamerConfig
from .rules import ProcessingRules, load_processing_rules


def _resolve_rename_directory(directory: str | Path, *, files_override: list[Path] | None) -> Path:
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
    if rules_override is not None:
        return rules_override
    return load_processing_rules(config.rules_file, raise_on_error=bool(config.rules_file))


def _mtime_key(path: Path) -> float:
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
    return collect_pdf_files_fn(
        path,
        recursive=config.recursive,
        max_depth=config.max_depth,
        include_patterns=config.include_patterns,
        exclude_patterns=config.exclude_patterns,
        skip_if_already_named=config.skip_if_already_named,
        files_override=files_override,
        rules=rules,
    )
