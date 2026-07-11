"""Batch rename orchestration and side-effect boundary.

This module coordinates discovery, extraction, filename generation, output
artifacts, optional hooks, and watch mode. Pure naming decisions live in
filename.py; filesystem mutation is delegated to rename_ops.py.
"""

from __future__ import annotations

import json
import logging
import os
import signal
import time
from collections.abc import Callable
from concurrent.futures import CancelledError, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from pathlib import Path

import requests

from . import loaders as _loaders
from .config import RenamerConfig as RenamerConfig
from .filename import FilenameGenerationRequest, generate_filename
from .pdf_extract import get_pdf_metadata
from .recoverable_errors import COMMON_RECOVERABLE_EXCEPTIONS
from .rename_ops import (
    MAX_RENAME_RETRIES,
    apply_single_rename,
    sanitize_filename_base,
)
from .renamer_discovery import (
    _collect_sorted_pdf_files_with,
    _load_effective_rules,
    _resolve_rename_directory,
)
from .renamer_extract import extract_pdf_content as _extract_pdf_content_default
from .renamer_files import collect_pdf_files as _collect_pdf_files
from .renamer_hooks import PostRenameAction
from .renamer_hooks import _apply_post_rename_actions as _apply_post_rename_actions
from .renamer_hooks import _make_post_rename_success_callback as _make_post_rename_success_callback
from .renamer_hooks import _run_post_rename_hook as _run_post_rename_hook
from .renamer_hooks import _write_pdf_title_metadata as _write_pdf_title_metadata
from .renamer_interactive import _interactive_rename_prompt
from .renamer_lookup import _lookup_override_category
from .renamer_output import (
    RenameOutputData,
    RenameSummaryData,
    _append_export_row,
    _sanitize_csv_cell,
    _write_rename_outputs,
    _write_summary_json,
)
from .renamer_output import _write_json_or_csv as _write_json_or_csv
from .renamer_parallel import (
    ParallelRenameDependencies,
    ParallelRenameRequest,
    RenameResult,
    produce_parallel_rename_results,
)
from .renamer_progress import _create_progress_reporter, _NullProgressReporter, _RichProgressReporter
from .renamer_watch import WatchLoopDependencies, run_watch_loop_impl
from .rules import (
    ProcessingRules,
    force_category_for_basename,
    load_processing_rules,
)

logger = logging.getLogger(__name__)
collect_pdf_files = _collect_pdf_files
_compat_aliases = (
    PostRenameAction,
    _NullProgressReporter,
    _RichProgressReporter,
    _apply_post_rename_actions,
    _run_post_rename_hook,
    _sanitize_csv_cell,
    _write_json_or_csv,
    _write_pdf_title_metadata,
    os,
    requests,
    signal,
    time,
)
_heuristic_scorer_cached = _loaders.heuristic_scorer_cached
_stopwords_cached = _loaders.stopwords_cached
_RECOVERABLE_RENAME_EXCEPTIONS = (
    AttributeError,
    CancelledError,
    *COMMON_RECOVERABLE_EXCEPTIONS,
)


def _stop_requested(config: RenamerConfig) -> bool:
    stop_event = config.stop_event
    return bool(stop_event is not None and hasattr(stop_event, "is_set") and stop_event.is_set())


def _extract_pdf_content(path: Path, config: RenamerConfig) -> tuple[str, bool]:
    """Compatibility shim for the extraction pipeline.
    The strategy logic lives in renamer_extract.py; this wrapper keeps the
    renamer-level patch points stable for tests and local overrides.
    """
    return _extract_pdf_content_default(path, config)


def _process_content_to_result(
    file_path: Path,
    content: str,
    config: RenamerConfig,
    rules: ProcessingRules | None = None,
    used_vision: bool = False,
) -> tuple[Path, str | None, dict[str, object] | None, BaseException | None]:
    """
    Generate filename from already-extracted content. Returns (path, new_base, meta, error).
    Caller must ensure content is non-empty if expecting a non-skip result.
    """
    try:
        override_cat = _lookup_override_category(
            file_path, config.override_category_map
        ) or force_category_for_basename(rules, file_path.name)
        pdf_meta = get_pdf_metadata(file_path) if config.use_pdf_metadata_for_date else None
        filename_str, meta = generate_filename(
            content,
            FilenameGenerationRequest(
                config=config,
                override_category=override_cat,
                pdf_metadata=pdf_meta,
                rules=rules,
                source_path=file_path,
            ),
        )
        new_base = sanitize_filename_base(filename_str)
        meta = meta or {}
        meta["used_vision_fallback"] = used_vision
        return (file_path, new_base, meta, None)
    except _RECOVERABLE_RENAME_EXCEPTIONS as exc:
        return (file_path, None, None, exc)


def _process_one_file(
    file_path: Path,
    config: RenamerConfig,
    rules: ProcessingRules | None = None,
) -> tuple[Path, str | None, dict[str, object] | None, BaseException | None]:
    """
    Extract text and generate filename for one file. Returns (path, new_base, meta, error).
    If error is not None or new_base is None, the file should be counted as failed/skipped.
    Empty or unextractable PDF: returns (path, None, None, None); caller logs and skips the file.
    """
    if _stop_requested(config):
        return (file_path, None, None, None)
    try:
        content, used_vision = _extract_pdf_content(file_path, config)
    except _RECOVERABLE_RENAME_EXCEPTIONS as exc:
        return (file_path, None, None, exc)
    if not content.strip():
        return (file_path, None, None, None)  # skipped empty
    try:
        return _process_content_to_result(file_path, content, config, rules=rules, used_vision=used_vision)
    except _RECOVERABLE_RENAME_EXCEPTIONS as exc:
        return (file_path, None, None, exc)


def process_one_file(
    file_path: Path,
    config: RenamerConfig,
    rules: ProcessingRules | None = None,
) -> tuple[Path, str | None, dict[str, object] | None, BaseException | None]:
    """Process one PDF into a rename suggestion tuple."""
    return _process_one_file(file_path, config, rules)


def suggest_rename_for_file(
    file_path: Path,
    config: RenamerConfig,
) -> tuple[str | None, dict[str, object] | None, BaseException | None]:
    """
    Run the pipeline for one file and return the suggested new basename and metadata.
    Does not rename or prompt. Returns (new_base, meta, error).
    new_base is None if content is empty or an error occurred.
    """
    rules = load_processing_rules(config.rules_file, raise_on_error=bool(config.rules_file))
    try:
        content, used_vision = _extract_pdf_content(file_path, config)
    except _RECOVERABLE_RENAME_EXCEPTIONS as exc:
        return (None, None, exc)
    if not content.strip():
        return (None, None, None)
    try:
        _path, new_base, meta, process_err = _process_content_to_result(
            file_path, content, config, rules=rules, used_vision=used_vision
        )
        if process_err is not None:
            return (None, None, process_err)
        return (new_base, meta, None)
    except _RECOVERABLE_RENAME_EXCEPTIONS as exc:
        return (None, None, exc)


def _produce_parallel_rename_results(
    files: list[Path],
    config: RenamerConfig,
    rules: ProcessingRules | None,
    progress_callback: Callable[[int, int, Path], None] | None,
    *,
    workers: int,
) -> list[RenameResult]:
    return produce_parallel_rename_results(
        ParallelRenameRequest(
            files=files,
            config=config,
            rules=rules,
            progress_callback=progress_callback,
            workers=workers,
        ),
        deps=ParallelRenameDependencies(
            process_one_file=process_one_file,
            stop_requested=_stop_requested,
            recoverable_exceptions=_RECOVERABLE_RENAME_EXCEPTIONS,
            logger=logger,
            executor_factory=ThreadPoolExecutor,
            wait_for_futures=wait,
        ),
    )


_RenameResult = RenameResult


def _produce_single_worker_rename_results(
    files: list[Path],
    config: RenamerConfig,
    rules: ProcessingRules | None,
    progress_callback: Callable[[int, int, Path], None] | None,
) -> list[tuple[Path, str | None, dict[str, object] | None, BaseException | None]]:
    results: list[_RenameResult] = []
    prefetched: Future[tuple[str, bool]] | None = None
    with ThreadPoolExecutor(max_workers=1) as executor:
        for i, file_path in enumerate(files):
            if _stop_requested(config):
                logger.info("Stop requested. Ending processing early.")
                break
            extracted = _single_worker_extracted_content(file_path, config, prefetched)
            prefetched = None
            if isinstance(extracted, BaseException):
                _append_single_worker_result(results, (file_path, None, None, extracted), len(files), progress_callback)
                continue
            content, used_vision = extracted
            prefetched = _prefetch_next_pdf_content(files, config, executor, i)
            if not content.strip():
                _append_single_worker_result(results, (file_path, None, None, None), len(files), progress_callback)
                continue
            result = _single_worker_processed_result(file_path, content, config, rules, used_vision=used_vision)
            _append_single_worker_result(results, result, len(files), progress_callback)
    return results


def _single_worker_extracted_content(
    file_path: Path,
    config: RenamerConfig,
    prefetched: Future[tuple[str, bool]] | None,
) -> tuple[str, bool] | BaseException:
    try:
        return prefetched.result() if prefetched is not None else _extract_pdf_content(file_path, config)
    except _RECOVERABLE_RENAME_EXCEPTIONS as exc:
        return exc


def _prefetch_next_pdf_content(
    files: list[Path],
    config: RenamerConfig,
    executor: ThreadPoolExecutor,
    index: int,
) -> Future[tuple[str, bool]] | None:
    return executor.submit(_extract_pdf_content, files[index + 1], config) if index + 1 < len(files) else None


def _single_worker_processed_result(
    file_path: Path,
    content: str,
    config: RenamerConfig,
    rules: ProcessingRules | None,
    *,
    used_vision: bool,
) -> _RenameResult:
    try:
        return _process_content_to_result(file_path, content, config, rules=rules, used_vision=used_vision)
    except _RECOVERABLE_RENAME_EXCEPTIONS as exc:
        return (file_path, None, None, exc)


def _append_single_worker_result(
    results: list[_RenameResult],
    result: _RenameResult,
    total: int,
    progress_callback: Callable[[int, int, Path], None] | None,
) -> None:
    results.append(result)
    if progress_callback is not None:
        progress_callback(len(results), total, result[0])


def _produce_rename_results(
    files: list[Path],
    config: RenamerConfig,
    rules: ProcessingRules | None = None,
    progress_callback: Callable[[int, int, Path], None] | None = None,
) -> list[tuple[Path, str | None, dict[str, object] | None, BaseException | None]]:
    """Produce (file_path, new_base, meta, exc) per file; parallel or single-worker with prefetch."""
    workers = max(1, config.workers or 1)
    if config.interactive:
        workers = 1
    if workers > 1:
        return _produce_parallel_rename_results(files, config, rules, progress_callback, workers=workers)
    return _produce_single_worker_rename_results(files, config, rules, progress_callback)


def produce_rename_results(
    files: list[Path],
    config: RenamerConfig,
    rules: ProcessingRules | None = None,
    progress_callback: Callable[[int, int, Path], None] | None = None,
) -> list[tuple[Path, str | None, dict[str, object] | None, BaseException | None]]:
    """Produce per-file rename suggestions for a prepared file list."""
    return _produce_rename_results(files, config, rules=rules, progress_callback=progress_callback)


def write_summary_json(
    summary_path: str | Path | None,
    summary_data: RenameSummaryData,
) -> None:
    """Write a rename run summary JSON file."""
    _write_summary_json(summary_path, summary_data)


@dataclass
class _RenameRunState(RenameOutputData):
    renamed_targets: set[Path] = field(default_factory=set)


def _collect_sorted_pdf_files(
    path: Path, config: RenamerConfig, *, files_override: list[Path] | None, rules: ProcessingRules | None
) -> list[Path]:
    return _collect_sorted_pdf_files_with(
        path,
        config,
        files_override=files_override,
        rules=rules,
        collect_pdf_files_fn=collect_pdf_files,
    )


def _record_processing_exception(file_path: Path, exc: BaseException, state: _RenameRunState) -> None:
    if isinstance(exc, json.JSONDecodeError):
        raise exc
    if isinstance(exc, ValueError) and "Invalid JSON in data file" in str(exc):
        raise exc
    if isinstance(exc, ValueError) and "No text extracted from" in str(exc):
        logger.warning("Skipping %s: %s", file_path.name, exc)
        state.skipped_count += 1
        return
    logger.exception("Failed to process %s: %s", file_path, exc)
    state.failed_count += 1
    state.failure_details.append({"file": str(file_path), "error": str(exc)})


def _maybe_prompt_for_interactive_rename(
    file_path: Path,
    new_base: str,
    meta: dict[str, object],
    config: RenamerConfig,
) -> tuple[bool, str]:
    if not config.interactive:
        return (True, new_base)
    target = file_path.with_name(new_base + file_path.suffix)
    if config.manual_mode:
        print(f"Suggested: {new_base}{file_path.suffix}")
        for key, value in meta.items():
            if key in ("category", "summary", "keywords", "category_source") and value:
                print(f"  {key}: {value}")
    reply, base, _target = _interactive_rename_prompt(
        file_path,
        target,
        new_base,
        edit_default_base=new_base if config.manual_mode else None,
    )
    return (reply != "n", base)


def _apply_rename_result(
    file_path: Path,
    base: str,
    meta: dict[str, object],
    config: RenamerConfig,
    state: _RenameRunState,
) -> None:
    _on_rename_success = _make_post_rename_success_callback(config, meta, state.export_rows)
    rows_before = len(state.export_rows)
    success, target = apply_single_rename(
        file_path,
        base,
        plan_file_path=config.plan_file_path,
        plan_entries=state.plan_entries,
        dry_run=config.dry_run,
        backup_dir=config.backup_dir,
        on_success=_on_rename_success,
        max_filename_chars=config.max_filename_chars,
    )
    if not success:
        logger.error("Skipping %s: could not rename after %s attempts", file_path.name, MAX_RENAME_RETRIES)
        state.failed_count += 1
        state.failure_details.append(
            {
                "file": str(file_path),
                "error": f"Could not rename after {MAX_RENAME_RETRIES} attempts.",
            }
        )
        return
    if config.dry_run:
        if config.export_metadata_path and len(state.export_rows) == rows_before:
            _append_export_row(state.export_rows, file_path=file_path, target=target, meta=meta)
        logger.info("Dry-run: would rename '%s' to '%s'", file_path.name, target.name)
    else:
        logger.info("Renamed '%s' to '%s'", file_path.name, target.name)
        if target != file_path:
            state.renamed_targets.add(target.resolve())
    state.renamed_count += 1


def _handle_rename_result(
    result: _RenameResult,
    config: RenamerConfig,
    state: _RenameRunState,
) -> None:
    file_path, new_base, meta, exc = result
    state.processed_count += 1
    if exc is not None:
        _record_processing_exception(file_path, exc, state)
        return
    if new_base is None:
        logger.info("PDF content is empty. Skipping %s.", file_path.name)
        state.skipped_count += 1
        return
    should_apply, base = _maybe_prompt_for_interactive_rename(file_path, new_base, meta or {}, config)
    if not should_apply:
        state.skipped_count += 1
        return
    try:
        _apply_rename_result(file_path, base, meta or {}, config, state)
    except _RECOVERABLE_RENAME_EXCEPTIONS as rename_error:
        logger.exception("Failed to process %s: %s", file_path, rename_error)
        state.failed_count += 1
        state.failure_details.append({"file": str(file_path), "error": str(rename_error)})


def _produce_results_with_progress(
    files: list[Path],
    config: RenamerConfig,
    rules: ProcessingRules | None,
) -> list[tuple[Path, str | None, dict[str, object] | None, BaseException | None]]:
    with _create_progress_reporter(len(files), config) as progress_reporter:
        if config.progress or config.quiet_progress:
            return produce_rename_results(files, config, rules=rules, progress_callback=progress_reporter.update)
        return produce_rename_results(files, config, rules=rules)


def rename_pdfs_in_directory(
    directory: str | Path,
    *,
    config: RenamerConfig,
    files_override: list[Path] | None = None,
    rules_override: ProcessingRules | None = None,
) -> set[Path]:
    """Rename all PDFs in a directory using the configured pipeline (extract, LLM/heuristic, rename).

    Write export metadata, plan file, and summary JSON after processing. Use files_override to
    process specific files instead of scanning the directory.
    """
    path, rules, files = _prepare_rename_run(
        directory,
        config,
        files_override=files_override,
        rules_override=rules_override,
    )
    if not files:
        _handle_empty_rename_run(path, config)
        return set()
    _log_rename_mode(config)
    state = _RenameRunState()
    results = _produce_results_with_progress(files, config, rules)
    _apply_rename_results(results, files, config, state)
    _write_final_rename_outputs(config, path, state)
    return state.renamed_targets


def _prepare_rename_run(
    directory: str | Path,
    config: RenamerConfig,
    *,
    files_override: list[Path] | None,
    rules_override: ProcessingRules | None,
) -> tuple[Path, ProcessingRules | None, list[Path]]:
    path = _resolve_rename_directory(directory, files_override=files_override)
    rules = _load_effective_rules(config, rules_override)
    files = _collect_sorted_pdf_files(path, config, files_override=files_override, rules=rules)
    return (path, rules, files)


def _handle_empty_rename_run(path: Path, config: RenamerConfig) -> None:
    logger.info("No matching PDF files found in %s", path)
    _write_summary_json(
        config.summary_json_path,
        RenameSummaryData(
            directory=path,
            processed=0,
            renamed=0,
            skipped=0,
            failed=0,
            dry_run=bool(config.dry_run),
            failures=[],
        ),
    )


def _log_rename_mode(config: RenamerConfig) -> None:
    if not config.use_llm:
        logger.info("Heuristic-only mode (LLM disabled). Category from heuristics; summary and keywords will be empty.")


def _apply_rename_results(
    results: list[_RenameResult],
    files: list[Path],
    config: RenamerConfig,
    state: _RenameRunState,
) -> None:
    for i, (file_path, new_base, meta, exc) in enumerate(results):
        if _stop_requested(config):
            logger.info("Stop requested. Ending rename/apply phase.")
            break
        logger.info("Processing %s/%s: %s", i + 1, len(files), file_path)
        _handle_rename_result((file_path, new_base, meta, exc), config, state)


def _write_final_rename_outputs(config: RenamerConfig, path: Path, state: _RenameRunState) -> None:
    _write_rename_outputs(
        config,
        path,
        RenameOutputData(
            export_rows=state.export_rows,
            plan_entries=state.plan_entries,
            processed_count=state.processed_count,
            renamed_count=state.renamed_count,
            skipped_count=state.skipped_count,
            failed_count=state.failed_count,
            failure_details=state.failure_details,
        ),
    )


def run_watch_loop(
    directory: str | Path,
    *,
    config: RenamerConfig,
    interval_seconds: float = 60.0,
) -> None:
    run_watch_loop_impl(
        directory,
        config=config,
        interval_seconds=interval_seconds,
        deps=WatchLoopDependencies(
            collect_pdf_files_fn=collect_pdf_files,
            load_processing_rules_fn=load_processing_rules,
            rename_pdfs_in_directory_fn=rename_pdfs_in_directory,
            logger=logger,
        ),
    )
