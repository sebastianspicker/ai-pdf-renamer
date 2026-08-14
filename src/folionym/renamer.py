"""Batch rename orchestration and side-effect boundary.

This module coordinates discovery, extraction, filename generation, output
artifacts, optional hooks, and watch mode. Pure naming decisions live in
filename.py; filesystem mutation is delegated to the rename_ops package.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Iterator
from concurrent.futures import CancelledError, ThreadPoolExecutor, wait
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from .config import RenamerConfig as _RenamerConfig
from .filename import FilenameGenerationRequest
from .filename import generate_filename as _generate_filename
from .filename_models import FilenameGenerationContext, FilenameGenerationDependencies
from .llm_backend import LLMClient, SerializedLLMClient, create_llm_client_from_config
from .pdf_extract import get_pdf_metadata
from .recoverable_errors import COMMON_RECOVERABLE_EXCEPTIONS
from .rename_ops import (
    MAX_RENAME_RETRIES,
    RenameApplyOptions,
    apply_single_rename,
    sanitize_filename_base,
)
from .renamer_discovery import (
    _collect_sorted_pdf_files_with,
    _load_effective_rules,
    _resolve_rename_directory,
)
from .renamer_extract import extract_pdf_content as _extract_pdf_content_default
from .renamer_extract import extract_pdf_content_with as _extract_pdf_content_with
from .renamer_files import collect_pdf_files as _collect_pdf_files
from .renamer_files import reject_source_symlink
from .renamer_hooks import _make_post_rename_success_callback as _make_post_rename_success_callback
from .renamer_interactive import _interactive_rename_prompt
from .renamer_lookup import _lookup_override_category
from .renamer_output import (
    RenameOutputData,
    RenameSummaryData,
    _append_export_row,
    _write_rename_outputs,
    _write_summary_json,
)
from .renamer_parallel import RenameResult
from .renamer_progress import _create_progress_reporter
from .renamer_results import (
    RenameResultDependencies,
    RenameResultRequest,
    failure_results,
    produce_rename_results_with,
)
from .renamer_watch import WatchLoopDependencies, run_watch_loop_impl
from .rules import (
    ProcessingRules,
    force_category_for_basename,
    load_processing_rules,
)

logger = logging.getLogger(__name__)
_RECOVERABLE_RENAME_EXCEPTIONS = (
    AttributeError,
    CancelledError,
    *COMMON_RECOVERABLE_EXCEPTIONS,
)


def _stop_requested(config: _RenamerConfig) -> bool:
    """Return whether the configured stop event exists and is set."""
    stop_event = config.output.hooks.stop_event
    return bool(stop_event is not None and hasattr(stop_event, "is_set") and stop_event.is_set())


def _extract_pdf_content(path: Path, config: _RenamerConfig) -> tuple[str, bool]:
    """Compatibility shim for the extraction pipeline.
    The strategy logic lives in renamer_extract.py; this wrapper keeps the
    renamer-level patch points stable for tests and local overrides.
    """
    return _extract_pdf_content_default(path, config)


def _extract_pdf_content_with_client(
    path: Path,
    config: _RenamerConfig,
    llm_client: LLMClient | None,
) -> tuple[str, bool]:
    """Reject source symlinks, then extract with the run-scoped client when supplied."""
    reject_source_symlink(path)
    if llm_client is None:
        return _extract_pdf_content(path, config)
    return _extract_pdf_content_with(path, config, llm_client=llm_client)


def _run_uses_llm_client(config: _RenamerConfig) -> bool:
    """Return whether this run can make an LLM or vision request."""
    return bool(config.llm.runtime.use_llm or config.llm.vision.vision_first or config.llm.vision.use_vision_fallback)


def _close_llm_client(client: SerializedLLMClient | None) -> None:
    """Close a run-scoped client while keeping backend cleanup best-effort."""
    if client is None:
        return
    try:
        client.close()
    except (AttributeError, OSError, RuntimeError) as exc:
        logger.warning("Could not close LLM backend cleanly: %s", exc)


@contextmanager
def _run_scoped_llm_client(config: _RenamerConfig) -> Iterator[LLMClient | None]:
    """Create at most one serialized LLM client and attempt to close it when the run exits."""
    raw_client = create_llm_client_from_config(config) if _run_uses_llm_client(config) else None
    client = SerializedLLMClient(raw_client) if raw_client is not None else None
    try:
        yield client
    finally:
        _close_llm_client(client)


@dataclass(frozen=True)
class ContentProcessingRequest:
    """Inputs for filename generation from previously extracted content."""

    file_path: Path
    content: str
    config: _RenamerConfig
    rules: ProcessingRules | None = None
    used_vision: bool = False
    llm_client: LLMClient | None = None


def _process_content_to_result(request: ContentProcessingRequest) -> RenameResult:
    """
    Generate filename from already-extracted content. Returns (path, new_base, meta, error).
    Caller must ensure content is non-empty if expecting a non-skip result.
    """
    try:
        override_cat = _lookup_override_category(
            request.file_path,
            request.config.output.template.override_category_map,
        ) or force_category_for_basename(request.rules, request.file_path.name)
        pdf_meta = get_pdf_metadata(request.file_path) if request.config.extraction.use_pdf_metadata_for_date else None
        filename_str, meta = _generate_filename(
            request.content,
            FilenameGenerationRequest(
                config=request.config,
                dependencies=FilenameGenerationDependencies(llm_client=request.llm_client),
                context=FilenameGenerationContext(
                    override_category=override_cat,
                    pdf_metadata=pdf_meta,
                    rules=request.rules,
                    source_path=request.file_path,
                ),
            ),
        )
        new_base = sanitize_filename_base(filename_str)
        meta = meta or {}
        meta["used_vision_fallback"] = request.used_vision
        return (request.file_path, new_base, meta, None)
    except _RECOVERABLE_RENAME_EXCEPTIONS as exc:
        return (request.file_path, None, None, exc)


def _process_one_file(
    file_path: Path,
    config: _RenamerConfig,
    rules: ProcessingRules | None = None,
    llm_client: LLMClient | None = None,
) -> RenameResult:
    """
    Extract text and generate filename for one file. Returns (path, new_base, meta, error).
    If error is not None or new_base is None, the file should be counted as failed/skipped.
    Empty or unextractable PDF: returns (path, None, None, None); caller logs and skips the file.
    """
    if _stop_requested(config):
        return (file_path, None, None, None)
    try:
        reject_source_symlink(file_path)
    except OSError as exc:
        return (file_path, None, None, exc)
    try:
        content, used_vision = _extract_pdf_content_with_client(file_path, config, llm_client)
    except _RECOVERABLE_RENAME_EXCEPTIONS as exc:
        return (file_path, None, None, exc)
    if not content.strip():
        return (file_path, None, None, None)  # skipped empty
    return _process_content_to_result(
        ContentProcessingRequest(
            file_path=file_path,
            content=content,
            config=config,
            rules=rules,
            used_vision=used_vision,
            llm_client=llm_client,
        )
    )


def process_one_file(
    file_path: Path,
    config: _RenamerConfig,
    rules: ProcessingRules | None = None,
    llm_client: LLMClient | None = None,
) -> RenameResult:
    """Process one PDF into a rename suggestion tuple."""
    return _process_one_file(file_path, config, rules, llm_client=llm_client)


def suggest_rename_for_file(
    file_path: Path,
    config: _RenamerConfig,
) -> tuple[str | None, dict[str, object] | None, BaseException | None]:
    """
    Run the pipeline for one file and return the suggested new basename and metadata.
    Does not rename or prompt. Returns (new_base, meta, error).
    new_base is None if content is empty or an error occurred.
    """
    try:
        reject_source_symlink(file_path)
    except OSError as exc:
        return (None, None, exc)
    rules_file = config.output.paths.rules_file
    rules = load_processing_rules(rules_file, raise_on_error=bool(rules_file))
    try:
        with _run_scoped_llm_client(config) as llm_client:
            content, used_vision = _extract_pdf_content_with_client(file_path, config, llm_client)
            if not content.strip():
                return (None, None, None)
            _path, new_base, meta, process_err = _process_content_to_result(
                ContentProcessingRequest(
                    file_path=file_path,
                    content=content,
                    config=config,
                    rules=rules,
                    used_vision=used_vision,
                    llm_client=llm_client,
                )
            )
            if process_err is not None:
                return (None, None, process_err)
            return (new_base, meta, None)
    except _RECOVERABLE_RENAME_EXCEPTIONS as exc:
        return (None, None, exc)


def _produce_rename_results(
    files: list[Path],
    config: _RenamerConfig,
    rules: ProcessingRules | None = None,
    progress_callback: Callable[[int, int, Path], None] | None = None,
    llm_client: LLMClient | None = None,
) -> list[tuple[Path, str | None, dict[str, object] | None, BaseException | None]]:
    """Produce input-ordered results on the caller thread or with bounded parallelism."""
    workers = max(1, config.output.traversal.workers or 1)
    if config.output.mode.interactive:
        workers = 1

    def process_one_file_with_client(
        file_path: Path,
        run_config: _RenamerConfig,
        run_rules: ProcessingRules | None,
        run_client: LLMClient | None,
    ) -> RenameResult:
        """Process one file using the shared run-scoped client when present."""
        if run_client is None:
            return process_one_file(file_path, run_config, run_rules)
        return _process_one_file(file_path, run_config, run_rules, llm_client=run_client)

    return produce_rename_results_with(
        RenameResultRequest(files, config, rules, progress_callback, workers, llm_client),
        RenameResultDependencies(
            process_one_file=process_one_file_with_client,
            stop_requested=_stop_requested,
            recoverable_exceptions=_RECOVERABLE_RENAME_EXCEPTIONS,
            logger=logger,
            executor_factory=ThreadPoolExecutor,
            wait_for_futures=wait,
        ),
    )


def produce_rename_results(
    files: list[Path],
    config: _RenamerConfig,
    rules: ProcessingRules | None = None,
    progress_callback: Callable[[int, int, Path], None] | None = None,
) -> list[tuple[Path, str | None, dict[str, object] | None, BaseException | None]]:
    """Produce per-file rename suggestions for a prepared file list."""
    if not files:
        return []
    try:
        with _run_scoped_llm_client(config) as llm_client:
            return _produce_rename_results(
                files,
                config,
                rules=rules,
                progress_callback=progress_callback,
                llm_client=llm_client,
            )
    except _RECOVERABLE_RENAME_EXCEPTIONS as exc:
        return failure_results(files, exc, progress_callback)


@dataclass
class _RenameRunState(RenameOutputData):
    """Mutable output state and resolved targets accumulated during one rename run."""

    renamed_targets: set[Path] = field(default_factory=set)


def _collect_sorted_pdf_files(
    path: Path, config: _RenamerConfig, *, files_override: list[Path] | None, rules: ProcessingRules | None
) -> list[Path]:
    """Collect configured PDF candidates and sort them newest first."""
    return _collect_sorted_pdf_files_with(
        path,
        config,
        files_override=files_override,
        rules=rules,
        collect_pdf_files_fn=_collect_pdf_files,
    )


def _record_processing_exception(file_path: Path, exc: BaseException, state: _RenameRunState) -> None:
    """Classify an exception as fatal data failure, skipped no-text input, or recorded file failure."""
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
    config: _RenamerConfig,
) -> tuple[bool, str]:
    """Prompt in interactive mode and return whether to apply the selected basename."""
    if not config.output.mode.interactive:
        return (True, new_base)
    target = file_path.with_name(new_base + file_path.suffix)
    if config.output.mode.manual_mode:
        print(f"Suggested: {new_base}{file_path.suffix}")
        for key, value in meta.items():
            if key in ("category", "summary", "keywords", "category_source") and value:
                print(f"  {key}: {value}")
    reply, base, _target = _interactive_rename_prompt(
        file_path,
        target,
        new_base,
        edit_default_base=new_base if config.output.mode.manual_mode else None,
    )
    return (reply != "n", base)


def _apply_rename_result(
    file_path: Path,
    base: str,
    meta: dict[str, object],
    config: _RenamerConfig,
    state: _RenameRunState,
) -> None:
    """Apply one suggestion and update output rows, counters, and resolved targets."""
    _on_rename_success = _make_post_rename_success_callback(config, meta, state.export_rows)
    rows_before = len(state.export_rows)
    success, target = apply_single_rename(
        file_path,
        base,
        RenameApplyOptions(
            plan_file_path=config.output.paths.plan_file_path,
            plan_entries=state.plan_entries,
            dry_run=config.output.mode.dry_run,
            backup_dir=config.output.paths.backup_dir,
            on_success=_on_rename_success,
            max_filename_chars=config.output.naming.max_filename_chars,
        ),
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
    if config.output.mode.dry_run:
        if config.output.paths.export_metadata_path and len(state.export_rows) == rows_before:
            _append_export_row(state.export_rows, file_path=file_path, target=target, meta=meta)
        logger.info("Dry-run: would rename '%s' to '%s'", file_path.name, target.name)
    else:
        logger.info("Renamed '%s' to '%s'", file_path.name, target.name)
        if target != file_path:
            state.renamed_targets.add(target.resolve())
    state.renamed_count += 1


def _handle_rename_result(
    result: RenameResult,
    config: _RenamerConfig,
    state: _RenameRunState,
) -> None:
    """Classify one produced result as a failure, skip, rejection, or applied rename."""
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
    config: _RenamerConfig,
    rules: ProcessingRules | None,
) -> list[tuple[Path, str | None, dict[str, object] | None, BaseException | None]]:
    """Produce results with progress while retaining deterministic result ordering."""
    with _create_progress_reporter(len(files), config) as progress_reporter:
        if config.output.progress_options.progress or config.output.progress_options.quiet_progress:
            return produce_rename_results(files, config, rules=rules, progress_callback=progress_reporter.update)
        return produce_rename_results(files, config, rules=rules)


def rename_pdfs_in_directory(
    directory: str | Path,
    *,
    config: _RenamerConfig,
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
    config: _RenamerConfig,
    *,
    files_override: list[Path] | None,
    rules_override: ProcessingRules | None,
) -> tuple[Path, ProcessingRules | None, list[Path]]:
    """Resolve the directory and rules, then collect PDF candidates for the run."""
    path = _resolve_rename_directory(directory, files_override=files_override)
    rules = _load_effective_rules(config, rules_override)
    files = _collect_sorted_pdf_files(path, config, files_override=files_override, rules=rules)
    return (path, rules, files)


def _handle_empty_rename_run(path: Path, config: _RenamerConfig) -> None:
    """Log an empty selection and emit an all-zero summary when configured."""
    logger.info("No matching PDF files found in %s", path)
    _write_summary_json(
        config.output.paths.summary_json_path,
        RenameSummaryData(
            directory=path,
            processed=0,
            renamed=0,
            skipped=0,
            failed=0,
            dry_run=bool(config.output.mode.dry_run),
            failures=[],
        ),
    )


def _log_rename_mode(config: _RenamerConfig) -> None:
    """Log heuristic-only mode and its missing summary and keyword fields."""
    if not config.llm.runtime.use_llm:
        logger.info("Heuristic-only mode (LLM disabled). Category from heuristics; summary and keywords will be empty.")


def _apply_rename_results(
    results: list[RenameResult],
    files: list[Path],
    config: _RenamerConfig,
    state: _RenameRunState,
) -> None:
    """Apply suggestions in order until stop is requested, updating run state."""
    for i, (file_path, new_base, meta, exc) in enumerate(results):
        if _stop_requested(config):
            logger.info("Stop requested. Ending rename/apply phase.")
            break
        logger.info("Processing %s/%s: %s", i + 1, len(files), file_path)
        _handle_rename_result((file_path, new_base, meta, exc), config, state)


def _write_final_rename_outputs(config: _RenamerConfig, path: Path, state: _RenameRunState) -> None:
    """Serialize accumulated export, plan, summary, and console output."""
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
    config: _RenamerConfig,
    interval_seconds: float = 60.0,
) -> None:
    """Delegate watch-mode scanning to the watch implementation with production dependencies."""
    run_watch_loop_impl(
        directory,
        config=config,
        interval_seconds=interval_seconds,
        deps=WatchLoopDependencies(
            collect_pdf_files_fn=_collect_pdf_files,
            load_processing_rules_fn=load_processing_rules,
            rename_pdfs_in_directory_fn=rename_pdfs_in_directory,
            logger=logger,
        ),
    )
