"""Batch rename orchestration and side-effect boundary.

This module coordinates discovery, extraction, filename generation, output
artifacts, optional hooks, and watch mode. Pure naming decisions live in
filename.py; filesystem mutation is delegated to the rename_ops package.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Protocol, Self

from ..infrastructure.errors import COMMON_RECOVERABLE_EXCEPTIONS
from ..naming.rules import ProcessingRules, load_processing_rules
from ..rename_ops import (
    MAX_RENAME_RETRIES,
    RenameApplyOptions,
    apply_single_rename,
    sanitize_filename_base,
)
from ..settings import RenamerConfig
from .artifacts import (
    RenameOutputData,
    RenameSummaryData,
    _append_export_row,
    _write_rename_outputs,
    _write_summary_json,
)
from .discovery import (
    _collect_sorted_pdf_files_with,
    _load_effective_rules,
    _resolve_rename_directory,
    collect_pdf_files,
)
from .hooks import _make_post_rename_success_callback
from .models import ApplyPolicy, Proposal
from .proposals import produce_proposals, stop_requested
from .watch import WatchLoopDependencies, run_watch_loop_impl

logger = logging.getLogger(__name__)
_RECOVERABLE_RENAME_EXCEPTIONS = (
    AttributeError,
    *COMMON_RECOVERABLE_EXCEPTIONS,
)


@dataclass
class _RenameRunState(RenameOutputData):
    """Mutable output state and resolved targets accumulated during one rename run."""

    renamed_targets: set[Path] = field(default_factory=set)


def apply_rename_with_policy(
    file_path: Path,
    base: str,
    options: RenameApplyOptions,
    policy: ApplyPolicy = ApplyPolicy.UNIQUE_AVAILABLE,
) -> tuple[bool, Path]:
    """Apply with an explicit collision policy while keeping mutation in ``rename_ops``."""
    return apply_single_rename(
        file_path,
        base,
        replace(options, exact_target=policy is ApplyPolicy.EXACT_REVIEWED),
    )


class _NullProgressReporter:
    """No-op progress reporter used when progress output is disabled."""

    def __enter__(self) -> _NullProgressReporter:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        return None

    def update(self, current: int, total: int, file_path: Path) -> None:
        return None


class _ProgressReporter(Protocol):
    def __enter__(self) -> Self: ...

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None: ...

    def update(self, current: int, total: int, file_path: Path) -> None: ...


def _create_progress_reporter(total: int, config: RenamerConfig) -> _ProgressReporter:
    """Create opt-in Rich progress while retaining a no-dependency fallback."""
    options = config.output.progress_options
    if not (options.progress or options.quiet_progress):
        return _NullProgressReporter()
    try:
        from rich.console import Console
        from rich.progress import BarColumn, Progress, ProgressColumn, TextColumn, TimeElapsedColumn

        columns: list[ProgressColumn] = [TextColumn("{task.completed}/{task.total}")]
        if not options.quiet_progress:
            columns.append(BarColumn(bar_width=None))
        columns.extend(
            [
                TextColumn("{task.percentage:>3.0f}%"),
                TextColumn("{task.fields[filename]}"),
                TimeElapsedColumn(),
            ]
        )
        progress = Progress(*columns, console=Console(stderr=True), transient=True)
        task_id = progress.add_task("Processing PDFs", total=total, filename="")

        class _RichProgressReporter:
            def __enter__(self) -> _RichProgressReporter:
                progress.start()
                return self

            def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
                progress.stop()

            def update(self, current: int, total: int, file_path: Path) -> None:
                progress.update(task_id, total=total, completed=current, filename=file_path.name)

        return _RichProgressReporter()
    except ImportError:
        logger.warning("Rich progress unavailable; continuing without progress UI.")
        return _NullProgressReporter()


def _interactive_rename_prompt(
    file_path: Path, target: Path, default_base: str, edit_default_base: str | None = None
) -> tuple[str, str, Path]:
    """Prompt for y/n/e=edit and return the selected action, basename, and target."""
    current_base = default_base
    current_target = target
    while True:
        try:
            prompt = f"Rename '{file_path.name}' to '{current_target.name}'? (y/n/e=edit, default y): "
            reply = input(prompt).strip().lower() or "y"
        except EOFError, KeyboardInterrupt:
            return ("n", current_base, current_target)
        if reply == "n":
            return ("n", current_base, current_target)
        if reply != "e":
            return ("y", current_base, current_target)
        try:
            edited = input(_edit_prompt_text(edit_default_base)).strip()
        except EOFError, KeyboardInterrupt:
            continue
        edited = edited or edit_default_base or ""
        if not edited:
            continue
        current_base = sanitize_filename_base(_strip_pdf_suffix(edited, file_path.suffix))
        current_target = file_path.with_name(current_base + file_path.suffix)
        return ("y", current_base, current_target)


def _edit_prompt_text(edit_default_base: str | None) -> str:
    """Build the edit prompt with an optional default basename."""
    default = f" [default: {edit_default_base}]" if edit_default_base else ""
    return f"New filename (without path){default}: "


def _strip_pdf_suffix(value: str, suffix: str) -> str:
    """Remove a matching PDF suffix case-insensitively."""
    return value.removesuffix(suffix) if value.lower().endswith(suffix.lower()) else value


def _collect_sorted_pdf_files(
    path: Path, config: RenamerConfig, *, files_override: list[Path] | None, rules: ProcessingRules | None
) -> list[Path]:
    """Collect configured PDF candidates and sort them newest first."""
    return _collect_sorted_pdf_files_with(
        path,
        config,
        files_override=files_override,
        rules=rules,
        collect_pdf_files_fn=collect_pdf_files,
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
    config: RenamerConfig,
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
    config: RenamerConfig,
    state: _RenameRunState,
) -> None:
    """Apply one suggestion and update output rows, counters, and resolved targets."""
    _on_rename_success = _make_post_rename_success_callback(config, meta, state.export_rows)
    rows_before = len(state.export_rows)
    success, target = apply_rename_with_policy(
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
        ApplyPolicy.UNIQUE_AVAILABLE,
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
    proposal: Proposal,
    config: RenamerConfig,
    state: _RenameRunState,
) -> None:
    """Classify one produced result as a failure, skip, rejection, or applied rename."""
    file_path = proposal.source
    new_base = proposal.proposed_base
    meta = proposal.metadata
    exc = proposal.error
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


ProduceProposalsFn = Callable[..., list[Proposal]]


def _produce_results_with_progress(
    files: list[Path],
    config: RenamerConfig,
    rules: ProcessingRules | None,
    produce_proposals_fn: ProduceProposalsFn | None,
) -> list[Proposal]:
    """Produce results with progress while retaining deterministic result ordering."""
    with _create_progress_reporter(len(files), config) as progress_reporter:
        if config.output.progress_options.progress or config.output.progress_options.quiet_progress:
            callback = progress_reporter.update
        else:
            callback = None
        if produce_proposals_fn is None:
            return produce_proposals(files, config, rules=rules, progress_callback=callback)
        return produce_proposals_fn(files, config, rules=rules, progress_callback=callback)


def rename_pdfs_in_directory(
    directory: str | Path,
    *,
    config: RenamerConfig,
    files_override: list[Path] | None = None,
    rules_override: ProcessingRules | None = None,
    produce_proposals_fn: ProduceProposalsFn | None = None,
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
    results = _produce_results_with_progress(files, config, rules, produce_proposals_fn)
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
    """Resolve the directory and rules, then collect PDF candidates for the run."""
    path = _resolve_rename_directory(directory, files_override=files_override)
    rules = _load_effective_rules(config, rules_override)
    files = _collect_sorted_pdf_files(path, config, files_override=files_override, rules=rules)
    return (path, rules, files)


def _handle_empty_rename_run(path: Path, config: RenamerConfig) -> None:
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


def _log_rename_mode(config: RenamerConfig) -> None:
    """Log heuristic-only mode and its missing summary and keyword fields."""
    if not config.llm.runtime.use_llm:
        logger.info("Heuristic-only mode (LLM disabled). Category from heuristics; summary and keywords will be empty.")


def _apply_rename_results(
    results: list[Proposal],
    files: list[Path],
    config: RenamerConfig,
    state: _RenameRunState,
) -> None:
    """Apply suggestions in order until stop is requested, updating run state."""
    for i, proposal in enumerate(results):
        if stop_requested(config):
            logger.info("Stop requested. Ending rename/apply phase.")
            break
        logger.info("Processing %s/%s: %s", i + 1, len(files), proposal.source)
        _handle_rename_result(proposal, config, state)


def _write_final_rename_outputs(config: RenamerConfig, path: Path, state: _RenameRunState) -> None:
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
    config: RenamerConfig,
    interval_seconds: float = 60.0,
) -> None:
    """Delegate watch-mode scanning to the watch implementation with production dependencies."""
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
