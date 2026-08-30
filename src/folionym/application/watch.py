"""Watch-mode helpers for the batch renamer."""

from __future__ import annotations

import contextlib
import logging
import signal
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import FrameType
from typing import Any

from ..naming.rules import ProcessingRules
from ..settings import RenamerConfig
from .discovery import _collect_pdf_files_with_config

_SignalHandler = Callable[[int, FrameType | None], Any] | int | signal.Handlers | None
_RECOVERABLE_WATCH_EXCEPTIONS = (
    AttributeError,
    KeyError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
)


@dataclass(frozen=True)
class WatchLoopDependencies:
    """Injectable discovery, rule-loading, rename, and logging dependencies."""

    collect_pdf_files_fn: Callable[..., list[Path]]
    load_processing_rules_fn: Callable[..., ProcessingRules | None]
    rename_pdfs_in_directory_fn: Callable[..., set[Path]]
    logger: logging.Logger


@dataclass(frozen=True)
class WatchLoopContext:
    """Inputs shared by every scan-loop iteration."""

    path: Path
    config: RenamerConfig
    interval_seconds: float
    deps: WatchLoopDependencies


@dataclass
class WatchLoopState:
    """Mutable watch state containing observed mtimes and prior rename targets."""

    seen: dict[Path, float]
    renamed_targets: set[Path]


@dataclass(frozen=True)
class WatchIterationContext:
    """Directory, configuration, and rules shared by one watch iteration."""

    path: Path
    config: RenamerConfig
    rules: ProcessingRules | None


def _cleanup_watch_state(state: WatchLoopState) -> WatchLoopState:
    """Remove missing paths from the observed and renamed-target state."""
    for path in [p for p in state.seen if not p.exists()]:
        del state.seen[path]
    state.renamed_targets = {path for path in state.renamed_targets if path.exists()}
    return state


def _changed_watch_files(
    path: Path,
    config: RenamerConfig,
    rules: ProcessingRules | None,
    state: WatchLoopState,
    collect_pdf_files_fn: Callable[..., list[Path]],
) -> list[Path]:
    """Return unseen or mtime-changed PDFs newest first, excluding prior rename targets."""
    files = _collect_pdf_files_with_config(
        path,
        config,
        files_override=None,
        rules=rules,
        collect_pdf_files_fn=collect_pdf_files_fn,
    )
    to_process: list[Path] = []
    for file_path in files:
        if file_path in state.renamed_targets:
            continue
        try:
            mtime = file_path.stat().st_mtime
        except OSError:
            continue
        if file_path not in state.seen or state.seen[file_path] != mtime:
            to_process.append(file_path)
            state.seen[file_path] = mtime
    to_process.sort(key=lambda file_path: state.seen.get(file_path, 0), reverse=True)
    return to_process


def _remember_watch_targets(
    state: WatchLoopState,
    actual_targets: set[Path],
) -> WatchLoopState:
    """Record successful targets and mtimes so watch mode ignores its own outputs."""
    if not actual_targets:
        return state
    state.renamed_targets.update(actual_targets)
    for renamed_target in actual_targets:
        with contextlib.suppress(OSError):
            state.seen[renamed_target] = renamed_target.stat().st_mtime
    return state


def _process_watch_files(
    context: WatchIterationContext,
    to_process: list[Path],
    state: WatchLoopState,
    stop_requested: Callable[[], bool],
    rename_pdfs_in_directory_fn: Callable[..., set[Path]],
) -> WatchLoopState:
    """Rename each changed file until stopped, updating watch state after each success."""
    for single in to_process:
        if stop_requested():
            break
        actual_targets = rename_pdfs_in_directory_fn(
            context.path,
            config=context.config,
            files_override=[single],
            rules_override=context.rules,
        )
        state = _remember_watch_targets(state, actual_targets)
    return state


def _run_watch_iteration(
    path: Path,
    config: RenamerConfig,
    state: WatchLoopState,
    stop_requested: Callable[[], bool],
    *,
    deps: WatchLoopDependencies,
) -> WatchLoopState:
    """Refresh state and rules, discover changed files, and process one iteration."""
    state = _cleanup_watch_state(state)
    rules_file = config.output.paths.rules_file
    rules = deps.load_processing_rules_fn(rules_file, raise_on_error=bool(rules_file))
    context = WatchIterationContext(path, config, rules)
    to_process = _changed_watch_files(path, config, rules, state, deps.collect_pdf_files_fn)
    return _process_watch_files(
        context,
        to_process,
        state,
        stop_requested,
        deps.rename_pdfs_in_directory_fn,
    )


def _run_watch_cycle(
    context: WatchLoopContext,
    state: WatchLoopState,
    stop_requested: Callable[[], bool],
) -> WatchLoopState:
    """Run one watch iteration, retaining state after recoverable failures."""
    try:
        next_state = _run_watch_iteration(
            context.path,
            context.config,
            state,
            stop_requested,
            deps=context.deps,
        )
    except _RECOVERABLE_WATCH_EXCEPTIONS as exc:
        if stop_requested():
            return state
        context.deps.logger.exception("Watch iteration failed: %s", exc)
        time.sleep(context.interval_seconds)
        return state
    if not stop_requested():
        time.sleep(context.interval_seconds)
    return next_state


def _install_watch_signal_handlers(
    handle_stop: Callable[[int, FrameType | None], None],
) -> tuple[bool, _SignalHandler, _SignalHandler]:
    """Install SIGINT and SIGTERM handlers only on the main thread and return the originals."""
    # signal.signal() is only legal on the main thread; embedded/TUI callers may
    # run watch mode from a worker thread.
    import threading as _threading

    is_main_thread = _threading.current_thread() is _threading.main_thread()
    if not is_main_thread:
        return (False, None, None)
    return (
        True,
        signal.signal(signal.SIGTERM, handle_stop),
        signal.signal(signal.SIGINT, handle_stop),
    )


def _restore_watch_signal_handlers(
    is_main_thread: bool,
    original_sigterm: _SignalHandler,
    original_sigint: _SignalHandler,
) -> None:
    """Restore only the signal handlers installed by this invocation."""
    # Restore only the signal handlers this invocation installed.
    if is_main_thread and original_sigterm is not None:
        signal.signal(signal.SIGTERM, original_sigterm)
    if is_main_thread and original_sigint is not None:
        signal.signal(signal.SIGINT, original_sigint)


def run_watch_loop_impl(
    directory: str | Path,
    *,
    config: RenamerConfig,
    interval_seconds: float,
    deps: WatchLoopDependencies,
) -> None:
    """Run rename in a loop, scanning the directory every interval_seconds. Processes new/changed PDFs."""
    path = Path(directory).resolve()
    if not path.is_dir():
        raise NotADirectoryError(f"Not a directory: {path}")

    stop_requested = False

    def handle_stop(sig: int, frame: FrameType | None) -> None:
        """Set the loop stop flag and log the received signal."""
        nonlocal stop_requested
        deps.logger.info("Watch mode: received signal %s, stopping...", sig)
        stop_requested = True

    is_main_thread, original_sigterm, original_sigint = _install_watch_signal_handlers(handle_stop)

    # Track successful rename targets so watch mode does not process its own outputs.
    context = WatchLoopContext(path, config, interval_seconds, deps)
    loop_state = WatchLoopState(seen={}, renamed_targets=set())
    deps.logger.info("Watch mode: scanning %s every %.1fs (Ctrl+C or SIGTERM to stop)", path, interval_seconds)
    try:
        while not stop_requested:
            loop_state = _run_watch_cycle(context, loop_state, lambda: stop_requested)
    finally:
        _restore_watch_signal_handlers(is_main_thread, original_sigterm, original_sigint)
        deps.logger.info("Watch stopped")
