"""Parallel rename-result production helpers."""

from __future__ import annotations

import logging
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import RenamerConfig
from .rules import ProcessingRules

RenameResult = tuple[Path, str | None, dict[str, object] | None, BaseException | None]
RenameFuture = Future[RenameResult]
RenameFutureMap = dict[RenameFuture, tuple[int, Path]]
ProcessOneFileFn = Callable[[Path, RenamerConfig, ProcessingRules | None], RenameResult]
StopRequestedFn = Callable[[RenamerConfig], bool]
ExecutorFactory = Callable[[int], Any]
WaitFn = Callable[..., tuple[set[RenameFuture], set[RenameFuture]]]


@dataclass(frozen=True)
class ParallelRenameDependencies:
    process_one_file: ProcessOneFileFn
    stop_requested: StopRequestedFn
    recoverable_exceptions: tuple[type[BaseException], ...]
    logger: logging.Logger
    executor_factory: ExecutorFactory = ThreadPoolExecutor
    wait_for_futures: WaitFn = wait


@dataclass(frozen=True)
class ParallelRenameRequest:
    files: list[Path]
    config: RenamerConfig
    rules: ProcessingRules | None
    progress_callback: Callable[[int, int, Path], None] | None
    workers: int


@dataclass(frozen=True)
class _ParallelRenameExecution:
    executor: Any
    futures: RenameFutureMap
    parallel_results: list[RenameResult | None]


@dataclass(frozen=True)
class _ParallelRenameContext:
    files: list[Path]
    config: RenamerConfig
    rules: ProcessingRules | None
    execution: _ParallelRenameExecution
    progress_callback: Callable[[int, int, Path], None] | None
    workers: int
    deps: ParallelRenameDependencies


def produce_parallel_rename_results(
    request: ParallelRenameRequest,
    deps: ParallelRenameDependencies,
) -> list[RenameResult]:
    executor = deps.executor_factory(request.workers)
    futures: RenameFutureMap = {}
    parallel_results: list[RenameResult | None] = [None] * len(request.files)
    execution = _ParallelRenameExecution(
        executor=executor,
        futures=futures,
        parallel_results=parallel_results,
    )
    stop_early = False
    try:
        stop_early = _run_parallel_rename_loop(
            _ParallelRenameContext(
                files=request.files,
                config=request.config,
                rules=request.rules,
                execution=execution,
                progress_callback=request.progress_callback,
                workers=request.workers,
                deps=deps,
            )
        )
    finally:
        _shutdown_parallel_executor(executor, futures, stop_early=stop_early)
    return [result for result in parallel_results if result is not None]


def _run_parallel_rename_loop(context: _ParallelRenameContext) -> bool:
    completed = 0
    next_index = 0
    while next_index < len(context.files) or context.execution.futures:
        if context.deps.stop_requested(context.config):
            context.deps.logger.info("Stop requested. Ending processing early.")
            return True
        next_index = _submit_rename_futures(context, next_index)
        if not context.execution.futures:
            continue
        done, _pending = context.deps.wait_for_futures(set(context.execution.futures), return_when=FIRST_COMPLETED)
        completed, stop_early = _store_completed_rename_futures(context, done, completed)
        if stop_early:
            return True
    return False


def _shutdown_parallel_executor(
    executor: ThreadPoolExecutor,
    futures: RenameFutureMap,
    *,
    stop_early: bool,
) -> None:
    if stop_early:
        for pending in futures:
            pending.cancel()
        executor.shutdown(wait=True, cancel_futures=True)
        return
    executor.shutdown(wait=True, cancel_futures=False)


def _submit_rename_futures(context: _ParallelRenameContext, next_index: int) -> int:
    while next_index < len(context.files) and len(context.execution.futures) < context.workers:
        path = context.files[next_index]
        future = context.execution.executor.submit(
            context.deps.process_one_file,
            path,
            context.config,
            context.rules,
        )
        context.execution.futures[future] = (next_index, path)
        next_index += 1
    return next_index


def _store_completed_rename_futures(
    context: _ParallelRenameContext,
    done: set[RenameFuture],
    completed: int,
) -> tuple[int, bool]:
    for future in done:
        idx, path = context.execution.futures.pop(future)
        try:
            result = future.result()
        except (KeyboardInterrupt, SystemExit):
            return (completed, True)
        except context.deps.recoverable_exceptions as exc:
            result = (path, None, None, exc)
        context.execution.parallel_results[idx] = result
        completed += 1
        if context.progress_callback is not None:
            context.progress_callback(completed, len(context.files), path)
    return (completed, False)
