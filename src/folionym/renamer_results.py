"""Rename-suggestion result production for sequential and parallel runs."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

from .config import RenamerConfig
from .llm_backend import LLMClient
from .renamer_parallel import (
    ExecutorFactory,
    ParallelRenameDependencies,
    ParallelRenameRequest,
    RenameResult,
    WaitFn,
    produce_parallel_rename_results,
)
from .rules import ProcessingRules

ProgressCallback = Callable[[int, int, Path], None]
ProcessOneFileWithClient = Callable[[Path, RenamerConfig, ProcessingRules | None, LLMClient | None], RenameResult]


@dataclass(frozen=True)
class RenameResultRequest:
    """Inputs for one sequential or parallel rename-suggestion run."""

    files: list[Path]
    config: RenamerConfig
    rules: ProcessingRules | None
    progress_callback: ProgressCallback | None
    workers: int
    llm_client: LLMClient | None


class RenameResultDependencies(NamedTuple):
    """Immutable dependency bundle for deterministic orchestration tests."""

    process_one_file: ProcessOneFileWithClient
    stop_requested: Callable[[RenamerConfig], bool]
    recoverable_exceptions: tuple[type[BaseException], ...]
    logger: logging.Logger
    executor_factory: ExecutorFactory
    wait_for_futures: WaitFn


def produce_rename_results_with(
    request: RenameResultRequest,
    deps: RenameResultDependencies,
) -> list[RenameResult]:
    """Produce input-ordered results on the caller thread or with bounded concurrency."""
    if request.workers > 1:
        return _produce_parallel_results(request, deps)
    return _produce_single_worker_results(request, deps)


def failure_results(
    files: list[Path],
    error: BaseException,
    progress_callback: ProgressCallback | None,
) -> list[RenameResult]:
    """Represent one run-initialization failure at each per-file boundary."""
    results: list[RenameResult] = []
    for file_path in files:
        _append_result(results, (file_path, None, None, error), len(files), progress_callback)
    return results


def _produce_parallel_results(
    request: RenameResultRequest,
    deps: RenameResultDependencies,
) -> list[RenameResult]:
    """Adapt the shared run-scoped LLM client into the parallel worker dependencies."""

    def process_with_run_client(
        file_path: Path,
        config: RenamerConfig,
        rules: ProcessingRules | None,
    ) -> RenameResult:
        """Process one file using the run-scoped LLM client."""
        return deps.process_one_file(file_path, config, rules, request.llm_client)

    return produce_parallel_rename_results(
        ParallelRenameRequest(
            files=request.files,
            config=request.config,
            rules=request.rules,
            progress_callback=request.progress_callback,
            workers=request.workers,
        ),
        deps=ParallelRenameDependencies(
            process_one_file=process_with_run_client,
            stop_requested=deps.stop_requested,
            recoverable_exceptions=deps.recoverable_exceptions,
            logger=deps.logger,
            executor_factory=deps.executor_factory,
            wait_for_futures=deps.wait_for_futures,
        ),
    )


def _produce_single_worker_results(
    request: RenameResultRequest,
    deps: RenameResultDependencies,
) -> list[RenameResult]:
    """Process directly on the caller thread; workers=1 never creates an executor."""
    results: list[RenameResult] = []
    for file_path in request.files:
        if deps.stop_requested(request.config):
            deps.logger.info("Stop requested. Ending processing early.")
            break
        try:
            result = deps.process_one_file(file_path, request.config, request.rules, request.llm_client)
        except deps.recoverable_exceptions as exc:
            result = (file_path, None, None, exc)
        _append_result(results, result, len(request.files), request.progress_callback)
    return results


def _append_result(
    results: list[RenameResult],
    result: RenameResult,
    total: int,
    progress_callback: ProgressCallback | None,
) -> None:
    """Append a result and report completed-count progress."""
    results.append(result)
    if progress_callback is not None:
        progress_callback(len(results), total, result[0])
