"""Parallel rename-result production helpers."""

from __future__ import annotations

import logging
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..llm.protocol import LLMClient
from ..naming.rules import ProcessingRules
from ..settings import RenamerConfig
from .models import Proposal

ProposalFuture = Future[Proposal]
ProposalFutureMap = dict[ProposalFuture, tuple[int, Path]]
ProcessOneFileFn = Callable[[Path, RenamerConfig, ProcessingRules | None, LLMClient | None], Proposal]
StopRequestedFn = Callable[[RenamerConfig], bool]
ExecutorFactory = Callable[[int], Any]
WaitFn = Callable[..., tuple[set[ProposalFuture], set[ProposalFuture]]]


@dataclass(frozen=True)
class ProposalProductionDependencies:
    """Dependencies and exception policy for proposal production."""

    process_one_file: ProcessOneFileFn
    stop_requested: StopRequestedFn
    recoverable_exceptions: tuple[type[BaseException], ...]
    logger: logging.Logger
    executor_factory: ExecutorFactory = ThreadPoolExecutor
    wait_for_futures: WaitFn = wait


@dataclass(frozen=True)
class ProposalProductionRequest:
    """Inputs controlling one sequential or parallel proposal-production run."""

    files: list[Path]
    config: RenamerConfig
    rules: ProcessingRules | None
    progress_callback: Callable[[int, int, Path], None] | None
    workers: int
    llm_client: LLMClient | None


@dataclass(frozen=True)
class _ProposalExecution:
    """Executor, pending futures, and input-indexed proposal slots for an active run."""

    executor: Any
    futures: ProposalFutureMap
    parallel_results: list[Proposal | None]


@dataclass(frozen=True)
class _ProposalContext:
    """Request and dependency context shared by the scheduling loop helpers."""

    files: list[Path]
    config: RenamerConfig
    rules: ProcessingRules | None
    execution: _ProposalExecution
    progress_callback: Callable[[int, int, Path], None] | None
    workers: int
    llm_client: LLMClient | None
    deps: ProposalProductionDependencies


def produce_proposals_with(
    request: ProposalProductionRequest,
    deps: ProposalProductionDependencies,
) -> list[Proposal]:
    """Produce typed proposals with bounded concurrency and stable input ordering."""
    if request.workers <= 1:
        return _produce_sequential_proposals(request, deps)
    executor = deps.executor_factory(request.workers)
    futures: ProposalFutureMap = {}
    parallel_results: list[Proposal | None] = [None] * len(request.files)
    execution = _ProposalExecution(
        executor=executor,
        futures=futures,
        parallel_results=parallel_results,
    )
    stop_early = False
    try:
        stop_early = _run_parallel_proposal_loop(
            _ProposalContext(
                files=request.files,
                config=request.config,
                rules=request.rules,
                execution=execution,
                progress_callback=request.progress_callback,
                workers=request.workers,
                llm_client=request.llm_client,
                deps=deps,
            )
        )
    finally:
        _shutdown_parallel_executor(executor, futures, stop_early=stop_early)
    return [proposal for proposal in parallel_results if proposal is not None]


def _run_parallel_proposal_loop(context: _ProposalContext) -> bool:
    """Submit bounded work, collect completed futures, and report whether processing stopped early."""
    completed = 0
    next_index = 0
    while next_index < len(context.files) or context.execution.futures:
        if context.deps.stop_requested(context.config):
            context.deps.logger.info("Stop requested. Ending processing early.")
            return True
        next_index = _submit_proposal_futures(context, next_index)
        if not context.execution.futures:
            continue
        done, _pending = context.deps.wait_for_futures(set(context.execution.futures), return_when=FIRST_COMPLETED)
        completed, stop_early = _store_completed_proposal_futures(context, done, completed)
        if stop_early:
            return True
    return False


def _shutdown_parallel_executor(
    executor: ThreadPoolExecutor,
    futures: ProposalFutureMap,
    *,
    stop_early: bool,
) -> None:
    """Cancel pending futures after early stop; otherwise wait for submitted work."""
    if stop_early:
        for pending in futures:
            pending.cancel()
        executor.shutdown(wait=True, cancel_futures=True)
        return
    executor.shutdown(wait=True, cancel_futures=False)


def _submit_proposal_futures(context: _ProposalContext, next_index: int) -> int:
    """Fill available worker slots and record each future's input index."""
    while next_index < len(context.files) and len(context.execution.futures) < context.workers:
        path = context.files[next_index]
        future = context.execution.executor.submit(
            context.deps.process_one_file,
            path,
            context.config,
            context.rules,
            context.llm_client,
        )
        context.execution.futures[future] = (next_index, path)
        next_index += 1
    return next_index


def _store_completed_proposal_futures(
    context: _ProposalContext,
    done: set[ProposalFuture],
    completed: int,
) -> tuple[int, bool]:
    """Store completed futures by input index, convert recoverable errors, and report progress."""
    for future in done:
        idx, path = context.execution.futures.pop(future)
        # fmt: off
        try:
            result = future.result()
        except (KeyboardInterrupt, SystemExit):
            return (completed, True)
        except context.deps.recoverable_exceptions as exc:
            result = Proposal(path, None, None, exc)
        # fmt: on
        context.execution.parallel_results[idx] = result
        completed += 1
        if context.progress_callback is not None:
            context.progress_callback(completed, len(context.files), path)
    return (completed, False)


def _produce_sequential_proposals(
    request: ProposalProductionRequest,
    deps: ProposalProductionDependencies,
) -> list[Proposal]:
    """Produce proposals on the caller thread, respecting cancellation between files."""
    proposals: list[Proposal] = []
    for path in request.files:
        if deps.stop_requested(request.config):
            deps.logger.info("Stop requested. Ending processing early.")
            break
        try:
            proposal = deps.process_one_file(path, request.config, request.rules, request.llm_client)
        except deps.recoverable_exceptions as exc:
            proposal = Proposal(path, None, None, exc)
        proposals.append(proposal)
        if request.progress_callback is not None:
            request.progress_callback(len(proposals), len(request.files), path)
    return proposals
