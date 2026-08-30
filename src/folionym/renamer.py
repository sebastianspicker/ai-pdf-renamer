"""Public compatibility facade for Folionym's application rename workflows."""

from __future__ import annotations

from collections.abc import Callable as _Callable
from pathlib import Path as _Path
from typing import TYPE_CHECKING as _TYPE_CHECKING

from .application.batch import rename_pdfs_in_directory as _rename_pdfs_in_directory
from .application.batch import run_watch_loop as _run_watch_loop
from .application.models import Proposal as _Proposal
from .application.proposals import process_one_file as _process_one_file
from .application.proposals import produce_proposals as _produce_proposals
from .application.proposals import suggest_rename_for_file as _suggest_rename_for_file

if _TYPE_CHECKING:
    from .llm.protocol import LLMClient as _LLMClient
    from .naming.rules import ProcessingRules as _ProcessingRules
    from .settings import RenamerConfig as _RenamerConfig

__all__ = [
    "process_one_file",
    "produce_rename_results",
    "rename_pdfs_in_directory",
    "run_watch_loop",
    "suggest_rename_for_file",
]


def process_one_file(
    file_path: _Path,
    config: _RenamerConfig,
    rules: _ProcessingRules | None = None,
    llm_client: _LLMClient | None = None,
) -> tuple[_Path, str | None, dict[str, object] | None, BaseException | None]:
    """Process one PDF into the documented tuple-shaped rename suggestion."""
    proposal = _process_one_file(file_path, config, rules, llm_client)
    return (proposal.source, proposal.proposed_base, proposal.metadata, proposal.error)


def suggest_rename_for_file(
    file_path: _Path, config: _RenamerConfig
) -> tuple[str | None, dict[str, object] | None, BaseException | None]:
    """Suggest one filename without renaming or prompting."""
    return _suggest_rename_for_file(file_path, config)


def produce_rename_results(
    files: list[_Path],
    config: _RenamerConfig,
    rules: _ProcessingRules | None = None,
    progress_callback: _Callable[[int, int, _Path], None] | None = None,
) -> list[tuple[_Path, str | None, dict[str, object] | None, BaseException | None]]:
    """Produce input-ordered tuple suggestions for a prepared file list."""
    return [
        (proposal.source, proposal.proposed_base, proposal.metadata, proposal.error)
        for proposal in _produce_proposals(files, config, rules, progress_callback)
    ]


def _produce_compatibility_proposals(
    files: list[_Path],
    config: _RenamerConfig,
    rules: _ProcessingRules | None = None,
    progress_callback: _Callable[[int, int, _Path], None] | None = None,
) -> list[_Proposal]:
    """Adapt the stable tuple-producing facade back into typed application proposals."""
    return [
        _Proposal(*result)
        for result in produce_rename_results(
            files,
            config,
            rules=rules,
            progress_callback=progress_callback,
        )
    ]


def rename_pdfs_in_directory(
    directory: str | _Path,
    *,
    config: _RenamerConfig,
    files_override: list[_Path] | None = None,
    rules_override: _ProcessingRules | None = None,
) -> set[_Path]:
    """Run the conventional unique-available batch rename workflow."""
    return _rename_pdfs_in_directory(
        directory,
        config=config,
        files_override=files_override,
        rules_override=rules_override,
        produce_proposals_fn=_produce_compatibility_proposals,
    )


def run_watch_loop(directory: str | _Path, *, config: _RenamerConfig, interval_seconds: float = 60.0) -> None:
    """Run the conventional directory watch loop."""
    _run_watch_loop(directory, config=config, interval_seconds=interval_seconds)
