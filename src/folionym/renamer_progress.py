"""Progress-reporter implementations for batch rename runs."""

from __future__ import annotations

import logging
from pathlib import Path

from .config import RenamerConfig

logger = logging.getLogger(__name__)


class _NullProgressReporter:
    """No-op context manager used when progress output is disabled or unavailable."""

    def __enter__(self) -> _NullProgressReporter:
        """Start the reporter lifecycle and return the reporter used by the caller."""
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        """End the reporter lifecycle without suppressing exceptions from the caller."""
        return None

    def update(self, current: int, total: int, file_path: Path) -> None:
        """Ignore a progress update."""
        return None


class _RichProgressReporter:
    """Rich-backed stderr progress reporter for one rename run."""

    def __init__(self, total: int, *, quiet: bool) -> None:
        """Build the requested progress columns and initialize the task."""
        from rich.console import Console
        from rich.progress import BarColumn, Progress, ProgressColumn, TextColumn, TimeElapsedColumn

        columns: list[ProgressColumn] = []
        if quiet:
            columns.extend(
                [
                    TextColumn("{task.percentage:>3.0f}%"),
                    TextColumn("{task.completed}/{task.total}"),
                ]
            )
        else:
            columns.extend(
                [
                    TextColumn("{task.completed}/{task.total}"),
                    BarColumn(bar_width=None),
                    TextColumn("{task.percentage:>3.0f}%"),
                ]
            )
        columns.extend([TextColumn("{task.fields[filename]}"), TimeElapsedColumn()])
        self._progress = Progress(*columns, console=Console(stderr=True), transient=True)
        self._task_id = self._progress.add_task("Processing PDFs", total=total, filename="")

    def __enter__(self) -> _RichProgressReporter:
        """Start the reporter lifecycle and return the reporter used by the caller."""
        self._progress.start()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        """End the reporter lifecycle without suppressing exceptions from the caller."""
        self._progress.stop()

    def update(self, current: int, total: int, file_path: Path) -> None:
        """Update the total, completion count, and displayed filename."""
        self._progress.update(self._task_id, total=total, completed=current, filename=file_path.name)


def _create_progress_reporter(total: int, config: RenamerConfig) -> _NullProgressReporter | _RichProgressReporter:
    """Create an opt-in progress reporter without affecting default CLI output."""
    progress_options = config.output.progress_options
    if not (progress_options.progress or progress_options.quiet_progress):
        return _NullProgressReporter()
    try:
        return _RichProgressReporter(total, quiet=progress_options.quiet_progress)
    except ImportError:
        logger.warning("Rich progress unavailable; continuing without progress UI.")
        return _NullProgressReporter()
