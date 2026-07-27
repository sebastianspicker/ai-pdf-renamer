"""Thread-to-UI message primitives used by the Textual TUI."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from rich.markup import escape as _escape_markup
from textual.message import Message as _MessageBase

if TYPE_CHECKING:
    from .tui import FolionymTUI


_TUI_WORKER_EXCEPTIONS = (AttributeError, KeyError, OSError, RuntimeError, TypeError, ValueError)


class _RunLog(_MessageBase):
    """A worker-originated line that must be rendered on Textual's UI thread."""

    def __init__(self, line: str) -> None:
        """Store one worker line for later rendering on Textual's UI thread."""
        super().__init__()
        self.line = line


class _RunFinished(_MessageBase):
    """A worker-originated terminal result for one TUI operation."""

    def __init__(self, ok: bool, message: str) -> None:
        """Store the terminal status that returns worker ownership to the UI."""
        super().__init__()
        self.ok = ok
        self.message = message


class _TextualLogHandler(logging.Handler):
    """Forward logs to the UI thread without retaining an application queue."""

    def __init__(self, app: FolionymTUI) -> None:
        """Bind the handler to the app used to marshal records onto the UI thread."""
        super().__init__()
        self._app = app

    def emit(self, record: logging.LogRecord) -> None:
        """Forward a log record to the UI without letting formatting failures stop the run."""
        try:
            line = _escape_markup(self.format(record)) + "\n"
            self._app.call_from_thread(self._app.post_message, _RunLog(line))
        except _TUI_WORKER_EXCEPTIONS:
            self.handleError(record)
