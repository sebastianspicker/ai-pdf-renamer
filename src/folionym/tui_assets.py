"""Constants, CSS, and log-line formatters for the Rich/Textual TUI."""

from __future__ import annotations

import re

from rich.markup import escape as _escape_markup
from textual.theme import Theme

PROCESS_RE = re.compile(r"Processing\s+(\d+)/(\d+)\s*:")
_RENAME_LOG_RE = re.compile(r"Renamed '(.+?)' to '(.+?)'")
_DRYRUN_LOG_RE = re.compile(r"Dry-run: would rename '(.+?)' to '(.+?)'")
PREVIEW_COLOR = "#2468E5"
SUCCESS_COLOR = "#087F70"
WARNING_COLOR = "#A45D00"
ERROR_COLOR = "#B42318"


def _format_rename_match(m: re.Match[str]) -> str:
    """Escape captured filenames before adding Rich markup to a rename message."""
    old = _escape_markup(m.group(1))
    new = _escape_markup(m.group(2))
    return (
        f"[{SUCCESS_COLOR}]Renamed[/{SUCCESS_COLOR}] [dim]{old}[/dim] "
        f"[{SUCCESS_COLOR} bold]->[/{SUCCESS_COLOR} bold] [bold]{new}[/bold]"
    )


def _format_dryrun_match(m: re.Match[str]) -> str:
    """Escape captured filenames before adding Rich markup to a dry-run message."""
    old = _escape_markup(m.group(1))
    new = _escape_markup(m.group(2))
    return (
        f"[{PREVIEW_COLOR}]Dry-run[/{PREVIEW_COLOR}] [dim]{old}[/dim] "
        f"[{PREVIEW_COLOR} bold]->[/{PREVIEW_COLOR} bold] [bold]{new}[/bold]"
    )


_LANGUAGES = [("German (de)", "de"), ("English (en)", "en")]
_CASES = [
    ("kebabCase", "kebabCase"),
    ("snakeCase", "snakeCase"),
    ("camelCase", "camelCase"),
]
_DATE_FORMATS = [("Day-Month-Year (dmy)", "dmy"), ("Month-Day-Year (mdy)", "mdy")]
_PRESETS = [
    ("(none)", ""),
    ("high-confidence-heuristic", "high-confidence-heuristic"),
    ("scanned", "scanned"),
    ("fast", "fast"),
    ("accurate", "accurate"),
    ("batch", "batch"),
]

FOLIONYM_THEME = Theme(
    name="folionym-ledger",
    primary=PREVIEW_COLOR,
    secondary="#647082",
    accent=PREVIEW_COLOR,
    foreground="#17202D",
    background="#FBFCFE",
    surface="#F2F5FA",
    panel="#E9EFF8",
    boost="#E0E6EF",
    success=SUCCESS_COLOR,
    warning=WARNING_COLOR,
    error=ERROR_COLOR,
    dark=False,
    variables={
        "muted-copy": "#566273",
        "quiet-border": "#D6DDE7",
        "strong-border": "#B7C0CE",
        "selection-soft": "#E7F0FF",
        "success-soft": "#E5F5F1",
        "warning-soft": "#FFF2D6",
        "error-soft": "#FDECEA",
    },
)
