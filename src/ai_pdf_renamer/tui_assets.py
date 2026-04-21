"""Constants, CSS, and log-line formatters for the Rich/Textual TUI."""

from __future__ import annotations

import re

from rich.markup import escape as _escape_markup

PROCESS_RE = re.compile(r"Processing\s+(\d+)/(\d+)\s*:")
_RENAME_LOG_RE = re.compile(r"Renamed '(.+?)' to '(.+?)'")
_DRYRUN_LOG_RE = re.compile(r"Dry-run: would rename '(.+?)' to '(.+?)'")


def _format_rename_match(m: re.Match[str]) -> str:
    old = _escape_markup(m.group(1))
    new = _escape_markup(m.group(2))
    return f"[green]Renamed[/green] [dim]{old}[/dim] [green bold]->[/green bold] [bold]{new}[/bold]"


def _format_dryrun_match(m: re.Match[str]) -> str:
    old = _escape_markup(m.group(1))
    new = _escape_markup(m.group(2))
    return f"[cyan]Dry-run[/cyan] [dim]{old}[/dim] [cyan bold]->[/cyan bold] [bold]{new}[/bold]"


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
_LLM_BACKENDS = [
    ("http (OpenAI-compatible server)", "http"),
    ("in-process (llama-cpp-python)", "in-process"),
    ("auto (in-process if path set)", "auto"),
]


_CSS = """
Screen {
    layout: vertical;
    background: $surface;
}

TabbedContent {
    height: 1fr;
}

/* --- Form layout --- */

.field-row {
    height: auto;
    margin-bottom: 1;
    layout: horizontal;
}

.field-label {
    width: 22;
    padding: 0 1 0 0;
    content-align: left middle;
    color: $text-muted;
}

.field-input {
    width: 1fr;
}

.section-title {
    height: auto;
    padding: 1 0 0 0;
    margin-bottom: 1;
    color: $accent;
    text-style: bold;
}

.section-sep {
    height: 1;
    border-top: solid $primary-darken-2;
    margin: 1 0;
}

.flags-row {
    height: auto;
    layout: horizontal;
}

.flag-check {
    margin-right: 2;
}

.form-container {
    padding: 1 2;
}

/* --- Run tab --- */

#run-tab {
    layout: vertical;
}

#run-status-bar {
    height: auto;
    layout: horizontal;
    padding: 0 1;
    margin-bottom: 1;
    background: $boost;
}

#run-status {
    height: 1;
    width: 1fr;
    padding: 0 1;
    color: $text-muted;
}

#run-status.status-idle {
    color: $text-muted;
}

#run-status.status-running {
    color: $warning;
    text-style: bold;
}

#run-status.status-done {
    color: $success;
    text-style: bold;
}

#run-status.status-error {
    color: $error;
    text-style: bold;
}

#run-status.status-cancel {
    color: $warning;
}

#run-file-counter {
    height: 1;
    width: auto;
    padding: 0 1;
    content-align: right middle;
    color: $text-muted;
}

#run-progress {
    height: 1;
    margin: 0 1 1 1;
}

#run-log {
    height: 1fr;
    margin: 0 1;
    border: round $primary-darken-2;
    padding: 0 1;
}

#run-buttons {
    height: auto;
    layout: horizontal;
    margin: 1 1;
    padding: 1 0 0 0;
    border-top: solid $primary-darken-3;
}

#run-buttons Button {
    margin-right: 1;
}

#btn-preview {
    min-width: 20;
}

#btn-apply {
    min-width: 20;
}

#btn-one {
    min-width: 20;
}

#btn-cancel {
    min-width: 16;
}

#run-summary {
    height: auto;
    padding: 0 2;
    margin: 0 1 1 1;
    color: $text-muted;
}
"""
