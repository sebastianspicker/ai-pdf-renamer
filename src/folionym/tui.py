"""Terminal UI for Folionym (Textual-based, replaces Tkinter gui.py).

Launch with: folionym-tui
Requires: pip install -e '.[tui]'
"""

from __future__ import annotations

import contextlib
import logging
import threading
from collections.abc import Callable
from pathlib import Path
from typing import ClassVar

from rich.markup import escape as _escape_markup
from rich.text import Text

try:
    from textual import events, on, work
    from textual.app import App, ComposeResult
    from textual.binding import Binding
    from textual.containers import Grid, Horizontal, Vertical
    from textual.css.query import QueryError
    from textual.screen import ModalScreen
    from textual.widgets import (
        Button,
        Checkbox,
        DataTable,
        Footer,
        Input,
        ProgressBar,
        RichLog,
        Select,
        Static,
        TabbedContent,
        TabPane,
    )
except ImportError as _e:  # pragma: no cover
    raise ImportError("textual is required for the TUI. Install with: pip install -e '.[tui]'") from _e

from .config import RenamerConfig
from .http_url import validate_http_endpoint
from .logging_utils import setup_logging
from .rename_ops import RenameApplyOptions, apply_single_rename, sanitize_filename_base
from .renamer import (
    _make_post_rename_success_callback,
    rename_pdfs_in_directory,
    suggest_rename_for_file,
)
from .tui_assets import (
    _CSS,
    _DRYRUN_LOG_RE,
    _PRESETS,
    _RENAME_LOG_RE,
    ERROR_COLOR,
    FOLIONYM_THEME,
    PREVIEW_COLOR,
    PROCESS_RE,
    SUCCESS_COLOR,
    WARNING_COLOR,
    _format_dryrun_match,
    _format_rename_match,
)
from .tui_forms import compose_advanced, compose_basic, compose_run
from .tui_state import (
    SETTINGS_PATH,
    _load_settings,
    _save_settings,
    build_config_from_snapshot,
)
from .tui_worker_messages import _RunFinished, _RunLog, _TextualLogHandler

__all__ = [
    "SETTINGS_PATH",
    "FolionymTUI",
    "_load_settings",
    "_save_settings",
    "main",
]

logger = logging.getLogger(__name__)

_TUI_WORKER_EXCEPTIONS = (AttributeError, KeyError, OSError, RuntimeError, TypeError, ValueError)


class ConfirmActionScreen(ModalScreen[bool]):
    """Confirm a consequential filesystem action without exposing hidden behavior."""

    BINDINGS: ClassVar[list[Binding | tuple[str, str] | tuple[str, str, str]]] = [Binding("escape", "cancel", "Cancel")]

    def __init__(self, title: str, detail: str, confirm_label: str) -> None:
        """Store action-specific copy for one reusable confirmation layout."""
        super().__init__()
        self._title = title
        self._detail = detail
        self._confirm_label = confirm_label

    def compose(self) -> ComposeResult:
        """Compose one focused confirmation with cancellation first in keyboard order."""
        with Grid(id="confirm-dialog"):
            yield Static(self._title, id="confirm-title", markup=False)
            yield Static(self._detail, id="confirm-detail", markup=False)
            yield Button("Cancel", id="confirm-cancel")
            yield Button(self._confirm_label, id="confirm-action", variant="error")

    def on_mount(self) -> None:
        """Put the non-destructive action first in the modal keyboard path."""
        self.query_one("#confirm-cancel", Button).focus()

    def action_cancel(self) -> None:
        """Dismiss without changing files."""
        self.dismiss(False)

    @on(Button.Pressed, "#confirm-cancel")
    def on_cancel(self) -> None:
        """Cancel the pending action."""
        self.dismiss(False)

    @on(Button.Pressed, "#confirm-action")
    def on_confirm(self) -> None:
        """Return an explicit confirmation to the owning app."""
        self.dismiss(True)


# ---------------------------------------------------------------------------
# Main App
# ---------------------------------------------------------------------------


class FolionymTUI(App[None]):
    """Terminal UI for Folionym."""

    TITLE = "Folionym"
    CSS = _CSS
    ENABLE_COMMAND_PALETTE = False
    BINDINGS: ClassVar[list[Binding | tuple[str, str] | tuple[str, str, str]]] = [
        Binding("ctrl+q", "quit", "Quit"),
        Binding("ctrl+p", "preview", "Preview (dry run)"),
        Binding("ctrl+a", "apply", "Apply renames"),
        Binding("ctrl+c", "cancel", "Cancel run"),
    ]

    def __init__(self) -> None:
        """Load persisted form state and initialize cancellation and run counters."""
        super().__init__()
        self.register_theme(FOLIONYM_THEME)
        self.theme = FOLIONYM_THEME.name
        self._stop_event = threading.Event()
        self._operation_running = False
        self._exit_after_run = False
        self._run_is_preview = True
        self._settings = _load_settings()
        self._run_counts = {"renamed": 0, "skipped": 0, "failed": 0}
        self._preview_records: list[tuple[str, str, str]] = []

    # ------------------------------------------------------------------
    # Compose
    # ------------------------------------------------------------------

    def compose(self) -> ComposeResult:
        """Compose the TUI layout while leaving application state in the controller."""
        with Horizontal(id="brand-bar"):
            yield Static("Folionym", id="brand-name")
            yield Static(
                str(self._settings.get("directory", "")) or "No folder selected",
                id="current-scope",
                markup=False,
            )
            yield Static("●  Processed locally", id="trust-state")
        with Vertical(id="app-workspace"):
            with Horizontal(id="workflow-header"):
                yield Button("1  Source\nChoose PDFs", id="nav-basic", classes="workflow-nav active")
                yield Button("2  Fine-tune\nRules & models", id="nav-advanced", classes="workflow-nav")
                yield Button("3  Review & rename\nPreview & apply", id="nav-run", classes="workflow-nav")
            with TabbedContent(initial="basic", id="workflow-tabs"):
                with TabPane("1  Setup", id="basic"):
                    yield from self._compose_basic()
                with TabPane("2  Fine-tune", id="advanced"):
                    yield from self._compose_advanced()
                with TabPane("3  Review & rename", id="run"):
                    yield from self._compose_run()
        yield Footer()

    def on_mount(self) -> None:
        """Initialize the review table and responsive shell after widgets are mounted."""
        table = self.query_one("#preview-table", DataTable)
        table.add_columns("STATUS", "SOURCE", "PROPOSED NAME")
        table.display = False
        self.query_one("#run-progress", ProgressBar).display = False
        self.query_one("#run-log", RichLog).display = False
        self._update_effective_configuration()
        self._sync_compact_layout(self.size.width)

    def on_resize(self, event: events.Resize) -> None:
        """Collapse secondary chrome when the supported compact terminal is active."""
        self._sync_compact_layout(event.size.width)

    def _sync_compact_layout(self, width: int) -> None:
        """Switch between the reference workbench and the compact terminal fallback."""
        self.screen.set_class(width < 100, "compact")

    def _activate_tab(self, pane_id: str) -> None:
        """Activate one workflow pane and keep the workflow header synchronized."""
        self.query_one("#workflow-tabs", TabbedContent).active = pane_id
        self._sync_workflow_nav(pane_id)
        if pane_id == "run":
            self._update_effective_configuration()

    def _sync_workflow_nav(self, pane_id: str) -> None:
        """Expose the active workflow stage without relying on color alone."""
        for candidate in ("basic", "advanced", "run"):
            with contextlib.suppress(QueryError):
                self.query_one(f"#nav-{candidate}", Button).set_class(candidate == pane_id, "active")

    @on(TabbedContent.TabActivated, "#workflow-tabs")
    def on_workflow_tab_activated(self, message: TabbedContent.TabActivated) -> None:
        """Keep the workflow header aligned with the active content pane."""
        if message.pane.id:
            self._sync_workflow_nav(message.pane.id)

    @on(Input.Changed, "#directory")
    def on_directory_changed(self, message: Input.Changed) -> None:
        """Keep the current local scope visible in the shared application header."""
        with contextlib.suppress(QueryError):
            self.query_one("#current-scope", Static).update(message.value.strip() or "No folder selected")

    def _compose_basic(self) -> ComposeResult:
        """Yield the Settings pane from the persisted form values."""
        yield from compose_basic(self._settings, _PRESETS)

    def _compose_advanced(self) -> ComposeResult:
        """Yield advanced limits, integrations, and LLM controls."""
        yield from compose_advanced(self._settings)

    def _compose_run(self) -> ComposeResult:
        """Yield progress, log, and action controls for active runs."""
        yield from compose_run()

    # ------------------------------------------------------------------
    # Settings
    # ------------------------------------------------------------------

    def get_str(self, widget_id: str, default: str = "") -> str:
        """Read and trim an input widget value by ID."""
        try:
            w = self.query_one(f"#{widget_id}", Input)
            return str(w.value).strip()
        except QueryError:
            logger.debug("Widget query failed for #%s (Input)", widget_id)
            return default

    def get_bool(self, widget_id: str, default: bool = False) -> bool:
        """Read a checkbox widget value by ID."""
        try:
            w = self.query_one(f"#{widget_id}", Checkbox)
            return bool(w.value)
        except QueryError:
            logger.debug("Widget query failed for #%s (Checkbox)", widget_id)
            return default

    def get_select(self, widget_id: str, default: str = "") -> str:
        """Read a select widget value by ID."""
        try:
            w = self.query_one(f"#{widget_id}", Select)
            v = w.value
            return str(v) if v is not Select.BLANK else default
        except QueryError:
            logger.debug("Widget query failed for #%s (Select)", widget_id)
            return default

    def snapshot(self) -> dict[str, object]:
        """Return the current TUI form state as build-config input."""
        return {
            "directory": self.get_str("directory"),
            "single_file": self.get_str("single_file"),
            "language": self.get_select("language", "de"),
            "case": self.get_select("case", "kebabCase"),
            "date_format": self.get_select("date_format", "dmy"),
            "preset": self.get_select("preset", ""),
            "project": self.get_str("project"),
            "version": self.get_str("version"),
            "template": self.get_str("template"),
            "backup_dir": self.get_str("backup_dir"),
            "rename_log": self.get_str("rename_log"),
            "export_metadata": self.get_str("export_metadata"),
            "summary_json": self.get_str("summary_json"),
            "rules_file": self.get_str("rules_file"),
            "post_rename_hook": self.get_str("post_rename_hook"),
            "llm_url": self.get_str("llm_url"),
            "llm_model": self.get_str("llm_model"),
            "llm_timeout": self.get_str("llm_timeout"),
            "max_tokens": self.get_str("max_tokens"),
            "max_content_chars": self.get_str("max_content_chars"),
            "max_content_tokens": self.get_str("max_content_tokens"),
            "workers": self.get_str("workers"),
            "max_filename_chars": self.get_str("max_filename_chars"),
            # Retain the persisted key for backward compatibility. The explicit
            # Preview and Apply actions own run mode, so no duplicate form toggle
            # is rendered.
            "dry_run": True,
            "use_llm": self.get_bool("use_llm", True),
            "use_ocr": self.get_bool("use_ocr"),
            "recursive": self.get_bool("recursive"),
            "skip_already_named": self.get_bool("skip_already_named"),
            "use_pdf_metadata_date": self.get_bool("use_pdf_metadata_date", True),
            "use_structured_fields": self.get_bool("use_structured_fields", True),
            "write_pdf_metadata": self.get_bool("write_pdf_metadata"),
            "use_vision_fallback": self.get_bool("use_vision_fallback"),
            "simple_naming_mode": self.get_bool("simple_naming_mode"),
            "vision_first": self.get_bool("vision_first"),
        }

    @property
    def run_active(self) -> bool:
        """Whether a preview/apply run is currently active."""
        return self._operation_running

    @run_active.setter
    def run_active(self, value: bool) -> None:
        """Report whether a worker is active so actions can enforce lifecycle boundaries."""
        self._operation_running = bool(value)

    @property
    def stop_requested(self) -> bool:
        """Whether cancellation has been requested for the active run."""
        return self._stop_event.is_set()

    def clear_stop_request(self) -> None:
        """Clear the cancellation flag before starting or simulating a run."""
        self._stop_event.clear()

    def build_config(self, *, dry_run: bool, manual_mode: bool = False) -> RenamerConfig:
        """Build a rename configuration from the current TUI form state."""
        return build_config_from_snapshot(
            self.snapshot(),
            self._stop_event,
            dry_run=dry_run,
            manual_mode=manual_mode,
        )

    def _endpoint_disclosure(self) -> tuple[str, str]:
        """Describe the configured content boundary without exposing endpoint details."""
        if not self.get_bool("use_llm", True):
            return "HEURISTICS ONLY", "Document text stays on this machine."
        endpoint_value = self.get_str("llm_url")
        if not endpoint_value:
            return "LOCAL HTTP MODEL", "Document text may be sent to the preset local endpoint."
        try:
            endpoint = validate_http_endpoint(endpoint_value)
        except ValueError:
            return "HTTP MODEL", "Review the endpoint before sending document-derived content."
        if endpoint.is_loopback:
            return "LOCAL HTTP MODEL", "Document text may be sent to the configured local endpoint."
        return "EXTERNAL HTTP MODEL", "Document-derived content may leave this machine."

    def _update_effective_configuration(self) -> None:
        """Render a concise, privacy-aware summary of the controls that shape the next run."""
        language = self.get_select("language", "de")
        case_style = self.get_select("case", "kebabCase")
        preset = self.get_select("preset", "") or "custom"
        ocr_state = "enabled" if self.get_bool("use_ocr") else "off"
        endpoint_label, disclosure = self._endpoint_disclosure()
        with contextlib.suppress(QueryError):
            self.query_one("#privacy-disclosure", Static).update(
                f"[b]{endpoint_label}[/b]\n{_escape_markup(disclosure)}"
            )
            self.query_one("#effective-config", Static).update(
                "\n".join(
                    (
                        f"LANGUAGE       {language.upper()}",
                        f"FILENAME STYLE {case_style}",
                        f"PRESET         {preset}",
                        f"OCR            {ocr_state}",
                    )
                )
            )

    def _clear_preview_records(self) -> None:
        """Reset structured preview presentation without discarding the chronological log."""
        self._preview_records.clear()
        with contextlib.suppress(QueryError):
            self.query_one("#preview-table", DataTable).clear(columns=False)
            self.query_one("#preview-table", DataTable).display = False
            self.query_one("#preview-empty", Static).display = True
            self.query_one("#inspector-source", Static).update("No document selected")
            self.query_one("#inspector-proposed", Static).update("Run a preview to inspect a proposed name.")
            self.query_one("#inspector-mode", Static).update("Waiting for preview output")

    def _record_preview_result(self, line: str) -> None:
        """Project recognized rename output into the structured preview table."""
        match = _DRYRUN_LOG_RE.search(line)
        status = "SUGGESTED"
        status_color = PREVIEW_COLOR
        mode = "Preview only. No file changed."
        if match is None:
            match = _RENAME_LOG_RE.search(line)
            status = "RENAMED"
            status_color = SUCCESS_COLOR
            mode = "Rename completed."
        if match is None:
            return
        source = Path(match.group(1)).name
        proposed = Path(match.group(2)).name
        self._preview_records.append((status, source, proposed))
        try:
            table = self.query_one("#preview-table", DataTable)
            table.add_row(Text(status, style=status_color), source, proposed, key=str(len(self._preview_records) - 1))
            table.display = True
            self.query_one("#preview-empty", Static).display = False
            self._show_preview_record(len(self._preview_records) - 1, mode=mode)
        except QueryError:
            return

    def _show_preview_record(self, index: int, *, mode: str | None = None) -> None:
        """Show the selected source-to-target transformation in the inspector."""
        if not 0 <= index < len(self._preview_records):
            return
        status, source, proposed = self._preview_records[index]
        self.query_one("#inspector-source", Static).update(_escape_markup(source))
        self.query_one("#inspector-proposed", Static).update(f"[b]{_escape_markup(proposed)}[/b]")
        self.query_one("#inspector-mode", Static).update(mode or status.title())

    # ------------------------------------------------------------------
    # Run worker
    # ------------------------------------------------------------------

    @work(thread=True, exclusive=True)
    def run_directory_worker(self, directory: str, config: RenamerConfig) -> None:
        """Run directory processing in a Textual-managed thread worker."""
        handler = _TextualLogHandler(self)
        root_logger = logging.getLogger()
        root_logger.addHandler(handler)
        try:
            rename_pdfs_in_directory(directory, config=config)
            self.call_from_thread(self.post_message, _RunFinished(True, "Completed"))
        except _TUI_WORKER_EXCEPTIONS as exc:
            self.call_from_thread(self.post_message, _RunFinished(False, str(exc)))
        finally:
            root_logger.removeHandler(handler)

    def _set_run_controls_active(self, running: bool) -> None:
        """Enable or disable run controls so only valid actions are reachable at any time."""
        for button_id in ("#btn-preview", "#btn-apply", "#btn-one"):
            with contextlib.suppress(QueryError):
                self.query_one(button_id, Button).disabled = running
        with contextlib.suppress(QueryError):
            self.query_one("#btn-cancel", Button).disabled = not running

    def _set_status(self, text: str, css_class: str = "status-idle") -> None:
        """Update the status label text and styling with a status indicator prefix."""
        _STATUS_INDICATORS = {
            "status-idle": "[dim]IDLE[/dim]",
            "status-running": f"[bold {WARNING_COLOR}]RUN[/bold {WARNING_COLOR}]",
            "status-done": f"[bold {SUCCESS_COLOR}]DONE[/bold {SUCCESS_COLOR}]",
            "status-error": f"[bold {ERROR_COLOR}]FAIL[/bold {ERROR_COLOR}]",
            "status-cancel": f"[{WARNING_COLOR}]STOP[/{WARNING_COLOR}]",
        }
        prefix = _STATUS_INDICATORS.get(css_class, "")
        status = self.query_one("#run-status", Static)
        status.update(f"{prefix}  {text}" if prefix else text)
        for cls in ("status-idle", "status-running", "status-done", "status-error", "status-cancel"):
            status.remove_class(cls)
        status.add_class(css_class)

    def _format_log_line(self, line: str) -> str:
        """Apply Rich markup to log lines for better visual clarity and track run counters."""
        stripped = line.rstrip()
        for predicate, formatter in self._log_line_formatters():
            if predicate(stripped):
                return formatter(stripped)
        return stripped

    def _log_line_formatters(self) -> tuple[tuple[Callable[[str], bool], Callable[[str], str]], ...]:
        """Return classifiers and formatters in priority order so each log line gets one presentation."""
        return (
            (self._is_rename_log_line, self._format_rename_log_line),
            (self._is_dryrun_log_line, self._format_dryrun_log_line),
            (self._is_skip_log_line, self._format_skip_log_line),
            (self._is_error_log_line, self._format_error_log_line),
            (self._is_progress_log_line, self._format_progress_log_line),
            (self._is_summary_log_line, self._format_summary_log_line),
            (self._is_info_log_line, self._format_info_log_line),
        )

    @staticmethod
    def _is_rename_log_line(stripped: str) -> bool:
        """Return whether a log line is a rename log line; formatter selection depends on this classification."""
        return "Renamed '" in stripped and "' to '" in stripped

    def _format_rename_log_line(self, stripped: str) -> str:
        """Format rename log line for safe, operator-readable TUI output."""
        self._increment_run_count("renamed")
        return _RENAME_LOG_RE.sub(_format_rename_match, stripped) if _RENAME_LOG_RE.search(stripped) else stripped

    @staticmethod
    def _is_dryrun_log_line(stripped: str) -> bool:
        """Return whether a log line is a dryrun log line; formatter selection depends on this classification."""
        return "Dry-run: would rename '" in stripped and "' to '" in stripped

    def _format_dryrun_log_line(self, stripped: str) -> str:
        """Format dryrun log line for safe, operator-readable TUI output."""
        self._increment_run_count("renamed")
        return _DRYRUN_LOG_RE.sub(_format_dryrun_match, stripped) if _DRYRUN_LOG_RE.search(stripped) else stripped

    def _format_skip_log_line(self, stripped: str) -> str:
        """Format skip log line for safe, operator-readable TUI output."""
        prefix = f"[{WARNING_COLOR}]SKIP[/{WARNING_COLOR}] [dim]"
        return self._format_counted_log_line(stripped, count_key="skipped", prefix=prefix)

    def _format_error_log_line(self, stripped: str) -> str:
        """Format error log line for safe, operator-readable TUI output."""
        prefix = f"[{ERROR_COLOR}]ERR[/{ERROR_COLOR}]  [bold {ERROR_COLOR}]"
        return self._format_counted_log_line(stripped, count_key="failed", prefix=prefix)

    @staticmethod
    def _is_progress_log_line(stripped: str) -> bool:
        """Return whether a log line is a progress log line; formatter selection depends on this classification."""
        return bool(PROCESS_RE.search(stripped))

    @staticmethod
    def _format_progress_log_line(stripped: str) -> str:
        """Format progress log line for safe, operator-readable TUI output."""
        return f"[dim]{stripped}[/dim]"

    @staticmethod
    def _is_summary_log_line(stripped: str) -> bool:
        """Return whether a log line is a summary log line; formatter selection depends on this classification."""
        return stripped.startswith("Summary:")

    @staticmethod
    def _format_summary_log_line(stripped: str) -> str:
        """Format summary log line for safe, operator-readable TUI output."""
        return f"[bold]{stripped}[/bold]"

    @staticmethod
    def _is_info_log_line(stripped: str) -> bool:
        """Return whether the line is an informational status line."""
        return "Heuristic-only mode" in stripped

    @staticmethod
    def _format_info_log_line(stripped: str) -> str:
        """Format info log line for safe, operator-readable TUI output."""
        return f"[{PREVIEW_COLOR}]INFO[/{PREVIEW_COLOR}] [dim]{stripped}[/dim]"

    def _format_counted_log_line(self, stripped: str, *, count_key: str, prefix: str) -> str:
        """Format counted log line for safe, operator-readable TUI output."""
        self._increment_run_count(count_key)
        suffix = "[/dim]" if count_key == "skipped" else f"[/bold {ERROR_COLOR}]"
        return f"{prefix}{stripped}{suffix}"

    def _increment_run_count(self, count_key: str) -> None:
        """Increment run count in one place so displayed progress remains consistent."""
        self._run_counts[count_key] += 1
        self._update_summary()

    @staticmethod
    def _is_skip_log_line(stripped: str) -> bool:
        """Return whether a log line is a skip log line; formatter selection depends on this classification."""
        return "Skipping " in stripped or "Skipped" in stripped or "content is empty" in stripped

    @staticmethod
    def _is_error_log_line(stripped: str) -> bool:
        """Return whether the line represents an error."""
        return "Failed" in stripped or "Error" in stripped or "failed" in stripped

    def _update_summary(self) -> None:
        """Update the run summary counters display."""
        c = self._run_counts
        parts = []
        if c["renamed"]:
            result_label = "suggestions" if self._run_is_preview else "renamed"
            parts.append(f"[{SUCCESS_COLOR}]{c['renamed']} {result_label}[/{SUCCESS_COLOR}]")
        if c["skipped"]:
            parts.append(f"[{WARNING_COLOR}]{c['skipped']} skipped[/{WARNING_COLOR}]")
        if c["failed"]:
            parts.append(f"[{ERROR_COLOR}]{c['failed']} failed[/{ERROR_COLOR}]")
        summary_text = "  |  ".join(parts) if parts else ""
        with contextlib.suppress(QueryError):
            self.query_one("#run-summary", Static).update(summary_text)
            self.query_one("#metric-suggestions", Static).update(
                f"[b]{c['renamed']}[/b] {'suggestions' if self._run_is_preview else 'renamed'}"
            )
            self.query_one("#metric-skipped", Static).update(f"[b]{c['skipped']}[/b] skipped")
            self.query_one("#metric-failed", Static).update(f"[b]{c['failed']}[/b] failed")

    @on(_RunLog)
    def on_run_log(self, message: _RunLog) -> None:
        """Render a worker log line after Textual returns to the UI thread."""
        try:
            log = self.query_one("#run-log", RichLog)
            progress = self.query_one("#run-progress", ProgressBar)
            counter = self.query_one("#run-file-counter", Static)
        except QueryError:
            return
        log.write(self._format_log_line(message.line))
        self._record_preview_result(message.line)
        self._update_progress_from_log_line(message.line, progress, counter)

    @on(_RunFinished)
    def on_run_finished(self, message: _RunFinished) -> None:
        """Finish a run once its managed worker reports a terminal result."""
        try:
            log = self.query_one("#run-log", RichLog)
            counter = self.query_one("#run-file-counter", Static)
        except QueryError:
            return
        self._operation_running = False
        self._set_run_controls_active(False)
        if message.ok:
            cancelled = self._stop_event.is_set()
            status = "Cancelled" if cancelled else "Completed"
            self._set_status(status, "status-cancel" if cancelled else "status-done")
            log.write(
                f"\n[bold {SUCCESS_COLOR}]Run completed.[/bold {SUCCESS_COLOR}]  {self._completion_summary_line()}"
            )
            self.notify(
                self._completion_summary_line(),
                title="Run cancelled" if cancelled else "Run completed",
                severity="warning" if cancelled else "information",
            )
        else:
            self._set_status(f"Failed: {message.message}", "status-error")
            log.write(f"[bold {ERROR_COLOR}]Run failed:[/bold {ERROR_COLOR}] {_escape_markup(message.message)}")
            self.notify(message.message, title="Run failed", severity="error")
        counter.update("")
        if self._exit_after_run:
            self.exit()

    def _completion_summary_line(self) -> str:
        """Summarize a completed run using counters accumulated from worker output."""
        summary_parts = []
        c = self._run_counts
        if c["renamed"]:
            result_label = "suggestions" if self._run_is_preview else "renamed"
            summary_parts.append(f"[{SUCCESS_COLOR}]{c['renamed']} {result_label}[/{SUCCESS_COLOR}]")
        if c["skipped"]:
            summary_parts.append(f"[{WARNING_COLOR}]{c['skipped']} skipped[/{WARNING_COLOR}]")
        if c["failed"]:
            summary_parts.append(f"[{ERROR_COLOR}]{c['failed']} failed[/{ERROR_COLOR}]")
        return "  ".join(summary_parts) if summary_parts else "no files processed"

    def _update_progress_from_log_line(self, line: str, progress: ProgressBar, counter: Static) -> None:
        """Advance progress only from recognized processing lines to avoid misleading UI state."""
        match = PROCESS_RE.search(line)
        if not match:
            return
        cur = int(match.group(1))
        tot = max(1, int(match.group(2)))
        progress.update(total=tot, progress=cur)
        self._set_status(f"Processing {cur}/{tot}...", "status-running")
        counter.update(f"{cur} of {tot} files")
        with contextlib.suppress(QueryError):
            self.query_one("#metric-files", Static).update(f"[b]{tot}[/b] PDFs")

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def action_preview(self) -> None:
        """Handle the preview action through the shared run lifecycle."""
        self.start_run(dry_run=True)

    def action_apply(self) -> None:
        """Ask for confirmation before entering the mutating run lifecycle."""
        self.request_apply()

    def action_cancel(self) -> None:
        """Handle the cancel action through the shared run lifecycle."""
        self.cancel_run()

    def _start_run(self, *, dry_run: bool) -> None:
        """Start a preview or apply run while keeping worker lifecycle ownership in the TUI."""
        self.start_run(dry_run=dry_run)

    def request_apply(self) -> None:
        """Confirm a batch rename while keeping Preview as the fast, safe path."""
        if self._operation_running:
            self.notify("A run is already in progress.", severity="warning")
            return
        directory = self.get_str("directory")
        target = directory or "the configured folder"
        detail = (
            f"Folionym will recompute suggestions for {target} and rename matching PDFs. "
            "LLM-backed names may differ from a previous preview."
        )
        self.push_screen(
            ConfirmActionScreen("Apply folder renames?", detail, "Apply renames"),
            self._finish_apply_confirmation,
        )

    def _finish_apply_confirmation(self, confirmed: bool | None) -> None:
        """Start an apply run only after an affirmative modal result."""
        if confirmed:
            self.start_run(dry_run=False)

    def request_process_one(self) -> None:
        """Validate and confirm an immediate single-PDF rename."""
        if self._operation_running:
            self.notify("A run is already in progress.", severity="warning")
            return
        selected = self._selected_single_pdf()
        if selected is None:
            return
        detail = (
            f"Folionym will generate a name for {selected.name} and rename it immediately. "
            "Use Preview folder when you need a non-mutating review."
        )
        self.push_screen(
            ConfirmActionScreen("Rename one PDF?", detail, "Rename PDF"),
            self._finish_single_confirmation,
        )

    def _finish_single_confirmation(self, confirmed: bool | None) -> None:
        """Start one-file processing only after an affirmative modal result."""
        if confirmed:
            self.process_one()

    def start_run(self, *, dry_run: bool) -> None:
        """Start a preview or apply run from the current TUI form state."""
        if self._operation_running:
            self.notify("A run is already in progress.", severity="warning")
            return
        directory = self.get_str("directory")
        if not directory:
            msg = "Set a folder path on the Setup tab first."
            self._show_setup_error("directory", msg)
            self.query_one("#run-log", RichLog).write(
                f"[bold {ERROR_COLOR}]No directory set.[/bold {ERROR_COLOR}] {msg}"
            )
            self.notify(msg, title="No directory set", severity="error")
            return
        if not Path(directory).is_dir():
            self._show_setup_error("directory", f"Folder not found: {directory}")
            self.query_one("#run-log", RichLog).write(
                f"[bold {ERROR_COLOR}]Directory not found:[/bold {ERROR_COLOR}] {_escape_markup(directory)}"
            )
            self.notify(directory, title="Directory not found", severity="error")
            return
        try:
            config = self.build_config(dry_run=dry_run)
        except ValueError as exc:
            err_msg = _escape_markup(str(exc))
            self.query_one("#run-log", RichLog).write(
                f"[bold {ERROR_COLOR}]Invalid config:[/bold {ERROR_COLOR}] {err_msg}"
            )
            self.notify(str(exc), title="Invalid config", severity="error")
            return
        setup_error = self.query_one("#setup-error", Static)
        setup_error.remove_class("visible")
        setup_error.update("")
        _save_settings(self.snapshot())
        self._activate_tab("run")
        self._stop_event.clear()
        self._exit_after_run = False
        self._operation_running = True
        self._run_is_preview = dry_run
        self._set_run_controls_active(True)
        self.call_after_refresh(self.query_one("#btn-cancel", Button).focus)
        self._run_counts = {"renamed": 0, "skipped": 0, "failed": 0}
        self._clear_preview_records()
        self._update_effective_configuration()
        with contextlib.suppress(QueryError):
            self.query_one("#run-summary", Static).update("")
        log = self.query_one("#run-log", RichLog)
        log.display = True
        mode_label = "Preview (dry run)" if dry_run else "Apply Renames"
        mode_color = PREVIEW_COLOR if dry_run else SUCCESS_COLOR
        log.write(f"\n[bold {mode_color}]{'=' * 50}[/bold {mode_color}]")
        log.write(f"[bold {mode_color}]  {mode_label}[/bold {mode_color}]")
        log.write(f"[bold {mode_color}]{'=' * 50}[/bold {mode_color}]")
        log.write(f"[dim]Directory: {_escape_markup(directory)}[/dim]")
        self._set_status(f"Starting {mode_label.lower()}...", "status-running")
        progress = self.query_one("#run-progress", ProgressBar)
        progress.display = True
        progress.update(total=100, progress=0)
        self.run_directory_worker(directory, config)

    def _show_setup_error(self, widget_id: str, message: str) -> None:
        """Keep a source-path error visible beside its field and move focus to the fix."""
        self._activate_tab("basic")
        error = self.query_one("#setup-error", Static)
        error.update(message)
        error.add_class("visible")
        with contextlib.suppress(QueryError):
            self.query_one(f"#{widget_id}", Input).focus()

    def cancel_run(self) -> None:
        """Set the cooperative stop flag; the active file finishes before cancellation completes."""
        if not self._operation_running:
            return
        self._stop_event.set()
        self._set_status("Cancelling...", "status-cancel")
        self.query_one("#run-log", RichLog).write(
            f"[{WARNING_COLOR}]Cancel requested -- finishing current file...[/{WARNING_COLOR}]"
        )

    async def action_quit(self) -> None:
        """Exit only after an active operation reaches its cancellation boundary."""
        if not self._operation_running:
            self.exit()
            return
        self._exit_after_run = True
        self.cancel_run()
        self.query_one("#run-log", RichLog).write(
            f"[{WARNING_COLOR}]Quit requested -- the app will close after the current file finishes.[/{WARNING_COLOR}]"
        )

    def _selected_single_pdf(self) -> Path | None:
        """Return the selected file only if it exists, is not a symlink, and has a .pdf extension."""
        log = self.query_one("#run-log", RichLog)
        file_path = self.get_str("single_file")
        if not file_path:
            msg = "Set a single PDF path on the Setup tab first."
            self._show_setup_error("single_file", msg)
            log.write(f"[bold {ERROR_COLOR}]No file set.[/bold {ERROR_COLOR}] {msg}")
            self.notify(msg, title="No file set", severity="error")
            return None
        fp = Path(file_path)
        if fp.is_symlink():
            self._show_setup_error("single_file", "Symbolic links are unsupported; choose the PDF itself.")
            log.write(
                f"[bold {ERROR_COLOR}]Unsupported file:[/bold {ERROR_COLOR}] "
                f"{_escape_markup(file_path)} is a symbolic link"
            )
            self.notify(file_path, title="Symbolic links are unsupported", severity="error")
            return None
        if not fp.exists():
            self._show_setup_error("single_file", f"File not found: {file_path}")
            log.write(f"[bold {ERROR_COLOR}]File not found:[/bold {ERROR_COLOR}] {_escape_markup(file_path)}")
            self.notify(file_path, title="File not found", severity="error")
            return None
        if fp.suffix.lower() != ".pdf":
            self._show_setup_error("single_file", "Choose a file with a .pdf extension.")
            log.write(
                f"[bold {ERROR_COLOR}]Not a PDF file:[/bold {ERROR_COLOR}] "
                f"{_escape_markup(file_path)} (expected .pdf extension)"
            )
            self.notify(file_path, title="Not a PDF file", severity="error")
            return None
        return fp

    def _start_single_file_ui(self, fp: Path) -> None:
        """Start one-file processing after selection is validated on the UI thread."""
        self._activate_tab("run")
        log = self.query_one("#run-log", RichLog)
        log.display = True
        self.query_one("#run-progress", ProgressBar).display = False
        log.write(f"\n[bold {PREVIEW_COLOR}]Single File Processing[/bold {PREVIEW_COLOR}]")
        log.write(f"[dim]File: {_escape_markup(fp.name)}[/dim]")
        self._set_status("Processing single file...", "status-running")
        self._operation_running = True
        self._set_run_controls_active(True)
        self._run_counts = {"renamed": 0, "skipped": 0, "failed": 0}
        self._clear_preview_records()
        self._update_effective_configuration()
        with contextlib.suppress(Exception):
            self.query_one("#run-summary", Static).update("")

    @work(thread=True, exclusive=True)
    def _single_file_worker(self, fp: Path, config: RenamerConfig) -> None:
        """Run one-file processing in a worker so filesystem work never blocks Textual rendering."""
        try:
            new_base, meta, err = suggest_rename_for_file(fp, config)
            if err is not None:
                self._post_worker_log(f"[bold {ERROR_COLOR}]Error:[/bold {ERROR_COLOR}] {_escape_markup(str(err))}\n")
                self.call_from_thread(self.post_message, _RunFinished(False, str(err)))
                return
            if new_base is None:
                self._post_worker_log(f"[{WARNING_COLOR}]Skipped -- no extractable content.[/{WARNING_COLOR}]\n")
                self.call_from_thread(self.post_message, _RunFinished(True, "Skipped"))
                return
            suggested = new_base + fp.suffix
            self._post_worker_log(f"[dim]Suggested:[/dim] [bold]{_escape_markup(suggested)}[/bold]\n")
            self._rename_single_file(fp, config, new_base, meta or {})
        except _TUI_WORKER_EXCEPTIONS as exc:
            self.call_from_thread(self.post_message, _RunFinished(False, str(exc)))

    def _post_worker_log(self, line: str) -> None:
        """Marshal a worker-produced log line onto Textual's UI thread via _RunLog."""
        self.call_from_thread(self.post_message, _RunLog(line))

    def _rename_single_file(self, fp: Path, config: RenamerConfig, new_base: str, meta: dict[str, object]) -> None:
        """Apply an accepted one-file suggestion with shared rename safeguards and callbacks."""
        export_rows: list[dict[str, object]] = []
        _on_rename_success = _make_post_rename_success_callback(config, meta, export_rows)
        success, target = apply_single_rename(
            fp,
            sanitize_filename_base(new_base),
            RenameApplyOptions(
                plan_file_path=None,
                plan_entries=[],
                dry_run=False,
                backup_dir=config.output.paths.backup_dir,
                on_success=_on_rename_success,
                max_filename_chars=config.output.naming.max_filename_chars,
            ),
        )
        if success:
            msg = (
                f"[{SUCCESS_COLOR}]Renamed[/{SUCCESS_COLOR}] [dim]{_escape_markup(fp.name)}[/dim]"
                f" [{SUCCESS_COLOR} bold]->[/{SUCCESS_COLOR} bold] [bold]{_escape_markup(target.name)}[/bold]\n"
            )
            self._post_worker_log(msg)
            self._log_single_file_meta(meta)
            self.call_from_thread(self.post_message, _RunFinished(True, "Completed"))
        else:
            self._post_worker_log(f"[bold {ERROR_COLOR}]Could not rename file.[/bold {ERROR_COLOR}]\n")
            self.call_from_thread(self.post_message, _RunFinished(False, "Could not rename file"))

    def _log_single_file_meta(self, meta: dict[str, object]) -> None:
        """Emit one-file metadata through the UI log so operators can inspect the outcome."""
        meta_parts = []
        for k in ("category", "summary", "keywords", "category_source"):
            v = meta.get(k)
            if v:
                meta_parts.append(f"[dim]{k}:[/dim] {_escape_markup(str(v))}")
        if meta_parts:
            self._post_worker_log("  " + "  |  ".join(meta_parts) + "\n")

    def process_one(self) -> None:
        """Process the currently selected single PDF."""
        if self._operation_running:
            self.notify("A run is already in progress.", severity="warning")
            return
        fp = self._selected_single_pdf()
        if fp is None:
            return
        self._stop_event.clear()
        self._exit_after_run = False
        try:
            config = self.build_config(dry_run=False, manual_mode=True)
        except ValueError as exc:
            self.query_one("#run-log", RichLog).write(
                f"[bold {ERROR_COLOR}]Invalid config:[/bold {ERROR_COLOR}] {_escape_markup(str(exc))}"
            )
            self.notify(str(exc), title="Invalid config", severity="error")
            return
        self._start_single_file_ui(fp)
        self._single_file_worker(fp, config)

    # ------------------------------------------------------------------
    # Button handlers
    # ------------------------------------------------------------------

    @on(Button.Pressed, "#nav-basic")
    def on_nav_basic(self) -> None:
        """Open the source and naming stage from the workflow rail."""
        self._activate_tab("basic")

    @on(Button.Pressed, "#nav-advanced")
    def on_nav_advanced(self) -> None:
        """Open expert controls from the workflow rail."""
        self._activate_tab("advanced")

    @on(Button.Pressed, "#nav-run")
    def on_nav_run(self) -> None:
        """Open the review workbench from the workflow rail."""
        self._activate_tab("run")

    @on(DataTable.RowHighlighted, "#preview-table")
    def on_preview_row_highlighted(self, message: DataTable.RowHighlighted) -> None:
        """Keep the document inspector synchronized with keyboard row focus."""
        self._show_preview_record(message.cursor_row)

    @on(Button.Pressed, "#btn-preview")
    def on_preview(self) -> None:
        """React to preview without duplicating action policy."""
        self.start_run(dry_run=True)

    @on(Button.Pressed, "#btn-apply")
    def on_apply(self) -> None:
        """React to apply through the confirmation policy."""
        self.request_apply()

    @on(Button.Pressed, "#btn-one")
    def on_one(self) -> None:
        """React to one-file rename through the confirmation policy."""
        self.request_process_one()

    @on(Button.Pressed, "#btn-cancel")
    def on_cancel(self) -> None:
        """React to cancel without duplicating action policy."""
        self.cancel_run()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Initialize logging and run the Textual application."""
    setup_logging(level=logging.INFO)
    app = FolionymTUI()
    app.run()


if __name__ == "__main__":
    main()
