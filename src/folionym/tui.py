"""Terminal UI for Folionym (Textual-based, replaces Tkinter gui.py); launch with ``folionym-tui``."""

from __future__ import annotations

import contextlib
import logging
import threading
from pathlib import Path, PurePath
from typing import ClassVar

from rich.markup import escape as _escape_markup
from rich.text import Text

try:
    from textual import events, on, work
    from textual.app import App, ComposeResult
    from textual.binding import Binding
    from textual.containers import Horizontal, Vertical
    from textual.css.query import QueryError
    from textual.widgets import (
        Button,
        DataTable,
        Footer,
        Input,
        ProgressBar,
        RichLog,
        Static,
        TabbedContent,
        TabPane,
    )
except ImportError as _e:  # pragma: no cover
    raise ImportError("textual is required for the TUI. Install with: pip install -e '.[tui]'") from _e
from .config import RenamerConfig
from .logging_utils import setup_logging
from .rename_ops import apply_single_rename, sanitize_filename_base
from .renamer import _make_post_rename_success_callback, suggest_rename_for_file
from .tui_assets import (
    _PRESETS,
    ERROR_COLOR,
    FOLIONYM_THEME,
    PREVIEW_COLOR,
    SUCCESS_COLOR,
    WARNING_COLOR,
)
from .tui_confirmation import ConfirmActionScreen
from .tui_forms import compose_advanced, compose_basic, compose_run
from .tui_operations import process_single_file, run_directory_rename
from .tui_presentation import (
    completion_summary,
    effective_configuration_lines,
    format_run_log_line,
    format_run_summary,
    metric_summary,
    parse_preview_record,
    parse_progress,
)
from .tui_selection import TuiSourceSelection
from .tui_state import SETTINGS_PATH, _load_settings, _save_settings
from .tui_values import TuiValueAccess
from .tui_worker_messages import _RunFinished, _RunLog, _TextualLogHandler

__all__ = ["SETTINGS_PATH", "FolionymTUI", "_load_settings", "_save_settings", "main"]
_TUI_WORKER_EXCEPTIONS = (AttributeError, KeyError, OSError, RuntimeError, TypeError, ValueError)


class FolionymTUI(TuiSourceSelection, TuiValueAccess, App[None]):
    """Coordinate Folionym's Textual setup, review, and rename workflow."""

    TITLE = "Folionym"
    CSS_PATH: ClassVar[list[str | PurePath]] = ["tui_base.tcss", "tui_run.tcss", "tui_responsive.tcss"]
    ENABLE_COMMAND_PALETTE = False
    BINDINGS: ClassVar[list[Binding | tuple[str, str] | tuple[str, str, str]]] = [
        Binding("ctrl+q", "quit", "Quit"),
        Binding("ctrl+p", "preview", "Preview (dry run)"),
        Binding("ctrl+a", "apply", "Apply renames"),
        Binding("ctrl+c", "cancel", "Cancel run"),
    ]

    def __init__(self) -> None:
        """Load settings and initialize cancellation, run, and preview state."""
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

    def compose(self) -> ComposeResult:
        """Compose the shared shell and its three workflow panes."""
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
                    yield from compose_basic(self._settings, _PRESETS)
                with TabPane("2  Fine-tune", id="advanced"):
                    yield from compose_advanced(self._settings)
                with TabPane("3  Review & rename", id="run"):
                    yield from compose_run()
        yield Footer()

    def on_mount(self) -> None:
        """Initialize review widgets and responsive state after mounting."""
        table = self.query_one("#preview-table", DataTable)
        table.add_columns("STATUS", "SOURCE", "PROPOSED NAME")
        table.display = False
        self.query_one("#run-progress", ProgressBar).display = False
        self.query_one("#run-log", RichLog).display = False
        self._update_effective_configuration()
        self._sync_compact_layout(self.size.width)

    def on_resize(self, event: events.Resize) -> None:
        """Apply the compact layout when the terminal width changes."""
        self._sync_compact_layout(event.size.width)

    def _sync_compact_layout(self, width: int) -> None:
        """Toggle compact screen styling at the supported width boundary."""
        self.screen.set_class(width < 100, "compact")

    def _activate_tab(self, pane_id: str) -> None:
        """Activate a workflow pane and synchronize its navigation state."""
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
        """Synchronize the workflow header after a tab activation event."""
        if message.pane.id:
            self._sync_workflow_nav(message.pane.id)

    @on(Input.Changed, "#directory")
    def on_directory_changed(self, message: Input.Changed) -> None:
        """Keep the selected local directory visible in the shared header."""
        with contextlib.suppress(QueryError):
            self.query_one("#current-scope", Static).update(message.value.strip() or "No folder selected")

    def _update_effective_configuration(self) -> None:
        """Render the concise privacy-aware configuration for the next run."""
        language = self.get_select("language", "de")
        case_style = self.get_select("case", "kebabCase")
        preset = self.get_select("preset", "")
        endpoint_label, disclosure = self._endpoint_disclosure()
        with contextlib.suppress(QueryError):
            self.query_one("#privacy-disclosure", Static).update(
                f"[b]{endpoint_label}[/b]\n{_escape_markup(disclosure)}"
            )
            self.query_one("#effective-config", Static).update(
                effective_configuration_lines(language, case_style, preset, self.get_bool("use_ocr"))
            )

    def _clear_preview_records(self) -> None:
        """Reset structured preview state while retaining the chronological log."""
        self._preview_records.clear()
        with contextlib.suppress(QueryError):
            self.query_one("#preview-table", DataTable).clear(columns=False)
            self.query_one("#preview-table", DataTable).display = False
            self.query_one("#preview-empty", Static).display = True
            self.query_one("#inspector-source", Static).update("No document selected")
            self.query_one("#inspector-proposed", Static).update("Run a preview to inspect a proposed name.")
            self.query_one("#inspector-mode", Static).update("Waiting for preview output")

    def _record_preview_result(self, line: str) -> None:
        """Project a recognized rename line into the structured review table."""
        record = parse_preview_record(line)
        if record is None:
            return
        self._preview_records.append((record.status, record.source, record.proposed))
        try:
            table = self.query_one("#preview-table", DataTable)
            table.add_row(
                Text(record.status, style=record.status_color),
                record.source,
                record.proposed,
                key=str(len(self._preview_records) - 1),
            )
            table.display = True
            self.query_one("#preview-empty", Static).display = False
            self._show_preview_record(len(self._preview_records) - 1, mode=record.mode)
        except QueryError:
            return

    def _show_preview_record(self, index: int, *, mode: str | None = None) -> None:
        """Show one selected source-to-target transformation in the inspector."""
        if not 0 <= index < len(self._preview_records):
            return
        status, source, proposed = self._preview_records[index]
        self.query_one("#inspector-source", Static).update(_escape_markup(source))
        self.query_one("#inspector-proposed", Static).update(f"[b]{_escape_markup(proposed)}[/b]")
        self.query_one("#inspector-mode", Static).update(mode or status.title())

    @work(thread=True, exclusive=True)
    def run_directory_worker(self, directory: str, config: RenamerConfig) -> None:
        """Run directory processing in a Textual-managed worker thread."""
        run_directory_rename(
            directory,
            config,
            handler=_TextualLogHandler(self),
            on_finished=lambda ok, message: self.call_from_thread(self.post_message, _RunFinished(ok, message)),
        )

    def _set_run_controls_active(self, running: bool) -> None:
        """Enable only the actions valid for the current worker state."""
        for button_id in ("#btn-preview", "#btn-apply", "#btn-one"):
            with contextlib.suppress(QueryError):
                self.query_one(button_id, Button).disabled = running
        with contextlib.suppress(QueryError):
            self.query_one("#btn-cancel", Button).disabled = not running

    def _set_status(self, text: str, css_class: str = "status-idle") -> None:
        """Update the run status text, indicator, and style class."""
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

    def _increment_run_count(self, count_key: str) -> None:
        """Increment one run counter and refresh its summary widgets."""
        self._run_counts[count_key] += 1
        self._update_summary()

    def _update_summary(self) -> None:
        """Render the current run counters in summary and metric widgets."""
        summary_text = format_run_summary(self._run_counts, self._run_is_preview, separator="  |  ")
        suggestions, skipped, failed = metric_summary(self._run_counts, self._run_is_preview)
        with contextlib.suppress(QueryError):
            self.query_one("#run-summary", Static).update(summary_text)
            self.query_one("#metric-suggestions", Static).update(suggestions)
            self.query_one("#metric-skipped", Static).update(skipped)
            self.query_one("#metric-failed", Static).update(failed)

    @on(_RunLog)
    def on_run_log(self, message: _RunLog) -> None:
        """Render a worker log event on Textual's UI thread."""
        try:
            log = self.query_one("#run-log", RichLog)
            progress = self.query_one("#run-progress", ProgressBar)
            counter = self.query_one("#run-file-counter", Static)
        except QueryError:
            return
        formatted, count_key = format_run_log_line(message.line)
        if count_key is not None:
            self._increment_run_count(count_key)
        log.write(formatted)
        self._record_preview_result(message.line)
        self._update_progress_from_log_line(message.line, progress, counter)

    @on(_RunFinished)
    def on_run_finished(self, message: _RunFinished) -> None:
        """Finalize controls, status, and notification after a worker result."""
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
        """Return the established completion summary for current counters."""
        return completion_summary(self._run_counts, self._run_is_preview)

    def _update_progress_from_log_line(self, line: str, progress: ProgressBar, counter: Static) -> None:
        """Advance progress only for recognized processing lines."""
        parsed = parse_progress(line)
        if parsed is None:
            return
        cur, tot = parsed
        progress.update(total=tot, progress=cur)
        self._set_status(f"Processing {cur}/{tot}...", "status-running")
        counter.update(f"{cur} of {tot} files")
        with contextlib.suppress(QueryError):
            self.query_one("#metric-files", Static).update(f"[b]{tot}[/b] PDFs")

    def action_preview(self) -> None:
        """Start the non-mutating directory preview action."""
        self.start_run(dry_run=True)

    def action_apply(self) -> None:
        """Request confirmation before the mutating directory action."""
        self.request_apply()

    def action_cancel(self) -> None:
        """Request cooperative cancellation for the active run."""
        self.cancel_run()

    def request_apply(self) -> None:
        """Open the confirmation screen for a directory rename."""
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
        """Start directory application only after affirmative confirmation."""
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
        """Start single-file processing only after affirmative confirmation."""
        if confirmed:
            self.process_one()

    def _run_configuration(self, *, dry_run: bool) -> tuple[str, RenamerConfig] | None:
        """Validate the selected directory and build one run configuration."""
        directory = self.get_str("directory")
        if not directory:
            msg = "Set a folder path on the Setup tab first."
            self._show_setup_error("directory", msg)
            self.query_one("#run-log", RichLog).write(
                f"[bold {ERROR_COLOR}]No directory set.[/bold {ERROR_COLOR}] {msg}"
            )
            self.notify(msg, title="No directory set", severity="error")
            return None
        if not Path(directory).is_dir():
            self._show_setup_error("directory", f"Folder not found: {directory}")
            self.query_one("#run-log", RichLog).write(
                f"[bold {ERROR_COLOR}]Directory not found:[/bold {ERROR_COLOR}] {_escape_markup(directory)}"
            )
            self.notify(directory, title="Directory not found", severity="error")
            return None
        try:
            config = self.build_config(dry_run=dry_run)
        except ValueError as exc:
            err_msg = _escape_markup(str(exc))
            self.query_one("#run-log", RichLog).write(
                f"[bold {ERROR_COLOR}]Invalid config:[/bold {ERROR_COLOR}] {err_msg}"
            )
            self.notify(str(exc), title="Invalid config", severity="error")
            return None
        return (directory, config)

    def _begin_run(self, directory: str, config: RenamerConfig, *, dry_run: bool) -> None:
        """Reset presentation state and launch one managed directory worker."""
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

    def start_run(self, *, dry_run: bool) -> None:
        """Start a directory run when no other operation is active."""
        if self._operation_running:
            self.notify("A run is already in progress.", severity="warning")
            return
        run = self._run_configuration(dry_run=dry_run)
        if run is not None:
            directory, config = run
            self._begin_run(directory, config, dry_run=dry_run)

    def cancel_run(self) -> None:
        """Signal cancellation while allowing the current file to finish."""
        if not self._operation_running:
            return
        self._stop_event.set()
        self._set_status("Cancelling...", "status-cancel")
        self.query_one("#run-log", RichLog).write(
            f"[{WARNING_COLOR}]Cancel requested -- finishing current file...[/{WARNING_COLOR}]"
        )

    async def action_quit(self) -> None:
        """Exit immediately when idle or after the active file completes."""
        if not self._operation_running:
            self.exit()
            return
        self._exit_after_run = True
        self.cancel_run()
        self.query_one("#run-log", RichLog).write(
            f"[{WARNING_COLOR}]Quit requested -- the app will close after the current file finishes.[/{WARNING_COLOR}]"
        )

    def _start_single_file_ui(self, fp: Path) -> None:
        """Prepare the run pane for one-file processing."""
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
        """Process one file in a managed worker and post its result events."""
        try:
            result = process_single_file(
                fp,
                config,
                suggest=suggest_rename_for_file,
                apply=apply_single_rename,
                sanitize=sanitize_filename_base,
                success_callback=_make_post_rename_success_callback,
            )
            for line in result.log_lines:
                self.call_from_thread(self.post_message, _RunLog(line))
            self.call_from_thread(self.post_message, _RunFinished(result.ok, result.message))
        except _TUI_WORKER_EXCEPTIONS as exc:
            self.call_from_thread(self.post_message, _RunFinished(False, str(exc)))

    def process_one(self) -> None:
        """Build configuration and process the currently selected PDF."""
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

    @on(Button.Pressed, "#nav-basic")
    def on_nav_basic(self) -> None:
        """Open the source and naming workflow stage."""
        self._activate_tab("basic")

    @on(Button.Pressed, "#nav-advanced")
    def on_nav_advanced(self) -> None:
        """Open the expert configuration workflow stage."""
        self._activate_tab("advanced")

    @on(Button.Pressed, "#nav-run")
    def on_nav_run(self) -> None:
        """Open the review and rename workflow stage."""
        self._activate_tab("run")

    @on(DataTable.RowHighlighted, "#preview-table")
    def on_preview_row_highlighted(self, message: DataTable.RowHighlighted) -> None:
        """Synchronize the inspector with keyboard row focus."""
        self._show_preview_record(message.cursor_row)

    @on(Button.Pressed, "#btn-preview")
    def on_preview(self) -> None:
        """Route the preview button through the shared run lifecycle."""
        self.start_run(dry_run=True)

    @on(Button.Pressed, "#btn-apply")
    def on_apply(self) -> None:
        """Route the apply button through confirmation policy."""
        self.request_apply()

    @on(Button.Pressed, "#btn-one")
    def on_one(self) -> None:
        """Route the one-file button through confirmation policy."""
        self.request_process_one()

    @on(Button.Pressed, "#btn-cancel")
    def on_cancel(self) -> None:
        """Route the cancel button through cooperative cancellation."""
        self.cancel_run()


def main() -> None:
    """Initialize logging and run the Textual application."""
    setup_logging(level=logging.INFO)
    FolionymTUI().run()


if __name__ == "__main__":
    main()
