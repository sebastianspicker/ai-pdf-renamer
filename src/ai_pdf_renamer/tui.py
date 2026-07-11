"""Terminal UI for AI-PDF-Renamer (Textual-based, replaces Tkinter gui.py).

Launch with: ai-pdf-renamer-tui
Requires: pip install -e '.[tui]'
"""

from __future__ import annotations

import contextlib
import logging
import queue
import threading
from collections.abc import Callable
from pathlib import Path
from typing import ClassVar

from rich.markup import escape as _escape_markup

try:
    from textual import on
    from textual.app import App, ComposeResult
    from textual.binding import Binding
    from textual.css.query import QueryError
    from textual.widgets import (
        Button,
        Checkbox,
        Footer,
        Header,
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

from .logging_utils import setup_logging
from .rename_ops import apply_single_rename, sanitize_filename_base
from .renamer import (
    RenamerConfig,
    _make_post_rename_success_callback,
    rename_pdfs_in_directory,
    suggest_rename_for_file,
)
from .tui_assets import (
    _CSS,
    _DRYRUN_LOG_RE,
    _PRESETS,
    _RENAME_LOG_RE,
    PROCESS_RE,
    _format_dryrun_match,
    _format_rename_match,
)
from .tui_forms import compose_advanced, compose_basic, compose_run
from .tui_state import (
    SETTINGS_PATH,
    _load_settings,
    _QueueHandler,
    _save_settings,
    build_config_from_snapshot,
)

__all__ = [
    "SETTINGS_PATH",
    "AIRenamerTUI",
    "_QueueHandler",
    "_load_settings",
    "_save_settings",
    "main",
]

logger = logging.getLogger(__name__)

_TUI_WORKER_EXCEPTIONS = (AttributeError, KeyError, OSError, RuntimeError, TypeError, ValueError)


# ---------------------------------------------------------------------------
# Main App
# ---------------------------------------------------------------------------


class AIRenamerTUI(App[None]):
    """Terminal UI for AI-PDF-Renamer."""

    TITLE = "AI-PDF-Renamer"
    CSS = _CSS
    BINDINGS: ClassVar[list[Binding | tuple[str, str] | tuple[str, str, str]]] = [
        Binding("ctrl+q", "quit", "Quit"),
        Binding("ctrl+p", "preview", "Preview (dry run)"),
        Binding("ctrl+a", "apply", "Apply renames"),
        Binding("ctrl+c", "cancel", "Cancel run"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self._log_queue: queue.Queue[str | None] = queue.Queue()
        self._result_queue: queue.Queue[tuple[bool, str]] = queue.Queue()
        self._stop_event = threading.Event()
        self._running = False
        self._settings = _load_settings()
        self._run_counts = {"renamed": 0, "skipped": 0, "failed": 0}

    # ------------------------------------------------------------------
    # Compose
    # ------------------------------------------------------------------

    def compose(self) -> ComposeResult:
        yield Header()
        with TabbedContent(initial="basic"):
            with TabPane("Settings", id="basic"):
                yield from self._compose_basic()
            with TabPane("Advanced", id="advanced"):
                yield from self._compose_advanced()
            with TabPane("Run", id="run"):
                yield from self._compose_run()
        yield Footer()

    def _compose_basic(self) -> ComposeResult:
        yield from compose_basic(self._settings, _PRESETS)

    def _compose_advanced(self) -> ComposeResult:
        yield from compose_advanced(self._settings)

    def _compose_run(self) -> ComposeResult:
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
            "llm_backend": self.get_select("llm_backend", "http"),
            "llm_url": self.get_str("llm_url"),
            "llm_model": self.get_str("llm_model"),
            "llm_model_path": self.get_str("llm_model_path"),
            "llm_timeout": self.get_str("llm_timeout"),
            "max_tokens": self.get_str("max_tokens"),
            "max_content_chars": self.get_str("max_content_chars"),
            "max_content_tokens": self.get_str("max_content_tokens"),
            "workers": self.get_str("workers"),
            "max_filename_chars": self.get_str("max_filename_chars"),
            "dry_run": self.get_bool("dry_run", True),
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
        return self._running

    @run_active.setter
    def run_active(self, value: bool) -> None:
        self._running = bool(value)

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

    # ------------------------------------------------------------------
    # Run worker
    # ------------------------------------------------------------------

    def run_directory_worker(self, directory: str, config: RenamerConfig) -> None:
        """Run the directory rename worker and queue its result."""
        handler = _QueueHandler(self._log_queue)
        root_logger = logging.getLogger()
        root_logger.addHandler(handler)
        try:
            rename_pdfs_in_directory(directory, config=config)
            self._result_queue.put((True, "Completed"))
        except _TUI_WORKER_EXCEPTIONS as exc:
            self._result_queue.put((False, str(exc)))
        finally:
            root_logger.removeHandler(handler)
            self._log_queue.put(None)  # sentinel

    def next_worker_result(self) -> tuple[bool, str]:
        """Return the next queued worker result without blocking."""
        return self._result_queue.get_nowait()

    def next_log_queue_item(self) -> str | None:
        """Return the next queued log line or sentinel without blocking."""
        return self._log_queue.get_nowait()

    def enqueue_log_queue_item(self, item: str | None) -> None:
        """Queue a log line or sentinel for log-drain processing."""
        self._log_queue.put(item)

    def enqueue_worker_result(self, result: tuple[bool, str]) -> None:
        """Queue a worker result for log-drain processing."""
        self._result_queue.put(result)

    def _set_status(self, text: str, css_class: str = "status-idle") -> None:
        """Update the status label text and styling with a status indicator prefix."""
        _STATUS_INDICATORS = {
            "status-idle": "[dim]IDLE[/dim]",
            "status-running": "[bold yellow]RUN[/bold yellow]",
            "status-done": "[bold green]DONE[/bold green]",
            "status-error": "[bold red]FAIL[/bold red]",
            "status-cancel": "[yellow]STOP[/yellow]",
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
        return "Renamed '" in stripped and "' to '" in stripped

    def _format_rename_log_line(self, stripped: str) -> str:
        self._increment_run_count("renamed")
        return _RENAME_LOG_RE.sub(_format_rename_match, stripped) if _RENAME_LOG_RE.search(stripped) else stripped

    @staticmethod
    def _is_dryrun_log_line(stripped: str) -> bool:
        return "Dry-run: would rename '" in stripped and "' to '" in stripped

    def _format_dryrun_log_line(self, stripped: str) -> str:
        self._increment_run_count("renamed")
        return _DRYRUN_LOG_RE.sub(_format_dryrun_match, stripped) if _DRYRUN_LOG_RE.search(stripped) else stripped

    def _format_skip_log_line(self, stripped: str) -> str:
        return self._format_counted_log_line(stripped, count_key="skipped", prefix="[yellow]SKIP[/yellow] [dim]")

    def _format_error_log_line(self, stripped: str) -> str:
        return self._format_counted_log_line(stripped, count_key="failed", prefix="[red]ERR[/red]  [bold red]")

    @staticmethod
    def _is_progress_log_line(stripped: str) -> bool:
        return bool(PROCESS_RE.search(stripped))

    @staticmethod
    def _format_progress_log_line(stripped: str) -> str:
        return f"[dim]{stripped}[/dim]"

    @staticmethod
    def _is_summary_log_line(stripped: str) -> bool:
        return stripped.startswith("Summary:")

    @staticmethod
    def _format_summary_log_line(stripped: str) -> str:
        return f"[bold]{stripped}[/bold]"

    @staticmethod
    def _is_info_log_line(stripped: str) -> bool:
        return "Heuristic-only mode" in stripped

    @staticmethod
    def _format_info_log_line(stripped: str) -> str:
        return f"[cyan]INFO[/cyan] [dim]{stripped}[/dim]"

    def _format_counted_log_line(self, stripped: str, *, count_key: str, prefix: str) -> str:
        self._increment_run_count(count_key)
        suffix = "[/dim]" if count_key == "skipped" else "[/bold red]"
        return f"{prefix}{stripped}{suffix}"

    def _increment_run_count(self, count_key: str) -> None:
        self._run_counts[count_key] += 1
        self._update_summary()

    @staticmethod
    def _is_skip_log_line(stripped: str) -> bool:
        return "Skipping " in stripped or "Skipped" in stripped or "content is empty" in stripped

    @staticmethod
    def _is_error_log_line(stripped: str) -> bool:
        return "Failed" in stripped or "Error" in stripped or "failed" in stripped

    def _update_summary(self) -> None:
        """Update the run summary counters display."""
        c = self._run_counts
        parts = []
        if c["renamed"]:
            parts.append(f"[green]{c['renamed']} renamed[/green]")
        if c["skipped"]:
            parts.append(f"[yellow]{c['skipped']} skipped[/yellow]")
        if c["failed"]:
            parts.append(f"[red]{c['failed']} failed[/red]")
        summary_text = "  |  ".join(parts) if parts else ""
        with contextlib.suppress(QueryError):
            self.query_one("#run-summary", Static).update(summary_text)

    def drain_log_queue(self) -> None:
        """Drain queued worker log lines into the TUI log widget."""
        log = self.query_one("#run-log", RichLog)
        progress = self.query_one("#run-progress", ProgressBar)
        counter = self.query_one("#run-file-counter", Static)
        try:
            while True:
                line = self._log_queue.get_nowait()
                if line is None:
                    self._finish_log_drain(log, counter)
                    return
                log.write(self._format_log_line(line))
                self._update_progress_from_log_line(line, progress, counter)
        except queue.Empty:
            if self._running:
                self.set_timer(0.1, self.drain_log_queue)

    def _finish_log_drain(self, log: RichLog, counter: Static) -> None:
        self._running = False
        ok, msg = self._result_queue.get_nowait() if not self._result_queue.empty() else (True, "Completed")
        if ok:
            self._set_status("Completed", "status-done")
            log.write(f"\n[bold green]Run completed.[/bold green]  {self._completion_summary_line()}")
        else:
            self._set_status(f"Failed: {msg}", "status-error")
            log.write(f"[bold red]Run failed:[/bold red] {_escape_markup(msg)}")
        counter.update("")

    def _completion_summary_line(self) -> str:
        summary_parts = []
        c = self._run_counts
        if c["renamed"]:
            summary_parts.append(f"[green]{c['renamed']} renamed[/green]")
        if c["skipped"]:
            summary_parts.append(f"[yellow]{c['skipped']} skipped[/yellow]")
        if c["failed"]:
            summary_parts.append(f"[red]{c['failed']} failed[/red]")
        return "  ".join(summary_parts) if summary_parts else "no files processed"

    def _update_progress_from_log_line(self, line: str, progress: ProgressBar, counter: Static) -> None:
        match = PROCESS_RE.search(line)
        if not match:
            return
        cur = int(match.group(1))
        tot = max(1, int(match.group(2)))
        progress.update(total=tot, progress=cur)
        self._set_status(f"Processing {cur}/{tot}...", "status-running")
        counter.update(f"{cur} of {tot} files")
        self.set_timer(0.1, self.drain_log_queue)

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def action_preview(self) -> None:
        self.start_run(dry_run=True)

    def action_apply(self) -> None:
        self.start_run(dry_run=False)

    def action_cancel(self) -> None:
        self.cancel_run()

    def _start_run(self, *, dry_run: bool) -> None:
        self.start_run(dry_run=dry_run)

    def start_run(self, *, dry_run: bool) -> None:
        """Start a preview or apply run from the current TUI form state."""
        if self._running:
            self.query_one("#run-log", RichLog).write("[yellow]A run is already in progress.[/yellow]")
            return
        directory = self.get_str("directory")
        if not directory:
            self.query_one("#run-log", RichLog).write(
                "[bold red]No directory set.[/bold red] Set a folder path on the Settings tab first."
            )
            return
        if not Path(directory).is_dir():
            self.query_one("#run-log", RichLog).write(
                f"[bold red]Directory not found:[/bold red] {_escape_markup(directory)}"
            )
            return
        try:
            config = self.build_config(dry_run=dry_run)
        except ValueError as exc:
            err_msg = _escape_markup(str(exc))
            self.query_one("#run-log", RichLog).write(f"[bold red]Invalid config:[/bold red] {err_msg}")
            return
        _save_settings(self.snapshot())
        self._stop_event.clear()
        self._running = True
        self._run_counts = {"renamed": 0, "skipped": 0, "failed": 0}
        with contextlib.suppress(QueryError):
            self.query_one("#run-summary", Static).update("")
        log = self.query_one("#run-log", RichLog)
        mode_label = "Preview (dry run)" if dry_run else "Apply Renames"
        mode_color = "cyan" if dry_run else "green"
        log.write(f"\n[bold {mode_color}]{'=' * 50}[/bold {mode_color}]")
        log.write(f"[bold {mode_color}]  {mode_label}[/bold {mode_color}]")
        log.write(f"[bold {mode_color}]{'=' * 50}[/bold {mode_color}]")
        log.write(f"[dim]Directory: {_escape_markup(directory)}[/dim]")
        self._set_status(f"Starting {mode_label.lower()}...", "status-running")
        progress = self.query_one("#run-progress", ProgressBar)
        progress.update(total=100, progress=0)
        worker = threading.Thread(target=self.run_directory_worker, args=(directory, config), daemon=True)
        worker.start()
        self.set_timer(0.1, self.drain_log_queue)

    def cancel_run(self) -> None:
        """Request cancellation for an active run."""
        if not self._running:
            return
        self._stop_event.set()
        self._set_status("Cancelling...", "status-cancel")
        self.query_one("#run-log", RichLog).write("[yellow]Cancel requested -- finishing current file...[/yellow]")

    def _selected_single_pdf(self) -> Path | None:
        log = self.query_one("#run-log", RichLog)
        file_path = self.get_str("single_file")
        if not file_path:
            log.write("[bold red]No file set.[/bold red] Set a single file path on the Settings tab first.")
            return None
        fp = Path(file_path)
        if not fp.exists():
            log.write(f"[bold red]File not found:[/bold red] {_escape_markup(file_path)}")
            return None
        if fp.suffix.lower() != ".pdf":
            log.write(f"[bold red]Not a PDF file:[/bold red] {_escape_markup(file_path)} (expected .pdf extension)")
            return None
        return fp

    def _start_single_file_ui(self, fp: Path) -> None:
        log = self.query_one("#run-log", RichLog)
        log.write("\n[bold cyan]Single File Processing[/bold cyan]")
        log.write(f"[dim]File: {_escape_markup(fp.name)}[/dim]")
        self._set_status("Processing single file...", "status-running")
        self._running = True
        self._run_counts = {"renamed": 0, "skipped": 0, "failed": 0}
        with contextlib.suppress(Exception):
            self.query_one("#run-summary", Static).update("")

    def _single_file_worker(self, fp: Path, config: RenamerConfig) -> None:
        try:
            new_base, meta, err = suggest_rename_for_file(fp, config)
            if err is not None:
                self._log_queue.put(f"[bold red]Error:[/bold red] {_escape_markup(str(err))}\n")
                self._result_queue.put((False, str(err)))
                return
            if new_base is None:
                self._log_queue.put("[yellow]Skipped -- no extractable content.[/yellow]\n")
                self._result_queue.put((True, "Skipped"))
                return
            suggested = new_base + fp.suffix
            self._log_queue.put(f"[dim]Suggested:[/dim] [bold]{_escape_markup(suggested)}[/bold]\n")
            self._rename_single_file(fp, config, new_base, meta or {})
        except _TUI_WORKER_EXCEPTIONS as exc:
            self._result_queue.put((False, str(exc)))
        finally:
            self._log_queue.put(None)  # sentinel

    def _rename_single_file(self, fp: Path, config: RenamerConfig, new_base: str, meta: dict[str, object]) -> None:
        export_rows: list[dict[str, object]] = []
        _on_rename_success = _make_post_rename_success_callback(config, meta, export_rows)
        success, target = apply_single_rename(
            fp,
            sanitize_filename_base(new_base),
            plan_file_path=None,
            plan_entries=[],
            dry_run=False,
            backup_dir=config.backup_dir,
            on_success=_on_rename_success,
            max_filename_chars=config.max_filename_chars,
        )
        if success:
            msg = (
                f"[green]Renamed[/green] [dim]{_escape_markup(fp.name)}[/dim]"
                f" [green bold]->[/green bold] [bold]{_escape_markup(target.name)}[/bold]\n"
            )
            self._log_queue.put(msg)
            self._log_single_file_meta(meta)
            self._result_queue.put((True, "Completed"))
        else:
            self._log_queue.put("[bold red]Could not rename file.[/bold red]\n")
            self._result_queue.put((False, "Could not rename file"))

    def _log_single_file_meta(self, meta: dict[str, object]) -> None:
        meta_parts = []
        for k in ("category", "summary", "keywords", "category_source"):
            v = meta.get(k)
            if v:
                meta_parts.append(f"[dim]{k}:[/dim] {_escape_markup(str(v))}")
        if meta_parts:
            self._log_queue.put("  " + "  |  ".join(meta_parts) + "\n")

    def process_one(self) -> None:
        """Process the currently selected single PDF."""
        if self._running:
            self.query_one("#run-log", RichLog).write("[yellow]A run is already in progress.[/yellow]")
            return
        fp = self._selected_single_pdf()
        if fp is None:
            return
        try:
            config = self.build_config(dry_run=False, manual_mode=True)
        except ValueError as exc:
            self.query_one("#run-log", RichLog).write(
                f"[bold red]Invalid config:[/bold red] {_escape_markup(str(exc))}"
            )
            return
        self._start_single_file_ui(fp)
        worker = threading.Thread(target=self._single_file_worker, args=(fp, config), daemon=True)
        worker.start()
        self.set_timer(0.1, self.drain_log_queue)

    # ------------------------------------------------------------------
    # Button handlers
    # ------------------------------------------------------------------

    @on(Button.Pressed, "#btn-preview")
    def on_preview(self) -> None:
        self.start_run(dry_run=True)

    @on(Button.Pressed, "#btn-apply")
    def on_apply(self) -> None:
        self.start_run(dry_run=False)

    @on(Button.Pressed, "#btn-one")
    def on_one(self) -> None:
        self.process_one()

    @on(Button.Pressed, "#btn-cancel")
    def on_cancel(self) -> None:
        self.cancel_run()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    setup_logging(level=logging.INFO)
    app = AIRenamerTUI()
    app.run()


if __name__ == "__main__":
    main()
