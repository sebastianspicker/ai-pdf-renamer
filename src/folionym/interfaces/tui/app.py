"""Textual interface for Folionym's local reviewed PDF rename workflow."""

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
    from textual.widget import Widget
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
except ImportError as _error:  # pragma: no cover
    raise ImportError("textual is required for the TUI. Install with: pip install -e '.[tui]'") from _error

from ...application.hooks import _make_post_rename_success_callback
from ...application.models import ApplyStatus, PreviewPlan
from ...application.proposals import suggest_rename_for_file
from ...application.reviewed_plan import apply_reviewed_plan, create_preview_plan
from ...infrastructure.logging import setup_logging
from ...rename_ops import apply_single_rename, sanitize_filename_base
from ...settings import RenamerConfig
from ..ui_settings import (
    SETTINGS_PATH,
    merged_ui_settings,
)
from ..ui_settings import (
    load_ui_settings as _load_settings,
)
from ..ui_settings import (
    save_ui_settings as _save_settings,
)
from .assets import _PRESETS, ERROR_COLOR, FOLIONYM_THEME, PREVIEW_COLOR, SUCCESS_COLOR, WARNING_COLOR
from .confirmation import ConfirmActionScreen
from .forms import compose_advanced, compose_basic, compose_run
from .operations import process_single_file
from .presentation import (
    completion_summary,
    effective_configuration_lines,
    format_run_log_line,
    format_run_summary,
    metric_summary,
)
from .reviewed_plan import DirectoryReviewedPlanController
from .selection import TuiSourceSelection
from .values import TuiValueAccess
from .worker_messages import _ApplyFinished, _PlanFailed, _PreviewFinished, _RunFinished, _RunLog, _RunProgress

__all__ = ["SETTINGS_PATH", "FolionymTUI", "_load_settings", "_save_settings", "main"]
_TUI_WORKER_EXCEPTIONS = (AttributeError, KeyError, OSError, RuntimeError, TypeError, ValueError)
_MATERIAL_CONTROL_IDS = frozenset(
    {
        "directory",
        "single_file",
        "language",
        "case",
        "date_format",
        "preset",
        "project",
        "version",
        "template",
        "backup_dir",
        "rename_log",
        "export_metadata",
        "summary_json",
        "rules_file",
        "post_rename_hook",
        "llm_url",
        "llm_model",
        "llm_timeout",
        "max_tokens",
        "max_content_chars",
        "max_content_tokens",
        "workers",
        "max_filename_chars",
        "use_llm",
        "use_ocr",
        "recursive",
        "skip_already_named",
        "use_pdf_metadata_date",
        "use_structured_fields",
        "write_pdf_metadata",
        "use_vision_fallback",
        "simple_naming_mode",
        "vision_first",
    }
)


class FolionymTUI(TuiSourceSelection, TuiValueAccess, App[None]):
    """Coordinate setup, immutable folder previews, and confirmed exact applies."""

    TITLE = "Folionym"
    CSS_PATH: ClassVar[list[str | PurePath]] = ["base.tcss", "run.tcss", "responsive.tcss"]
    ENABLE_COMMAND_PALETTE = False
    BINDINGS: ClassVar[list[Binding | tuple[str, str] | tuple[str, str, str]]] = [
        Binding("ctrl+q", "quit", "Quit"),
        Binding("ctrl+p", "preview", "Preview"),
        Binding("ctrl+a", "apply", "Apply reviewed names"),
        Binding("ctrl+c", "cancel", "Cancel run"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.register_theme(FOLIONYM_THEME)
        self.theme = FOLIONYM_THEME.name
        self._stop_event = threading.Event()
        self._operation_running = False
        self._exit_after_run = False
        self._run_is_preview = True
        self._settings = merged_ui_settings(_load_settings())
        self._run_counts = {"renamed": 0, "skipped": 0, "failed": 0}
        self._reviewed_plan = DirectoryReviewedPlanController()

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
        """Apply compact styling at the supported terminal-width boundary."""
        self._sync_compact_layout(event.size.width)

    def _sync_compact_layout(self, width: int) -> None:
        self.screen.set_class(width < 100, "compact")

    def _activate_tab(self, pane_id: str) -> None:
        self.query_one("#workflow-tabs", TabbedContent).active = pane_id
        self._sync_workflow_nav(pane_id)
        if pane_id == "run":
            self._update_effective_configuration()

    def _sync_workflow_nav(self, pane_id: str) -> None:
        for candidate in ("basic", "advanced", "run"):
            with contextlib.suppress(QueryError):
                self.query_one(f"#nav-{candidate}", Button).set_class(candidate == pane_id, "active")

    @on(TabbedContent.TabActivated, "#workflow-tabs")
    def on_workflow_tab_activated(self, message: TabbedContent.TabActivated) -> None:
        if message.pane.id:
            self._sync_workflow_nav(message.pane.id)

    @on(Input.Changed)
    def on_input_changed(self, message: Input.Changed) -> None:
        """Invalidate a reviewed plan whenever a source or text setting changes."""
        if message.input.id == "directory":
            with contextlib.suppress(QueryError):
                self.query_one("#current-scope", Static).update(message.value.strip() or "No folder selected")
        self._invalidate_for_control(message.input.id)

    @on(Checkbox.Changed)
    def on_checkbox_changed(self, message: Checkbox.Changed) -> None:
        self._invalidate_for_control(message.checkbox.id)

    @on(Select.Changed)
    def on_select_changed(self, message: Select.Changed) -> None:
        self._invalidate_for_control(message.select.id)

    def _invalidate_for_control(self, control_id: str | None) -> None:
        if control_id not in _MATERIAL_CONTROL_IDS or self._operation_running or self._reviewed_plan.plan is None:
            return
        self._reviewed_plan.invalidate("Source or settings changed after Preview.")
        self._clear_preview_table()
        self._set_apply_guidance("Preview invalidated by a source or settings change. Preview again before Apply.")
        self._set_status("Preview invalidated", "status-idle")

    def _update_effective_configuration(self) -> None:
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

    def _set_apply_guidance(self, message: str) -> None:
        with contextlib.suppress(QueryError):
            self.query_one("#run-guidance", Static).update(f"[b]REVIEWED APPLY[/b]  {_escape_markup(message)}")

    def _clear_preview_table(self) -> None:
        with contextlib.suppress(QueryError):
            self.query_one("#preview-table", DataTable).clear(columns=False)
            self.query_one("#preview-table", DataTable).display = False
            self.query_one("#preview-empty", Static).display = True
            self.query_one("#inspector-source", Static).update("No document selected")
            self.query_one("#inspector-proposed", Static).update("Run a preview to inspect a proposed name.")
            self.query_one("#inspector-mode", Static).update("Waiting for preview output")

    def _render_retained_plan(self, plan: PreviewPlan) -> None:
        """Render rows directly from the retained typed plan, never worker logs."""
        self._clear_preview_table()
        rows = self._reviewed_plan.rows()
        if not rows:
            return
        table = self.query_one("#preview-table", DataTable)
        colors = {"READY": SUCCESS_COLOR, "REVIEW": WARNING_COLOR, "SKIPPED": WARNING_COLOR, "FAILED": ERROR_COLOR}
        for row in rows:
            table.add_row(
                Text(row.status, style=colors.get(row.status, "")),
                row.source_name,
                row.proposed_name,
                key=row.item_id,
            )
        table.display = True
        self.query_one("#preview-empty", Static).display = False
        self._show_preview_item(0)

    def _show_preview_item(self, index: int) -> None:
        plan = self._reviewed_plan.plan
        if plan is None or not 0 <= index < len(plan.items):
            return
        item = plan.items[index]
        self.query_one("#inspector-source", Static).update(_escape_markup(item.source.name))
        self.query_one("#inspector-proposed", Static).update(
            f"[b]{_escape_markup(item.proposed_name or 'No reviewed filename')}[/b]"
        )
        self.query_one("#inspector-mode", Static).update(item.reason or item.status.value.upper())

    def _progress_from_worker(self, current: int, total: int, source: Path) -> None:
        progress = self.query_one("#run-progress", ProgressBar)
        progress.update(total=max(1, total), progress=current)
        self._set_status(f"Processing {current}/{total}: {source.name}", "status-running")
        self.query_one("#run-file-counter", Static).update(f"{current} of {total} files")
        self.query_one("#metric-files", Static).update(f"[b]{total}[/b] PDFs")

    def _post_progress_from_worker(self, current: int, total: int, source: Path) -> None:
        """Marshal canonical plan progress into the Textual UI thread."""
        self.call_from_thread(self.post_message, _RunProgress(current, total, source))

    @work(thread=True, exclusive=True)
    def preview_directory_worker(self, directory: str, config: RenamerConfig) -> None:
        """Create one canonical immutable preview plan in a Textual worker."""
        try:
            plan = create_preview_plan(
                directory,
                config,
                progress_callback=self._post_progress_from_worker,
            )
            self.call_from_thread(self.post_message, _PreviewFinished(plan, complete=not self._stop_event.is_set()))
        except _TUI_WORKER_EXCEPTIONS as exc:
            self.call_from_thread(self.post_message, _PlanFailed("Preview", str(exc)))

    @work(thread=True, exclusive=True)
    def apply_reviewed_plan_worker(self, plan: PreviewPlan, item_ids: tuple[str, ...]) -> None:
        """Apply retained exact targets only, without a second name-generation pass."""
        try:
            report = apply_reviewed_plan(
                plan,
                item_ids,
                stop_event=self._stop_event,
                progress_callback=self._post_progress_from_worker,
            )
            self.call_from_thread(self.post_message, _ApplyFinished(report))
        except _TUI_WORKER_EXCEPTIONS as exc:
            self.call_from_thread(self.post_message, _PlanFailed("Apply", str(exc)))

    def _set_run_controls_active(self, running: bool) -> None:
        for button_id in ("#btn-preview", "#btn-apply", "#btn-one"):
            with contextlib.suppress(QueryError):
                self.query_one(button_id, Button).disabled = running
        for control_id in _MATERIAL_CONTROL_IDS:
            with contextlib.suppress(QueryError):
                self.query_one(f"#{control_id}", Widget).disabled = running
        with contextlib.suppress(QueryError):
            self.query_one("#btn-cancel", Button).disabled = not running

    def _set_status(self, text: str, css_class: str = "status-idle") -> None:
        indicators = {
            "status-idle": "[dim]IDLE[/dim]",
            "status-running": f"[bold {WARNING_COLOR}]RUN[/bold {WARNING_COLOR}]",
            "status-done": f"[bold {SUCCESS_COLOR}]DONE[/bold {SUCCESS_COLOR}]",
            "status-error": f"[bold {ERROR_COLOR}]FAIL[/bold {ERROR_COLOR}]",
            "status-cancel": f"[{WARNING_COLOR}]STOP[/{WARNING_COLOR}]",
        }
        prefix = indicators.get(css_class, "")
        status = self.query_one("#run-status", Static)
        status.update(f"{prefix}  {text}" if prefix else text)
        for candidate in indicators:
            status.remove_class(candidate)
        status.add_class(css_class)

    def _set_counts(self, *, renamed: int, skipped: int, failed: int) -> None:
        self._run_counts = {"renamed": renamed, "skipped": skipped, "failed": failed}
        summary_text = format_run_summary(self._run_counts, self._run_is_preview, separator="  |  ")
        suggestions, skipped_text, failed_text = metric_summary(self._run_counts, self._run_is_preview)
        with contextlib.suppress(QueryError):
            self.query_one("#run-summary", Static).update(summary_text)
            self.query_one("#metric-suggestions", Static).update(suggestions)
            self.query_one("#metric-skipped", Static).update(skipped_text)
            self.query_one("#metric-failed", Static).update(failed_text)

    @on(_RunProgress)
    def on_run_progress(self, message: _RunProgress) -> None:
        self._progress_from_worker(message.current, message.total, message.source)

    @on(_PreviewFinished)
    def on_preview_finished(self, message: _PreviewFinished) -> None:
        """Retain and render the plan even when Preview was cancelled partway through."""
        self._operation_running = False
        self._set_run_controls_active(False)
        self._reviewed_plan.retain(message.plan, complete=message.complete)
        self._render_retained_plan(message.plan)
        ready = sum(item.status.value == "ready" for item in message.plan.items)
        skipped = sum(item.status.value == "skipped" for item in message.plan.items)
        failed = sum(item.status.value == "failed" for item in message.plan.items)
        self._set_counts(renamed=ready, skipped=skipped, failed=failed)
        log = self.query_one("#run-log", RichLog)
        if message.complete:
            self._set_status("Preview ready. Apply uses these exact reviewed names.", "status-done")
            self._set_apply_guidance("Apply uses only included READY rows and never recomputes names.")
            log.write(f"[bold {SUCCESS_COLOR}]Preview ready.[/bold {SUCCESS_COLOR}] Exact reviewed names are retained.")
            self.notify(
                "Apply will use the retained exact reviewed names.", title="Preview ready", severity="information"
            )
        else:
            self._set_status("Preview cancelled or incomplete. Apply is disabled.", "status-cancel")
            self._set_apply_guidance("Preview was incomplete. Preview again before Apply.")
            log.write(
                f"[bold {WARNING_COLOR}]Preview incomplete.[/bold {WARNING_COLOR}] Retained rows cannot be applied."
            )
            self.notify(
                "Preview was incomplete; run Preview again before Apply.",
                title="Preview incomplete",
                severity="warning",
            )
        self.query_one("#run-file-counter", Static).update("")
        if self._exit_after_run:
            self.exit()

    @on(_ApplyFinished)
    def on_apply_finished(self, message: _ApplyFinished) -> None:
        """Present exact-plan outcomes, including partial cancellation, explicitly."""
        self._operation_running = False
        self._set_run_controls_active(False)
        report = message.report
        renamed = report.count(ApplyStatus.RENAMED)
        skipped = report.count(ApplyStatus.SKIPPED) + report.count(ApplyStatus.UNCHANGED)
        failed = report.count(ApplyStatus.FAILED) + report.count(ApplyStatus.CANCELLED)
        self._run_is_preview = False
        self._set_counts(renamed=renamed, skipped=skipped, failed=failed)
        cancelled = report.count(ApplyStatus.CANCELLED) > 0 or self._stop_event.is_set()
        log = self.query_one("#run-log", RichLog)
        for item in report.items:
            target = item.target_name or "no target"
            detail = item.reason or item.status.value
            log.write(f"[dim]{_escape_markup(item.source_name)}[/dim] → [b]{_escape_markup(target)}[/b]  {detail}")
        self._reviewed_plan.invalidate("The reviewed plan was consumed by Apply. Preview again before another Apply.")
        if cancelled:
            self._set_status("Apply cancelled; completed exact targets are shown below.", "status-cancel")
            self._set_apply_guidance("Apply was partial or cancelled. Preview again before another Apply.")
            self.notify(
                "Completed and cancelled exact-plan outcomes are shown.", title="Apply cancelled", severity="warning"
            )
        else:
            self._set_status("Exact reviewed apply completed.", "status-done")
            self._set_apply_guidance("The reviewed plan was consumed. Preview again before another Apply.")
            self.notify(self._completion_summary_line(), title="Apply completed", severity="information")
        self.query_one("#run-file-counter", Static).update("")
        if self._exit_after_run:
            self.exit()

    @on(_PlanFailed)
    def on_plan_failed(self, message: _PlanFailed) -> None:
        self._operation_running = False
        self._set_run_controls_active(False)
        self._set_status(f"{message.action} failed: {message.message}", "status-error")
        self._set_apply_guidance("Preview again before Apply.")
        self.query_one("#run-log", RichLog).write(
            f"[bold {ERROR_COLOR}]{message.action} failed:[/bold {ERROR_COLOR}] {_escape_markup(message.message)}"
        )
        self.notify(message.message, title=f"{message.action} failed", severity="error")
        if self._exit_after_run:
            self.exit()

    @on(_RunLog)
    def on_run_log(self, message: _RunLog) -> None:
        """Present only immediate single-file worker logs; folder rows never use logs."""
        formatted, count_key = format_run_log_line(message.line)
        if count_key == "renamed":
            self._run_counts["renamed"] += 1
        elif count_key in {"skipped", "failed"}:
            self._run_counts[count_key] += 1
        self.query_one("#run-log", RichLog).write(formatted)
        self._set_counts(**self._run_counts)

    @on(_RunFinished)
    def on_run_finished(self, message: _RunFinished) -> None:
        """Finalize the explicit immediate single-file flow."""
        self._operation_running = False
        self._set_run_controls_active(False)
        if message.ok:
            cancelled = self._stop_event.is_set()
            self._set_status("Cancelled" if cancelled else "Completed", "status-cancel" if cancelled else "status-done")
            self.query_one("#run-log", RichLog).write(
                f"[bold {SUCCESS_COLOR}]Run completed.[/bold {SUCCESS_COLOR}]  {self._completion_summary_line()}"
            )
        else:
            self._set_status(f"Failed: {message.message}", "status-error")
            self.query_one("#run-log", RichLog).write(
                f"[bold {ERROR_COLOR}]Run failed:[/bold {ERROR_COLOR}] {_escape_markup(message.message)}"
            )
        self.query_one("#run-file-counter", Static).update("")
        if self._exit_after_run:
            self.exit()

    def _completion_summary_line(self) -> str:
        return completion_summary(self._run_counts, self._run_is_preview)

    def action_preview(self) -> None:
        self.start_preview()

    def action_apply(self) -> None:
        self.request_apply()

    def action_cancel(self) -> None:
        self.cancel_run()

    def request_apply(self) -> None:
        """Confirm an exact reviewed apply only when a current complete plan exists."""
        if self._operation_running:
            self.notify("A run is already in progress.", severity="warning")
            return
        try:
            plan, item_ids = self._reviewed_plan.require_apply()
        except ValueError as exc:
            self._set_apply_guidance(str(exc))
            self.notify(str(exc), title="Preview required", severity="warning")
            return
        detail = (
            f"Folionym will rename {len(item_ids)} included ready PDFs in {plan.source} "
            "to their exact reviewed targets. "
            "It will not generate names again."
        )
        self.push_screen(
            ConfirmActionScreen("Apply reviewed names?", detail, "Apply exact names"), self._finish_apply_confirmation
        )

    def _finish_apply_confirmation(self, confirmed: bool | None) -> None:
        if confirmed:
            self.start_apply()

    def request_process_one(self) -> None:
        """Validate and confirm the intentionally immediate one-PDF rename flow."""
        if self._operation_running:
            self.notify("A run is already in progress.", severity="warning")
            return
        selected = self._selected_single_pdf()
        if selected is None:
            return
        detail = (
            f"Folionym will generate a unique available name for {selected.name} and rename it immediately. "
            "Use Preview folder when you need a reviewed exact-plan apply."
        )
        self.push_screen(ConfirmActionScreen("Rename one PDF?", detail, "Rename PDF"), self._finish_single_confirmation)

    def _finish_single_confirmation(self, confirmed: bool | None) -> None:
        if confirmed:
            self.process_one()

    def _preview_configuration(self) -> tuple[str, RenamerConfig] | None:
        directory = self.get_str("directory")
        if not directory:
            message = "Set a folder path on the Setup tab first."
            self._show_setup_error("directory", message)
            self.query_one("#run-log", RichLog).write(
                f"[bold {ERROR_COLOR}]No directory set.[/bold {ERROR_COLOR}] {message}"
            )
            self.notify(message, title="No directory set", severity="error")
            return None
        if not Path(directory).is_dir():
            self._show_setup_error("directory", f"Folder not found: {directory}")
            self.query_one("#run-log", RichLog).write(
                f"[bold {ERROR_COLOR}]Directory not found:[/bold {ERROR_COLOR}] {_escape_markup(directory)}"
            )
            self.notify(directory, title="Directory not found", severity="error")
            return None
        try:
            return directory, self.build_config(dry_run=True)
        except ValueError as exc:
            self.query_one("#run-log", RichLog).write(
                f"[bold {ERROR_COLOR}]Invalid config:[/bold {ERROR_COLOR}] {_escape_markup(str(exc))}"
            )
            self.notify(str(exc), title="Invalid config", severity="error")
            return None

    def _begin_directory_run(self, *, label: str, preview: bool) -> None:
        self._activate_tab("run")
        self._stop_event.clear()
        self._exit_after_run = False
        self._operation_running = True
        self._run_is_preview = preview
        self._set_run_controls_active(True)
        self.call_after_refresh(self.query_one("#btn-cancel", Button).focus)
        self._set_counts(renamed=0, skipped=0, failed=0)
        self._update_effective_configuration()
        log = self.query_one("#run-log", RichLog)
        log.display = True
        color = PREVIEW_COLOR if preview else SUCCESS_COLOR
        log.write(f"\n[bold {color}]{'=' * 50}[/bold {color}]")
        log.write(f"[bold {color}]  {label}[/bold {color}]")
        log.write(f"[bold {color}]{'=' * 50}[/bold {color}]")
        self._set_status(f"Starting {label.lower()}...", "status-running")
        progress = self.query_one("#run-progress", ProgressBar)
        progress.display = True
        progress.update(total=100, progress=0)

    def start_preview(self) -> None:
        if self._operation_running:
            self.notify("A run is already in progress.", severity="warning")
            return
        run = self._preview_configuration()
        if run is None:
            return
        directory, config = run
        self._reviewed_plan.clear()
        self._clear_preview_table()
        self._set_apply_guidance("Preview is running. Apply remains unavailable until it completes.")
        self._begin_directory_run(label="Preview", preview=True)
        self.preview_directory_worker(directory, config)

    def start_apply(self) -> None:
        if self._operation_running:
            self.notify("A run is already in progress.", severity="warning")
            return
        try:
            plan, item_ids = self._reviewed_plan.require_apply()
        except ValueError as exc:
            self.notify(str(exc), title="Preview required", severity="warning")
            return
        self._begin_directory_run(label="Apply exact reviewed names", preview=False)
        self._set_apply_guidance("Applying retained exact targets. No names are being recomputed.")
        self.apply_reviewed_plan_worker(plan, item_ids)

    def cancel_run(self) -> None:
        if not self._operation_running:
            return
        self._stop_event.set()
        self._set_status("Cancelling after the current item...", "status-cancel")
        self.query_one("#run-log", RichLog).write(
            f"[{WARNING_COLOR}]Cancel requested -- finishing the current item...[/{WARNING_COLOR}]"
        )

    async def action_quit(self) -> None:
        if not self._operation_running:
            self.exit()
            return
        self._exit_after_run = True
        self.cancel_run()
        self.query_one("#run-log", RichLog).write(
            f"[{WARNING_COLOR}]Quit requested -- the app will close after the current item finishes.[/{WARNING_COLOR}]"
        )

    def _start_single_file_ui(self, fp: Path) -> None:
        self._reviewed_plan.invalidate("A single-file rename was started. Preview again before folder Apply.")
        self._activate_tab("run")
        log = self.query_one("#run-log", RichLog)
        log.display = True
        self.query_one("#run-progress", ProgressBar).display = False
        log.write(f"\n[bold {PREVIEW_COLOR}]Single File Processing[/bold {PREVIEW_COLOR}]")
        log.write(f"[dim]File: {_escape_markup(fp.name)}[/dim]")
        self._set_status("Processing single file...", "status-running")
        self._operation_running = True
        self._set_run_controls_active(True)
        self._set_counts(renamed=0, skipped=0, failed=0)
        self._clear_preview_table()
        self._update_effective_configuration()
        self._set_apply_guidance("Single-file mode uses immediate unique-available rename after confirmation.")

    @work(thread=True, exclusive=True)
    def _single_file_worker(self, fp: Path, config: RenamerConfig) -> None:
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
        self._activate_tab("basic")

    @on(Button.Pressed, "#nav-advanced")
    def on_nav_advanced(self) -> None:
        self._activate_tab("advanced")

    @on(Button.Pressed, "#nav-run")
    def on_nav_run(self) -> None:
        self._activate_tab("run")

    @on(DataTable.RowHighlighted, "#preview-table")
    def on_preview_row_highlighted(self, message: DataTable.RowHighlighted) -> None:
        self._show_preview_item(message.cursor_row)

    @on(Button.Pressed, "#btn-preview")
    def on_preview(self) -> None:
        self.start_preview()

    @on(Button.Pressed, "#btn-apply")
    def on_apply(self) -> None:
        self.request_apply()

    @on(Button.Pressed, "#btn-one")
    def on_one(self) -> None:
        self.request_process_one()

    @on(Button.Pressed, "#btn-cancel")
    def on_cancel(self) -> None:
        self.cancel_run()


def main() -> None:
    """Initialize logging and run the Textual application."""
    setup_logging(level=logging.INFO)
    FolionymTUI().run()


if __name__ == "__main__":
    main()
