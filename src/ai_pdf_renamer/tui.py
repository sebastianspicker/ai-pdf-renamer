"""Terminal UI for AI-PDF-Renamer (Textual-based, replaces Tkinter gui.py).

Launch with: ai-pdf-renamer-tui
Requires: pip install -e '.[tui]'
"""

from __future__ import annotations

import contextlib
import json
import logging
import queue
import re
import threading
from pathlib import Path
from typing import ClassVar

from rich.markup import escape as _escape_markup

try:
    from textual import on
    from textual.app import App, ComposeResult
    from textual.binding import Binding
    from textual.containers import Horizontal, ScrollableContainer
    from textual.widgets import (
        Button,
        Checkbox,
        Footer,
        Header,
        Input,
        Label,
        ProgressBar,
        RichLog,
        Select,
        Static,
        TabbedContent,
        TabPane,
    )
except ImportError as _e:  # pragma: no cover
    raise ImportError("textual is required for the TUI. Install with: pip install -e '.[tui]'") from _e

from .config_resolver import build_config
from .logging_utils import setup_logging
from .rename_ops import apply_single_rename, sanitize_filename_base
from .renamer import RenamerConfig, _apply_post_rename_actions, rename_pdfs_in_directory, suggest_rename_for_file

logger = logging.getLogger(__name__)

SETTINGS_PATH = Path.home() / ".ai_pdf_renamer_gui.json"
PROCESS_RE = re.compile(r"Processing\s+(\d+)/(\d+)\s*:")

# Patterns for enriching log output with color
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
]
_LLM_BACKENDS = [
    ("http (OpenAI-compatible server)", "http"),
    ("in-process (llama-cpp-python)", "in-process"),
    ("auto (in-process if path set)", "auto"),
]


# ---------------------------------------------------------------------------
# Logging handler that forwards to a queue
# ---------------------------------------------------------------------------


class _QueueHandler(logging.Handler):
    def __init__(self, q: queue.Queue[str | None]) -> None:
        super().__init__()
        self._queue = q

    def emit(self, record: logging.LogRecord) -> None:
        try:
            # Escape Rich markup so filenames/paths in log messages are not interpreted as tags.
            self._queue.put(_escape_markup(self.format(record)) + "\n")
        except Exception:
            self.handleError(record)


# ---------------------------------------------------------------------------
# Settings persistence
# ---------------------------------------------------------------------------


def _load_settings() -> dict[str, object]:
    if not SETTINGS_PATH.exists():
        return {}
    try:
        raw = SETTINGS_PATH.read_text(encoding="utf-8")
        data = json.loads(raw)
        if isinstance(data, dict):
            return data
    except (OSError, json.JSONDecodeError):
        pass
    return {}


def _save_settings(data: dict[str, object]) -> None:
    try:
        SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
        # Write with restricted permissions (owner read/write only) since the file may contain
        # LLM endpoint URLs, hook commands, and directory paths.
        SETTINGS_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        with contextlib.suppress(OSError):  # Best-effort; not supported on all platforms (e.g. Windows)
            SETTINGS_PATH.chmod(0o600)
    except OSError:
        logger.debug("Could not save TUI settings", exc_info=True)


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

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
        with ScrollableContainer(classes="form-container"):
            yield Static("Input Source", classes="section-title")
            with Horizontal(classes="field-row"):
                yield Label("Folder:", classes="field-label")
                yield Input(
                    id="directory",
                    placeholder="/path/to/PDFs",
                    classes="field-input",
                    value=str(self._settings.get("directory", "")),
                )
            with Horizontal(classes="field-row"):
                yield Label("Single file:", classes="field-label")
                yield Input(
                    id="single_file",
                    placeholder="/path/to/file.pdf",
                    classes="field-input",
                    value=str(self._settings.get("single_file", "")),
                )
            yield Static("", classes="section-sep")
            yield Static("Naming Options", classes="section-title")
            with Horizontal(classes="field-row"):
                yield Label("Language:", classes="field-label")
                yield Select(
                    _LANGUAGES,
                    id="language",
                    value=str(self._settings.get("language", "de")),
                )
            with Horizontal(classes="field-row"):
                yield Label("Case:", classes="field-label")
                yield Select(
                    _CASES,
                    id="case",
                    value=str(self._settings.get("case", "kebabCase")),
                )
            with Horizontal(classes="field-row"):
                yield Label("Date format:", classes="field-label")
                yield Select(
                    _DATE_FORMATS,
                    id="date_format",
                    value=str(self._settings.get("date_format", "dmy")),
                )
            with Horizontal(classes="field-row"):
                yield Label("Preset:", classes="field-label")
                yield Select(
                    _PRESETS,
                    id="preset",
                    value=str(self._settings.get("preset", "")),
                )
            with Horizontal(classes="field-row"):
                yield Label("Project:", classes="field-label")
                yield Input(
                    id="project",
                    placeholder="optional project name",
                    classes="field-input",
                    value=str(self._settings.get("project", "")),
                )
            with Horizontal(classes="field-row"):
                yield Label("Version:", classes="field-label")
                yield Input(
                    id="version",
                    placeholder="optional version",
                    classes="field-input",
                    value=str(self._settings.get("version", "")),
                )
            yield Static("", classes="section-sep")
            yield Static("Processing Flags", classes="section-title")
            with Horizontal(classes="flags-row"):
                yield Checkbox(
                    "Dry run", id="dry_run", value=bool(self._settings.get("dry_run", True)), classes="flag-check"
                )
                yield Checkbox(
                    "Use LLM", id="use_llm", value=bool(self._settings.get("use_llm", True)), classes="flag-check"
                )
                yield Checkbox(
                    "OCR", id="use_ocr", value=bool(self._settings.get("use_ocr", False)), classes="flag-check"
                )
                yield Checkbox(
                    "Recursive",
                    id="recursive",
                    value=bool(self._settings.get("recursive", False)),
                    classes="flag-check",
                )
                yield Checkbox(
                    "Skip already named",
                    id="skip_already_named",
                    value=bool(self._settings.get("skip_already_named", False)),
                    classes="flag-check",
                )

    def _compose_advanced(self) -> ComposeResult:
        with ScrollableContainer(classes="form-container"):
            yield Static("Output & Integration", classes="section-title")
            for label, field_id, placeholder in [
                ("Template", "template", "{date}_{category}_{keywords}"),
                ("Backup dir", "backup_dir", "/path/to/backups"),
                ("Rename log", "rename_log", "rename_log.tsv"),
                ("Export metadata", "export_metadata", "metadata.json"),
                ("Summary JSON", "summary_json", "summaries.json"),
                ("Rules file", "rules_file", "processing_rules.json"),
                ("Post-rename hook", "post_rename_hook", "/path/to/script.sh"),
            ]:
                with Horizontal(classes="field-row"):
                    yield Label(f"{label}:", classes="field-label")
                    yield Input(
                        id=field_id,
                        placeholder=placeholder,
                        classes="field-input",
                        value=str(self._settings.get(field_id, "")),
                    )
            yield Static("", classes="section-sep")
            yield Static("LLM Configuration", classes="section-title")
            with Horizontal(classes="field-row"):
                yield Label("LLM backend:", classes="field-label")
                yield Select(
                    _LLM_BACKENDS,
                    id="llm_backend",
                    value=str(self._settings.get("llm_backend", "http")),
                )
            for label, field_id, placeholder in [
                ("LLM URL", "llm_url", "http://127.0.0.1:8080/v1/completions"),
                ("LLM model", "llm_model", "default"),
                ("LLM model path", "llm_model_path", "/path/to/model.gguf"),
                ("LLM timeout (s)", "llm_timeout", "60"),
            ]:
                with Horizontal(classes="field-row"):
                    yield Label(f"{label}:", classes="field-label")
                    yield Input(
                        id=field_id,
                        placeholder=placeholder,
                        classes="field-input",
                        value=str(self._settings.get(field_id, "")),
                    )
            yield Static("", classes="section-sep")
            yield Static("Limits", classes="section-title")
            for label, field_id, placeholder in [
                ("Max extract tokens", "max_tokens", ""),
                ("Max content chars", "max_content_chars", ""),
                ("Max content tokens", "max_content_tokens", ""),
                ("Workers", "workers", "1"),
                ("Max filename chars", "max_filename_chars", ""),
            ]:
                with Horizontal(classes="field-row"):
                    yield Label(f"{label}:", classes="field-label")
                    yield Input(
                        id=field_id,
                        placeholder=placeholder,
                        classes="field-input",
                        value=str(self._settings.get(field_id, "")),
                    )
            yield Static("", classes="section-sep")
            yield Static("Advanced Flags", classes="section-title")
            with Horizontal(classes="flags-row"):
                yield Checkbox(
                    "PDF metadata date",
                    id="use_pdf_metadata_date",
                    value=bool(self._settings.get("use_pdf_metadata_date", True)),
                    classes="flag-check",
                )
                yield Checkbox(
                    "Structured fields",
                    id="use_structured_fields",
                    value=bool(self._settings.get("use_structured_fields", True)),
                    classes="flag-check",
                )
                yield Checkbox(
                    "Write PDF title",
                    id="write_pdf_metadata",
                    value=bool(self._settings.get("write_pdf_metadata", False)),
                    classes="flag-check",
                )
                yield Checkbox(
                    "Vision fallback",
                    id="use_vision_fallback",
                    value=bool(self._settings.get("use_vision_fallback", False)),
                    classes="flag-check",
                )
                yield Checkbox(
                    "Simple naming",
                    id="simple_naming_mode",
                    value=bool(self._settings.get("simple_naming_mode", False)),
                    classes="flag-check",
                )
                yield Checkbox(
                    "Vision first",
                    id="vision_first",
                    value=bool(self._settings.get("vision_first", False)),
                    classes="flag-check",
                )

    def _compose_run(self) -> ComposeResult:
        with Horizontal(id="run-status-bar"):
            yield Static("[dim]IDLE[/dim]  Ready to process", id="run-status", classes="status-idle")
            yield Static("", id="run-file-counter")
        yield ProgressBar(id="run-progress", show_eta=False)
        yield Static("", id="run-summary")
        yield RichLog(id="run-log", highlight=True, markup=True)
        with Horizontal(id="run-buttons"):
            yield Button("Preview (dry run)", id="btn-preview", variant="primary")
            yield Button("Apply renames", id="btn-apply", variant="success")
            yield Button("Process one file", id="btn-one", variant="default")
            yield Button("Cancel", id="btn-cancel", variant="error")

    # ------------------------------------------------------------------
    # Settings
    # ------------------------------------------------------------------

    def _get_str(self, widget_id: str, default: str = "") -> str:
        try:
            w = self.query_one(f"#{widget_id}", Input)
            return str(w.value).strip()
        except Exception:  # Textual query/widget errors (no stubs to narrow)
            logger.debug("Widget query failed for #%s (Input)", widget_id)
            return default

    def _get_bool(self, widget_id: str, default: bool = False) -> bool:
        try:
            w = self.query_one(f"#{widget_id}", Checkbox)
            return bool(w.value)
        except Exception:  # Textual query/widget errors (no stubs to narrow)
            logger.debug("Widget query failed for #%s (Checkbox)", widget_id)
            return default

    def _get_select(self, widget_id: str, default: str = "") -> str:
        try:
            w = self.query_one(f"#{widget_id}", Select)
            v = w.value
            return str(v) if v is not Select.BLANK else default
        except Exception:  # Textual query/widget errors (no stubs to narrow)
            logger.debug("Widget query failed for #%s (Select)", widget_id)
            return default

    def _snapshot(self) -> dict[str, object]:
        return {
            "directory": self._get_str("directory"),
            "single_file": self._get_str("single_file"),
            "language": self._get_select("language", "de"),
            "case": self._get_select("case", "kebabCase"),
            "date_format": self._get_select("date_format", "dmy"),
            "preset": self._get_select("preset", ""),
            "project": self._get_str("project"),
            "version": self._get_str("version"),
            "template": self._get_str("template"),
            "backup_dir": self._get_str("backup_dir"),
            "rename_log": self._get_str("rename_log"),
            "export_metadata": self._get_str("export_metadata"),
            "summary_json": self._get_str("summary_json"),
            "rules_file": self._get_str("rules_file"),
            "post_rename_hook": self._get_str("post_rename_hook"),
            "llm_backend": self._get_select("llm_backend", "http"),
            "llm_url": self._get_str("llm_url"),
            "llm_model": self._get_str("llm_model"),
            "llm_model_path": self._get_str("llm_model_path"),
            "llm_timeout": self._get_str("llm_timeout"),
            "max_tokens": self._get_str("max_tokens"),
            "max_content_chars": self._get_str("max_content_chars"),
            "max_content_tokens": self._get_str("max_content_tokens"),
            "workers": self._get_str("workers"),
            "max_filename_chars": self._get_str("max_filename_chars"),
            "dry_run": self._get_bool("dry_run", True),
            "use_llm": self._get_bool("use_llm", True),
            "use_ocr": self._get_bool("use_ocr"),
            "recursive": self._get_bool("recursive"),
            "skip_already_named": self._get_bool("skip_already_named"),
            "use_pdf_metadata_date": self._get_bool("use_pdf_metadata_date", True),
            "use_structured_fields": self._get_bool("use_structured_fields", True),
            "write_pdf_metadata": self._get_bool("write_pdf_metadata"),
            "use_vision_fallback": self._get_bool("use_vision_fallback"),
            "simple_naming_mode": self._get_bool("simple_naming_mode"),
            "vision_first": self._get_bool("vision_first"),
        }

    def _build_config(self, *, dry_run: bool, manual_mode: bool = False) -> RenamerConfig:
        snap = self._snapshot()
        raw = {
            "language": snap["language"],
            "desired_case": snap["case"],
            "project": snap["project"],
            "version": snap["version"],
            "date_locale": snap["date_format"],
            "dry_run": dry_run,
            "use_llm": snap["use_llm"],
            "use_ocr": snap["use_ocr"],
            "use_pdf_metadata_for_date": snap["use_pdf_metadata_date"],
            "use_structured_fields": snap["use_structured_fields"],
            "skip_if_already_named": snap["skip_already_named"],
            "recursive": snap["recursive"],
            "backup_dir": snap["backup_dir"],
            "rename_log_path": snap["rename_log"],
            "export_metadata_path": snap["export_metadata"],
            "summary_json_path": snap["summary_json"],
            "rules_file": snap["rules_file"],
            "post_rename_hook": snap["post_rename_hook"],
            "llm_backend": snap["llm_backend"],
            "llm_base_url": snap["llm_url"],
            "llm_model": snap["llm_model"],
            "llm_model_path": snap["llm_model_path"],
            "llm_timeout_s": snap["llm_timeout"],
            "max_tokens_for_extraction": snap["max_tokens"],
            "max_content_chars": snap["max_content_chars"],
            "max_content_tokens": snap["max_content_tokens"],
            "workers": snap["workers"],
            "max_filename_chars": snap["max_filename_chars"],
            "write_pdf_metadata": snap["write_pdf_metadata"],
            "filename_template": snap["template"],
            "use_vision_fallback": snap["use_vision_fallback"],
            "simple_naming_mode": snap["simple_naming_mode"],
            "vision_first": snap["vision_first"],
            "preset": snap["preset"],
            "manual_mode": manual_mode,
            "interactive": manual_mode,
            "stop_event": self._stop_event,
        }
        return build_config(raw)

    # ------------------------------------------------------------------
    # Run worker
    # ------------------------------------------------------------------

    def _run_worker(self, directory: str, config: RenamerConfig) -> None:
        handler = _QueueHandler(self._log_queue)
        root_logger = logging.getLogger()
        root_logger.addHandler(handler)
        try:
            rename_pdfs_in_directory(directory, config=config)
            self._result_queue.put((True, "Completed"))
        except Exception as exc:
            self._result_queue.put((False, str(exc)))
        finally:
            root_logger.removeHandler(handler)
            self._log_queue.put(None)  # sentinel

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
        # Highlight rename results: 'old' -> 'new'
        if "Renamed '" in stripped and "' to '" in stripped:
            self._run_counts["renamed"] += 1
            self._update_summary()
            return _RENAME_LOG_RE.sub(_format_rename_match, stripped) if _RENAME_LOG_RE.search(stripped) else stripped
        if "Dry-run: would rename '" in stripped and "' to '" in stripped:
            self._run_counts["renamed"] += 1
            self._update_summary()
            return _DRYRUN_LOG_RE.sub(_format_dryrun_match, stripped) if _DRYRUN_LOG_RE.search(stripped) else stripped
        # Highlight skip messages
        if "Skipping " in stripped or "Skipped" in stripped or "content is empty" in stripped:
            self._run_counts["skipped"] += 1
            self._update_summary()
            return f"[yellow]SKIP[/yellow] [dim]{stripped}[/dim]"
        # Highlight errors/failures
        if "Failed" in stripped or "Error" in stripped or "failed" in stripped:
            self._run_counts["failed"] += 1
            self._update_summary()
            return f"[red]ERR[/red]  [bold red]{stripped}[/bold red]"
        # Highlight processing progress
        if PROCESS_RE.search(stripped):
            return f"[dim]{stripped}[/dim]"
        # Highlight summary line
        if stripped.startswith("Summary:"):
            return f"[bold]{stripped}[/bold]"
        # Highlight heuristic-only mode notice
        if "Heuristic-only mode" in stripped:
            return f"[cyan]INFO[/cyan] [dim]{stripped}[/dim]"
        return stripped

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
        with contextlib.suppress(Exception):
            self.query_one("#run-summary", Static).update(summary_text)

    def _drain_log_queue(self) -> None:
        log = self.query_one("#run-log", RichLog)
        progress = self.query_one("#run-progress", ProgressBar)
        counter = self.query_one("#run-file-counter", Static)
        try:
            while True:
                line = self._log_queue.get_nowait()
                if line is None:
                    self._running = False
                    ok, msg = self._result_queue.get_nowait() if not self._result_queue.empty() else (True, "Completed")
                    c = self._run_counts
                    if ok:
                        self._set_status("Completed", "status-done")
                        summary_parts = []
                        if c["renamed"]:
                            summary_parts.append(f"[green]{c['renamed']} renamed[/green]")
                        if c["skipped"]:
                            summary_parts.append(f"[yellow]{c['skipped']} skipped[/yellow]")
                        if c["failed"]:
                            summary_parts.append(f"[red]{c['failed']} failed[/red]")
                        summary_line = "  ".join(summary_parts) if summary_parts else "no files processed"
                        log.write(f"\n[bold green]Run completed.[/bold green]  {summary_line}")
                    else:
                        self._set_status(f"Failed: {msg}", "status-error")
                        log.write(f"[bold red]Run failed:[/bold red] {_escape_markup(msg)}")
                    counter.update("")
                    return
                log.write(self._format_log_line(line))
                m = PROCESS_RE.search(line)
                if m:
                    cur = int(m.group(1))
                    tot = max(1, int(m.group(2)))
                    progress.update(total=tot, progress=cur)
                    self._set_status(f"Processing {cur}/{tot}...", "status-running")
                    counter.update(f"{cur} of {tot} files")
        except queue.Empty:
            pass
        self.set_timer(0.1, self._drain_log_queue)

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def action_preview(self) -> None:
        self._start_run(dry_run=True)

    def action_apply(self) -> None:
        self._start_run(dry_run=False)

    def action_cancel(self) -> None:
        self._cancel()

    def _start_run(self, *, dry_run: bool) -> None:
        if self._running:
            self.query_one("#run-log", RichLog).write("[yellow]A run is already in progress.[/yellow]")
            return
        directory = self._get_str("directory")
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
            config = self._build_config(dry_run=dry_run)
        except ValueError as exc:
            err_msg = _escape_markup(str(exc))
            self.query_one("#run-log", RichLog).write(f"[bold red]Invalid config:[/bold red] {err_msg}")
            return
        _save_settings(self._snapshot())
        self._stop_event.clear()
        self._running = True
        self._run_counts = {"renamed": 0, "skipped": 0, "failed": 0}
        with contextlib.suppress(Exception):
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
        worker = threading.Thread(target=self._run_worker, args=(directory, config), daemon=True)
        worker.start()
        self.set_timer(0.1, self._drain_log_queue)

    def _cancel(self) -> None:
        if not self._running:
            return
        self._stop_event.set()
        self._set_status("Cancelling...", "status-cancel")
        self.query_one("#run-log", RichLog).write("[yellow]Cancel requested -- finishing current file...[/yellow]")

    def _process_one(self) -> None:
        if self._running:
            self.query_one("#run-log", RichLog).write("[yellow]A run is already in progress.[/yellow]")
            return
        file_path = self._get_str("single_file")
        if not file_path:
            self.query_one("#run-log", RichLog).write(
                "[bold red]No file set.[/bold red] Set a single file path on the Settings tab first."
            )
            return
        fp = Path(file_path)
        if not fp.exists():
            self.query_one("#run-log", RichLog).write(
                f"[bold red]File not found:[/bold red] {_escape_markup(file_path)}"
            )
            return
        if fp.suffix.lower() != ".pdf":
            self.query_one("#run-log", RichLog).write(
                f"[bold red]Not a PDF file:[/bold red] {_escape_markup(file_path)} (expected .pdf extension)"
            )
            return
        try:
            config = self._build_config(dry_run=False, manual_mode=True)
        except ValueError as exc:
            self.query_one("#run-log", RichLog).write(
                f"[bold red]Invalid config:[/bold red] {_escape_markup(str(exc))}"
            )
            return
        log = self.query_one("#run-log", RichLog)
        log.write("\n[bold cyan]Single File Processing[/bold cyan]")
        log.write(f"[dim]File: {_escape_markup(fp.name)}[/dim]")
        self._set_status("Processing single file...", "status-running")
        # P1 Bug 14: Wrap in thread to avoid blocking main thread
        self._running = True
        self._run_counts = {"renamed": 0, "skipped": 0, "failed": 0}
        with contextlib.suppress(Exception):
            self.query_one("#run-summary", Static).update("")

        def _single_file_worker() -> None:
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
                export_rows: list[dict[str, object]] = []

                # P1 Bug 8: Add on_success callback like batch mode
                def _on_rename_success(
                    _fp: Path,
                    _target: Path,
                    _current_base: str,
                    _meta: dict[str, object] = meta or {},
                    _rows: list[dict[str, object]] = export_rows,
                ) -> None:
                    _apply_post_rename_actions(config, _fp, _target, _current_base, _meta, _rows)

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
                    if meta:
                        meta_parts = []
                        for k in ("category", "summary", "keywords", "category_source"):
                            v = meta.get(k)
                            if v:
                                meta_parts.append(f"[dim]{k}:[/dim] {_escape_markup(str(v))}")
                        if meta_parts:
                            self._log_queue.put("  " + "  |  ".join(meta_parts) + "\n")
                    self._result_queue.put((True, "Completed"))
                else:
                    self._log_queue.put("[bold red]Could not rename file.[/bold red]\n")
                    self._result_queue.put((False, "Could not rename file"))
            except Exception as exc:
                self._result_queue.put((False, str(exc)))
            finally:
                self._log_queue.put(None)  # sentinel

        worker = threading.Thread(target=_single_file_worker, daemon=True)
        worker.start()
        self.set_timer(0.1, self._drain_log_queue)

    # ------------------------------------------------------------------
    # Button handlers
    # ------------------------------------------------------------------

    @on(Button.Pressed, "#btn-preview")
    def on_preview(self) -> None:
        self._start_run(dry_run=True)

    @on(Button.Pressed, "#btn-apply")
    def on_apply(self) -> None:
        self._start_run(dry_run=False)

    @on(Button.Pressed, "#btn-one")
    def on_one(self) -> None:
        self._process_one()

    @on(Button.Pressed, "#btn-cancel")
    def on_cancel(self) -> None:
        self._cancel()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    setup_logging(level=logging.INFO)
    app = AIRenamerTUI()
    app.run()


if __name__ == "__main__":
    main()
