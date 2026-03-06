"""Terminal UI for AI-PDF-Renamer (Textual-based, replaces Tkinter gui.py).

Launch with: ai-pdf-renamer-tui
Requires: pip install -e '.[tui]'
"""

from __future__ import annotations

import json
import logging
import queue
import re
import threading
from pathlib import Path
from typing import ClassVar

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
from .renamer import RenamerConfig, rename_pdfs_in_directory, suggest_rename_for_file

logger = logging.getLogger(__name__)

SETTINGS_PATH = Path.home() / ".ai_pdf_renamer_gui.json"
PROCESS_RE = re.compile(r"Processing\s+(\d+)/(\d+)\s*:")

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
            self._queue.put(self.format(record) + "\n")
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
        SETTINGS_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    except OSError:
        logger.debug("Could not save TUI settings", exc_info=True)


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

_CSS = """
Screen {
    layout: vertical;
}

TabbedContent {
    height: 1fr;
}

.field-row {
    height: auto;
    margin-bottom: 1;
    layout: horizontal;
}

.field-label {
    width: 22;
    padding: 0 1 0 0;
    content-align: left middle;
}

.field-input {
    width: 1fr;
}

.section-sep {
    height: 1;
    border-top: solid $primary-darken-2;
    margin: 1 0;
}

.flags-row {
    height: auto;
    layout: horizontal;
    flex-wrap: wrap;
}

.flag-check {
    margin-right: 2;
}

#run-tab {
    layout: vertical;
}

#run-status {
    height: 1;
    padding: 0 1;
}

#run-progress {
    height: 2;
    margin: 0 0 1 0;
}

#run-log {
    height: 1fr;
}

#run-buttons {
    height: auto;
    layout: horizontal;
    margin-top: 1;
}

#run-buttons Button {
    margin-right: 1;
}
"""


# ---------------------------------------------------------------------------
# Main App
# ---------------------------------------------------------------------------


class AIRenamerTUI(App[None]):
    """Terminal UI for AI-PDF-Renamer."""

    TITLE = "AI-PDF-Renamer"
    CSS = _CSS
    BINDINGS: ClassVar[list[Binding]] = [
        Binding("ctrl+q", "quit", "Quit"),
        Binding("ctrl+p", "preview", "Preview"),
        Binding("ctrl+a", "apply", "Apply"),
        Binding("ctrl+c", "cancel", "Cancel"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self._log_queue: queue.Queue[str | None] = queue.Queue()
        self._result_queue: queue.Queue[tuple[bool, str]] = queue.Queue()
        self._stop_event = threading.Event()
        self._running = False
        self._settings = _load_settings()

    # ------------------------------------------------------------------
    # Compose
    # ------------------------------------------------------------------

    def compose(self) -> ComposeResult:
        yield Header()
        with TabbedContent(initial="basic"):
            with TabPane("Basic", id="basic"):
                yield from self._compose_basic()
            with TabPane("Advanced", id="advanced"):
                yield from self._compose_advanced()
            with TabPane("Run", id="run"):
                yield from self._compose_run()
        yield Footer()

    def _compose_basic(self) -> ComposeResult:
        with ScrollableContainer():
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
        with ScrollableContainer():
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
        yield Static("Idle", id="run-status")
        yield ProgressBar(id="run-progress", show_eta=False)
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
            return w.value.strip()
        except Exception:
            return default

    def _get_bool(self, widget_id: str, default: bool = False) -> bool:
        try:
            w = self.query_one(f"#{widget_id}", Checkbox)
            return w.value
        except Exception:
            return default

    def _get_select(self, widget_id: str, default: str = "") -> str:
        try:
            w = self.query_one(f"#{widget_id}", Select)
            v = w.value
            return str(v) if v is not Select.BLANK else default
        except Exception:
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

    def _drain_log_queue(self) -> None:
        log = self.query_one("#run-log", RichLog)
        progress = self.query_one("#run-progress", ProgressBar)
        status = self.query_one("#run-status", Static)
        try:
            while True:
                line = self._log_queue.get_nowait()
                if line is None:
                    self._running = False
                    ok, msg = self._result_queue.get_nowait() if not self._result_queue.empty() else (True, "Completed")
                    status.update("Idle")
                    if not ok:
                        log.write(f"[red]Run failed: {msg}[/red]")
                    return
                log.write(line.rstrip())
                m = PROCESS_RE.search(line)
                if m:
                    cur = int(m.group(1))
                    tot = max(1, int(m.group(2)))
                    progress.update(total=tot, progress=cur)
                    status.update(f"Processing {cur}/{tot}")
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
            self.query_one("#run-log", RichLog).write("[red]Please set a directory on the Basic tab.[/red]")
            return
        try:
            config = self._build_config(dry_run=dry_run)
        except ValueError as exc:
            self.query_one("#run-log", RichLog).write(f"[red]Invalid config: {exc}[/red]")
            return
        _save_settings(self._snapshot())
        self._stop_event.clear()
        self._running = True
        log = self.query_one("#run-log", RichLog)
        log.write(f"\n[bold]--- {'Preview' if dry_run else 'Apply'} ---[/bold]")
        status = self.query_one("#run-status", Static)
        status.update("Running…")
        progress = self.query_one("#run-progress", ProgressBar)
        progress.update(total=100, progress=0)
        worker = threading.Thread(target=self._run_worker, args=(directory, config), daemon=True)
        worker.start()
        self.set_timer(0.1, self._drain_log_queue)

    def _cancel(self) -> None:
        if not self._running:
            return
        self._stop_event.set()
        self.query_one("#run-status", Static).update("Cancel requested…")

    def _process_one(self) -> None:
        if self._running:
            self.query_one("#run-log", RichLog).write("[yellow]A run is already in progress.[/yellow]")
            return
        file_path = self._get_str("single_file")
        if not file_path:
            self.query_one("#run-log", RichLog).write("[red]Set a single file path on the Basic tab.[/red]")
            return
        fp = Path(file_path)
        if not fp.exists() or fp.suffix.lower() != ".pdf":
            self.query_one("#run-log", RichLog).write(f"[red]Not an existing PDF: {file_path}[/red]")
            return
        try:
            config = self._build_config(dry_run=False, manual_mode=True)
        except ValueError as exc:
            self.query_one("#run-log", RichLog).write(f"[red]Invalid config: {exc}[/red]")
            return
        log = self.query_one("#run-log", RichLog)
        log.write(f"Processing: {fp.name} …")
        new_base, meta, err = suggest_rename_for_file(fp, config)
        if err is not None:
            log.write(f"[red]Error: {err}[/red]")
            return
        if new_base is None:
            log.write("[yellow]Skipped: no extractable content.[/yellow]")
            return
        suggested = new_base + fp.suffix
        log.write(f"Suggested: [bold]{suggested}[/bold]")
        success, target = apply_single_rename(
            fp,
            sanitize_filename_base(new_base),
            plan_file_path=None,
            plan_entries=[],
            dry_run=False,
            backup_dir=config.backup_dir,
            on_success=None,
            max_filename_chars=config.max_filename_chars,
        )
        if success:
            log.write(f"[green]Renamed: {fp.name} -> {target.name}[/green]")
            if meta:
                log.write(f"Meta: {meta}")
        else:
            log.write("[red]Could not rename file.[/red]")

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
