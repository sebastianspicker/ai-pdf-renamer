"""Textual widget composition helpers for the TUI form panes."""

from __future__ import annotations

from collections.abc import Sequence

from textual.app import ComposeResult
from textual.containers import Horizontal, ScrollableContainer
from textual.widgets import Button, Checkbox, Input, Label, ProgressBar, RichLog, Select, Static

from .tui_assets import _CASES, _DATE_FORMATS, _LANGUAGES, _LLM_BACKENDS


def _text_input_row(label: str, field_id: str, placeholder: str, settings: dict[str, object]) -> ComposeResult:
    with Horizontal(classes="field-row"):
        yield Label(f"{label}:", classes="field-label")
        yield Input(
            id=field_id,
            placeholder=placeholder,
            classes="field-input",
            value=str(settings.get(field_id, "")),
        )


def _compose_input_source(settings: dict[str, object]) -> ComposeResult:
    yield Static("Input Source", classes="section-title")
    yield from _text_input_row("Folder", "directory", "/path/to/PDFs", settings)
    yield from _text_input_row("Single file", "single_file", "/path/to/file.pdf", settings)


def _compose_naming_options(settings: dict[str, object], presets: Sequence[tuple[str, str]]) -> ComposeResult:
    yield Static("", classes="section-sep")
    yield Static("Naming Options", classes="section-title")
    with Horizontal(classes="field-row"):
        yield Label("Language:", classes="field-label")
        yield Select(_LANGUAGES, id="language", value=str(settings.get("language", "de")))
    with Horizontal(classes="field-row"):
        yield Label("Case:", classes="field-label")
        yield Select(_CASES, id="case", value=str(settings.get("case", "kebabCase")))
    with Horizontal(classes="field-row"):
        yield Label("Date format:", classes="field-label")
        yield Select(_DATE_FORMATS, id="date_format", value=str(settings.get("date_format", "dmy")))
    with Horizontal(classes="field-row"):
        yield Label("Preset:", classes="field-label")
        yield Select(presets, id="preset", value=str(settings.get("preset", "")))
    yield from _text_input_row("Project", "project", "optional project name", settings)
    yield from _text_input_row("Version", "version", "optional version", settings)


def _compose_processing_flags(settings: dict[str, object]) -> ComposeResult:
    yield Static("", classes="section-sep")
    yield Static("Processing Flags", classes="section-title")
    with Horizontal(classes="flags-row"):
        yield Checkbox("Dry run", id="dry_run", value=bool(settings.get("dry_run", True)), classes="flag-check")
        yield Checkbox("Use LLM", id="use_llm", value=bool(settings.get("use_llm", True)), classes="flag-check")
        yield Checkbox("OCR", id="use_ocr", value=bool(settings.get("use_ocr", False)), classes="flag-check")
        yield Checkbox("Recursive", id="recursive", value=bool(settings.get("recursive", False)), classes="flag-check")
        yield Checkbox(
            "Skip already named",
            id="skip_already_named",
            value=bool(settings.get("skip_already_named", False)),
            classes="flag-check",
        )


def compose_basic(settings: dict[str, object], presets: Sequence[tuple[str, str]]) -> ComposeResult:
    with ScrollableContainer(classes="form-container"):
        yield from _compose_input_source(settings)
        yield from _compose_naming_options(settings, presets)
        yield from _compose_processing_flags(settings)


def _compose_output_integration(settings: dict[str, object]) -> ComposeResult:
    yield Static("Output & Integration", classes="section-title")
    for label, field_id, placeholder in [
        ("Template", "template", "{date}_{category}_{keywords}"),
        ("Backup dir", "backup_dir", "/path/to/backups"),
        ("Rename log", "rename_log", "rename_log.tsv"),
        ("Export metadata", "export_metadata", "metadata.json"),
        ("Summary JSON", "summary_json", "summaries.json"),
        ("Rules file", "rules_file", "processing_rules.json"),
        ("Post-rename hook", "post_rename_hook", "https://example.invalid/hook"),
    ]:
        yield from _text_input_row(label, field_id, placeholder, settings)


def _compose_llm_configuration(settings: dict[str, object]) -> ComposeResult:
    yield Static("", classes="section-sep")
    yield Static("LLM Configuration", classes="section-title")
    with Horizontal(classes="field-row"):
        yield Label("LLM backend:", classes="field-label")
        yield Select(_LLM_BACKENDS, id="llm_backend", value=str(settings.get("llm_backend", "http")))
    for label, field_id, placeholder in [
        ("LLM URL", "llm_url", "http://127.0.0.1:8080/v1/completions"),
        ("LLM model", "llm_model", "default"),
        ("LLM model path", "llm_model_path", "/path/to/model.gguf"),
        ("LLM timeout (s)", "llm_timeout", "60"),
    ]:
        yield from _text_input_row(label, field_id, placeholder, settings)


def _compose_limits(settings: dict[str, object]) -> ComposeResult:
    yield Static("", classes="section-sep")
    yield Static("Limits", classes="section-title")
    for label, field_id, placeholder in [
        ("Max extract tokens", "max_tokens", ""),
        ("Max content chars", "max_content_chars", ""),
        ("Max content tokens", "max_content_tokens", ""),
        ("Workers", "workers", "1"),
        ("Max filename chars", "max_filename_chars", ""),
    ]:
        yield from _text_input_row(label, field_id, placeholder, settings)


def _compose_advanced_flags(settings: dict[str, object]) -> ComposeResult:
    yield Static("", classes="section-sep")
    yield Static("Advanced Flags", classes="section-title")
    with Horizontal(classes="flags-row"):
        yield Checkbox(
            "PDF metadata date",
            id="use_pdf_metadata_date",
            value=bool(settings.get("use_pdf_metadata_date", True)),
            classes="flag-check",
        )
        yield Checkbox(
            "Structured fields",
            id="use_structured_fields",
            value=bool(settings.get("use_structured_fields", True)),
            classes="flag-check",
        )
        yield Checkbox(
            "Write PDF title",
            id="write_pdf_metadata",
            value=bool(settings.get("write_pdf_metadata", False)),
            classes="flag-check",
        )
        yield Checkbox(
            "Vision fallback",
            id="use_vision_fallback",
            value=bool(settings.get("use_vision_fallback", False)),
            classes="flag-check",
        )
        yield Checkbox(
            "Simple naming",
            id="simple_naming_mode",
            value=bool(settings.get("simple_naming_mode", False)),
            classes="flag-check",
        )
        yield Checkbox(
            "Vision first",
            id="vision_first",
            value=bool(settings.get("vision_first", False)),
            classes="flag-check",
        )


def compose_advanced(settings: dict[str, object]) -> ComposeResult:
    with ScrollableContainer(classes="form-container"):
        yield from _compose_output_integration(settings)
        yield from _compose_llm_configuration(settings)
        yield from _compose_limits(settings)
        yield from _compose_advanced_flags(settings)


def compose_run() -> ComposeResult:
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
