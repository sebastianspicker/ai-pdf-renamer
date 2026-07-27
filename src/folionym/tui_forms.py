"""Textual widget composition helpers for the TUI form panes."""

from __future__ import annotations

from collections.abc import Sequence

from textual.app import ComposeResult
from textual.containers import Grid, Horizontal, ScrollableContainer, Vertical
from textual.widgets import Button, Checkbox, DataTable, Input, Label, ProgressBar, RichLog, Select, Static

from .tui_assets import _CASES, _DATE_FORMATS, _LANGUAGES


def _text_input_row(
    label: str,
    field_id: str,
    placeholder: str,
    settings: dict[str, object],
    *,
    numeric: bool = False,
) -> ComposeResult:
    """Yield a labelled text input with persisted form state as its initial value.

    Numeric fields restrict keystrokes to digits so invalid values are caught while
    typing rather than only after a Preview/Apply attempt.
    """
    with Horizontal(classes="field-row"):
        yield Label(f"{label}:", classes="field-label")
        yield Input(
            id=field_id,
            placeholder=placeholder,
            classes="field-input",
            value=str(settings.get(field_id, "")),
            type="integer" if numeric else "text",
            valid_empty=True,
        )


def _compose_input_source(settings: dict[str, object]) -> ComposeResult:
    """Yield directory and single-file inputs initialized from saved settings."""
    yield Static("Source", classes="section-title")
    yield Static("Choose a folder for batch work or one PDF for a focused rename.", classes="section-note")
    yield from _text_input_row("Folder batch", "directory", "/path/to/PDFs", settings)
    yield from _text_input_row("Single PDF", "single_file", "/path/to/file.pdf", settings)


def _compose_naming_options(settings: dict[str, object], presets: Sequence[tuple[str, str]]) -> ComposeResult:
    """Yield language, casing, preset, and optional identity controls."""
    yield Static("", classes="section-sep")
    yield Static("Filename rules", classes="section-title")
    yield Static("These values shape the proposed filename; Preview never changes files.", classes="section-note")
    with Horizontal(classes="field-row"):
        yield Label("Language:", classes="field-label")
        yield Select(_LANGUAGES, id="language", value=str(settings.get("language", "de")))
    with Horizontal(classes="field-row"):
        yield Label("Letter case:", classes="field-label")
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
    """Yield the high-frequency processing flags shown on the Settings pane."""
    yield Static("", classes="section-sep")
    yield Static("Processing", classes="section-title")
    yield Static("Optional extraction and classification behavior for the next run.", classes="section-note")
    with Grid(classes="flags-row"):
        yield Checkbox("Use HTTP LLM", id="use_llm", value=bool(settings.get("use_llm", True)), classes="flag-check")
        yield Checkbox("OCR", id="use_ocr", value=bool(settings.get("use_ocr", False)), classes="flag-check")
        yield Checkbox("Recursive", id="recursive", value=bool(settings.get("recursive", False)), classes="flag-check")
        yield Checkbox(
            "Skip already named",
            id="skip_already_named",
            value=bool(settings.get("skip_already_named", False)),
            classes="flag-check",
        )


def compose_basic(settings: dict[str, object], presets: Sequence[tuple[str, str]]) -> ComposeResult:
    """Assemble the scrollable Settings pane from pure widget helpers."""
    with ScrollableContainer(classes="form-container"):
        yield Static(
            "[b]PDF[/b]  →  extract · classify · name  →  [b]YYYYMMDD-category-keywords.pdf[/b]",
            id="rename-rail",
        )
        yield Static("", id="setup-error")
        yield from _compose_input_source(settings)
        yield from _compose_naming_options(settings, presets)
        yield from _compose_processing_flags(settings)


def _compose_output_integration(settings: dict[str, object]) -> ComposeResult:
    """Yield output paths and the optional post-rename integration hook."""
    yield Static("Records and integration", classes="section-title")
    yield Static("Leave a path empty when that artifact or integration is not required.", classes="section-note")
    for label, field_id, placeholder in [
        ("Template", "template", "{date}_{category}_{keywords}"),
        ("Backup folder", "backup_dir", "/path/to/backups"),
        ("Rename log", "rename_log", "rename_log.tsv"),
        ("Export metadata", "export_metadata", "metadata.json"),
        ("Summary JSON", "summary_json", "summaries.json"),
        ("Rules file", "rules_file", "processing_rules.json"),
        ("After-rename URL", "post_rename_hook", "https://example.invalid/hook"),
    ]:
        yield from _text_input_row(label, field_id, placeholder, settings)


def _compose_llm_configuration(settings: dict[str, object]) -> ComposeResult:
    """Yield HTTP-only LLM endpoint, model, and timeout controls."""
    yield Static("", classes="section-sep")
    yield Static("Model connection", classes="section-title")
    yield Static("Document-derived content may be sent to this HTTP endpoint.", classes="section-note")
    yield from _text_input_row("LLM URL", "llm_url", "http://127.0.0.1:11434/v1/completions", settings)
    yield from _text_input_row("LLM model", "llm_model", "default", settings)
    yield from _text_input_row("LLM timeout (s)", "llm_timeout", "60", settings, numeric=True)


def _compose_limits(settings: dict[str, object]) -> ComposeResult:
    """Yield extraction, content, worker, and filename limits."""
    yield Static("", classes="section-sep")
    yield Static("Limits", classes="section-title")
    for label, field_id, placeholder in [
        ("Max extract tokens", "max_tokens", ""),
        ("Max content chars", "max_content_chars", ""),
        ("Max content tokens", "max_content_tokens", ""),
        ("Workers", "workers", "1"),
        ("Max filename chars", "max_filename_chars", ""),
    ]:
        yield from _text_input_row(label, field_id, placeholder, settings, numeric=True)


def _compose_advanced_flags(settings: dict[str, object]) -> ComposeResult:
    """Yield lower-frequency metadata, vision, and naming-mode flags."""
    yield Static("", classes="section-sep")
    yield Static("Document handling", classes="section-title")
    yield Static("Use these controls only when the source documents require them.", classes="section-note")
    with Grid(classes="flags-row"):
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
    """Assemble the scrollable Advanced pane from pure widget helpers."""
    with ScrollableContainer(classes="form-container"):
        yield from _compose_output_integration(settings)
        yield from _compose_llm_configuration(settings)
        yield from _compose_limits(settings)
        yield from _compose_advanced_flags(settings)


def compose_run() -> ComposeResult:
    """Assemble the review workbench while retaining the chronological runtime log."""
    with Horizontal(id="review-workspace"):
        with Vertical(id="preview-pane"):
            with Horizontal(id="run-status-bar"):
                yield Static("Review & rename", id="review-title")
                yield Static("[dim]IDLE[/dim]  Ready to process", id="run-status", classes="status-idle")
                yield Static("", id="run-file-counter")
            yield Static(
                "Compare current and proposed filenames before applying changes.",
                id="review-description",
            )
            with Grid(id="run-metrics"):
                yield Static("[b]0[/b] PDFs", id="metric-files", classes="run-metric")
                yield Static("[b]0[/b] suggestions", id="metric-suggestions", classes="run-metric")
                yield Static("[b]0[/b] skipped", id="metric-skipped", classes="run-metric")
                yield Static("[b]0[/b] failed", id="metric-failed", classes="run-metric")
            yield ProgressBar(total=100, id="run-progress", show_eta=False)
            yield Static("No preview rows yet. Choose Preview folder to inspect proposed names.", id="preview-empty")
            yield DataTable(id="preview-table", cursor_type="row", zebra_stripes=False)
            yield Static("Activity", id="activity-label")
            yield RichLog(id="run-log", max_lines=5_000, min_width=1, wrap=True, highlight=False, markup=True)
            yield Static("", id="run-summary")
        with Vertical(id="review-inspector"):
            yield Static("Selected document", classes="inspector-label")
            yield Static("No document selected", id="inspector-source")
            yield Static("", classes="inspector-rule")
            yield Static("Proposed filename", classes="inspector-label")
            yield Static("Run a preview to inspect a proposed name.", id="inspector-proposed")
            yield Static("", classes="inspector-rule")
            yield Static("Processing", classes="inspector-label")
            yield Static("Waiting for preview output", id="inspector-mode")
            yield Static("", id="privacy-disclosure")
            yield Static("Effective configuration", classes="inspector-label")
            yield Static("", id="effective-config")
    yield Static(
        "[b]CAUTION[/b]  Apply recomputes suggestions; model-backed names may differ.",
        id="run-guidance",
    )
    with Grid(id="run-buttons"):
        yield Button("Preview again  ^P", id="btn-preview", variant="primary", tooltip="Ctrl+P")
        yield Button("Apply to folder...  ^A", id="btn-apply", variant="warning", tooltip="Ctrl+A")
        yield Button("Rename one PDF...", id="btn-one", variant="default")
        yield Button("Cancel  ^C", id="btn-cancel", variant="error", tooltip="Ctrl+C", disabled=True)
