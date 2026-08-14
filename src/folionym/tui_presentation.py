"""Framework-free parsing and presentation for the terminal run view."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from .http_url import validate_http_endpoint
from .tui_assets import (
    _DRYRUN_LOG_RE,
    _RENAME_LOG_RE,
    ERROR_COLOR,
    PREVIEW_COLOR,
    PROCESS_RE,
    SUCCESS_COLOR,
    WARNING_COLOR,
    _format_dryrun_match,
    _format_rename_match,
)

RunCounts = dict[str, int]


@dataclass(frozen=True)
class PreviewRecord:
    """A recognized rename proposal prepared for the Textual preview widgets."""

    status: str
    source: str
    proposed: str
    status_color: str
    mode: str


def snapshot_from_readers(
    get_str: Callable[[str, str], str],
    get_bool: Callable[[str, bool], bool],
    get_select: Callable[[str, str], str],
) -> dict[str, object]:
    """Collect the persisted configuration shape from caller-supplied widget readers."""
    return {
        "directory": get_str("directory", ""),
        "single_file": get_str("single_file", ""),
        "language": get_select("language", "de"),
        "case": get_select("case", "kebabCase"),
        "date_format": get_select("date_format", "dmy"),
        "preset": get_select("preset", ""),
        "project": get_str("project", ""),
        "version": get_str("version", ""),
        "template": get_str("template", ""),
        "backup_dir": get_str("backup_dir", ""),
        "rename_log": get_str("rename_log", ""),
        "export_metadata": get_str("export_metadata", ""),
        "summary_json": get_str("summary_json", ""),
        "rules_file": get_str("rules_file", ""),
        "post_rename_hook": get_str("post_rename_hook", ""),
        "llm_url": get_str("llm_url", ""),
        "llm_model": get_str("llm_model", ""),
        "llm_timeout": get_str("llm_timeout", ""),
        "max_tokens": get_str("max_tokens", ""),
        "max_content_chars": get_str("max_content_chars", ""),
        "max_content_tokens": get_str("max_content_tokens", ""),
        "workers": get_str("workers", ""),
        "max_filename_chars": get_str("max_filename_chars", ""),
        "dry_run": True,
        "use_llm": get_bool("use_llm", True),
        "use_ocr": get_bool("use_ocr", False),
        "recursive": get_bool("recursive", False),
        "skip_already_named": get_bool("skip_already_named", False),
        "use_pdf_metadata_date": get_bool("use_pdf_metadata_date", True),
        "use_structured_fields": get_bool("use_structured_fields", True),
        "write_pdf_metadata": get_bool("write_pdf_metadata", False),
        "use_vision_fallback": get_bool("use_vision_fallback", False),
        "simple_naming_mode": get_bool("simple_naming_mode", False),
        "vision_first": get_bool("vision_first", False),
    }


def endpoint_disclosure(use_llm: bool, endpoint_value: str) -> tuple[str, str]:
    """Describe the configured content boundary without exposing endpoint details."""
    if not use_llm:
        return "HEURISTICS ONLY", "Document text stays on this machine."
    if not endpoint_value:
        return "LOCAL HTTP MODEL", "Document text may be sent to the preset local endpoint."
    try:
        endpoint = validate_http_endpoint(endpoint_value)
    except ValueError:
        return "HTTP MODEL", "Review the endpoint before sending document-derived content."
    if endpoint.is_loopback:
        return "LOCAL HTTP MODEL", "Document text may be sent to the configured local endpoint."
    return "EXTERNAL HTTP MODEL", "Document-derived content may leave this machine."


def effective_configuration_lines(language: str, case_style: str, preset: str, ocr_enabled: bool) -> str:
    """Format the concise control summary used by the next-run disclosure."""
    return "\n".join(
        (
            f"LANGUAGE       {language.upper()}",
            f"FILENAME STYLE {case_style}",
            f"PRESET         {preset or 'custom'}",
            f"OCR            {'enabled' if ocr_enabled else 'off'}",
        )
    )


def parse_preview_record(line: str) -> PreviewRecord | None:
    """Recognize a rename line and return its structured review projection."""
    match = _DRYRUN_LOG_RE.search(line)
    if match is not None:
        return PreviewRecord(
            "SUGGESTED",
            match.group(1).rsplit("/", 1)[-1],
            match.group(2).rsplit("/", 1)[-1],
            PREVIEW_COLOR,
            "Preview only. No file changed.",
        )
    match = _RENAME_LOG_RE.search(line)
    if match is None:
        return None
    return PreviewRecord(
        "RENAMED",
        match.group(1).rsplit("/", 1)[-1],
        match.group(2).rsplit("/", 1)[-1],
        SUCCESS_COLOR,
        "Rename completed.",
    )


def _format_rename_line(stripped: str) -> tuple[str, str | None] | None:
    """Format applied and preview rename lines when recognized."""
    if "Renamed '" in stripped and "' to '" in stripped:
        formatted = _RENAME_LOG_RE.sub(_format_rename_match, stripped) if _RENAME_LOG_RE.search(stripped) else stripped
        return formatted, "renamed"
    if "Dry-run: would rename '" in stripped and "' to '" in stripped:
        formatted = _DRYRUN_LOG_RE.sub(_format_dryrun_match, stripped) if _DRYRUN_LOG_RE.search(stripped) else stripped
        return formatted, "renamed"
    return None


def _format_status_outcome(stripped: str) -> tuple[str, str | None] | None:
    """Format skip and failure outcomes when recognized."""
    if "Skipping " in stripped or "Skipped" in stripped or "content is empty" in stripped:
        return f"[{WARNING_COLOR}]SKIP[/{WARNING_COLOR}] [dim]{stripped}[/dim]", "skipped"
    if "Failed" in stripped or "Error" in stripped or "failed" in stripped:
        return f"[{ERROR_COLOR}]ERR[/{ERROR_COLOR}]  [bold {ERROR_COLOR}]{stripped}[/bold {ERROR_COLOR}]", "failed"
    return None


def _format_context_line(stripped: str) -> tuple[str, str | None]:
    """Format progress and informational lines without changing counters."""
    if PROCESS_RE.search(stripped):
        return f"[dim]{stripped}[/dim]", None
    if stripped.startswith("Summary:"):
        return f"[bold]{stripped}[/bold]", None
    if "Heuristic-only mode" in stripped:
        return f"[{PREVIEW_COLOR}]INFO[/{PREVIEW_COLOR}] [dim]{stripped}[/dim]", None
    return stripped, None


def format_run_log_line(line: str) -> tuple[str, str | None]:
    """Return Rich markup and an optional counter key for one worker log line."""
    stripped = line.rstrip()
    return _format_rename_line(stripped) or _format_status_outcome(stripped) or _format_context_line(stripped)


def format_run_summary(counts: RunCounts, preview: bool, *, separator: str) -> str:
    """Format colored counters for the run view or terminal completion line."""
    result_label = "suggestions" if preview else "renamed"
    parts = []
    if counts["renamed"]:
        parts.append(f"[{SUCCESS_COLOR}]{counts['renamed']} {result_label}[/{SUCCESS_COLOR}]")
    if counts["skipped"]:
        parts.append(f"[{WARNING_COLOR}]{counts['skipped']} skipped[/{WARNING_COLOR}]")
    if counts["failed"]:
        parts.append(f"[{ERROR_COLOR}]{counts['failed']} failed[/{ERROR_COLOR}]")
    return separator.join(parts)


def metric_summary(counts: RunCounts, preview: bool) -> tuple[str, str, str]:
    """Format the three persistent run metrics without duplicating their labels."""
    result_label = "suggestions" if preview else "renamed"
    return (
        f"[b]{counts['renamed']}[/b] {result_label}",
        f"[b]{counts['skipped']}[/b] skipped",
        f"[b]{counts['failed']}[/b] failed",
    )


def completion_summary(counts: RunCounts, preview: bool) -> str:
    """Return the final completion copy, including the established empty-run fallback."""
    return format_run_summary(counts, preview, separator="  ") or "no files processed"


def parse_progress(line: str) -> tuple[int, int] | None:
    """Parse a producer progress line while protecting the progress widget from zero totals."""
    match = PROCESS_RE.search(line)
    if not match:
        return None
    return int(match.group(1)), max(1, int(match.group(2)))
