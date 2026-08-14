"""Unit tests for Textual-independent TUI presentation helpers."""

from __future__ import annotations

from folionym.tui_presentation import (
    completion_summary,
    format_run_log_line,
    format_run_summary,
    parse_preview_record,
    parse_progress,
)


def test_rename_and_preview_lines_keep_their_existing_counter_category() -> None:
    """Both successful output forms count as renamed work while retaining their labels."""
    renamed, renamed_count = format_run_log_line("Renamed 'old.pdf' to 'new.pdf'\n")
    preview, preview_count = format_run_log_line("Dry-run: would rename 'old.pdf' to 'new.pdf'\n")

    assert renamed_count == preview_count == "renamed"
    assert "Renamed" in renamed
    assert "Dry-run" in preview


def test_preview_record_preserves_preview_and_apply_presentation() -> None:
    """Preview and apply output project the established status and inspection mode."""
    preview = parse_preview_record("Dry-run: would rename 'scan.pdf' to 'invoice.pdf'")
    applied = parse_preview_record("Renamed 'scan.pdf' to 'invoice.pdf'")

    assert preview is not None
    assert (preview.status, preview.source, preview.proposed, preview.mode) == (
        "SUGGESTED",
        "scan.pdf",
        "invoice.pdf",
        "Preview only. No file changed.",
    )
    assert applied is not None
    assert (applied.status, applied.mode) == ("RENAMED", "Rename completed.")


def test_progress_and_shared_summary_keep_empty_and_preview_copy() -> None:
    """Run presentation protects zero totals and calls dry-run results suggestions."""
    counts = {"renamed": 2, "skipped": 1, "failed": 0}

    assert parse_progress("Processing 3/0: sample.pdf") == (3, 1)
    assert "2 suggestions" in format_run_summary(counts, True, separator="  |  ")
    assert "2 suggestions" in completion_summary(counts, True)
    assert completion_summary({"renamed": 0, "skipped": 0, "failed": 0}, False) == "no files processed"
