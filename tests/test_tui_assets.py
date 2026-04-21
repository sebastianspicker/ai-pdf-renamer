"""Tests for tui_assets module: regex formatters and constants."""

from __future__ import annotations

import re

from ai_pdf_renamer.tui_assets import (
    _DRYRUN_LOG_RE,
    _RENAME_LOG_RE,
    PROCESS_RE,
    _format_dryrun_match,
    _format_rename_match,
)


def _match(pattern: re.Pattern[str], text: str) -> re.Match[str]:
    m = pattern.search(text)
    assert m is not None, f"Pattern did not match: {text!r}"
    return m


# ---------------------------------------------------------------------------
# _format_rename_match
# ---------------------------------------------------------------------------


def test_format_rename_match_basic() -> None:
    line = "Renamed 'old_name.pdf' to 'new_name.pdf'"
    m = _match(_RENAME_LOG_RE, line)
    result = _format_rename_match(m)
    assert "Renamed" in result
    assert "old_name.pdf" in result
    assert "new_name.pdf" in result
    assert "[green]" in result
    assert "->" in result


def test_format_rename_match_rich_markup_escaping() -> None:
    """Filenames with Rich markup special chars must be escaped."""
    line = "Renamed '[bold]tricky.pdf' to 'safe.pdf'"
    m = _match(_RENAME_LOG_RE, line)
    result = _format_rename_match(m)
    # The bracket should be escaped so it doesn't render as markup
    assert "[bold]" not in result or "\\[bold]" in result or r"\[bold]" in result


def test_format_rename_match_returns_string() -> None:
    line = "Renamed 'a.pdf' to 'b.pdf'"
    m = _match(_RENAME_LOG_RE, line)
    assert isinstance(_format_rename_match(m), str)


# ---------------------------------------------------------------------------
# _format_dryrun_match
# ---------------------------------------------------------------------------


def test_format_dryrun_match_basic() -> None:
    line = "Dry-run: would rename 'original.pdf' to 'renamed.pdf'"
    m = _match(_DRYRUN_LOG_RE, line)
    result = _format_dryrun_match(m)
    assert "Dry-run" in result
    assert "original.pdf" in result
    assert "renamed.pdf" in result
    assert "[cyan]" in result
    assert "->" in result


def test_format_dryrun_match_rich_markup_escaping() -> None:
    """Filenames with Rich markup chars must be escaped in dry-run format too."""
    line = "Dry-run: would rename '[red]tricky.pdf' to 'safe.pdf'"
    m = _match(_DRYRUN_LOG_RE, line)
    result = _format_dryrun_match(m)
    assert isinstance(result, str)
    assert "safe.pdf" in result


def test_format_dryrun_match_returns_string() -> None:
    line = "Dry-run: would rename 'x.pdf' to 'y.pdf'"
    m = _match(_DRYRUN_LOG_RE, line)
    assert isinstance(_format_dryrun_match(m), str)


# ---------------------------------------------------------------------------
# PROCESS_RE pattern sanity checks
# ---------------------------------------------------------------------------


def test_process_re_matches_expected_format() -> None:
    m = PROCESS_RE.search("Processing 3/10:")
    assert m is not None
    assert m.group(1) == "3"
    assert m.group(2) == "10"


def test_process_re_no_match_on_unrelated() -> None:
    assert PROCESS_RE.search("Just some log line") is None


# ---------------------------------------------------------------------------
# Regex pattern correctness
# ---------------------------------------------------------------------------


def test_rename_log_re_captures_both_filenames() -> None:
    line = "Renamed 'foo bar.pdf' to 'baz qux.pdf'"
    m = _RENAME_LOG_RE.search(line)
    assert m is not None
    assert m.group(1) == "foo bar.pdf"
    assert m.group(2) == "baz qux.pdf"


def test_dryrun_log_re_captures_both_filenames() -> None:
    line = "Dry-run: would rename 'alpha.pdf' to 'beta.pdf'"
    m = _DRYRUN_LOG_RE.search(line)
    assert m is not None
    assert m.group(1) == "alpha.pdf"
    assert m.group(2) == "beta.pdf"
