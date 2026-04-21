# ruff: noqa: F401

"""Tests for cli.py helper functions: config loading, override maps, and doctor checks."""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import queue
import re
import tempfile
import types
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from ai_pdf_renamer.cli import (
    ConfigLoadError,
    _load_config_file,
    _load_override_category_map,
    run_doctor_checks,
)
from ai_pdf_renamer.heuristics import (
    HeuristicRule,
    HeuristicScorer,
    _combine_resolve_conflict,
    _tokenize_for_overlap,
    load_heuristic_rules,
    normalize_llm_category,
)
from ai_pdf_renamer.tui import _CSS, SETTINGS_PATH, AIRenamerTUI, _load_settings, _QueueHandler, _save_settings


def _raise_os_error(*args: Any, **kwargs: Any) -> Any:
    raise OSError("Simulated read error")


_PATCHED_CSS = _CSS.replace("flex-wrap: wrap;", "")


def _make_app(settings: dict[str, object] | None = None) -> AIRenamerTUI:
    """Create an AIRenamerTUI with patched CSS and optional pre-loaded settings."""
    if settings is not None:
        with patch("ai_pdf_renamer.tui._load_settings", return_value=settings):
            app = AIRenamerTUI()
    else:
        with patch("ai_pdf_renamer.tui._load_settings", return_value={}):
            app = AIRenamerTUI()
    app.CSS = _PATCHED_CSS  # type: ignore[assignment]
    return app


class TestLoadHeuristicRules:
    """Tests for load_heuristic_rules data file loading."""

    def test_load_heuristic_rules_missing_file(self, tmp_path: Path) -> None:
        """Non-existent file raises ValueError."""
        missing = tmp_path / "nonexistent.json"
        with pytest.raises(ValueError, match="Could not read data file"):
            load_heuristic_rules(missing)

    def test_load_heuristic_rules_invalid_json(self, tmp_path: Path) -> None:
        """Malformed JSON raises ValueError."""
        bad_file = tmp_path / "bad.json"
        bad_file.write_text("{not valid json", encoding="utf-8")
        with pytest.raises(ValueError, match="Invalid JSON"):
            load_heuristic_rules(bad_file)

    def test_load_heuristic_rules_skips_bad_entries(self, tmp_path: Path) -> None:
        """Entries missing 'regex' key are silently skipped."""
        rules_file = tmp_path / "rules.json"
        data = {
            "patterns": [
                {"category": "invoice", "score": 2.0},  # missing regex
                {"regex": "test", "category": "receipt", "score": 1.0},  # valid
            ]
        }
        rules_file.write_text(json.dumps(data), encoding="utf-8")
        rules = load_heuristic_rules(rules_file)
        assert len(rules) == 1
        assert rules[0].category == "receipt"

    def test_load_heuristic_rules_non_numeric_score(self, tmp_path: Path) -> None:
        """When score is a string, it defaults to 0.0."""
        rules_file = tmp_path / "rules.json"
        data = {
            "patterns": [
                {"regex": "invoice", "category": "invoice", "score": "not_a_number"},
            ]
        }
        rules_file.write_text(json.dumps(data), encoding="utf-8")
        rules = load_heuristic_rules(rules_file)
        assert len(rules) == 1
        assert rules[0].score == 0.0

    def test_load_heuristic_rules_valid(self, tmp_path: Path) -> None:
        """Valid JSON with rules returns a list of HeuristicRule objects."""
        rules_file = tmp_path / "rules.json"
        data = {
            "patterns": [
                {"regex": "invoice", "category": "invoice", "score": 3.0, "language": "en"},
                {"regex": "rechnung", "category": "invoice", "score": 4.0, "language": "de", "parent": "finance"},
            ]
        }
        rules_file.write_text(json.dumps(data), encoding="utf-8")
        rules = load_heuristic_rules(rules_file)
        assert len(rules) == 2
        assert rules[0].category == "invoice"
        assert rules[0].score == 3.0
        assert rules[0].language == "en"
        assert rules[1].language == "de"
        assert rules[1].parent == "finance"
        # Verify it can be used with HeuristicScorer
        scorer = HeuristicScorer(rules=rules)
        assert scorer.best_category("This is an invoice") == "invoice"

    def test_load_heuristic_rules_invalid_regex_fails_fast(self, tmp_path: Path) -> None:
        """Invalid regex patterns raise ValueError instead of being skipped silently."""
        rules_file = tmp_path / "rules.json"
        rules_file.write_text(
            json.dumps({"patterns": [{"regex": "(", "category": "invoice", "score": 1.0}]}),
            encoding="utf-8",
        )

        with pytest.raises(ValueError, match="Invalid regex"):
            load_heuristic_rules(rules_file)


class TestCategoryAliases:
    """Tests for normalize_llm_category alias resolution."""

    def test_category_aliases_loaded(self) -> None:
        """normalize_llm_category resolves known aliases from the data file."""
        # "rechnung" should map to "invoice" via category_aliases.json
        assert normalize_llm_category("rechnung") == "invoice"
        assert normalize_llm_category("Rechnung") == "invoice"
        # "quittung" should map to "receipt"
        assert normalize_llm_category("quittung") == "receipt"
        # "lohnabrechnung" should map to "payslip"
        assert normalize_llm_category("lohnabrechnung") == "payslip"
        # Identity mappings
        assert normalize_llm_category("invoice") == "invoice"

    def test_normalize_llm_category_custom_aliases(self) -> None:
        """normalize_llm_category with explicit _aliases parameter."""
        aliases = {"custom_cat": "mapped_cat"}
        assert normalize_llm_category("custom_cat", _aliases=aliases) == "mapped_cat"
        # Unknown key returns as-is
        assert normalize_llm_category("other_cat", _aliases=aliases) == "other_cat"

    def test_normalize_llm_category_empty(self) -> None:
        """Empty or None input returns empty string."""
        assert normalize_llm_category("") == ""
        assert normalize_llm_category(None) == ""  # type: ignore[arg-type]


class TestKeywordOverlap:
    """Tests for keyword overlap conflict resolution in _combine_resolve_conflict."""

    def test_keyword_overlap_llm_wins(self) -> None:
        """Context with more LLM category words returns LLM category."""
        # Context contains "bank" and "statement", matching "bank_statement" (LLM)
        # but not "invoice" (heuristic)
        result = _combine_resolve_conflict(
            "bank_statement",
            "invoice",
            prefer_llm=False,
            context_for_overlap="bank statement account balance transaction",
            use_embeddings_for_conflict=False,
            use_keyword_overlap=True,
            heuristic_score=None,
            heuristic_score_weight=0.0,
        )
        assert result == "bank_statement"

    def test_keyword_overlap_heuristic_wins(self) -> None:
        """Context with more heuristic category words returns heuristic category."""
        # Context contains "invoice", matching heuristic "invoice"
        # but not "bank_statement" (LLM)
        result = _combine_resolve_conflict(
            "bank_statement",
            "invoice",
            prefer_llm=False,
            context_for_overlap="invoice total amount due payment rechnung",
            use_embeddings_for_conflict=False,
            use_keyword_overlap=True,
            heuristic_score=None,
            heuristic_score_weight=0.0,
        )
        assert result == "invoice"

    def test_keyword_overlap_tie_returns_heuristic(self) -> None:
        """When overlap is a tie, heuristic wins."""
        result = _combine_resolve_conflict(
            "receipt",
            "invoice",
            prefer_llm=False,
            context_for_overlap="unrelated content with no category words",
            use_embeddings_for_conflict=False,
            use_keyword_overlap=True,
            heuristic_score=None,
            heuristic_score_weight=0.0,
        )
        assert result == "invoice"

    def test_keyword_overlap_with_score_weight(self) -> None:
        """Heuristic score weight adds bonus to heuristic overlap count."""
        # Both have 1 token overlap, but heuristic_score_weight tips it
        result = _combine_resolve_conflict(
            "bank_statement",
            "invoice",
            prefer_llm=False,
            context_for_overlap="bank invoice",
            use_embeddings_for_conflict=False,
            use_keyword_overlap=True,
            heuristic_score=5.0,
            heuristic_score_weight=1.0,
        )
        # heuristic overlap = 1 + (1.0 * 5.0) = 6.0; llm overlap = 1
        assert result == "invoice"


class TestBestCategoryEmptyText:
    """Test HeuristicScorer with empty/degenerate text input."""

    def test_best_category_with_confidence_empty_text(self) -> None:
        """Empty text returns unknown with zero scores."""
        rules = [
            HeuristicRule(pattern=re.compile("invoice", re.IGNORECASE), category="invoice", score=2.0),
        ]
        scorer = HeuristicScorer(rules=rules)
        cat, score, _runner, _runner_score = scorer.best_category_with_confidence("")
        assert cat == "unknown"
        assert score == 0.0
        assert _runner == "unknown"
        assert _runner_score == 0.0

    def test_best_category_with_confidence_none_text(self) -> None:
        """None text returns unknown with zero scores."""
        rules = [
            HeuristicRule(pattern=re.compile("test"), category="test", score=1.0),
        ]
        scorer = HeuristicScorer(rules=rules)
        cat, score, _runner, _runner_score = scorer.best_category_with_confidence(None)  # type: ignore[arg-type]
        assert cat == "unknown"
        assert score == 0.0


class TestTokenizeForOverlap:
    """Tests for _tokenize_for_overlap helper."""

    def test_tokenize_basic(self) -> None:
        """Splits on whitespace and underscore, lowercases."""
        tokens = _tokenize_for_overlap("Bank_Statement PDF")
        assert tokens == {"bank", "statement", "pdf"}

    def test_tokenize_empty(self) -> None:
        """Empty or None input returns empty set."""
        assert _tokenize_for_overlap("") == set()
        assert _tokenize_for_overlap(None) == set()  # type: ignore[arg-type]


async def test_app_starts_and_has_title() -> None:
    """App starts headlessly and exposes the expected title."""
    app = _make_app()
    async with app.run_test(size=(120, 40)) as _pilot:
        assert app.title == "AI-PDF-Renamer"


async def test_app_has_tabs() -> None:
    """The three tab panes (Basic, Advanced, Run) are composed."""
    from textual.widgets import TabPane

    app = _make_app()
    async with app.run_test(size=(120, 40)) as _pilot:
        panes = app.query(TabPane)
        ids = {p.id for p in panes}
        assert "basic" in ids
        assert "advanced" in ids
        assert "run" in ids
        assert len(panes) == 3


async def test_app_has_buttons() -> None:
    """Preview, Apply, One, and Cancel buttons exist with their IDs."""
    from textual.widgets import Button

    app = _make_app()
    async with app.run_test(size=(120, 40)) as _pilot:
        buttons = app.query(Button)
        ids = {b.id for b in buttons}
        assert "btn-preview" in ids
        assert "btn-apply" in ids
        assert "btn-one" in ids
        assert "btn-cancel" in ids


async def test_app_default_widget_values() -> None:
    """Default widget values match expected defaults when no settings are loaded."""
    from textual.widgets import Checkbox, Input, Select

    app = _make_app()
    async with app.run_test(size=(120, 40)) as _pilot:
        # Inputs default to empty
        assert app.query_one("#directory", Input).value == ""
        assert app.query_one("#single_file", Input).value == ""
        assert app.query_one("#project", Input).value == ""

        # Selects default to their first-option values
        assert app.query_one("#language", Select).value == "de"
        assert app.query_one("#case", Select).value == "kebabCase"
        assert app.query_one("#date_format", Select).value == "dmy"

        # Checkboxes
        assert app.query_one("#dry_run", Checkbox).value is True
        assert app.query_one("#use_llm", Checkbox).value is True
        assert app.query_one("#use_ocr", Checkbox).value is False
        assert app.query_one("#recursive", Checkbox).value is False


async def test_get_str_returns_value() -> None:
    """_get_str reads the current value of an Input widget."""
    from textual.widgets import Input

    app = _make_app()
    async with app.run_test(size=(120, 40)) as _pilot:
        app.query_one("#directory", Input).value = "/tmp/pdfs"
        assert app._get_str("directory") == "/tmp/pdfs"


async def test_get_bool_returns_value() -> None:
    """_get_bool reads the current value of a Checkbox widget."""
    from textual.widgets import Checkbox

    app = _make_app()
    async with app.run_test(size=(120, 40)) as _pilot:
        cb = app.query_one("#dry_run", Checkbox)
        cb.value = False
        assert app._get_bool("dry_run") is False
        cb.value = True
        assert app._get_bool("dry_run") is True


async def test_get_select_returns_value() -> None:
    """_get_select reads the current value of a Select widget."""
    app = _make_app()
    async with app.run_test(size=(120, 40)) as _pilot:
        val = app._get_select("language")
        assert val == "de"

        val = app._get_select("case")
        assert val == "kebabCase"


async def test_get_str_missing_widget() -> None:
    """_get_str returns the default when the widget ID doesn't exist."""
    app = _make_app()
    async with app.run_test(size=(120, 40)) as _pilot:
        assert app._get_str("nonexistent_widget", "fallback") == "fallback"
        assert app._get_str("nonexistent_widget") == ""


async def test_snapshot_returns_dict() -> None:
    """_snapshot returns a dict containing all expected top-level keys."""
    app = _make_app()
    async with app.run_test(size=(120, 40)) as _pilot:
        snap = app._snapshot()
        assert isinstance(snap, dict)
        expected_keys = {
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
            "llm_backend",
            "llm_url",
            "llm_model",
            "llm_model_path",
            "llm_timeout",
            "max_tokens",
            "max_content_chars",
            "max_content_tokens",
            "workers",
            "max_filename_chars",
            "dry_run",
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
        assert expected_keys.issubset(snap.keys())


async def test_snapshot_has_language_field() -> None:
    """_snapshot includes 'language' with the widget's current value."""
    app = _make_app()
    async with app.run_test(size=(120, 40)) as _pilot:
        snap = app._snapshot()
        assert "language" in snap
        assert snap["language"] == "de"


async def test_load_settings_populates_widgets() -> None:
    """Pre-existing settings JSON populates widget values on startup."""
    from textual.widgets import Checkbox, Input, Select

    settings: dict[str, object] = {
        "directory": "/home/user/docs",
        "language": "en",
        "case": "snakeCase",
        "dry_run": False,
        "use_ocr": True,
        "project": "my-project",
    }
    # Write settings to a temp file and patch the path
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tf:
        json.dump(settings, tf)
        tf_path = Path(tf.name)

    try:
        with patch("ai_pdf_renamer.tui.SETTINGS_PATH", tf_path):
            app = AIRenamerTUI()
            app.CSS = _PATCHED_CSS  # type: ignore[assignment]
            async with app.run_test(size=(120, 40)) as _pilot:
                assert app.query_one("#directory", Input).value == "/home/user/docs"
                assert app.query_one("#language", Select).value == "en"
                assert app.query_one("#case", Select).value == "snakeCase"
                assert app.query_one("#dry_run", Checkbox).value is False
                assert app.query_one("#use_ocr", Checkbox).value is True
                assert app.query_one("#project", Input).value == "my-project"
    finally:
        tf_path.unlink(missing_ok=True)


async def test_cancel_sets_stop_event() -> None:
    """Pressing Cancel while a run is in progress sets _stop_event."""
    app = _make_app()
    async with app.run_test(size=(120, 40)) as _pilot:
        # Simulate an in-progress run
        app._running = True
        app._stop_event.clear()

        # Invoke the cancel handler directly (button may not be visible on active tab)
        app.on_cancel()

        assert app._stop_event.is_set()


async def test_cancel_noop_when_not_running() -> None:
    """Cancel does nothing when no run is in progress."""
    app = _make_app()
    async with app.run_test(size=(120, 40)) as _pilot:
        app._running = False
        app._stop_event.clear()

        app._cancel()

        assert not app._stop_event.is_set()
