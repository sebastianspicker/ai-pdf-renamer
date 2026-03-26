"""Round 5 coverage tests: TUI settings persistence and heuristics data loading."""

from __future__ import annotations

import json
import logging
import queue
import re
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ai_pdf_renamer.heuristics import (
    HeuristicRule,
    HeuristicScorer,
    _combine_resolve_conflict,
    _tokenize_for_overlap,
    load_heuristic_rules,
    normalize_llm_category,
)
from ai_pdf_renamer.tui import SETTINGS_PATH, _load_settings, _QueueHandler, _save_settings


class TestQueueHandler:
    """Tests for _QueueHandler logging handler."""

    def test_queue_handler_emit(self) -> None:
        """Emit a LogRecord; verify formatted message appears in the queue."""
        q: queue.Queue[str | None] = queue.Queue()
        handler = _QueueHandler(q)
        handler.setFormatter(logging.Formatter("%(message)s"))

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="hello world",
            args=(),
            exc_info=None,
        )
        handler.emit(record)

        result = q.get_nowait()
        assert result is not None
        assert "hello world" in result
        # Should end with newline
        assert result.endswith("\n")

    def test_queue_handler_emit_error(self) -> None:
        """When formatting raises, handleError is called instead of crashing."""
        q: queue.Queue[str | None] = queue.Queue()
        handler = _QueueHandler(q)

        # Use a formatter that will cause an error during format()
        bad_formatter = logging.Formatter("%(message)s")
        handler.setFormatter(bad_formatter)

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="hello %s %s",
            args=("only_one",),  # Too few args for format string
            exc_info=None,
        )

        # Monkey-patch handleError to track if it gets called
        called = MagicMock()
        handler.handleError = called  # type: ignore[assignment]

        # Force the emit path to raise by making _escape_markup fail
        with patch("ai_pdf_renamer.tui._escape_markup", side_effect=RuntimeError("boom")):
            handler.emit(record)

        called.assert_called_once_with(record)
        # Queue should remain empty since emit failed
        assert q.empty()


class TestSettingsPath:
    """Test that SETTINGS_PATH is a proper Path object."""

    def test_settings_path_exists(self) -> None:
        """SETTINGS_PATH should be a Path pointing to the expected location."""
        assert isinstance(SETTINGS_PATH, Path)
        assert SETTINGS_PATH.name == ".ai_pdf_renamer_gui.json"


class TestSettingsPersistence:
    """Tests for _load_settings / _save_settings standalone functions."""

    def test_load_settings_missing_file(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """When settings file does not exist, _load_settings returns empty dict."""
        monkeypatch.setattr("ai_pdf_renamer.tui.SETTINGS_PATH", tmp_path / "nonexistent.json")
        result = _load_settings()
        assert result == {}

    def test_load_settings_valid(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """When settings file has valid JSON dict, _load_settings returns it."""
        settings_file = tmp_path / "settings.json"
        data = {"language": "en", "dry_run": True}
        settings_file.write_text(json.dumps(data), encoding="utf-8")
        monkeypatch.setattr("ai_pdf_renamer.tui.SETTINGS_PATH", settings_file)
        result = _load_settings()
        assert result == data

    def test_load_settings_invalid_json(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """When settings file has invalid JSON, _load_settings returns empty dict."""
        settings_file = tmp_path / "settings.json"
        settings_file.write_text("not valid json{{{", encoding="utf-8")
        monkeypatch.setattr("ai_pdf_renamer.tui.SETTINGS_PATH", settings_file)
        result = _load_settings()
        assert result == {}

    def test_load_settings_non_dict(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """When settings file contains a JSON list (not dict), return empty dict."""
        settings_file = tmp_path / "settings.json"
        settings_file.write_text("[1, 2, 3]", encoding="utf-8")
        monkeypatch.setattr("ai_pdf_renamer.tui.SETTINGS_PATH", settings_file)
        result = _load_settings()
        assert result == {}

    def test_save_settings_creates_file(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """_save_settings writes JSON to the settings path."""
        settings_file = tmp_path / "subdir" / "settings.json"
        monkeypatch.setattr("ai_pdf_renamer.tui.SETTINGS_PATH", settings_file)
        data = {"directory": "/tmp/pdfs", "use_llm": False}
        _save_settings(data)
        assert settings_file.exists()
        loaded = json.loads(settings_file.read_text(encoding="utf-8"))
        assert loaded == data

    def test_save_settings_roundtrip(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Save then load returns the same data."""
        settings_file = tmp_path / "settings.json"
        monkeypatch.setattr("ai_pdf_renamer.tui.SETTINGS_PATH", settings_file)
        data: dict[str, object] = {"language": "de", "case": "kebabCase", "dry_run": True}
        _save_settings(data)
        result = _load_settings()
        assert result == data


# ---------------------------------------------------------------------------
# Part B: heuristics.py data loading and keyword overlap
# ---------------------------------------------------------------------------


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
