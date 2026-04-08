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

# ---------------------------------------------------------------------------
# _load_config_file
# ---------------------------------------------------------------------------


class TestLoadConfigFile:
    """Tests for _load_config_file: JSON, YAML, error handling."""

    def test_load_json_config(self, tmp_path: Path) -> None:
        """Valid JSON config file returns the parsed dict."""
        cfg = {"language": "en", "desired_case": "snakeCase", "dry_run": True}
        p = tmp_path / "config.json"
        p.write_text(json.dumps(cfg), encoding="utf-8")

        result = _load_config_file(p)

        assert result == cfg

    def test_load_json_config_invalid(self, tmp_path: Path) -> None:
        """Invalid JSON content returns an empty dict."""
        p = tmp_path / "bad.json"
        p.write_text("{ not valid json }", encoding="utf-8")

        result = _load_config_file(p)

        assert result == {}

    def test_load_yaml_config(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Valid YAML config file (with yaml importable) returns the parsed dict."""
        p = tmp_path / "config.yaml"
        p.write_text("language: en\ndesired_case: snakeCase\n", encoding="utf-8")

        # Create a fake yaml module with safe_load
        fake_yaml = types.ModuleType("yaml")
        fake_yaml.safe_load = lambda raw: {"language": "en", "desired_case": "snakeCase"}  # type: ignore[attr-defined]

        # Patch the import so `import yaml` inside _load_config_file resolves to our fake
        import builtins

        real_import = builtins.__import__

        def patched_import(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "yaml":
                return fake_yaml
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", patched_import)

        result = _load_config_file(p)

        assert result == {"language": "en", "desired_case": "snakeCase"}

    def test_load_yaml_config_yml_extension(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """YAML with .yml extension is also handled."""
        p = tmp_path / "config.yml"
        p.write_text("language: de\n", encoding="utf-8")

        fake_yaml = types.ModuleType("yaml")
        fake_yaml.safe_load = lambda raw: {"language": "de"}  # type: ignore[attr-defined]

        import builtins

        real_import = builtins.__import__

        def patched_import(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "yaml":
                return fake_yaml
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", patched_import)

        result = _load_config_file(p)

        assert result == {"language": "de"}

    def test_load_yaml_no_module(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """When yaml is not installed (ImportError), returns empty dict."""
        p = tmp_path / "config.yaml"
        p.write_text("language: en\n", encoding="utf-8")

        import builtins

        real_import = builtins.__import__

        def patched_import(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "yaml":
                raise ImportError("No module named 'yaml'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", patched_import)

        result = _load_config_file(p)

        assert result == {}

    def test_load_config_missing_file(self, tmp_path: Path) -> None:
        """Nonexistent path returns an empty dict."""
        result = _load_config_file(tmp_path / "does_not_exist.json")

        assert result == {}

    def test_load_config_read_error(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """OSError on read_text returns an empty dict."""
        p = tmp_path / "config.json"
        p.write_text("{}", encoding="utf-8")

        # Make read_text raise OSError
        monkeypatch.setattr(Path, "read_text", _raise_os_error)

        result = _load_config_file(p)

        assert result == {}

    def test_load_config_unknown_suffix(self, tmp_path: Path) -> None:
        """Unsupported file extension (e.g. .toml) returns an empty dict."""
        p = tmp_path / "config.toml"
        p.write_text('[section]\nkey = "value"\n', encoding="utf-8")

        result = _load_config_file(p)

        assert result == {}

    def test_load_json_config_non_dict(self, tmp_path: Path) -> None:
        """JSON file with a non-dict top-level (e.g. list) returns empty dict."""
        p = tmp_path / "config.json"
        p.write_text("[1, 2, 3]", encoding="utf-8")

        result = _load_config_file(p)

        assert result == {}


def _raise_os_error(*args: Any, **kwargs: Any) -> Any:
    raise OSError("Simulated read error")


# ---------------------------------------------------------------------------
# _load_override_category_map
# ---------------------------------------------------------------------------


class TestLoadOverrideCategoryMap:
    """Tests for _load_override_category_map: CSV parsing and error handling."""

    def test_override_map_valid_csv(self, tmp_path: Path) -> None:
        """CSV with filename,category columns returns a proper mapping."""
        p = tmp_path / "overrides.csv"
        p.write_text("filename,category\ninvoice.pdf,finance\nreport.pdf,report\n", encoding="utf-8")

        result = _load_override_category_map(p)

        assert result == {"invoice.pdf": "finance", "report.pdf": "report"}

    def test_override_map_missing_file(self, tmp_path: Path) -> None:
        """Nonexistent file returns an empty dict."""
        result = _load_override_category_map(tmp_path / "nonexistent.csv")

        assert result == {}

    def test_override_map_empty_rows(self, tmp_path: Path) -> None:
        """CSV with header but no data rows returns an empty dict."""
        p = tmp_path / "empty.csv"
        p.write_text("filename,category\n", encoding="utf-8")

        result = _load_override_category_map(p)

        assert result == {}

    def test_override_map_path_column(self, tmp_path: Path) -> None:
        """CSV with 'path' column instead of 'filename' is also accepted."""
        p = tmp_path / "overrides.csv"
        p.write_text("path,category\ncontract.pdf,legal\n", encoding="utf-8")

        result = _load_override_category_map(p)

        assert result == {"contract.pdf": "legal"}

    def test_override_map_file_column(self, tmp_path: Path) -> None:
        """CSV with 'file' column is accepted as a fallback."""
        p = tmp_path / "overrides.csv"
        p.write_text("file,category\nphoto.pdf,media\n", encoding="utf-8")

        result = _load_override_category_map(p)

        assert result == {"photo.pdf": "media"}

    def test_override_map_strips_whitespace(self, tmp_path: Path) -> None:
        """Leading and trailing whitespace in values is stripped."""
        p = tmp_path / "overrides.csv"
        p.write_text("filename,category\n  invoice.pdf  ,  finance  \n", encoding="utf-8")

        result = _load_override_category_map(p)

        assert result == {"invoice.pdf": "finance"}

    def test_override_map_skips_empty_name(self, tmp_path: Path) -> None:
        """Rows with empty filename are skipped."""
        p = tmp_path / "overrides.csv"
        p.write_text("filename,category\n,finance\nvalid.pdf,report\n", encoding="utf-8")

        result = _load_override_category_map(p)

        assert result == {"valid.pdf": "report"}

    def test_override_map_skips_empty_category(self, tmp_path: Path) -> None:
        """Rows with empty category are skipped."""
        p = tmp_path / "overrides.csv"
        p.write_text("filename,category\ninvoice.pdf,\nvalid.pdf,report\n", encoding="utf-8")

        result = _load_override_category_map(p)

        assert result == {"valid.pdf": "report"}


# ---------------------------------------------------------------------------
# run_doctor_checks
# ---------------------------------------------------------------------------


class TestRunDoctorChecks:
    """Tests for run_doctor_checks: data files and optional deps."""

    @staticmethod
    def _make_args(**overrides: Any) -> argparse.Namespace:
        """Create a minimal argparse.Namespace for doctor checks."""
        defaults: dict[str, Any] = {
            "use_llm": False,  # Skip LLM probing by default in tests
        }
        defaults.update(overrides)
        return argparse.Namespace(**defaults)

    def test_doctor_data_files_ok(self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
        """When data files exist and are valid JSON, doctor reports OK."""
        import ai_pdf_renamer.cli as cli_module

        valid_json = '{"key": "value"}'

        def fake_data_path(filename: str) -> Path:
            # Return a real temp path would need files; instead return an object
            # whose read_text returns valid JSON.
            p = Path(__file__).parent / "fake_data" / filename
            return p

        # Create a mock Path-like that responds to read_text
        class FakePath:
            def __init__(self, name: str) -> None:
                self._name = name

            def read_text(self, encoding: str = "utf-8") -> str:
                return valid_json

            def __str__(self) -> str:
                return f"/fake/{self._name}"

        def mock_data_path(filename: str) -> Any:
            return FakePath(filename)

        monkeypatch.setattr(cli_module, "data_path", mock_data_path)
        # Stub out all find_spec calls to return None (no optional deps)
        monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)

        args = self._make_args()
        result = run_doctor_checks(args)
        captured = capsys.readouterr()

        assert result == 0
        assert "heuristic_scores.json" in captured.out
        assert "meta_stopwords.json" in captured.out
        assert "passed" in captured.out.lower()

    def test_doctor_data_files_missing(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """When data_path raises FileNotFoundError, doctor reports FAIL and returns 1."""
        import ai_pdf_renamer.cli as cli_module

        def mock_data_path(filename: str) -> Any:
            raise FileNotFoundError(f"Data file {filename!r} not found.")

        monkeypatch.setattr(cli_module, "data_path", mock_data_path)
        monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)

        args = self._make_args()
        result = run_doctor_checks(args)
        captured = capsys.readouterr()

        assert result == 1
        assert "heuristic_scores.json" in captured.out
        assert "meta_stopwords.json" in captured.out
        assert "failed" in captured.out.lower()

    def test_doctor_data_files_invalid_json(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """When data files contain invalid JSON, doctor reports FAIL."""
        import ai_pdf_renamer.cli as cli_module

        class FakePath:
            def __init__(self, name: str) -> None:
                self._name = name

            def read_text(self, encoding: str = "utf-8") -> str:
                return "{ broken json"

            def __str__(self) -> str:
                return f"/fake/{self._name}"

        def mock_data_path(filename: str) -> Any:
            return FakePath(filename)

        monkeypatch.setattr(cli_module, "data_path", mock_data_path)
        monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)

        args = self._make_args()
        result = run_doctor_checks(args)
        captured = capsys.readouterr()

        assert result == 1
        assert "FAIL" in captured.out

    def test_doctor_optional_deps_all_present(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """When all optional deps are found, doctor prints OK for each."""
        import ai_pdf_renamer.cli as cli_module

        # Stub data_path to succeed
        class FakePath:
            def __init__(self, name: str) -> None:
                self._name = name

            def read_text(self, encoding: str = "utf-8") -> str:
                return "{}"

            def __str__(self) -> str:
                return f"/fake/{self._name}"

        monkeypatch.setattr(cli_module, "data_path", lambda filename: FakePath(filename))

        # Make find_spec return a truthy ModuleSpec-like for known optional deps
        sentinel = types.ModuleType("sentinel")

        def fake_find_spec(name: str) -> Any:
            if name in ("fitz", "ocrmypdf", "tiktoken", "llama_cpp"):
                return sentinel  # truthy
            return None

        monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)

        args = self._make_args()
        result = run_doctor_checks(args)
        captured = capsys.readouterr()

        assert result == 0
        assert "PyMuPDF" in captured.out
        assert "ocrmypdf" in captured.out
        assert "tiktoken" in captured.out
        assert "llama-cpp-python" in captured.out

    def test_doctor_optional_deps_all_missing(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """When all optional deps are missing, doctor prints WARN/INFO (but still passes)."""
        import ai_pdf_renamer.cli as cli_module

        class FakePath:
            def __init__(self, name: str) -> None:
                self._name = name

            def read_text(self, encoding: str = "utf-8") -> str:
                return "{}"

            def __str__(self) -> str:
                return f"/fake/{self._name}"

        monkeypatch.setattr(cli_module, "data_path", lambda filename: FakePath(filename))
        monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)

        args = self._make_args()
        result = run_doctor_checks(args)
        captured = capsys.readouterr()

        # Missing optional deps are WARN/INFO, not FAIL -- so doctor still passes
        assert result == 0
        assert "PyMuPDF" in captured.out
        assert "ocrmypdf" in captured.out
        assert "tiktoken" in captured.out
        assert "llama-cpp-python" in captured.out

    def test_doctor_llm_skipped_message(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """When use_llm is False, doctor prints skip message for LLM checks."""
        import ai_pdf_renamer.cli as cli_module

        class FakePath:
            def __init__(self, name: str) -> None:
                self._name = name

            def read_text(self, encoding: str = "utf-8") -> str:
                return "{}"

            def __str__(self) -> str:
                return f"/fake/{self._name}"

        monkeypatch.setattr(cli_module, "data_path", lambda filename: FakePath(filename))
        monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)

        args = self._make_args(use_llm=False)
        result = run_doctor_checks(args)
        captured = capsys.readouterr()

        assert "skipped" in captured.out.lower()
        assert result == 0


# --- Merged from test_round5_tui_heuristics.py ---


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


# --- Merged from test_round7_tui.py ---

# Textual 8.x dropped the ``flex-wrap`` CSS property.  Remove it from the
# app's stylesheet so that ``run_test()`` can compose widgets successfully.
_PATCHED_CSS = _CSS.replace("flex-wrap: wrap;", "")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# 1. Basic app lifecycle
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_app_starts_and_has_title() -> None:
    """App starts headlessly and exposes the expected title."""
    app = _make_app()
    async with app.run_test(size=(120, 40)) as _pilot:
        assert app.title == "AI-PDF-Renamer"


@pytest.mark.asyncio
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


@pytest.mark.asyncio
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


@pytest.mark.asyncio
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


# ---------------------------------------------------------------------------
# 2. Widget getters
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_str_returns_value() -> None:
    """_get_str reads the current value of an Input widget."""
    from textual.widgets import Input

    app = _make_app()
    async with app.run_test(size=(120, 40)) as _pilot:
        app.query_one("#directory", Input).value = "/tmp/pdfs"
        assert app._get_str("directory") == "/tmp/pdfs"


@pytest.mark.asyncio
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


@pytest.mark.asyncio
async def test_get_select_returns_value() -> None:
    """_get_select reads the current value of a Select widget."""
    app = _make_app()
    async with app.run_test(size=(120, 40)) as _pilot:
        val = app._get_select("language")
        assert val == "de"

        val = app._get_select("case")
        assert val == "kebabCase"


@pytest.mark.asyncio
async def test_get_str_missing_widget() -> None:
    """_get_str returns the default when the widget ID doesn't exist."""
    app = _make_app()
    async with app.run_test(size=(120, 40)) as _pilot:
        assert app._get_str("nonexistent_widget", "fallback") == "fallback"
        assert app._get_str("nonexistent_widget") == ""


# ---------------------------------------------------------------------------
# 3. Snapshot and config
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
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


@pytest.mark.asyncio
async def test_snapshot_has_language_field() -> None:
    """_snapshot includes 'language' with the widget's current value."""
    app = _make_app()
    async with app.run_test(size=(120, 40)) as _pilot:
        snap = app._snapshot()
        assert "language" in snap
        assert snap["language"] == "de"


# ---------------------------------------------------------------------------
# 4. Settings persistence
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
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


# ---------------------------------------------------------------------------
# 5. Cancel action
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
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


@pytest.mark.asyncio
async def test_cancel_noop_when_not_running() -> None:
    """Cancel does nothing when no run is in progress."""
    app = _make_app()
    async with app.run_test(size=(120, 40)) as _pilot:
        app._running = False
        app._stop_event.clear()

        app._cancel()

        assert not app._stop_event.is_set()
