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

    def test_load_json_config_invalid_raises_when_requested(self, tmp_path: Path) -> None:
        """Invalid JSON is fatal for explicit CLI config loading."""
        p = tmp_path / "bad.json"
        p.write_text("{ not valid json }", encoding="utf-8")

        with pytest.raises(ConfigLoadError):
            _load_config_file(p, raise_on_error=True)

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

    def test_load_yaml_invalid_raises_when_requested(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Invalid YAML is fatal for explicit CLI config loading."""
        p = tmp_path / "config.yaml"
        p.write_text("language: [oops", encoding="utf-8")

        fake_yaml = types.ModuleType("yaml")

        def _raise_yaml_error(raw: str) -> dict[str, object]:
            raise ValueError("bad yaml")

        fake_yaml.safe_load = _raise_yaml_error  # type: ignore[attr-defined]

        import builtins

        real_import = builtins.__import__

        def patched_import(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "yaml":
                return fake_yaml
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", patched_import)

        with pytest.raises(ConfigLoadError):
            _load_config_file(p, raise_on_error=True)

    def test_load_config_missing_file(self, tmp_path: Path) -> None:
        """Nonexistent path returns an empty dict."""
        result = _load_config_file(tmp_path / "does_not_exist.json")

        assert result == {}

    def test_load_config_missing_file_raises_when_requested(self, tmp_path: Path) -> None:
        """Missing config files are fatal for explicit CLI config loading."""
        with pytest.raises(ConfigLoadError):
            _load_config_file(tmp_path / "does_not_exist.json", raise_on_error=True)

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

    def test_load_config_unknown_suffix_raises_when_requested(self, tmp_path: Path) -> None:
        """Unsupported config formats are fatal for explicit CLI config loading."""
        p = tmp_path / "config.toml"
        p.write_text('[section]\nkey = "value"\n', encoding="utf-8")

        with pytest.raises(ConfigLoadError):
            _load_config_file(p, raise_on_error=True)

    def test_load_json_config_non_dict(self, tmp_path: Path) -> None:
        """JSON file with a non-dict top-level (e.g. list) returns empty dict."""
        p = tmp_path / "config.json"
        p.write_text("[1, 2, 3]", encoding="utf-8")

        result = _load_config_file(p)

        assert result == {}

    def test_load_json_config_non_dict_raises_when_requested(self, tmp_path: Path) -> None:
        """Non-object JSON configs are fatal for explicit CLI config loading."""
        p = tmp_path / "config.json"
        p.write_text("[1, 2, 3]", encoding="utf-8")

        with pytest.raises(ConfigLoadError):
            _load_config_file(p, raise_on_error=True)


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


class TestRunDoctorChecks:
    """Tests for run_doctor_checks: data files and optional deps."""

    class FakePath:
        def __init__(self, name: str, payload: str = "{}") -> None:
            self._name = name
            self._payload = payload

        def read_text(self, encoding: str = "utf-8") -> str:
            return self._payload

        def __str__(self) -> str:
            return f"/fake/{self._name}"

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

        def mock_data_path(filename: str) -> Any:
            return self.FakePath(filename, valid_json)

        monkeypatch.setattr(cli_module, "data_path", mock_data_path)
        monkeypatch.setattr(cli_module, "load_heuristic_rules", lambda _path: [])
        # Stub out all find_spec calls to return None (no optional deps)
        monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)

        args = self._make_args()
        result = run_doctor_checks(args)
        captured = capsys.readouterr()

        assert result == 0
        assert "heuristic_scores.json" in captured.out
        assert "meta_stopwords.json" in captured.out
        assert "passed" in captured.out.lower()

    def test_doctor_reports_pattern_statistics(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Doctor prints loaded pattern and category counts for heuristic scores."""
        import ai_pdf_renamer.cli as cli_module

        heuristic_payload = json.dumps(
            {
                "patterns": [
                    {"regex": "invoice", "category": "invoice", "score": 2},
                    {"regex": "contract", "category": "contract", "score": 2},
                ]
            }
        )
        stopwords_payload = json.dumps({"stopwords": []})

        def fake_data_path(filename: str) -> Any:
            if filename == "heuristic_scores.json":
                return self.FakePath(filename, heuristic_payload)
            return self.FakePath(filename, stopwords_payload)

        monkeypatch.setattr(cli_module, "data_path", fake_data_path)
        monkeypatch.setattr(
            cli_module,
            "load_heuristic_rules",
            lambda _path: [
                HeuristicRule(pattern=re.compile("invoice"), category="invoice", score=2.0),
                HeuristicRule(pattern=re.compile("contract"), category="contract", score=2.0),
            ],
        )
        monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)

        result = run_doctor_checks(self._make_args())
        captured = capsys.readouterr()

        assert result == 0
        assert "patterns=2" in captured.out
        assert "categories=2" in captured.out

    def test_doctor_fails_on_invalid_heuristic_rules(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Malformed heuristic patterns must report FAIL instead of a green OK."""
        import ai_pdf_renamer.cli as cli_module

        heuristic_payload = json.dumps({"patterns": [{"regex": "(", "category": "broken", "score": 1}]})
        stopwords_payload = json.dumps({"stopwords": []})

        def fake_data_path(filename: str) -> Any:
            if filename == "heuristic_scores.json":
                return self.FakePath(filename, heuristic_payload)
            return self.FakePath(filename, stopwords_payload)

        def raise_invalid_regex(_path: Any) -> Any:
            raise ValueError("Invalid regex")

        monkeypatch.setattr(cli_module, "data_path", fake_data_path)
        monkeypatch.setattr(cli_module, "load_heuristic_rules", raise_invalid_regex)
        monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)

        result = run_doctor_checks(self._make_args())
        captured = capsys.readouterr()

        assert result == 1
        assert "FAIL" in captured.out
        assert "heuristic_scores.json" in captured.out
        assert "Invalid regex" in captured.out

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

        def mock_data_path(filename: str) -> Any:
            return self.FakePath(filename, "{ broken json")

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
        monkeypatch.setattr(cli_module, "data_path", lambda filename: self.FakePath(filename))
        monkeypatch.setattr(cli_module, "load_heuristic_rules", lambda _path: [])

        # Make find_spec return a truthy ModuleSpec-like for known optional deps
        sentinel = types.ModuleType("sentinel")

        def fake_find_spec(name: str) -> Any:
            if name in ("fitz", "ocrmypdf", "tiktoken", "llama_cpp", "textual"):
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
        assert "textual" in captured.out

    def test_doctor_optional_deps_all_missing(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """When all optional deps are missing, doctor prints WARN/INFO (but still passes)."""
        import ai_pdf_renamer.cli as cli_module

        monkeypatch.setattr(cli_module, "data_path", lambda filename: self.FakePath(filename))
        monkeypatch.setattr(cli_module, "load_heuristic_rules", lambda _path: [])
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
        assert "textual" in captured.out
        assert ".[pdf]" in captured.out
        assert ".[tui]" in captured.out

    def test_doctor_llm_skipped_message(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """When use_llm is False, doctor prints skip message for LLM checks."""
        import ai_pdf_renamer.cli as cli_module

        monkeypatch.setattr(cli_module, "data_path", lambda filename: self.FakePath(filename))
        monkeypatch.setattr(cli_module, "load_heuristic_rules", lambda _path: [])
        monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)

        args = self._make_args(use_llm=False)
        result = run_doctor_checks(args)
        captured = capsys.readouterr()

        assert "skipped" in captured.out.lower()
        assert result == 0

    def test_doctor_probes_configured_default_endpoint(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import ai_pdf_renamer.cli as cli_module

        monkeypatch.setattr(cli_module, "data_path", lambda filename: self.FakePath(filename))
        monkeypatch.setattr(cli_module, "load_heuristic_rules", lambda _path: [])
        monkeypatch.setattr(importlib.util, "find_spec", lambda name: object())

        captured_calls: list[tuple[str, str, bool]] = []

        class FakeClient:
            base_url = "http://127.0.0.1:11434/v1/completions"
            model = "qwen2.5:3b"

        class FakeConfig:
            llm_backend = "http"
            llm_use_chat_api = True
            llm_base_url = "http://127.0.0.1:11434/v1/completions"

        monkeypatch.setattr(cli_module, "build_config", lambda _raw: FakeConfig())
        monkeypatch.setattr(
            "ai_pdf_renamer.llm_backend.create_llm_client_from_config",
            lambda _cfg: FakeClient(),
        )
        monkeypatch.setattr(
            cli_module,
            "_probe_llm_endpoint",
            lambda url, model, *, label, fail_is_warn=False, use_chat_api=True: captured_calls.append(
                (url, label, fail_is_warn)
            )
            or True,
        )

        result = run_doctor_checks(self._make_args(use_llm=True))

        assert result == 0
        assert captured_calls == [("http://127.0.0.1:11434/v1/completions", "Configured/default endpoint", False)]


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
            msg="hello %s",
            args=("world",),
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
