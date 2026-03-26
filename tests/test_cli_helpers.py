"""Tests for cli.py helper functions: config loading, override maps, and doctor checks."""

from __future__ import annotations

import argparse
import importlib.util
import json
import types
from pathlib import Path
from typing import Any

import pytest

from ai_pdf_renamer.cli import (
    _load_config_file,
    _load_override_category_map,
    run_doctor_checks,
)

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
