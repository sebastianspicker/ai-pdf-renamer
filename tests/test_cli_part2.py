# ruff: noqa: F401

from __future__ import annotations

import argparse
import json
import runpy
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

import ai_pdf_renamer.cli as cli_mod
from ai_pdf_renamer.cli import _resolve_dirs, _resolve_option
from ai_pdf_renamer.config import RenamerConfig
from ai_pdf_renamer.tui import _CSS, AIRenamerTUI

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


def _ns(**kwargs: Any) -> argparse.Namespace:
    """Create an argparse.Namespace with sensible defaults for tests."""
    defaults: dict[str, Any] = {
        "dirs": None,
        "dirs_from_file": None,
        "single_file": None,
        "manual_file": None,
        "doctor": False,
        "watch": False,
        "watch_interval": 60,
        "language": "de",
        "desired_case": "kebabCase",
        "project": "",
        "version": "",
        "config": None,
    }
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


class TestResolveDirs:
    """Tests for _resolve_dirs: --dirs-from-file, --single-file, and error paths."""

    def test_resolve_dirs_from_file(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Directories listed in --dirs-from-file are returned."""
        monkeypatch.setattr(cli_mod, "_is_interactive", lambda: False)

        dir_a = tmp_path / "dir_a"
        dir_b = tmp_path / "dir_b"
        dir_a.mkdir()
        dir_b.mkdir()

        dirs_file = tmp_path / "dirs.txt"
        dirs_file.write_text(f"{dir_a}\n{dir_b}\n", encoding="utf-8")

        args = _ns(dirs_from_file=str(dirs_file))
        dirs, single = _resolve_dirs(args)

        assert str(dir_a.resolve()) in dirs
        assert str(dir_b.resolve()) in dirs
        assert single is None

    def test_resolve_dirs_from_file_too_many(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """When --dirs-from-file has >10000 lines, only the first 10000 are used and a warning is logged."""
        monkeypatch.setattr(cli_mod, "_is_interactive", lambda: False)

        dirs_file = tmp_path / "many_dirs.txt"
        # Write 15000 lines; each is a unique directory path (they don't need to exist for this test)
        lines = [f"/fake/dir_{i}" for i in range(15_000)]
        dirs_file.write_text("\n".join(lines), encoding="utf-8")

        args = _ns(dirs_from_file=str(dirs_file))

        import logging

        with caplog.at_level(logging.WARNING, logger="ai_pdf_renamer.cli"):
            dirs, _ = _resolve_dirs(args)

        # Should be capped at 10000
        assert len(dirs) == 10_000
        # Warning should mention the cap
        assert any("10000" in rec.message or "10,000" in rec.message for rec in caplog.records) or any(
            "10000" in msg or "first" in msg.lower() for msg in [r.getMessage() for r in caplog.records]
        )

    def test_resolve_dirs_from_file_missing(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """When --dirs-from-file points to a nonexistent file, SystemExit is raised."""
        monkeypatch.setattr(cli_mod, "_is_interactive", lambda: False)

        args = _ns(dirs_from_file=str(tmp_path / "nonexistent.txt"))

        with pytest.raises(SystemExit) as exc_info:
            _resolve_dirs(args)

        assert exc_info.value.code == 1

    def test_resolve_dirs_single_file(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """--single-file with an existing file returns a single-element list of its parent dir."""
        monkeypatch.setattr(cli_mod, "_is_interactive", lambda: False)

        pdf = tmp_path / "test.pdf"
        pdf.write_bytes(b"%PDF-1.4 dummy")

        args = _ns(single_file=str(pdf))
        dirs, single = _resolve_dirs(args)

        assert len(dirs) == 1
        assert dirs[0] == str(tmp_path.resolve())
        assert single == str(pdf)

    def test_resolve_dirs_single_file_missing(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """--single-file with a nonexistent file raises SystemExit."""
        monkeypatch.setattr(cli_mod, "_is_interactive", lambda: False)

        args = _ns(single_file=str(tmp_path / "missing.pdf"))

        with pytest.raises(SystemExit) as exc_info:
            _resolve_dirs(args)

        assert exc_info.value.code == 1


class TestMainErrorHandling:
    """Tests for main() covering --doctor, --watch, and exception-wrapping paths."""

    def test_main_doctor_flag(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """main(["--doctor"]) calls run_doctor_checks and exits with its return code."""
        monkeypatch.setattr(cli_mod, "setup_logging", lambda **k: None)
        monkeypatch.setattr(cli_mod, "run_doctor_checks", lambda args: 0)

        with pytest.raises(SystemExit) as exc_info:
            cli_mod.main(["--doctor"])

        assert exc_info.value.code == 0

    def test_main_doctor_flag_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """main(["--doctor"]) propagates non-zero return from run_doctor_checks."""
        monkeypatch.setattr(cli_mod, "setup_logging", lambda **k: None)
        monkeypatch.setattr(cli_mod, "run_doctor_checks", lambda args: 1)

        with pytest.raises(SystemExit) as exc_info:
            cli_mod.main(["--doctor"])

        assert exc_info.value.code == 1

    def test_main_watch_multiple_dirs(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """main() with --watch and multiple --dir exits with 'only one directory' message."""
        monkeypatch.setattr(cli_mod, "_is_interactive", lambda: False)
        monkeypatch.setattr(cli_mod, "setup_logging", lambda **k: None)

        dir_a = tmp_path / "a"
        dir_b = tmp_path / "b"
        dir_a.mkdir()
        dir_b.mkdir()

        with pytest.raises(SystemExit) as exc_info:
            cli_mod.main(
                [
                    "--watch",
                    "--dir",
                    str(dir_a),
                    str(dir_b),
                    "--language",
                    "de",
                    "--case",
                    "kebabCase",
                    "--project",
                    "",
                    "--version",
                    "",
                ]
            )

        # --watch with multiple dirs should exit with error
        assert "one directory" in str(exc_info.value).lower() or "only one" in str(exc_info.value).lower()

    def test_main_file_not_found(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """When rename_pdfs_in_directory raises FileNotFoundError, main() exits with SystemExit."""
        monkeypatch.setattr(cli_mod, "_is_interactive", lambda: False)
        monkeypatch.setattr(cli_mod, "setup_logging", lambda **k: None)

        def _raise_fnf(*args: Any, **kwargs: Any) -> None:
            raise FileNotFoundError("Directory does not exist: /fake/path")

        monkeypatch.setattr(cli_mod, "rename_pdfs_in_directory", _raise_fnf)

        with pytest.raises(SystemExit) as exc_info:
            cli_mod.main(
                [
                    "--dir",
                    str(tmp_path),
                    "--language",
                    "de",
                    "--case",
                    "kebabCase",
                    "--project",
                    "",
                    "--version",
                    "",
                ]
            )

        assert exc_info.value.code == 1

    def test_main_value_error(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """When rename_pdfs_in_directory raises ValueError, main() exits with SystemExit."""
        monkeypatch.setattr(cli_mod, "_is_interactive", lambda: False)
        monkeypatch.setattr(cli_mod, "setup_logging", lambda **k: None)

        def _raise_ve(*args: Any, **kwargs: Any) -> None:
            raise ValueError("Invalid configuration value")

        monkeypatch.setattr(cli_mod, "rename_pdfs_in_directory", _raise_ve)

        with pytest.raises(SystemExit) as exc_info:
            cli_mod.main(
                [
                    "--dir",
                    str(tmp_path),
                    "--language",
                    "de",
                    "--case",
                    "kebabCase",
                    "--project",
                    "",
                    "--version",
                    "",
                ]
            )

        assert exc_info.value.code == 1

    def test_main_uses_config_defaults_for_unset_cli_values(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Config-file values win over parser defaults when the CLI omits those options."""
        monkeypatch.setattr(cli_mod, "setup_logging", lambda **k: None)
        monkeypatch.setattr(cli_mod, "_is_interactive", lambda: False)

        config_path = tmp_path / "config.json"
        config_path.write_text(
            json.dumps(
                {
                    "language": "en",
                    "desired_case": "snakeCase",
                    "dry_run": True,
                    "workers": 7,
                    "use_llm": False,
                    "use_structured_fields": False,
                    "include_patterns": ["*.pdf"],
                }
            ),
            encoding="utf-8",
        )

        captured: dict[str, RenamerConfig] = {}

        def _capture(directory: str, *, config: RenamerConfig, files_override: list[Path] | None = None) -> None:
            captured["config"] = config

        monkeypatch.setattr(cli_mod, "rename_pdfs_in_directory", _capture)

        cli_mod.main(["--dir", str(tmp_path), "--config", str(config_path)])

        config = captured["config"]
        assert config.language == "en"
        assert config.desired_case == "snakeCase"
        assert config.dry_run is True
        assert config.workers == 7
        assert config.use_llm is False
        assert config.use_structured_fields is False
        assert config.include_patterns == ["*.pdf"]

    def test_main_exits_on_invalid_config_file(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """A malformed --config file is fatal and does not continue with defaults."""
        monkeypatch.setattr(cli_mod, "setup_logging", lambda **k: None)
        monkeypatch.setattr(cli_mod, "_is_interactive", lambda: False)

        bad_config = tmp_path / "bad.json"
        bad_config.write_text("{bad", encoding="utf-8")

        rename_mock = MagicMock()
        monkeypatch.setattr(cli_mod, "rename_pdfs_in_directory", rename_mock)

        with pytest.raises(SystemExit) as exc_info:
            cli_mod.main(["--dir", str(tmp_path), "--config", str(bad_config)])

        assert exc_info.value.code == 1
        rename_mock.assert_not_called()

    def test_main_exits_on_invalid_rules_file(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """A malformed --rules-file is fatal and does not continue with no rules."""
        monkeypatch.setattr(cli_mod, "setup_logging", lambda **k: None)
        monkeypatch.setattr(cli_mod, "_is_interactive", lambda: False)

        bad_rules = tmp_path / "bad-rules.json"
        bad_rules.write_text("{bad", encoding="utf-8")

        with pytest.raises(SystemExit) as exc_info:
            cli_mod.main(["--dir", str(tmp_path), "--dry-run", "--rules-file", str(bad_rules)])

        assert exc_info.value.code == 1


def test_cli_module_entrypoint_runs_help(capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> None:
    """Executing the module directly should invoke main() and print argparse help."""
    monkeypatch.setattr(sys, "argv", ["ai_pdf_renamer.cli", "--help"])
    existing_module = sys.modules.pop("ai_pdf_renamer.cli", None)

    try:
        with pytest.raises(SystemExit) as exc_info:
            runpy.run_module("ai_pdf_renamer.cli", run_name="__main__")
    finally:
        if existing_module is not None:
            sys.modules["ai_pdf_renamer.cli"] = existing_module

    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert "usage: ai-pdf-renamer" in captured.out
