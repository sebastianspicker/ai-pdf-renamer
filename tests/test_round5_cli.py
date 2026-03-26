"""Tests for cli.py uncovered paths: _resolve_option, _resolve_dirs, and main() error handling."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pytest

import ai_pdf_renamer.cli as cli_mod
from ai_pdf_renamer.cli import _resolve_dirs, _resolve_option

# ---------------------------------------------------------------------------
# Helper: build a minimal argparse.Namespace
# ---------------------------------------------------------------------------


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


# ===========================================================================
# _resolve_option interactive prompts (lines 109-119)
# ===========================================================================


class TestResolveOptionFreePrompt:
    """Tests for the free_prompt branch in _resolve_option."""

    def test_resolve_option_valid_choice(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When input() returns a non-empty string, it is accepted as the value."""
        monkeypatch.setattr(cli_mod, "_is_interactive", lambda: True)
        monkeypatch.setattr("builtins.input", lambda _prompt: "my-project")

        args = _ns(project=None)
        result = _resolve_option(
            args,
            "project",
            {},
            "project",
            "",
            free_prompt="Project name (optional): ",
        )

        assert result == "my-project"

    def test_resolve_option_empty_input_returns_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When input() returns empty string, the default is used."""
        monkeypatch.setattr(cli_mod, "_is_interactive", lambda: True)
        monkeypatch.setattr("builtins.input", lambda _prompt: "")

        args = _ns(project=None)
        result = _resolve_option(
            args,
            "project",
            {},
            "project",
            "fallback-default",
            free_prompt="Project name (optional): ",
        )

        assert result == "fallback-default"

    def test_resolve_option_eof_exits(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When input() raises EOFError, SystemExit(1) is raised."""
        monkeypatch.setattr(cli_mod, "_is_interactive", lambda: True)

        def _raise_eof(_prompt: str) -> str:
            raise EOFError

        monkeypatch.setattr("builtins.input", _raise_eof)

        args = _ns(project=None)
        with pytest.raises(SystemExit) as exc_info:
            _resolve_option(
                args,
                "project",
                {},
                "project",
                "",
                free_prompt="Project name (optional): ",
            )

        assert exc_info.value.code == 1

    def test_resolve_option_keyboard_interrupt(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When input() raises KeyboardInterrupt, SystemExit(130) is raised."""
        monkeypatch.setattr(cli_mod, "_is_interactive", lambda: True)

        def _raise_ki(_prompt: str) -> str:
            raise KeyboardInterrupt

        monkeypatch.setattr("builtins.input", _raise_ki)

        args = _ns(project=None)
        with pytest.raises(SystemExit) as exc_info:
            _resolve_option(
                args,
                "project",
                {},
                "project",
                "",
                free_prompt="Project name (optional): ",
            )

        assert exc_info.value.code == 130


# ===========================================================================
# _resolve_dirs (lines 275-308)
# ===========================================================================


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


# ===========================================================================
# main() error handling (lines 371-423 equivalent via _run_renamer_or_watch)
# ===========================================================================


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
