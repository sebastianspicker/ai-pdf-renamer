"""Cover CLI and configuration runtime failure paths."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from tests.conftest import make_cli_main_args


def _assert_cli_rename_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, error: BaseException) -> None:
    """Exercise the CLI exit boundary for a raised rename failure."""
    import folionym.cli as cli

    monkeypatch.setattr(cli, "setup_logging", lambda **_kwargs: None)
    monkeypatch.setattr(cli, "_is_interactive", lambda: False)

    def fake_rename(*_args: Any, **_kwargs: Any) -> None:
        raise error

    monkeypatch.setattr(cli, "rename_pdfs_in_directory", fake_rename)
    with pytest.raises(SystemExit) as exc_info:
        cli.main(make_cli_main_args(tmp_path))
    assert exc_info.value.code == 1


class TestLoadConfigJsonNonDict:
    """Test _load_config_file with JSON array at top level (line 43)."""

    def test_load_config_json_non_dict(self, tmp_path: Path) -> None:
        """JSON array at top level -> returns {}."""
        from folionym.cli import _load_config_file

        p = tmp_path / "config.json"
        p.write_text("[1, 2, 3]", encoding="utf-8")

        result = _load_config_file(p)
        assert result == {}


class TestResolveLogConfig:
    """Test _resolve_log_config (lines 161-171)."""

    def test_resolve_log_config_defaults(self) -> None:
        """No args set -> defaults to XDG-style log path and INFO."""
        from folionym.cli import _resolve_log_config

        args = argparse.Namespace()
        log_file, log_level = _resolve_log_config(args)
        assert log_file.endswith("folionym/error.log")
        assert log_level == logging.INFO

    def test_resolve_log_config_verbose(self) -> None:
        """--verbose -> DEBUG level."""
        from folionym.cli import _resolve_log_config

        args = argparse.Namespace(verbose=True, quiet=False, log_file=None, log_level=None)
        _log_file, log_level = _resolve_log_config(args)
        assert log_level == logging.DEBUG

    def test_resolve_log_config_quiet(self) -> None:
        """--quiet -> WARNING level."""
        from folionym.cli import _resolve_log_config

        args = argparse.Namespace(verbose=False, quiet=True, log_file=None, log_level=None)
        _log_file, log_level = _resolve_log_config(args)
        assert log_level == logging.WARNING

    def test_resolve_log_config_explicit_level(self) -> None:
        """--log-level ERROR -> ERROR level."""
        from folionym.cli import _resolve_log_config

        args = argparse.Namespace(verbose=False, quiet=False, log_file=None, log_level="ERROR")
        _log_file, log_level = _resolve_log_config(args)
        assert log_level == logging.ERROR

    def test_resolve_log_config_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """FOLIONYM_LOG_LEVEL env var -> that level."""
        from folionym.cli import _resolve_log_config

        monkeypatch.setenv("FOLIONYM_LOG_LEVEL", "DEBUG")
        args = argparse.Namespace(verbose=False, quiet=False, log_file=None, log_level=None)
        _log_file, log_level = _resolve_log_config(args)
        assert log_level == logging.DEBUG

    def test_resolve_log_config_log_file_from_args(self) -> None:
        """--log-file custom.log -> custom.log."""
        from folionym.cli import _resolve_log_config

        args = argparse.Namespace(verbose=False, quiet=False, log_file="custom.log", log_level=None)
        log_file, _log_level = _resolve_log_config(args)
        assert log_file == "custom.log"

    def test_resolve_log_config_log_file_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """FOLIONYM_LOG_FILE env var -> that file."""
        from folionym.cli import _resolve_log_config

        monkeypatch.setenv("FOLIONYM_LOG_FILE", "env_log.log")
        args = argparse.Namespace(verbose=False, quiet=False, log_file=None, log_level=None)
        log_file, _log_level = _resolve_log_config(args)
        assert log_file == "env_log.log"


class TestResolveDirsInteractivePrompt:
    """Test _resolve_dirs interactive prompt (lines 295-306)."""

    def test_resolve_dirs_interactive_prompt(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Interactive mode with no --dir prompts user; input is used."""
        from folionym.cli_runtime import resolve_dirs

        monkeypatch.setattr("builtins.input", lambda _prompt: str(tmp_path))

        args = argparse.Namespace(dirs=None, single_file=None, manual_file=None, dirs_from_file=None)
        dirs, single_file = resolve_dirs(
            args,
            is_interactive=lambda: True,
            console=MagicMock(),
            logger=MagicMock(),
        )
        assert dirs == [str(tmp_path.resolve())]
        assert single_file is None


class TestResolveDirsNoTtyNoDir:
    """Test _resolve_dirs non-interactive with no --dir (lines 307-311)."""

    def test_resolve_dirs_no_tty_no_dir(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Non-interactive, no --dir -> SystemExit."""
        from folionym.cli_runtime import resolve_dirs

        args = argparse.Namespace(dirs=None, single_file=None, manual_file=None, dirs_from_file=None)
        with pytest.raises(SystemExit) as exc_info:
            resolve_dirs(
                args,
                is_interactive=lambda: False,
                console=MagicMock(),
                logger=MagicMock(),
            )
        assert exc_info.value.code == 1


class TestMainConfigFileLoaded:
    """Test main() with --config loading a JSON file (line 437)."""

    def test_main_config_file_loaded(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Pass --config with valid JSON file, verify config values are used."""
        import folionym.cli as cli

        monkeypatch.setattr(cli, "setup_logging", lambda **k: None)
        monkeypatch.setattr(cli, "_is_interactive", lambda: False)

        config_data = {"language": "en", "desired_case": "snakeCase"}
        config_file = tmp_path / "myconfig.json"
        config_file.write_text(json.dumps(config_data), encoding="utf-8")

        captured: dict[str, Any] = {}

        def fake_rename(directory: str, *, config: Any, files_override: Any = None) -> None:
            captured["config"] = config

        monkeypatch.setattr(cli, "rename_pdfs_in_directory", fake_rename)

        cli.main(
            [
                "--dir",
                str(tmp_path),
                "--config",
                str(config_file),
                "--project",
                "",
                "--version",
                "",
            ]
        )

        assert captured["config"].output.naming.language == "en"
        assert captured["config"].output.naming.desired_case == "snakeCase"


class TestLoadOverrideCategoryMapWarning:
    """Test _load_override_category_map OSError warning (lines 185-186)."""

    def test_load_override_category_map_os_error(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """OSError when reading CSV -> warning logged, empty dict returned."""
        from folionym.cli import _load_override_category_map

        p = tmp_path / "overrides.csv"
        p.write_text("filename,category\ninvoice.pdf,finance\n", encoding="utf-8")

        # Make open() raise OSError
        def fake_open(*args: Any, **kwargs: Any) -> Any:
            raise OSError("Permission denied")

        monkeypatch.setattr("builtins.open", fake_open)

        with caplog.at_level(logging.WARNING):
            result = _load_override_category_map(p)

        assert result == {}
        assert any("Could not read override-category file" in r.message for r in caplog.records)


class TestMainDoctorPath:
    """Test main() --doctor path (line 433-434)."""

    def test_main_doctor_exits(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """--doctor calls run_doctor_checks and exits."""
        import folionym.cli as cli

        monkeypatch.setattr(cli, "setup_logging", lambda **k: None)
        monkeypatch.setattr(cli, "run_doctor_checks", lambda args: 0)

        with pytest.raises(SystemExit) as exc_info:
            cli.main(["--doctor", "--dir", "."])

        assert exc_info.value.code == 0


class TestMainRequestsError:
    """Test main() requests/OSError error handling (line 410-411).

    Note: requests.RequestException inherits from OSError, so it's caught
    by the ``except (FileNotFoundError, NotADirectoryError, OSError)`` handler.
    """

    def test_main_requests_error(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """requests.RequestException (subclass of OSError) -> SystemExit with its message."""
        import requests

        _assert_cli_rename_error(monkeypatch, tmp_path, requests.RequestException("Connection refused"))


class TestMainGenericError:
    """Test main() unexpected runtime error handling."""

    def test_main_generic_error(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Unexpected RuntimeError during rename -> SystemExit with exit code 1."""
        _assert_cli_rename_error(monkeypatch, tmp_path, RuntimeError("Unexpected failure"))
