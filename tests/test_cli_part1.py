# ruff: noqa: F401,F811

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


def test_cli_rejects_empty_dir(monkeypatch) -> None:
    import ai_pdf_renamer.cli as cli

    monkeypatch.setattr(cli, "setup_logging", lambda **k: None)

    with pytest.raises(SystemExit) as excinfo:
        cli.main(
            [
                "--dir",
                "",
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

    assert excinfo.value.code == 1


def test_cli_exits_on_missing_directory(monkeypatch, tmp_path) -> None:
    import ai_pdf_renamer.cli as cli

    monkeypatch.setattr(cli, "setup_logging", lambda **k: None)

    missing = tmp_path / "missing"
    with pytest.raises(SystemExit) as excinfo:
        cli.main(
            [
                "--dir",
                str(missing),
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

    assert excinfo.value.code == 1


def test_cli_reprompts_on_invalid_choices(monkeypatch, tmp_path) -> None:
    import builtins

    import ai_pdf_renamer.cli as cli

    monkeypatch.setattr(cli, "setup_logging", lambda **k: None)
    monkeypatch.setattr(cli, "_is_interactive", lambda: True)

    inputs = iter(["fr", "en", "badcase", "snakecase"])
    monkeypatch.setattr(builtins, "input", lambda _prompt: next(inputs))

    captured: dict[str, object] = {}

    def _fake_rename(directory, *, config, files_override=None):
        captured["directory"] = directory
        captured["config"] = config

    monkeypatch.setattr(cli, "rename_pdfs_in_directory", _fake_rename)

    cli.main(
        [
            "--dir",
            str(tmp_path),
            "--project",
            "",
            "--version",
            "",
        ]
    )

    config = captured["config"]
    assert config.language == "en"
    assert config.desired_case == "snakeCase"


def test_cli_exits_on_invalid_json_in_data_file(monkeypatch, tmp_path) -> None:
    import json

    import ai_pdf_renamer.cli as cli

    monkeypatch.setattr(cli, "setup_logging", lambda **k: None)

    def _raise_json_error(*args, **kwargs):
        raise json.JSONDecodeError("Expecting value", doc="", pos=0)

    monkeypatch.setattr(cli, "rename_pdfs_in_directory", _raise_json_error)

    with pytest.raises(SystemExit) as excinfo:
        cli.main(
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

    assert excinfo.value.code == 1


def test_cli_exits_on_missing_data_file(monkeypatch, tmp_path) -> None:
    """When a required data file is missing, CLI exits with a clear message."""
    import ai_pdf_renamer.cli as cli

    monkeypatch.setattr(cli, "setup_logging", lambda **k: None)

    def _raise_file_not_found(*args, **kwargs):
        raise FileNotFoundError(
            "Data file 'heuristic_scores.json' not found. "
            "Looked in: /nonexistent and /packaged. "
            "Set AI_PDF_RENAMER_DATA_DIR or run from the project root."
        )

    monkeypatch.setattr(cli, "rename_pdfs_in_directory", _raise_file_not_found)

    with pytest.raises(SystemExit) as excinfo:
        cli.main(
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

    assert excinfo.value.code == 1


def test_cli_exits_on_broken_json_in_data_file_integration(monkeypatch, tmp_path) -> None:
    """Broken JSON in data dir + PDF with content: CLI exits with clear message."""
    import ai_pdf_renamer.cli as cli
    import ai_pdf_renamer.renamer as renamer

    monkeypatch.setattr(cli, "setup_logging", lambda **k: None)
    monkeypatch.setenv("AI_PDF_RENAMER_DATA_DIR", str(tmp_path))

    (tmp_path / "meta_stopwords.json").write_text('{"stopwords": []}', encoding="utf-8")
    (tmp_path / "heuristic_scores.json").write_text("{ invalid }", encoding="utf-8")

    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()

    try:
        import fitz

        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 72), "Sample document text for renaming.")
        doc.save(str(pdf_dir / "dummy.pdf"))
        doc.close()
    except Exception:
        pytest.skip("PyMuPDF (fitz) required for integration test")

    renamer._stopwords_cached.cache_clear()
    renamer._heuristic_scorer_cached.cache_clear()

    with pytest.raises(SystemExit) as excinfo:
        cli.main(
            [
                "--dir",
                str(pdf_dir),
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

    assert excinfo.value.code == 1


def test_cli_non_interactive_uses_defaults_without_hanging(monkeypatch, tmp_path) -> None:
    """With no TTY, CLI uses defaults and does not prompt (suitable for CI/cron)."""
    import ai_pdf_renamer.cli as cli

    monkeypatch.setattr(cli, "setup_logging", lambda **k: None)
    monkeypatch.setattr(cli, "_is_interactive", lambda: False)

    captured: dict[str, object] = {}

    def _fake_rename(directory, *, config, files_override=None):
        captured["directory"] = directory
        captured["config"] = config

    monkeypatch.setattr(cli, "rename_pdfs_in_directory", _fake_rename)

    # Omit --language, --case, --project, --version; only --dir.
    cli.main(["--dir", str(tmp_path)])

    assert captured.get("directory") == str(tmp_path)
    config = captured["config"]
    assert config.language == "de"
    assert config.desired_case == "kebabCase"
    assert config.project == ""
    assert config.version == ""


def test_cli_renames_with_mocked_llm_no_network(monkeypatch, tmp_path) -> None:
    """Rename flow without real LLM when get_document_* are mocked (CI-safe)."""
    import ai_pdf_renamer.cli as cli

    monkeypatch.setattr(cli, "setup_logging", lambda **k: None)
    monkeypatch.setattr(cli, "_is_interactive", lambda: False)

    import ai_pdf_renamer.filename as filename_mod

    monkeypatch.setattr(
        filename_mod,
        "get_document_summary",
        lambda *args, **kwargs: "Mocked summary for testing",
    )
    monkeypatch.setattr(
        filename_mod,
        "get_document_keywords",
        lambda *args, **kwargs: ["mock", "keywords"],
    )
    monkeypatch.setattr(
        filename_mod,
        "get_document_category",
        lambda *args, **kwargs: "document",
    )
    monkeypatch.setattr(
        filename_mod,
        "get_final_summary_tokens",
        lambda *args, **kwargs: ["mocked", "tokens"],
    )

    try:
        import fitz

        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 72), "Sample text so PDF is not empty.")
        doc.save(str(tmp_path / "sample.pdf"))
        doc.close()
    except Exception:
        pytest.skip("PyMuPDF (fitz) required for this test")

    cli.main(
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

    pdfs = list(tmp_path.glob("*.pdf"))
    assert len(pdfs) == 1
    # Renamed file should no longer be named sample.pdf (content-based name)
    assert pdfs[0].name != "sample.pdf" or "mock" in pdfs[0].name.lower()


async def test_build_config_returns_renamer_config() -> None:
    """_build_config() returns a RenamerConfig instance."""
    app = _make_app()
    async with app.run_test(size=(120, 40)) as _pilot:
        config = app._build_config(dry_run=True)
        assert isinstance(config, RenamerConfig)
        assert config.dry_run is True


async def test_build_config_maps_directory() -> None:
    """Setting the #directory Input is reflected in the snapshot; dry_run maps correctly."""
    from textual.widgets import Input

    app = _make_app()
    async with app.run_test(size=(120, 40)) as _pilot:
        app.query_one("#directory", Input).value = "/tmp/test-pdfs"
        config = app._build_config(dry_run=False)
        assert config.dry_run is False
        # directory is consumed by _start_run, not stored as a config field;
        # verify the snapshot captures it and build_config succeeds.
        snap = app._snapshot()
        assert snap["directory"] == "/tmp/test-pdfs"


async def test_build_config_dry_run_false() -> None:
    """_build_config with dry_run=False produces config.dry_run is False."""
    app = _make_app()
    async with app.run_test(size=(120, 40)) as _pilot:
        config = app._build_config(dry_run=False)
        assert isinstance(config, RenamerConfig)
        assert config.dry_run is False


async def test_build_config_manual_mode() -> None:
    """_build_config with manual_mode=True propagates the flag."""
    app = _make_app()
    async with app.run_test(size=(120, 40)) as _pilot:
        config = app._build_config(dry_run=False, manual_mode=True)
        assert isinstance(config, RenamerConfig)
        assert config.manual_mode is True
        assert config.interactive is True


async def test_run_worker_success() -> None:
    """_run_worker puts (True, 'Completed') when rename_pdfs_in_directory succeeds."""
    app = _make_app()
    async with app.run_test(size=(120, 40)) as _pilot:
        config = app._build_config(dry_run=True)

        def mock_rename(directory: Any, *, config: Any, **kw: Any) -> None:
            pass  # success — no exception

        with patch("ai_pdf_renamer.tui.rename_pdfs_in_directory", mock_rename):
            app._run_worker("/tmp/test", config)

        ok, msg = app._result_queue.get_nowait()
        assert ok is True
        assert msg == "Completed"
        # Sentinel None should be in the log queue
        sentinel = app._log_queue.get_nowait()
        assert sentinel is None


async def test_run_worker_failure() -> None:
    """_run_worker puts (False, error_msg) when rename_pdfs_in_directory raises."""
    app = _make_app()
    async with app.run_test(size=(120, 40)) as _pilot:
        config = app._build_config(dry_run=True)

        def mock_rename_fail(directory: Any, *, config: Any, **kw: Any) -> None:
            raise ValueError("something went wrong")

        with patch("ai_pdf_renamer.tui.rename_pdfs_in_directory", mock_rename_fail):
            app._run_worker("/tmp/test", config)

        ok, msg = app._result_queue.get_nowait()
        assert ok is False
        assert "something went wrong" in msg


async def test_start_run_empty_dir_logs_error() -> None:
    """_start_run with no directory writes error to RichLog and does not launch a thread."""
    app = _make_app()
    async with app.run_test(size=(120, 40)) as _pilot:
        # Reset the Textual-internal _running so we pass the "already running" guard.
        app._running = False

        with patch("ai_pdf_renamer.tui.threading.Thread") as mock_thread_cls:
            app._start_run(dry_run=True)
            # No thread should have been created — directory is empty.
            mock_thread_cls.assert_not_called()


async def test_start_run_valid_dir(tmp_path: Path) -> None:
    """_start_run with a valid directory starts a background thread."""
    from textual.widgets import Input

    app = _make_app()
    async with app.run_test(size=(120, 40)) as _pilot:
        app.query_one("#directory", Input).value = str(tmp_path)
        # Reset the Textual-internal _running so we pass the "already running" guard.
        app._running = False

        mock_thread = MagicMock()
        with (
            patch("ai_pdf_renamer.tui.threading.Thread", return_value=mock_thread),
            patch("ai_pdf_renamer.tui._save_settings"),
        ):
            app._start_run(dry_run=True)

        assert app._running is True
        mock_thread.start.assert_called_once()


async def test_start_run_already_running() -> None:
    """_start_run while already running writes a warning and does nothing else."""
    app = _make_app()
    async with app.run_test(size=(120, 40)) as _pilot:
        app._running = True
        with patch("ai_pdf_renamer.tui.threading.Thread") as mock_thread_cls:
            app._start_run(dry_run=True)
            # Should not have attempted to start a thread.
            mock_thread_cls.assert_not_called()
        assert app._running is True


async def test_process_one_no_file_set() -> None:
    """_process_one with no single_file path writes an error to the log."""
    app = _make_app()
    async with app.run_test(size=(120, 40)) as _pilot:
        # Reset _running so we pass the guard.
        app._running = False

        with patch("ai_pdf_renamer.tui.suggest_rename_for_file") as mock_suggest:
            app._process_one()
            # suggest should never be called — early return.
            mock_suggest.assert_not_called()


async def test_process_one_success(tmp_path: Path) -> None:
    """_process_one with a valid PDF and successful suggestion logs success."""
    from textual.widgets import Input

    pdf_file = tmp_path / "test-doc.pdf"
    pdf_file.write_bytes(b"%PDF-1.4 minimal")

    app = _make_app()
    async with app.run_test(size=(120, 40)) as _pilot:
        app.query_one("#single_file", Input).value = str(pdf_file)
        app._running = False

        mock_suggest = MagicMock(return_value=("new_name", {"category": "invoice"}, None))
        mock_apply = MagicMock(return_value=(True, tmp_path / "new_name.pdf"))

        with (
            patch("ai_pdf_renamer.tui.suggest_rename_for_file", mock_suggest),
            patch("ai_pdf_renamer.tui.apply_single_rename", mock_apply),
            patch("ai_pdf_renamer.tui.sanitize_filename_base", return_value="new_name"),
        ):
            app._process_one()

        mock_suggest.assert_called_once()
        mock_apply.assert_called_once()


async def test_process_one_failure(tmp_path: Path) -> None:
    """_process_one logs an error when suggest_rename_for_file returns an error."""
    from textual.widgets import Input

    pdf_file = tmp_path / "fail-doc.pdf"
    pdf_file.write_bytes(b"%PDF-1.4 minimal")

    app = _make_app()
    async with app.run_test(size=(120, 40)) as _pilot:
        app.query_one("#single_file", Input).value = str(pdf_file)
        app._running = False

        mock_suggest = MagicMock(return_value=(None, None, ValueError("extraction error")))

        with patch("ai_pdf_renamer.tui.suggest_rename_for_file", mock_suggest):
            app._process_one()

        mock_suggest.assert_called_once()


async def test_process_one_already_running() -> None:
    """_process_one while already running writes a warning and exits early."""
    app = _make_app()
    async with app.run_test(size=(120, 40)) as _pilot:
        app._running = True
        with patch("ai_pdf_renamer.tui.suggest_rename_for_file") as mock_suggest:
            app._process_one()
            mock_suggest.assert_not_called()


async def test_process_one_not_pdf(tmp_path: Path) -> None:
    """_process_one with a non-PDF file writes an error to the log."""
    from textual.widgets import Input

    txt_file = tmp_path / "readme.txt"
    txt_file.write_text("not a PDF")

    app = _make_app()
    async with app.run_test(size=(120, 40)) as _pilot:
        app.query_one("#single_file", Input).value = str(txt_file)
        app._running = False

        with patch("ai_pdf_renamer.tui.suggest_rename_for_file") as mock_suggest:
            app._process_one()
            # Should not reach suggest — non-PDF early return.
            mock_suggest.assert_not_called()


async def test_process_one_skipped(tmp_path: Path) -> None:
    """_process_one logs 'Skipped' when suggest returns None base with no error."""
    from textual.widgets import Input

    pdf_file = tmp_path / "empty-doc.pdf"
    pdf_file.write_bytes(b"%PDF-1.4 minimal")

    app = _make_app()
    async with app.run_test(size=(120, 40)) as _pilot:
        app.query_one("#single_file", Input).value = str(pdf_file)
        app._running = False

        mock_suggest = MagicMock(return_value=(None, None, None))

        with patch("ai_pdf_renamer.tui.suggest_rename_for_file", mock_suggest):
            app._process_one()

        mock_suggest.assert_called_once()


async def test_process_one_rename_fails(tmp_path: Path) -> None:
    """_process_one logs failure when apply_single_rename returns (False, ...)."""
    from textual.widgets import Input

    pdf_file = tmp_path / "test-doc.pdf"
    pdf_file.write_bytes(b"%PDF-1.4 minimal")

    app = _make_app()
    async with app.run_test(size=(120, 40)) as _pilot:
        app.query_one("#single_file", Input).value = str(pdf_file)
        app._running = False

        mock_suggest = MagicMock(return_value=("new_name", None, None))
        mock_apply = MagicMock(return_value=(False, pdf_file))

        with (
            patch("ai_pdf_renamer.tui.suggest_rename_for_file", mock_suggest),
            patch("ai_pdf_renamer.tui.apply_single_rename", mock_apply),
            patch("ai_pdf_renamer.tui.sanitize_filename_base", return_value="new_name"),
        ):
            app._process_one()

        mock_suggest.assert_called_once()
        mock_apply.assert_called_once()


async def test_process_one_nonexistent_file(tmp_path: Path) -> None:
    """_process_one with a path to a file that does not exist writes an error."""
    from textual.widgets import Input

    pdf_file = tmp_path / "ghost.pdf"
    # Do NOT create the file.

    app = _make_app()
    async with app.run_test(size=(120, 40)) as _pilot:
        app.query_one("#single_file", Input).value = str(pdf_file)
        app._running = False

        with patch("ai_pdf_renamer.tui.suggest_rename_for_file") as mock_suggest:
            app._process_one()
            mock_suggest.assert_not_called()


async def test_preview_button_calls_start_run() -> None:
    """on_preview invokes _start_run(dry_run=True)."""
    app = _make_app()
    async with app.run_test(size=(120, 40)) as _pilot:
        with patch.object(app, "_start_run") as mock_start:
            app.on_preview()
            mock_start.assert_called_once_with(dry_run=True)


async def test_apply_button_calls_start_run() -> None:
    """on_apply invokes _start_run(dry_run=False)."""
    app = _make_app()
    async with app.run_test(size=(120, 40)) as _pilot:
        with patch.object(app, "_start_run") as mock_start:
            app.on_apply()
            mock_start.assert_called_once_with(dry_run=False)


async def test_one_button_calls_process_one() -> None:
    """on_one invokes _process_one()."""
    app = _make_app()
    async with app.run_test(size=(120, 40)) as _pilot:
        with patch.object(app, "_process_one") as mock_proc:
            app.on_one()
            mock_proc.assert_called_once()


async def test_drain_log_queue_writes_lines() -> None:
    """_drain_log_queue reads queued lines and writes to RichLog."""
    app = _make_app()
    async with app.run_test(size=(120, 40)) as _pilot:
        app._log_queue.put("Processing 1/3: file1.pdf\n")
        app._log_queue.put("Processing 2/3: file2.pdf\n")
        app._log_queue.put(None)  # sentinel
        app._result_queue.put((True, "Completed"))

        app._running = True
        app._drain_log_queue()

        # After drain, _running should be False (sentinel received)
        assert app._running is False


async def test_drain_log_queue_failure_result() -> None:
    """_drain_log_queue handles failure result from the worker."""
    app = _make_app()
    async with app.run_test(size=(120, 40)) as _pilot:
        app._log_queue.put(None)  # sentinel immediately
        app._result_queue.put((False, "something broke"))

        app._running = True
        app._drain_log_queue()

        assert app._running is False


async def test_drain_log_queue_reschedules_on_empty() -> None:
    """_drain_log_queue reschedules itself when the queue is empty (no sentinel)."""
    app = _make_app()
    async with app.run_test(size=(120, 40)) as _pilot:
        # Queue is empty — drain should just reschedule
        app._running = True
        with patch.object(app, "set_timer") as mock_timer:
            app._drain_log_queue()
            mock_timer.assert_called_once()
        # _running still True since no sentinel was received
        assert app._running is True


async def test_get_bool_missing_widget_returns_default() -> None:
    """_get_bool returns default when the widget does not exist."""
    app = _make_app()
    async with app.run_test(size=(120, 40)) as _pilot:
        assert app._get_bool("nonexistent_checkbox") is False
        assert app._get_bool("nonexistent_checkbox", True) is True


async def test_get_select_missing_widget_returns_default() -> None:
    """_get_select returns default when the widget does not exist."""
    app = _make_app()
    async with app.run_test(size=(120, 40)) as _pilot:
        assert app._get_select("nonexistent_select") == ""
        assert app._get_select("nonexistent_select", "fallback") == "fallback"


def test_main_function_exists() -> None:
    """tui.main is a callable function."""
    from ai_pdf_renamer import tui

    assert callable(tui.main)


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
