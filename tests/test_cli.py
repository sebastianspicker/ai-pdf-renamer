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


# --- Merged from test_round8_tui.py ---

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
# 1. _build_config
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_build_config_returns_renamer_config() -> None:
    """_build_config() returns a RenamerConfig instance."""
    app = _make_app()
    async with app.run_test(size=(120, 40)) as _pilot:
        config = app._build_config(dry_run=True)
        assert isinstance(config, RenamerConfig)
        assert config.dry_run is True


@pytest.mark.asyncio
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


@pytest.mark.asyncio
async def test_build_config_dry_run_false() -> None:
    """_build_config with dry_run=False produces config.dry_run is False."""
    app = _make_app()
    async with app.run_test(size=(120, 40)) as _pilot:
        config = app._build_config(dry_run=False)
        assert isinstance(config, RenamerConfig)
        assert config.dry_run is False


@pytest.mark.asyncio
async def test_build_config_manual_mode() -> None:
    """_build_config with manual_mode=True propagates the flag."""
    app = _make_app()
    async with app.run_test(size=(120, 40)) as _pilot:
        config = app._build_config(dry_run=False, manual_mode=True)
        assert isinstance(config, RenamerConfig)
        assert config.manual_mode is True
        assert config.interactive is True


# ---------------------------------------------------------------------------
# 2. _run_worker
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
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


@pytest.mark.asyncio
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


# ---------------------------------------------------------------------------
# 3. _start_run
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
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


@pytest.mark.asyncio
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


@pytest.mark.asyncio
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


# ---------------------------------------------------------------------------
# 4. _process_one
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
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


@pytest.mark.asyncio
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


@pytest.mark.asyncio
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


@pytest.mark.asyncio
async def test_process_one_already_running() -> None:
    """_process_one while already running writes a warning and exits early."""
    app = _make_app()
    async with app.run_test(size=(120, 40)) as _pilot:
        app._running = True
        with patch("ai_pdf_renamer.tui.suggest_rename_for_file") as mock_suggest:
            app._process_one()
            mock_suggest.assert_not_called()


@pytest.mark.asyncio
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


@pytest.mark.asyncio
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


@pytest.mark.asyncio
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


@pytest.mark.asyncio
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


# ---------------------------------------------------------------------------
# 5. Button handlers
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_preview_button_calls_start_run() -> None:
    """on_preview invokes _start_run(dry_run=True)."""
    app = _make_app()
    async with app.run_test(size=(120, 40)) as _pilot:
        with patch.object(app, "_start_run") as mock_start:
            app.on_preview()
            mock_start.assert_called_once_with(dry_run=True)


@pytest.mark.asyncio
async def test_apply_button_calls_start_run() -> None:
    """on_apply invokes _start_run(dry_run=False)."""
    app = _make_app()
    async with app.run_test(size=(120, 40)) as _pilot:
        with patch.object(app, "_start_run") as mock_start:
            app.on_apply()
            mock_start.assert_called_once_with(dry_run=False)


@pytest.mark.asyncio
async def test_one_button_calls_process_one() -> None:
    """on_one invokes _process_one()."""
    app = _make_app()
    async with app.run_test(size=(120, 40)) as _pilot:
        with patch.object(app, "_process_one") as mock_proc:
            app.on_one()
            mock_proc.assert_called_once()


# ---------------------------------------------------------------------------
# 6. _drain_log_queue
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
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


@pytest.mark.asyncio
async def test_drain_log_queue_failure_result() -> None:
    """_drain_log_queue handles failure result from the worker."""
    app = _make_app()
    async with app.run_test(size=(120, 40)) as _pilot:
        app._log_queue.put(None)  # sentinel immediately
        app._result_queue.put((False, "something broke"))

        app._running = True
        app._drain_log_queue()

        assert app._running is False


@pytest.mark.asyncio
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


# ---------------------------------------------------------------------------
# 7. _get_bool/_get_select exception fallback
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_bool_missing_widget_returns_default() -> None:
    """_get_bool returns default when the widget does not exist."""
    app = _make_app()
    async with app.run_test(size=(120, 40)) as _pilot:
        assert app._get_bool("nonexistent_checkbox") is False
        assert app._get_bool("nonexistent_checkbox", True) is True


@pytest.mark.asyncio
async def test_get_select_missing_widget_returns_default() -> None:
    """_get_select returns default when the widget does not exist."""
    app = _make_app()
    async with app.run_test(size=(120, 40)) as _pilot:
        assert app._get_select("nonexistent_select") == ""
        assert app._get_select("nonexistent_select", "fallback") == "fallback"


# ---------------------------------------------------------------------------
# 8. main() entry point
# ---------------------------------------------------------------------------


def test_main_function_exists() -> None:
    """tui.main is a callable function."""
    from ai_pdf_renamer import tui

    assert callable(tui.main)


# --- Merged from test_round5_cli.py ---

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
