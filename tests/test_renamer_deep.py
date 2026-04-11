"""Deep coverage tests for renamer.py helper functions.

Covers _write_pdf_title_metadata, _run_post_rename_hook,
_interactive_rename_prompt, rename_pdfs_in_directory integration,
and run_watch_loop.
"""

from __future__ import annotations

import contextlib
import json
import logging
import subprocess
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from ai_pdf_renamer.config import RenamerConfig
from ai_pdf_renamer.renamer import (
    _interactive_rename_prompt,
    _run_post_rename_hook,
    _write_pdf_title_metadata,
    rename_pdfs_in_directory,
    run_watch_loop,
)


def _cfg(**overrides: Any) -> RenamerConfig:
    """Build a RenamerConfig with sensible test defaults."""
    defaults: dict[str, Any] = {
        "use_llm": False,
        "use_single_llm_call": False,
        "dry_run": False,
    }
    defaults.update(overrides)
    return RenamerConfig(**defaults)


# ---------------------------------------------------------------------------
# _write_pdf_title_metadata
# ---------------------------------------------------------------------------


class TestWritePdfTitleMetadata:
    def test_write_pdf_title_metadata_success(self, tmp_path: Path) -> None:
        """Verify atomic save: tempfile created, doc saved, then os.replace called."""
        pdf = tmp_path / "test.pdf"
        pdf.write_bytes(b"%PDF-1.4 dummy")

        # Create the tmp file so stat().st_size works after mock save
        tmp_pdf = tmp_path / "tmp.pdf"
        tmp_pdf.write_bytes(b"%PDF-1.4 saved content")

        mock_doc = MagicMock()
        mock_fitz = MagicMock()
        mock_fitz.open.return_value = mock_doc

        mock_tempfile = MagicMock()
        mock_tempfile.mkstemp.return_value = (99, str(tmp_pdf))

        with (
            patch.dict("sys.modules", {"fitz": mock_fitz, "tempfile": mock_tempfile}),
            patch("ai_pdf_renamer.renamer.os.close") as mock_os_close,
            patch("ai_pdf_renamer.renamer.os.replace") as mock_os_replace,
        ):
            _write_pdf_title_metadata(pdf, "My Title")

        mock_fitz.open.assert_called_once_with(pdf)
        mock_doc.set_metadata.assert_called_once_with({"title": "My Title"})
        mock_doc.save.assert_called_once()
        mock_doc.close.assert_called_once()
        mock_os_close.assert_called_once_with(99)
        mock_os_replace.assert_called_once()

    def test_write_pdf_title_metadata_no_fitz(self, tmp_path: Path) -> None:
        """When fitz is not importable, no error should be raised."""
        pdf = tmp_path / "test.pdf"
        pdf.write_bytes(b"%PDF-1.4 dummy")

        # Remove fitz from modules so import fails inside the function
        with patch.dict("sys.modules", {"fitz": None}):
            # Should not raise
            _write_pdf_title_metadata(pdf, "Some Title")

    def test_write_pdf_title_metadata_save_fails(self, tmp_path: Path) -> None:
        """When doc.save raises, the error is caught and logged as a warning."""
        pdf = tmp_path / "test.pdf"
        pdf.write_bytes(b"%PDF-1.4 dummy")

        mock_doc = MagicMock()
        mock_doc.save.side_effect = RuntimeError("cannot save")

        mock_fitz = MagicMock()
        mock_fitz.open.return_value = mock_doc

        mock_tempfile = MagicMock()
        mock_tempfile.mkstemp.return_value = (99, str(tmp_path / "tmp.pdf"))

        with (
            patch.dict("sys.modules", {"fitz": mock_fitz, "tempfile": mock_tempfile}),
            patch("ai_pdf_renamer.renamer.os.close"),
        ):
            # Should not raise (error is logged as warning)
            _write_pdf_title_metadata(pdf, "Fallback Title")

        mock_doc.save.assert_called_once()
        mock_doc.close.assert_called_once()


# ---------------------------------------------------------------------------
# _run_post_rename_hook
# ---------------------------------------------------------------------------


class TestRunPostRenameHook:
    def test_hook_http_post(self, tmp_path: Path) -> None:
        """HTTP hook: mock requests.Session.post. Verify POST called with JSON payload."""
        old = tmp_path / "old.pdf"
        new = tmp_path / "new.pdf"
        meta: dict[str, object] = {"category": "invoice", "summary": "test"}

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()

        mock_session = MagicMock()
        mock_session.post.return_value = mock_response
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)

        with patch("ai_pdf_renamer.renamer.requests.Session", return_value=mock_session):
            _run_post_rename_hook("http://localhost:9999/hook", old, new, meta)

        mock_session.post.assert_called_once()
        call_args = mock_session.post.call_args
        assert call_args[0][0] == "http://localhost:9999/hook"
        payload = call_args[1]["json"]
        assert payload["old_path"] == str(old)
        assert payload["new_path"] == str(new)
        assert payload["meta"] == meta

    def test_hook_shell_command(self, tmp_path: Path) -> None:
        """Simple command without metacharacters: shell=False, args from shlex.split."""
        old = tmp_path / "old.pdf"
        new = tmp_path / "new.pdf"
        meta: dict[str, object] = {}

        with patch("ai_pdf_renamer.renamer.subprocess.run") as mock_run:
            _run_post_rename_hook("echo hello world", old, new, meta)

        mock_run.assert_called_once()
        call_args = mock_run.call_args
        assert call_args[1]["shell"] is False
        assert call_args[0][0] == ["echo", "hello", "world"]

    def test_hook_with_metacharacters(self, tmp_path: Path) -> None:
        """Command containing pipe should use shell invocation path."""
        old = tmp_path / "old.pdf"
        new = tmp_path / "new.pdf"
        meta: dict[str, object] = {}

        with patch("ai_pdf_renamer.renamer.subprocess.run") as mock_run:
            _run_post_rename_hook("echo hello | grep hello", old, new, meta)

        mock_run.assert_called_once()
        call_args = mock_run.call_args
        # shell=False is always passed, but the args list should contain the shell executable
        assert call_args[1]["shell"] is False
        args = call_args[0][0]
        # The args should be [shell_exe, "-c", cmd] on unix
        assert len(args) == 3
        assert args[1] == "-c"
        assert "echo hello | grep hello" in args[2]

    def test_hook_timeout(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        """subprocess.run raises TimeoutExpired: verify warning logged."""
        old = tmp_path / "old.pdf"
        new = tmp_path / "new.pdf"
        meta: dict[str, object] = {}

        with (
            patch(
                "ai_pdf_renamer.renamer.subprocess.run",
                side_effect=subprocess.TimeoutExpired(cmd="slow", timeout=120),
            ),
            caplog.at_level(logging.WARNING, logger="ai_pdf_renamer.renamer"),
        ):
            _run_post_rename_hook("slow_command", old, new, meta)

        assert any("timed out" in record.message for record in caplog.records)

    def test_hook_http_failure(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        """requests post raises ConnectionError: verify warning logged."""
        old = tmp_path / "old.pdf"
        new = tmp_path / "new.pdf"
        meta: dict[str, object] = {}

        import requests

        mock_session = MagicMock()
        mock_session.post.side_effect = requests.ConnectionError("refused")
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)

        with (
            patch("ai_pdf_renamer.renamer.requests.Session", return_value=mock_session),
            caplog.at_level(logging.WARNING, logger="ai_pdf_renamer.renamer"),
        ):
            _run_post_rename_hook("http://127.0.0.1:9999/hook", old, new, meta)

        assert any("HTTP call failed" in record.message for record in caplog.records)


# ---------------------------------------------------------------------------
# _interactive_rename_prompt
# ---------------------------------------------------------------------------


class TestInteractiveRenamePrompt:
    def test_interactive_accept(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Input returns 'y': verify returns ('y', base, target)."""
        pdf = tmp_path / "doc.pdf"
        target = tmp_path / "new-name.pdf"
        monkeypatch.setattr("builtins.input", lambda _prompt: "y")

        reply, base, result_target = _interactive_rename_prompt(pdf, target, "new-name")
        assert reply == "y"
        assert base == "new-name"
        assert result_target == target

    def test_interactive_reject(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Input returns 'n': verify returns ('n', ...)."""
        pdf = tmp_path / "doc.pdf"
        target = tmp_path / "new-name.pdf"
        monkeypatch.setattr("builtins.input", lambda _prompt: "n")

        reply, base, _result_target = _interactive_rename_prompt(pdf, target, "new-name")
        assert reply == "n"
        assert base == "new-name"

    def test_interactive_edit(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Input returns 'e' then 'CUSTOM_NAME': verify returns ('y', 'CUSTOM_NAME', ...)."""
        pdf = tmp_path / "doc.pdf"
        target = tmp_path / "new-name.pdf"
        responses = iter(["e", "CUSTOM_NAME"])
        monkeypatch.setattr("builtins.input", lambda _prompt: next(responses))

        reply, base, result_target = _interactive_rename_prompt(pdf, target, "new-name")
        assert reply == "y"
        assert base == "CUSTOM_NAME"
        assert result_target.name == "CUSTOM_NAME.pdf"

    def test_interactive_eof(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Input raises EOFError: verify returns ('n', ...)."""
        pdf = tmp_path / "doc.pdf"
        target = tmp_path / "new-name.pdf"

        def _raise_eof(_prompt: str) -> str:
            raise EOFError

        monkeypatch.setattr("builtins.input", _raise_eof)

        reply, base, _result_target = _interactive_rename_prompt(pdf, target, "new-name")
        assert reply == "n"
        assert base == "new-name"


# ---------------------------------------------------------------------------
# rename_pdfs_in_directory integration
# ---------------------------------------------------------------------------


class TestRenamePdfsInDirectory:
    def test_rename_no_files(self, tmp_path: Path) -> None:
        """Empty directory: verify summary written with 0 counts."""
        summary = tmp_path / "summary.json"
        cfg = _cfg(summary_json_path=str(summary))

        rename_pdfs_in_directory(tmp_path, config=cfg)

        data = json.loads(summary.read_text(encoding="utf-8"))
        assert data["processed"] == 0
        assert data["renamed"] == 0
        assert data["skipped"] == 0
        assert data["failed"] == 0

    def test_rename_error_tracked(self, tmp_path: Path) -> None:
        """When _produce_rename_results returns an error tuple, verify failed count."""
        pdf = tmp_path / "bad.pdf"
        pdf.write_bytes(b"%PDF-1.4 dummy")
        summary = tmp_path / "summary.json"
        cfg = _cfg(summary_json_path=str(summary))

        error_result = [(pdf, None, None, ValueError("something went wrong"))]

        with (
            patch("ai_pdf_renamer.renamer._collect_pdf_files", return_value=[pdf]),
            patch("ai_pdf_renamer.renamer._produce_rename_results", return_value=error_result),
            patch("ai_pdf_renamer.renamer.load_processing_rules", return_value=None),
        ):
            rename_pdfs_in_directory(tmp_path, config=cfg)

        data = json.loads(summary.read_text(encoding="utf-8"))
        assert data["failed"] == 1
        assert data["processed"] == 1

    def test_rename_dry_run(self, tmp_path: Path) -> None:
        """Dry run: verify apply_single_rename receives dry_run=True."""
        pdf = tmp_path / "test.pdf"
        pdf.write_bytes(b"%PDF-1.4 dummy")
        summary = tmp_path / "summary.json"
        cfg = _cfg(summary_json_path=str(summary), dry_run=True)

        results = [(pdf, "renamed-doc", {"category": "test"}, None)]

        with (
            patch("ai_pdf_renamer.renamer._collect_pdf_files", return_value=[pdf]),
            patch("ai_pdf_renamer.renamer._produce_rename_results", return_value=results),
            patch("ai_pdf_renamer.renamer.load_processing_rules", return_value=None),
            patch(
                "ai_pdf_renamer.renamer.apply_single_rename", return_value=(True, pdf.with_name("renamed-doc.pdf"))
            ) as mock_rename,
        ):
            rename_pdfs_in_directory(tmp_path, config=cfg)

        mock_rename.assert_called_once()
        call_kwargs = mock_rename.call_args[1]
        assert call_kwargs["dry_run"] is True

    def test_rename_data_file_error_propagates(self, tmp_path: Path) -> None:
        """ValueError with 'Invalid JSON in data file' should propagate (re-raised)."""
        pdf = tmp_path / "test.pdf"
        pdf.write_bytes(b"%PDF-1.4 dummy")
        cfg = _cfg()

        error_result = [(pdf, None, None, ValueError("Invalid JSON in data file: bad.json"))]

        with (
            patch("ai_pdf_renamer.renamer._collect_pdf_files", return_value=[pdf]),
            patch("ai_pdf_renamer.renamer._produce_rename_results", return_value=error_result),
            patch("ai_pdf_renamer.renamer.load_processing_rules", return_value=None),
            pytest.raises(ValueError, match="Invalid JSON in data file"),
        ):
            rename_pdfs_in_directory(tmp_path, config=cfg)


# ---------------------------------------------------------------------------
# run_watch_loop
# ---------------------------------------------------------------------------


class TestRunWatchLoop:
    def test_watch_loop_stops_on_keyboard_interrupt(self, tmp_path: Path) -> None:
        """Mock time.sleep to raise KeyboardInterrupt. Verify clean exit."""
        cfg = _cfg()

        call_count = 0

        def _sleep_then_interrupt(seconds: float) -> None:
            nonlocal call_count
            call_count += 1
            if call_count >= 1:
                raise KeyboardInterrupt

        with (
            patch("ai_pdf_renamer.renamer._collect_pdf_files", return_value=[]),
            patch("ai_pdf_renamer.renamer.time.sleep", side_effect=_sleep_then_interrupt),
            contextlib.suppress(KeyboardInterrupt),
        ):
            run_watch_loop(tmp_path, config=cfg, interval_seconds=0.01)

        # Verify at least one sleep was called (loop ran at least once)
        assert call_count >= 1
