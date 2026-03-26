"""Round 6 coverage tests for renamer.py.

Targets remaining uncovered paths: post-rename hook HTTP, _process_content_to_result,
watch loop, and rename pipeline edge cases (files_override, export_metadata).
"""

from __future__ import annotations

import contextlib
import json
import logging
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

import ai_pdf_renamer.renamer as renamer
from ai_pdf_renamer.config import RenamerConfig


def _cfg(**overrides: Any) -> RenamerConfig:
    """Build a RenamerConfig with sensible test defaults."""
    defaults: dict[str, Any] = {
        "use_llm": False,
        "use_single_llm_call": False,
    }
    defaults.update(overrides)
    return RenamerConfig(**defaults)


# ---------------------------------------------------------------------------
# Post-rename hook HTTP path (lines 134-191)
# ---------------------------------------------------------------------------


class TestPostRenameHookHttp:
    def test_hook_http_remote_warns(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, tmp_path: Path
    ) -> None:
        """Hook URL with non-loopback http:// host logs an 'unencrypted' warning."""
        old = tmp_path / "old.pdf"
        new = tmp_path / "new.pdf"
        old.write_bytes(b"%PDF-1.4 dummy")
        meta: dict[str, object] = {"category": "invoice"}

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()

        mock_session = MagicMock()
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)
        mock_session.post = MagicMock(return_value=mock_resp)

        with (
            patch("requests.Session", return_value=mock_session),
            caplog.at_level(logging.WARNING, logger="ai_pdf_renamer.renamer"),
        ):
            renamer._run_post_rename_hook(
                "http://192.168.1.1:8080/hook",
                old,
                new,
                meta,
            )

        assert any("unencrypted" in rec.message.lower() for rec in caplog.records), (
            f"Expected 'unencrypted' warning in log records: {[r.message for r in caplog.records]}"
        )

    def test_hook_http_post_with_meta(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Hook URL with loopback host posts JSON payload with old_path, new_path, and meta."""
        old = tmp_path / "old.pdf"
        new = tmp_path / "new.pdf"
        old.write_bytes(b"%PDF-1.4 dummy")
        meta: dict[str, object] = {"category": "invoice", "amount": "100.00"}

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()

        mock_session = MagicMock()
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)
        mock_session.post = MagicMock(return_value=mock_resp)

        with patch("requests.Session", return_value=mock_session):
            renamer._run_post_rename_hook(
                "http://127.0.0.1:8080/hook",
                old,
                new,
                meta,
            )

        mock_session.post.assert_called_once()
        call_args = mock_session.post.call_args
        assert call_args[0][0] == "http://127.0.0.1:8080/hook"
        payload = call_args[1]["json"]
        assert payload["old_path"] == str(old)
        assert payload["new_path"] == str(new)
        assert payload["meta"] == meta
        assert payload["meta"]["category"] == "invoice"
        assert payload["meta"]["amount"] == "100.00"


# ---------------------------------------------------------------------------
# _process_content_to_result (lines 251-279)
# ---------------------------------------------------------------------------


class TestProcessContentToResult:
    def test_process_content_to_result_success(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """generate_filename returns a valid tuple; result includes used_vision_fallback."""
        pdf = tmp_path / "doc.pdf"
        pdf.write_bytes(b"%PDF-1.4 dummy")
        cfg = _cfg()

        monkeypatch.setattr(
            renamer,
            "generate_filename",
            lambda content, *, config, override_category=None, pdf_metadata=None, rules=None: (
                "20260101-invoice-acme",
                {"category": "invoice"},
            ),
        )
        monkeypatch.setattr(renamer, "get_pdf_metadata", lambda path: None)

        path_out, new_base, meta, exc = renamer._process_content_to_result(
            pdf, "some invoice content", cfg, rules=None, used_vision=False
        )

        assert path_out == pdf
        assert new_base == "20260101-invoice-acme"
        assert meta is not None
        assert meta["category"] == "invoice"
        assert meta["used_vision_fallback"] is False
        assert exc is None

    def test_process_content_to_result_error(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """When generate_filename raises ValueError, error is captured in result tuple."""
        pdf = tmp_path / "broken.pdf"
        pdf.write_bytes(b"%PDF-1.4 dummy")
        cfg = _cfg()

        def _raise_value_error(*args: Any, **kwargs: Any) -> Any:
            raise ValueError("bad content format")

        monkeypatch.setattr(renamer, "generate_filename", _raise_value_error)
        monkeypatch.setattr(renamer, "get_pdf_metadata", lambda path: None)

        path_out, new_base, meta, exc = renamer._process_content_to_result(
            pdf, "content", cfg, rules=None, used_vision=False
        )

        assert path_out == pdf
        assert new_base is None
        assert meta is None
        assert isinstance(exc, ValueError)
        assert "bad content format" in str(exc)


# ---------------------------------------------------------------------------
# Watch loop (lines 674-744)
# ---------------------------------------------------------------------------


class TestWatchLoop:
    def test_watch_loop_processes_new_files(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Watch loop detects a new PDF and calls rename_pdfs_in_directory for it."""
        pdf = tmp_path / "new-doc.pdf"
        pdf.write_bytes(b"%PDF-1.4 dummy")
        cfg = _cfg(dry_run=True)

        rename_calls: list[dict[str, Any]] = []

        def fake_rename(directory: Any, *, config: Any, files_override: Any = None) -> None:
            rename_calls.append({"dir": directory, "files": files_override})

        monkeypatch.setattr(renamer, "rename_pdfs_in_directory", fake_rename)
        monkeypatch.setattr(
            renamer,
            "_collect_pdf_files",
            lambda *args, **kwargs: [pdf],
        )

        sleep_count = 0

        def fake_sleep(seconds: float) -> None:
            nonlocal sleep_count
            sleep_count += 1
            if sleep_count >= 1:
                raise KeyboardInterrupt("stop watch loop")

        monkeypatch.setattr("time.sleep", fake_sleep)

        with contextlib.suppress(KeyboardInterrupt):
            renamer.run_watch_loop(tmp_path, config=cfg, interval_seconds=0.01)

        assert len(rename_calls) >= 1, "rename_pdfs_in_directory should have been called at least once"
        assert rename_calls[0]["files"] == [pdf]

    def test_watch_loop_skips_unchanged_files(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Watch loop does NOT re-process a file if mtime has not changed between iterations."""
        pdf = tmp_path / "stable.pdf"
        pdf.write_bytes(b"%PDF-1.4 dummy")
        cfg = _cfg(dry_run=True)

        rename_calls: list[dict[str, Any]] = []

        def fake_rename(directory: Any, *, config: Any, files_override: Any = None) -> None:
            rename_calls.append({"dir": directory, "files": files_override})

        monkeypatch.setattr(renamer, "rename_pdfs_in_directory", fake_rename)

        # _collect_pdf_files returns the same file every iteration
        monkeypatch.setattr(
            renamer,
            "_collect_pdf_files",
            lambda *args, **kwargs: [pdf],
        )

        iteration = 0

        def fake_sleep(seconds: float) -> None:
            nonlocal iteration
            iteration += 1
            if iteration >= 2:
                raise KeyboardInterrupt("stop after 2 sleep cycles")

        monkeypatch.setattr("time.sleep", fake_sleep)

        with contextlib.suppress(KeyboardInterrupt):
            renamer.run_watch_loop(tmp_path, config=cfg, interval_seconds=0.01)

        # File is only processed on the first iteration since mtime does not change.
        assert len(rename_calls) == 1, (
            f"Expected exactly 1 rename call (file unchanged on 2nd iteration), got {len(rename_calls)}"
        )


# ---------------------------------------------------------------------------
# Rename pipeline edge cases (lines 508-671)
# ---------------------------------------------------------------------------


class TestRenamePipelineEdgeCases:
    def test_rename_pdfs_files_override(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """files_override is passed to _collect_pdf_files and those files are processed."""
        pdf = tmp_path / "override-target.pdf"
        pdf.write_bytes(b"%PDF-1.4 dummy")
        cfg = _cfg(dry_run=True)

        collected_overrides: list[list[Path] | None] = []
        original_collect = renamer._collect_pdf_files

        def spy_collect(*args: Any, **kwargs: Any) -> list[Path]:
            collected_overrides.append(kwargs.get("files_override"))
            return original_collect(*args, **kwargs)

        monkeypatch.setattr(renamer, "_collect_pdf_files", spy_collect)
        monkeypatch.setattr(
            renamer,
            "_produce_rename_results",
            lambda files, config, rules=None: [(pdf, "20260101-overridden-file", {"category": "test"}, None)],
        )

        renamer.rename_pdfs_in_directory(
            tmp_path,
            config=cfg,
            files_override=[pdf],
        )

        assert len(collected_overrides) == 1
        assert collected_overrides[0] == [pdf]

    def test_rename_pdfs_with_export_metadata(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """When config.export_metadata_path is set, the export file is written after processing."""
        pdf = tmp_path / "exportable.pdf"
        pdf.write_bytes(b"%PDF-1.4 dummy")
        export_path = tmp_path / "export.json"
        cfg = _cfg(dry_run=True, export_metadata_path=str(export_path))

        monkeypatch.setattr(
            renamer,
            "_collect_pdf_files",
            lambda *args, **kwargs: [pdf],
        )
        monkeypatch.setattr(
            renamer,
            "_produce_rename_results",
            lambda files, config, rules=None: [
                (
                    pdf,
                    "20260101-exported-doc",
                    {
                        "category": "report",
                        "summary": "quarterly report",
                        "keywords": "finance, q1",
                        "category_source": "heuristic",
                        "llm_failed": False,
                        "used_vision_fallback": False,
                        "invoice_id": "",
                        "amount": "",
                        "company": "Acme",
                    },
                    None,
                )
            ],
        )

        # In dry_run mode, apply_single_rename returns True without calling on_success.
        # We mock it to invoke on_success so the export path is exercised.
        def fake_apply(
            file_path: Path,
            base: str,
            *,
            plan_file_path: Any = None,
            plan_entries: Any = None,
            dry_run: bool = False,
            backup_dir: Any = None,
            on_success: Any = None,
            max_filename_chars: Any = None,
        ) -> tuple[bool, Path]:
            target = file_path.with_name(base + file_path.suffix)
            if on_success is not None:
                on_success(file_path, target, base)
            return (True, target)

        monkeypatch.setattr(renamer, "apply_single_rename", fake_apply)

        renamer.rename_pdfs_in_directory(tmp_path, config=cfg)

        assert export_path.exists(), "Export metadata file should have been written"
        data = json.loads(export_path.read_text(encoding="utf-8"))
        assert len(data) == 1
        row = data[0]
        assert row["category"] == "report"
        assert row["summary"] == "quarterly report"
        assert row["company"] == "Acme"
        assert row["new_name"] == "20260101-exported-doc.pdf"
