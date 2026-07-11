"""Additional tests for renamer.py pipeline orchestration and hook helpers."""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

import ai_pdf_renamer.renamer as renamer
import ai_pdf_renamer.renamer_hooks as renamer_hooks
from ai_pdf_renamer.config import RenamerConfig
from tests.conftest import make_config as _cfg
from tests.conftest import make_summary_data, patch_renamer_process_result


class TestProduceRenameResults:
    def test_produce_results_sequential(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """workers=1: all files processed sequentially via prefetch path."""
        files = []
        for i in range(3):
            p = tmp_path / f"file{i}.pdf"
            p.write_bytes(b"%PDF-1.4 dummy")
            files.append(p)

        cfg = _cfg(workers=1)

        call_log: list[Path] = []

        def fake_extract(path: Path, config: RenamerConfig) -> tuple[str, bool]:
            call_log.append(path)
            return (f"content of {path.name}", False)

        monkeypatch.setattr(renamer, "_extract_pdf_content", fake_extract)
        patch_renamer_process_result(
            monkeypatch,
            renamer,
            lambda file_path: f"renamed-{file_path.stem}",
            meta={"category": "test"},
        )

        results = renamer.produce_rename_results(files, cfg, rules=None)

        assert len(results) == 3
        for i, (path_out, new_base, meta, exc) in enumerate(results):
            assert path_out == files[i]
            assert new_base == f"renamed-{files[i].stem}"
            assert meta == {"category": "test"}
            assert exc is None

    def test_produce_results_parallel(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """workers=2: results collected from ThreadPoolExecutor."""
        files = []
        for i in range(4):
            p = tmp_path / f"doc{i}.pdf"
            p.write_bytes(b"%PDF-1.4 dummy")
            files.append(p)

        cfg = _cfg(workers=2)

        def fake_process(
            file_path: Path, config: RenamerConfig, rules: Any = None
        ) -> tuple[Path, str | None, dict[str, object] | None, BaseException | None]:
            return (file_path, f"parallel-{file_path.stem}", {"worker": "pool"}, None)

        monkeypatch.setattr(renamer, "process_one_file", fake_process)

        results = renamer.produce_rename_results(files, cfg, rules=None)

        assert len(results) == 4
        returned_paths = {r[0] for r in results}
        assert returned_paths == set(files)
        for _path_out, new_base, meta, exc in results:
            assert new_base is not None
            assert new_base.startswith("parallel-")
            assert meta == {"worker": "pool"}
            assert exc is None

    def test_produce_results_parallel_with_exception(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """workers=2: when _process_one_file raises inside the future, error is captured."""
        files = []
        for i in range(2):
            p = tmp_path / f"err{i}.pdf"
            p.write_bytes(b"%PDF-1.4 dummy")
            files.append(p)

        cfg = _cfg(workers=2)

        def fake_process(
            file_path: Path, config: RenamerConfig, rules: Any = None
        ) -> tuple[Path, str | None, dict[str, object] | None, BaseException | None]:
            raise RuntimeError(f"worker error for {file_path.name}")

        monkeypatch.setattr(renamer, "process_one_file", fake_process)

        results = renamer.produce_rename_results(files, cfg, rules=None)

        assert len(results) == 2
        for _path_out, new_base, meta, exc in results:
            assert new_base is None
            assert meta is None
            assert isinstance(exc, RuntimeError)
            assert "worker error" in str(exc)

    def test_produce_results_sequential_extraction_error(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """workers=1: extraction error for one file does not stop other files."""
        files = []
        for i in range(3):
            p = tmp_path / f"mix{i}.pdf"
            p.write_bytes(b"%PDF-1.4 dummy")
            files.append(p)

        cfg = _cfg(workers=1)

        def fake_extract(path: Path, config: RenamerConfig) -> tuple[str, bool]:
            if path.name == "mix1.pdf":
                raise RuntimeError("corrupt PDF")
            return ("good content", False)

        monkeypatch.setattr(renamer, "_extract_pdf_content", fake_extract)
        monkeypatch.setattr(
            renamer,
            "_process_content_to_result",
            lambda file_path, content, config, rules=None, used_vision=False: (
                file_path,
                f"ok-{file_path.stem}",
                {},
                None,
            ),
        )

        results = renamer.produce_rename_results(files, cfg, rules=None)

        assert len(results) == 3
        # mix0 and mix2 succeed, mix1 fails
        assert results[0][1] == "ok-mix0"
        assert results[0][3] is None
        assert results[1][1] is None
        assert isinstance(results[1][3], RuntimeError)
        assert results[2][1] == "ok-mix2"
        assert results[2][3] is None


class TestRenamePdfsStopEvent:
    def test_rename_pdfs_stop_event_skips_processing(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """When stop_event is already set, processing loop exits before any renames."""
        pdf = tmp_path / "doc.pdf"
        pdf.write_bytes(b"%PDF-1.4 dummy")

        stop_event = threading.Event()
        stop_event.set()

        summary_path = tmp_path / "summary.json"
        cfg = _cfg(
            dry_run=True,
            summary_json_path=str(summary_path),
            stop_event=stop_event,
        )

        # produce_rename_results returns one result, but the apply loop should
        # break immediately because stop_event is set.
        monkeypatch.setattr(
            renamer,
            "produce_rename_results",
            lambda *a, **k: [(pdf, "new-name", {}, None)],
        )

        renamer.rename_pdfs_in_directory(tmp_path, config=cfg)

        data = json.loads(summary_path.read_text(encoding="utf-8"))
        # Stop event was set, so processed should be 0 (loop breaks immediately).
        assert data["processed"] == 0
        assert data["renamed"] == 0


class TestWriteSummaryJsonMerged:
    def test_write_summary_json_with_failures(self, tmp_path: Path) -> None:
        """Verify JSON structure includes failure_details list."""
        summary_path = tmp_path / "summary.json"
        failures = [
            {"file": str(tmp_path / "bad1.pdf"), "error": "corrupt header"},
            {"file": str(tmp_path / "bad2.pdf"), "error": "timeout contacting LLM"},
        ]

        renamer.write_summary_json(
            summary_path,
            make_summary_data(
                tmp_path,
                processed=5,
                renamed=2,
                skipped=1,
                failed=2,
                dry_run=False,
                failures=failures,
            ),
        )

        data = json.loads(summary_path.read_text(encoding="utf-8"))

        assert data["processed"] == 5
        assert data["renamed"] == 2
        assert data["skipped"] == 1
        assert data["failed"] == 2
        assert data["dry_run"] is False
        assert data["directory"] == str(tmp_path)
        assert len(data["failures"]) == 2
        assert data["failures"][0]["file"] == str(tmp_path / "bad1.pdf")
        assert data["failures"][0]["error"] == "corrupt header"
        assert data["failures"][1]["file"] == str(tmp_path / "bad2.pdf")
        assert data["failures"][1]["error"] == "timeout contacting LLM"

    def test_write_summary_json_none_path(self) -> None:
        """When summary_path is None, no file is written (no-op)."""
        # Should not raise.
        renamer.write_summary_json(
            None,
            make_summary_data(Path("input")),
        )

    def test_write_summary_json_creates_parent_dirs(self, tmp_path: Path) -> None:
        """Parent directories are created if they don't exist."""
        summary_path = tmp_path / "nested" / "deep" / "summary.json"

        renamer.write_summary_json(
            summary_path,
            make_summary_data(tmp_path, processed=1, renamed=1),
        )

        assert summary_path.exists()
        data = json.loads(summary_path.read_text(encoding="utf-8"))
        assert data["processed"] == 1
        assert data["dry_run"] is True


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
            patch("ai_pdf_renamer.renamer_hooks.requests.Session", return_value=mock_session),
            caplog.at_level(logging.WARNING, logger="ai_pdf_renamer.renamer_hooks"),
        ):
            renamer_hooks.run_post_rename_hook(
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

        with patch("ai_pdf_renamer.renamer_hooks.requests.Session", return_value=mock_session):
            renamer_hooks.run_post_rename_hook(
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
