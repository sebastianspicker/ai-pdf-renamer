"""Additional tests for renamer.py pipeline orchestration and hook helpers."""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

import folionym.renamer as renamer
import folionym.renamer_hooks as renamer_hooks
from folionym.config import RenamerConfig
from folionym.llm_backend import SerializedLLMClient
from tests.conftest import make_config as _cfg
from tests.helpers import patch_renamer_process_result


def _write_pdf_series(tmp_path: Path, prefix: str, count: int) -> list[Path]:
    """Create an ordered set of minimal PDF inputs for a batch scenario."""
    files = [tmp_path / f"{prefix}{index}.pdf" for index in range(count)]
    for file_path in files:
        file_path.write_bytes(b"%PDF-1.4 dummy")
    return files


def _patch_parallel_process(monkeypatch: pytest.MonkeyPatch, result_for: object) -> None:
    """Patch the executor work item while retaining its production call signature."""

    def fake_process(
        file_path: Path, _config: RenamerConfig, _rules: Any = None
    ) -> tuple[Path, str | None, dict[str, object] | None, BaseException | None]:
        return result_for(file_path)

    monkeypatch.setattr(renamer, "process_one_file", fake_process)


class TestProduceRenameResults:
    def test_heuristic_only_run_does_not_construct_llm_backend(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        pdf = tmp_path / "doc.pdf"
        pdf.write_bytes(b"%PDF-1.4 dummy")
        cfg = _cfg(workers=1, use_llm=False)

        monkeypatch.setattr(renamer, "_extract_pdf_content", lambda path, config: ("text", False))
        patch_renamer_process_result(monkeypatch, renamer, "renamed", meta={"category": "test"})
        monkeypatch.setattr(
            renamer,
            "create_llm_client_from_config",
            lambda _config: pytest.fail("heuristic-only run must not create an LLM backend"),
        )

        results = renamer.produce_rename_results([pdf], cfg)

        assert results[0][1] == "renamed"

    def test_llm_backend_is_owned_once_and_closed_after_batch(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        files = [tmp_path / f"doc{i}.pdf" for i in range(2)]
        for file_path in files:
            file_path.write_bytes(b"%PDF-1.4 dummy")
        cfg = _cfg(workers=1, use_llm=True)
        client = MagicMock()
        seen_clients: list[object] = []

        monkeypatch.setattr(renamer, "create_llm_client_from_config", lambda _config: client)

        def extract_with_run_client(path: Path, config: RenamerConfig, llm_client: object) -> tuple[str, bool]:
            seen_clients.append(llm_client)
            return ("text", False)

        monkeypatch.setattr(renamer, "_extract_pdf_content_with", extract_with_run_client)
        monkeypatch.setattr(
            renamer,
            "_process_content_to_result",
            lambda request: (
                request.file_path,
                request.file_path.stem,
                {"category": "test"},
                None,
            ),
        )

        results = renamer.produce_rename_results(files, cfg)

        assert [result[1] for result in results] == ["doc0", "doc1"]
        assert seen_clients
        assert all(isinstance(run_client, SerializedLLMClient) for run_client in seen_clients)
        client.close.assert_called_once_with()

    def test_llm_backend_initialization_failure_becomes_per_file_results(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        files = [tmp_path / f"doc{i}.pdf" for i in range(2)]
        for file_path in files:
            file_path.write_bytes(b"%PDF-1.4 dummy")
        cfg = _cfg(workers=2, use_llm=True)
        progress: list[tuple[int, int, Path]] = []

        def fail_factory(_config: RenamerConfig) -> object:
            raise RuntimeError("backend initialization failed")

        monkeypatch.setattr(renamer, "create_llm_client_from_config", fail_factory)

        results = renamer.produce_rename_results(files, cfg, progress_callback=lambda *args: progress.append(args))

        assert [result[0] for result in results] == files
        assert all(isinstance(result[3], RuntimeError) for result in results)
        assert all(str(result[3]) == "backend initialization failed" for result in results)
        assert progress == [(1, 2, files[0]), (2, 2, files[1])]

    def test_suggestion_reports_llm_backend_initialization_failure(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        pdf = tmp_path / "doc.pdf"
        pdf.write_bytes(b"%PDF-1.4 dummy")
        cfg = _cfg(use_llm=True)

        def fail_factory(_config: RenamerConfig) -> object:
            raise RuntimeError("backend initialization failed")

        monkeypatch.setattr(renamer, "create_llm_client_from_config", fail_factory)

        new_base, meta, error = renamer.suggest_rename_for_file(pdf, cfg)

        assert new_base is None
        assert meta is None
        assert isinstance(error, RuntimeError)
        assert str(error) == "backend initialization failed"

    def test_produce_results_sequential(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """workers=1: all files process synchronously without an executor."""
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
        monkeypatch.setattr(
            renamer,
            "ThreadPoolExecutor",
            lambda *_args, **_kwargs: pytest.fail("workers=1 must not create an executor"),
        )
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
        files = _write_pdf_series(tmp_path, "doc", 4)

        cfg = _cfg(workers=2)

        _patch_parallel_process(
            monkeypatch,
            lambda file_path: (file_path, f"parallel-{file_path.stem}", {"worker": "pool"}, None),
        )

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
        files = _write_pdf_series(tmp_path, "err", 2)

        cfg = _cfg(workers=2)

        def worker_error(file_path: Path) -> tuple[Path, str | None, dict[str, object] | None, BaseException | None]:
            raise RuntimeError(f"worker error for {file_path.name}")

        _patch_parallel_process(monkeypatch, worker_error)

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
            lambda request: (
                request.file_path,
                f"ok-{request.file_path.stem}",
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
            patch("folionym.renamer_hooks.requests.Session", return_value=mock_session),
            caplog.at_level(logging.WARNING, logger="folionym.renamer_hooks"),
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

        with patch("folionym.renamer_hooks.requests.Session", return_value=mock_session):
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
