"""Tests for renamer.py pipeline helper functions.

Covers _sanitize_csv_cell, _write_json_or_csv, _write_summary_json,
and _write_rename_outputs without requiring full pipeline orchestration.
"""

from __future__ import annotations

import contextlib
import csv
import json
import logging
import threading
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

import ai_pdf_renamer.renamer as renamer
from ai_pdf_renamer.config import RenamerConfig
from ai_pdf_renamer.renamer import _sanitize_csv_cell, _write_json_or_csv, _write_rename_outputs, _write_summary_json

# ---------------------------------------------------------------------------
# _sanitize_csv_cell
# ---------------------------------------------------------------------------


class TestSanitizeCsvCell:
    def test_csv_cell_formula_equals(self) -> None:
        assert _sanitize_csv_cell("=SUM()") == "'=SUM()"

    def test_csv_cell_formula_plus(self) -> None:
        assert _sanitize_csv_cell("+100") == "'+100"

    def test_csv_cell_formula_minus(self) -> None:
        assert _sanitize_csv_cell("-100") == "'-100"

    def test_csv_cell_formula_at(self) -> None:
        assert _sanitize_csv_cell("@attack") == "'@attack"

    def test_csv_cell_formula_pipe(self) -> None:
        assert _sanitize_csv_cell("|cmd") == "'|cmd"

    def test_csv_cell_whitespace_then_formula(self) -> None:
        assert _sanitize_csv_cell("  =cmd") == "'  =cmd"

    def test_csv_cell_inline_tabs_replaced(self) -> None:
        assert _sanitize_csv_cell("a\tb") == "a b"

    def test_csv_cell_inline_cr_lf_replaced(self) -> None:
        assert _sanitize_csv_cell("a\r\nb") == "a  b"

    def test_csv_cell_normal_string(self) -> None:
        assert _sanitize_csv_cell("hello") == "hello"

    def test_csv_cell_none(self) -> None:
        assert _sanitize_csv_cell(None) is None

    def test_csv_cell_integer(self) -> None:
        assert _sanitize_csv_cell(42) == 42

    def test_csv_cell_empty(self) -> None:
        assert _sanitize_csv_cell("") == ""


# ---------------------------------------------------------------------------
# _write_json_or_csv
# ---------------------------------------------------------------------------


class TestWriteJsonOrCsv:
    def test_write_json(self, tmp_path: Path) -> None:
        out = tmp_path / "out.json"
        rows = [{"a": 1, "b": "two"}, {"a": 3, "b": "four"}]
        _write_json_or_csv(out, rows, None)
        data = json.loads(out.read_text(encoding="utf-8"))
        assert data == rows

    def test_write_csv(self, tmp_path: Path) -> None:
        out = tmp_path / "out.csv"
        rows = [{"name": "Alice", "age": "30"}, {"name": "Bob", "age": "25"}]
        _write_json_or_csv(out, rows, ["name", "age"])
        text = out.read_text(encoding="utf-8")
        reader = csv.DictReader(text.splitlines())
        parsed = list(reader)
        assert len(parsed) == 2
        assert parsed[0]["name"] == "Alice"
        assert parsed[1]["age"] == "25"

    def test_write_csv_with_sanitization(self, tmp_path: Path) -> None:
        out = tmp_path / "data.csv"
        rows = [{"val": "=EVIL()", "safe": "ok"}]
        _write_json_or_csv(out, rows, ["val", "safe"])
        text = out.read_text(encoding="utf-8")
        reader = csv.DictReader(text.splitlines())
        parsed = list(reader)
        assert parsed[0]["val"] == "'=EVIL()"
        assert parsed[0]["safe"] == "ok"


# ---------------------------------------------------------------------------
# _write_summary_json
# ---------------------------------------------------------------------------


class TestWriteSummaryJson:
    def test_write_summary_json(self, tmp_path: Path) -> None:
        out = tmp_path / "summary.json"
        _write_summary_json(
            out,
            directory=tmp_path,
            processed=10,
            renamed=7,
            skipped=2,
            failed=1,
            dry_run=False,
            failures=[{"file": "bad.pdf", "error": "oops"}],
        )
        data = json.loads(out.read_text(encoding="utf-8"))
        assert data["processed"] == 10
        assert data["renamed"] == 7
        assert data["skipped"] == 2
        assert data["failed"] == 1
        assert data["dry_run"] is False
        assert data["directory"] == str(tmp_path)
        assert len(data["failures"]) == 1
        assert data["failures"][0]["file"] == "bad.pdf"

    def test_write_summary_json_none_path(self, tmp_path: Path) -> None:
        # When path is None, nothing should be written and no error raised.
        _write_summary_json(
            None,
            directory=tmp_path,
            processed=0,
            renamed=0,
            skipped=0,
            failed=0,
            dry_run=True,
            failures=[],
        )
        # No file created anywhere; just verify no exception.


# ---------------------------------------------------------------------------
# _write_rename_outputs
# ---------------------------------------------------------------------------


def _make_config(**overrides: object) -> MagicMock:
    """Build a MagicMock that behaves like RenamerConfig with sensible defaults."""
    cfg = MagicMock()
    cfg.export_metadata_path = overrides.get("export_metadata_path")
    cfg.plan_file_path = overrides.get("plan_file_path")
    cfg.summary_json_path = overrides.get("summary_json_path")
    cfg.dry_run = overrides.get("dry_run", False)
    return cfg


class TestWriteRenameOutputs:
    def test_write_rename_outputs_export(self, tmp_path: Path) -> None:
        export_path = tmp_path / "export.json"
        cfg = _make_config(export_metadata_path=str(export_path))
        rows = [
            {
                "path": "/tmp/a.pdf",
                "new_name": "renamed.pdf",
                "category": "invoice",
                "summary": "test",
                "keywords": "k1",
                "category_source": "heuristic",
                "llm_failed": False,
                "used_vision_fallback": False,
                "invoice_id": "INV-1",
                "amount": "100",
                "company": "Acme",
            }
        ]
        _write_rename_outputs(
            cfg,
            tmp_path,
            export_rows=rows,
            plan_entries=[],
            processed_count=1,
            renamed_count=1,
            skipped_count=0,
            failed_count=0,
            failure_details=[],
        )
        data = json.loads(export_path.read_text(encoding="utf-8"))
        assert len(data) == 1
        assert data[0]["new_name"] == "renamed.pdf"

    def test_write_rename_outputs_plan(self, tmp_path: Path) -> None:
        plan_path = tmp_path / "plan.json"
        cfg = _make_config(plan_file_path=str(plan_path))
        entries = [{"old": "/tmp/a.pdf", "new": "/tmp/renamed.pdf"}]
        _write_rename_outputs(
            cfg,
            tmp_path,
            export_rows=[],
            plan_entries=entries,
            processed_count=1,
            renamed_count=0,
            skipped_count=0,
            failed_count=0,
            failure_details=[],
        )
        data = json.loads(plan_path.read_text(encoding="utf-8"))
        assert len(data) == 1
        assert data[0]["old"] == "/tmp/a.pdf"

    def test_write_rename_outputs_no_data(self, tmp_path: Path) -> None:
        export_path = tmp_path / "export.json"
        plan_path = tmp_path / "plan.json"
        cfg = _make_config(
            export_metadata_path=str(export_path),
            plan_file_path=str(plan_path),
        )
        _write_rename_outputs(
            cfg,
            tmp_path,
            export_rows=[],
            plan_entries=[],
            processed_count=0,
            renamed_count=0,
            skipped_count=0,
            failed_count=0,
            failure_details=[],
        )
        # With empty rows, neither export nor plan file should be written.
        assert not export_path.exists()
        assert not plan_path.exists()


# --- Merged from test_round5_renamer.py ---


def _cfg(**overrides: Any) -> RenamerConfig:
    """Build a RenamerConfig with sensible test defaults."""
    defaults: dict[str, Any] = {
        "use_llm": False,
        "use_single_llm_call": False,
    }
    defaults.update(overrides)
    return RenamerConfig(**defaults)


# ---------------------------------------------------------------------------
# _process_one_file
# ---------------------------------------------------------------------------


class TestProcessOneFile:
    def test_process_one_file_success(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Extraction returns content; processing returns a valid result."""
        pdf = tmp_path / "invoice.pdf"
        pdf.write_bytes(b"%PDF-1.4 dummy")
        cfg = _cfg()

        monkeypatch.setattr(
            renamer,
            "_extract_pdf_content",
            lambda path, config: ("sample text about an invoice", False),
        )
        monkeypatch.setattr(
            renamer,
            "_process_content_to_result",
            lambda file_path, content, config, rules=None, used_vision=False: (
                file_path,
                "20260101-invoice-sample",
                {"category": "invoice"},
                None,
            ),
        )

        result = renamer._process_one_file(pdf, cfg)
        path_out, new_base, meta, exc = result

        assert path_out == pdf
        assert new_base == "20260101-invoice-sample"
        assert meta == {"category": "invoice"}
        assert exc is None

    def test_process_one_file_extraction_error(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """When _extract_pdf_content raises, error is captured in the result tuple."""
        pdf = tmp_path / "broken.pdf"
        pdf.write_bytes(b"%PDF-1.4 dummy")
        cfg = _cfg()

        monkeypatch.setattr(
            renamer,
            "_extract_pdf_content",
            lambda path, config: (_ for _ in ()).throw(RuntimeError("extraction failed")),
        )

        result = renamer._process_one_file(pdf, cfg)
        path_out, new_base, meta, exc = result

        assert path_out == pdf
        assert new_base is None
        assert meta is None
        assert isinstance(exc, RuntimeError)
        assert "extraction failed" in str(exc)

    def test_process_one_file_processing_error(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Extraction succeeds but _process_content_to_result raises."""
        pdf = tmp_path / "bad.pdf"
        pdf.write_bytes(b"%PDF-1.4 dummy")
        cfg = _cfg()

        monkeypatch.setattr(
            renamer,
            "_extract_pdf_content",
            lambda path, config: ("some valid content here", False),
        )

        def _raise_on_process(*args: Any, **kwargs: Any) -> Any:
            raise ValueError("filename generation blew up")

        monkeypatch.setattr(renamer, "_process_content_to_result", _raise_on_process)

        result = renamer._process_one_file(pdf, cfg)
        path_out, new_base, meta, exc = result

        assert path_out == pdf
        assert new_base is None
        assert meta is None
        assert isinstance(exc, ValueError)
        assert "filename generation blew up" in str(exc)

    def test_process_one_file_empty_content(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """When extracted content is empty/whitespace, returns (path, None, None, None) skip."""
        pdf = tmp_path / "empty.pdf"
        pdf.write_bytes(b"%PDF-1.4 dummy")
        cfg = _cfg()

        monkeypatch.setattr(
            renamer,
            "_extract_pdf_content",
            lambda path, config: ("   ", False),
        )

        result = renamer._process_one_file(pdf, cfg)
        path_out, new_base, meta, exc = result

        assert path_out == pdf
        assert new_base is None
        assert meta is None
        assert exc is None

    def test_process_one_file_stop_requested(self, tmp_path: Path) -> None:
        """When stop_event is set, returns early without extracting."""
        pdf = tmp_path / "doc.pdf"
        pdf.write_bytes(b"%PDF-1.4 dummy")
        stop_event = threading.Event()
        stop_event.set()
        cfg = _cfg(stop_event=stop_event)

        result = renamer._process_one_file(pdf, cfg)
        path_out, new_base, meta, exc = result

        assert path_out == pdf
        assert new_base is None
        assert meta is None
        assert exc is None


# ---------------------------------------------------------------------------
# suggest_rename_for_file
# ---------------------------------------------------------------------------


class TestSuggestRenameForFile:
    def test_suggest_rename_success(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Full pipeline: extraction + filename generation returns suggestion."""
        pdf = tmp_path / "contract.pdf"
        pdf.write_bytes(b"%PDF-1.4 dummy")
        cfg = _cfg()

        monkeypatch.setattr(
            renamer,
            "_extract_pdf_content",
            lambda path, config: ("Contract for services dated 2025-06-15", False),
        )
        monkeypatch.setattr(
            renamer,
            "_process_content_to_result",
            lambda file_path, content, config, rules=None, used_vision=False: (
                file_path,
                "20250615-contract-services",
                {"category": "contract", "summary": "services contract"},
                None,
            ),
        )
        monkeypatch.setattr(renamer, "load_processing_rules", lambda path: None)

        new_base, meta, exc = renamer.suggest_rename_for_file(pdf, cfg)

        assert new_base == "20250615-contract-services"
        assert meta is not None
        assert meta["category"] == "contract"
        assert exc is None

    def test_suggest_rename_empty_content(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """When extracted content is empty, returns (None, None, None)."""
        pdf = tmp_path / "blank.pdf"
        pdf.write_bytes(b"%PDF-1.4 dummy")
        cfg = _cfg()

        monkeypatch.setattr(
            renamer,
            "_extract_pdf_content",
            lambda path, config: ("", False),
        )
        monkeypatch.setattr(renamer, "load_processing_rules", lambda path: None)

        new_base, meta, exc = renamer.suggest_rename_for_file(pdf, cfg)

        assert new_base is None
        assert meta is None
        assert exc is None

    def test_suggest_rename_error(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """When _extract_pdf_content raises, error is captured."""
        pdf = tmp_path / "missing.pdf"
        pdf.write_bytes(b"%PDF-1.4 dummy")
        cfg = _cfg()

        def _raise(*args: Any, **kwargs: Any) -> Any:
            raise OSError("cannot read PDF")

        monkeypatch.setattr(renamer, "_extract_pdf_content", _raise)
        monkeypatch.setattr(renamer, "load_processing_rules", lambda path: None)

        new_base, meta, exc = renamer.suggest_rename_for_file(pdf, cfg)

        assert new_base is None
        assert meta is None
        assert isinstance(exc, OSError)
        assert "cannot read PDF" in str(exc)

    def test_suggest_rename_process_error(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """When _process_content_to_result returns an error, it propagates via (None, None, exc)."""
        pdf = tmp_path / "doc.pdf"
        pdf.write_bytes(b"%PDF-1.4 dummy")
        cfg = _cfg()

        monkeypatch.setattr(
            renamer,
            "_extract_pdf_content",
            lambda path, config: ("valid content", False),
        )
        inner_exc = RuntimeError("generate failed")
        monkeypatch.setattr(
            renamer,
            "_process_content_to_result",
            lambda file_path, content, config, rules=None, used_vision=False: (
                file_path,
                None,
                None,
                inner_exc,
            ),
        )
        monkeypatch.setattr(renamer, "load_processing_rules", lambda path: None)

        new_base, meta, exc = renamer.suggest_rename_for_file(pdf, cfg)

        assert new_base is None
        assert meta is None
        assert exc is inner_exc


# ---------------------------------------------------------------------------
# _produce_rename_results
# ---------------------------------------------------------------------------


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
        monkeypatch.setattr(
            renamer,
            "_process_content_to_result",
            lambda file_path, content, config, rules=None, used_vision=False: (
                file_path,
                f"renamed-{file_path.stem}",
                {"category": "test"},
                None,
            ),
        )

        results = renamer._produce_rename_results(files, cfg, rules=None)

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

        monkeypatch.setattr(renamer, "_process_one_file", fake_process)

        results = renamer._produce_rename_results(files, cfg, rules=None)

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

        monkeypatch.setattr(renamer, "_process_one_file", fake_process)

        results = renamer._produce_rename_results(files, cfg, rules=None)

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

        results = renamer._produce_rename_results(files, cfg, rules=None)

        assert len(results) == 3
        # mix0 and mix2 succeed, mix1 fails
        assert results[0][1] == "ok-mix0"
        assert results[0][3] is None
        assert results[1][1] is None
        assert isinstance(results[1][3], RuntimeError)
        assert results[2][1] == "ok-mix2"
        assert results[2][3] is None


# ---------------------------------------------------------------------------
# rename_pdfs_in_directory — stop_event early exit
# ---------------------------------------------------------------------------


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

        # _produce_rename_results returns one result, but the apply loop should
        # break immediately because stop_event is set.
        monkeypatch.setattr(
            renamer,
            "_produce_rename_results",
            lambda *a, **k: [(pdf, "new-name", {}, None)],
        )

        renamer.rename_pdfs_in_directory(tmp_path, config=cfg)

        data = json.loads(summary_path.read_text(encoding="utf-8"))
        # Stop event was set, so processed should be 0 (loop breaks immediately).
        assert data["processed"] == 0
        assert data["renamed"] == 0


# ---------------------------------------------------------------------------
# _write_summary_json — with failure details
# ---------------------------------------------------------------------------


class TestWriteSummaryJson_merged:
    def test_write_summary_json_with_failures(self, tmp_path: Path) -> None:
        """Verify JSON structure includes failure_details list."""
        summary_path = tmp_path / "summary.json"
        failures = [
            {"file": "/tmp/bad1.pdf", "error": "corrupt header"},
            {"file": "/tmp/bad2.pdf", "error": "timeout contacting LLM"},
        ]

        renamer._write_summary_json(
            summary_path,
            directory=tmp_path,
            processed=5,
            renamed=2,
            skipped=1,
            failed=2,
            dry_run=False,
            failures=failures,
        )

        data = json.loads(summary_path.read_text(encoding="utf-8"))

        assert data["processed"] == 5
        assert data["renamed"] == 2
        assert data["skipped"] == 1
        assert data["failed"] == 2
        assert data["dry_run"] is False
        assert data["directory"] == str(tmp_path)
        assert len(data["failures"]) == 2
        assert data["failures"][0]["file"] == "/tmp/bad1.pdf"
        assert data["failures"][0]["error"] == "corrupt header"
        assert data["failures"][1]["file"] == "/tmp/bad2.pdf"
        assert data["failures"][1]["error"] == "timeout contacting LLM"

    def test_write_summary_json_none_path(self) -> None:
        """When summary_path is None, no file is written (no-op)."""
        # Should not raise.
        renamer._write_summary_json(
            None,
            directory=Path("/tmp"),
            processed=0,
            renamed=0,
            skipped=0,
            failed=0,
            dry_run=True,
            failures=[],
        )

    def test_write_summary_json_creates_parent_dirs(self, tmp_path: Path) -> None:
        """Parent directories are created if they don't exist."""
        summary_path = tmp_path / "nested" / "deep" / "summary.json"

        renamer._write_summary_json(
            summary_path,
            directory=tmp_path,
            processed=1,
            renamed=1,
            skipped=0,
            failed=0,
            dry_run=True,
            failures=[],
        )

        assert summary_path.exists()
        data = json.loads(summary_path.read_text(encoding="utf-8"))
        assert data["processed"] == 1
        assert data["dry_run"] is True


# --- Merged from test_round6_renamer.py ---


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
