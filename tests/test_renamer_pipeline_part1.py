"""Tests for renamer.py pipeline helper functions.

Covers _sanitize_csv_cell, _write_json_or_csv, _write_summary_json,
and _write_rename_outputs without requiring full pipeline orchestration.
"""

from __future__ import annotations

import csv
import json
import threading
from pathlib import Path
from typing import Any

import pytest

import ai_pdf_renamer.renamer as renamer
from ai_pdf_renamer.renamer import (
    RenameOutputData,
    RenameSummaryData,
    _sanitize_csv_cell,
    _write_json_or_csv,
    _write_rename_outputs,
    _write_summary_json,
)
from tests.conftest import make_config as _cfg
from tests.conftest import make_renamer_output_config as _make_config
from tests.conftest import make_summary_data, patch_renamer_process_result
from tests.conftest import write_dummy_pdf as _write_dummy_pdf


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


class TestWriteSummaryJson:
    def test_write_summary_json(self, tmp_path: Path) -> None:
        out = tmp_path / "summary.json"
        _write_summary_json(
            out,
            RenameSummaryData(
                directory=tmp_path,
                processed=10,
                renamed=7,
                skipped=2,
                failed=1,
                dry_run=False,
                failures=[{"file": "bad.pdf", "error": "oops"}],
            ),
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
            make_summary_data(tmp_path),
        )


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
            RenameOutputData(export_rows=rows, processed_count=1, renamed_count=1),
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
            RenameOutputData(plan_entries=entries, processed_count=1),
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
            RenameOutputData(),
        )
        # With empty rows, neither export nor plan file should be written.
        assert not export_path.exists()
        assert not plan_path.exists()


class TestProcessOneFile:
    def test_process_one_file_success(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Extraction returns content; processing returns a valid result."""
        pdf = _write_dummy_pdf(tmp_path / "invoice.pdf")
        cfg = _cfg()

        monkeypatch.setattr(
            renamer,
            "_extract_pdf_content",
            lambda path, config: ("sample text about an invoice", False),
        )
        patch_renamer_process_result(
            monkeypatch,
            renamer,
            "20260101-invoice-sample",
            meta={"category": "invoice"},
        )

        new_base, meta, exc = renamer.suggest_rename_for_file(pdf, cfg)

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

        new_base, meta, exc = renamer.suggest_rename_for_file(pdf, cfg)

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

        new_base, meta, exc = renamer.suggest_rename_for_file(pdf, cfg)

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

        new_base, meta, exc = renamer.suggest_rename_for_file(pdf, cfg)

        assert new_base is None
        assert meta is None
        assert exc is None

    def test_process_one_file_stop_requested(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """When stop_event is set, returns early without extracting."""
        pdf = _write_dummy_pdf(tmp_path / "doc.pdf")
        calls: dict[str, int] = {"extract": 0}

        def fake_extract(*args: Any, **kwargs: Any) -> tuple[str, bool]:
            calls["extract"] += 1
            return ("content", False)

        monkeypatch.setattr(renamer, "_extract_pdf_content", fake_extract)
        stop_event = threading.Event()
        stop_event.set()
        cfg = _cfg(stop_event=stop_event)

        renamed = renamer.rename_pdfs_in_directory(tmp_path, config=cfg, files_override=[pdf])

        assert renamed == set()
        assert calls["extract"] == 0


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
        patch_renamer_process_result(
            monkeypatch,
            renamer,
            "20250615-contract-services",
            meta={"category": "contract", "summary": "services contract"},
        )
        monkeypatch.setattr(renamer, "load_processing_rules", lambda path, **kwargs: None)

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
        monkeypatch.setattr(renamer, "load_processing_rules", lambda path, **kwargs: None)

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
        monkeypatch.setattr(renamer, "load_processing_rules", lambda path, **kwargs: None)

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
        patch_renamer_process_result(monkeypatch, renamer, None, error=inner_exc)
        monkeypatch.setattr(renamer, "load_processing_rules", lambda path, **kwargs: None)

        new_base, meta, exc = renamer.suggest_rename_for_file(pdf, cfg)

        assert new_base is None
        assert meta is None
        assert exc is inner_exc
