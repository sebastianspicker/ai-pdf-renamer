"""Tests for renamer.py pipeline helper functions.

Covers _sanitize_csv_cell, _write_json_or_csv, _write_summary_json,
and _write_rename_outputs without requiring full pipeline orchestration.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from unittest.mock import MagicMock

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
