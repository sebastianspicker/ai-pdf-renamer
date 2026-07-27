"""Integration tests for PDF extraction, filename generation, and dry-run output."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from tests.helpers import make_filename_config


def test_heuristic_only_filename_from_real_pdf(tmp_path: Path) -> None:
    fitz = pytest.importorskip("fitz")

    pdf_path = tmp_path / "test_invoice.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text(fitz.Point(72, 100), "Rechnung Nr. 12345 Betrag: 100,00 EUR")
    doc.save(str(pdf_path))
    doc.close()

    from folionym.filename import FilenameGenerationRequest, generate_filename
    from folionym.filename_models import FilenameGenerationContext
    from folionym.pdf_extract import pdf_to_text

    content = pdf_to_text(pdf_path)
    assert "Rechnung" in content or "12345" in content

    config = make_filename_config(
        use_llm=False,
        language="de",
        use_timestamp_fallback=False,
    )
    filename, meta = generate_filename(
        content,
        FilenameGenerationRequest(config=config, context=FilenameGenerationContext(today=date(2026, 3, 22))),
    )

    assert "20260322" in filename
    assert meta.get("category_source") == "heuristic"


def test_directory_dry_run_preserves_pdf_and_writes_summary(tmp_path: Path) -> None:
    fitz = pytest.importorskip("fitz")

    pdf_path = tmp_path / "rechnung.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text(fitz.Point(72, 100), "Rechnung Nr. 99999 Betrag: 250,00 EUR")
    doc.save(str(pdf_path))
    doc.close()

    summary_json = tmp_path / "summary.json"
    config = make_filename_config(
        use_llm=False,
        language="de",
        dry_run=True,
        summary_json_path=str(summary_json),
        use_timestamp_fallback=False,
    )

    from folionym.renamer import rename_pdfs_in_directory

    rename_pdfs_in_directory(str(tmp_path), config=config)

    assert pdf_path.exists()
    assert summary_json.exists()
    summary = json.loads(summary_json.read_text(encoding="utf-8"))
    assert summary["dry_run"] is True
    assert summary["processed"] >= 1
    assert summary["renamed"] >= 0
