"""Targeted coverage tests for renamer_files, data_paths, llm_prompts, and llm_schema."""

from __future__ import annotations

from pathlib import Path

import pytest

from ai_pdf_renamer.data_paths import data_dir, data_path
from ai_pdf_renamer.llm_prompts import (
    _summary_doc_type_hint,
    build_analysis_prompt,
)
from ai_pdf_renamer.llm_schema import (
    DEFAULT_LLM_CATEGORY,
    DEFAULT_LLM_SUMMARY,
    validate_llm_document_result,
)
from ai_pdf_renamer.renamer_files import collect_pdf_files

# ---------------------------------------------------------------------------
# 1. renamer_files.py
# ---------------------------------------------------------------------------


def test_collect_files_override(tmp_path: Path) -> None:
    """files_override list is returned as-is (no directory scan)."""
    # Create files in tmp_path but pass explicit override list
    real_pdf = tmp_path / "override.pdf"
    real_pdf.write_bytes(b"%PDF-1.4 dummy")
    non_pdf = tmp_path / "readme.txt"
    non_pdf.write_text("hi")

    result = collect_pdf_files(
        tmp_path,
        files_override=[real_pdf, non_pdf],
    )
    # Only the .pdf file survives the suffix filter
    assert result == [real_pdf]


def test_collect_skip_already_named(tmp_path: Path) -> None:
    """Files matching YYYYMMDD-*.pdf are skipped when skip_if_already_named=True."""
    already = tmp_path / "20250101-invoice.pdf"
    already.write_bytes(b"%PDF")
    normal = tmp_path / "report.pdf"
    normal.write_bytes(b"%PDF")

    result = collect_pdf_files(tmp_path, skip_if_already_named=True)
    assert normal in result
    assert already not in result


def test_collect_max_depth(tmp_path: Path) -> None:
    """max_depth=1 returns only top-level and depth-1 PDFs in recursive mode."""
    top = tmp_path / "top.pdf"
    top.write_bytes(b"%PDF")
    lvl1 = tmp_path / "sub" / "lvl1.pdf"
    lvl1.parent.mkdir()
    lvl1.write_bytes(b"%PDF")
    lvl2 = tmp_path / "sub" / "deep" / "lvl2.pdf"
    lvl2.parent.mkdir(parents=True)
    lvl2.write_bytes(b"%PDF")

    result = collect_pdf_files(tmp_path, recursive=True, max_depth=1)
    names = {p.name for p in result}
    assert "top.pdf" in names
    assert "lvl1.pdf" in names
    assert "lvl2.pdf" not in names


def test_collect_hidden_files_skipped(tmp_path: Path) -> None:
    """Dot-prefixed files are excluded in non-recursive mode."""
    hidden = tmp_path / ".hidden.pdf"
    hidden.write_bytes(b"%PDF")
    visible = tmp_path / "visible.pdf"
    visible.write_bytes(b"%PDF")

    result = collect_pdf_files(tmp_path)
    names = {p.name for p in result}
    assert "visible.pdf" in names
    assert ".hidden.pdf" not in names


def test_collect_include_pattern(tmp_path: Path) -> None:
    """include_patterns filters to matching filenames only."""
    inv = tmp_path / "invoice_123.pdf"
    inv.write_bytes(b"%PDF")
    other = tmp_path / "report.pdf"
    other.write_bytes(b"%PDF")

    result = collect_pdf_files(tmp_path, include_patterns=["invoice*"])
    assert [p.name for p in result] == ["invoice_123.pdf"]


# ---------------------------------------------------------------------------
# 2. data_paths.py
# ---------------------------------------------------------------------------


def test_data_path_env_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """AI_PDF_RENAMER_DATA_DIR env var redirects data_dir()."""
    monkeypatch.setenv("AI_PDF_RENAMER_DATA_DIR", str(tmp_path))
    result = data_dir()
    assert result == tmp_path.resolve()


def test_data_path_unknown_file() -> None:
    """data_path() raises ValueError for an unrecognised filename."""
    with pytest.raises(ValueError, match="Unsupported data file"):
        data_path("totally_bogus.json")  # type: ignore[arg-type]


def test_data_dir_fallback() -> None:
    """data_dir() always returns a valid Path (even without env override)."""
    result = data_dir()
    assert isinstance(result, Path)


# ---------------------------------------------------------------------------
# 3. llm_prompts.py
# ---------------------------------------------------------------------------


def test_build_analysis_prompt_german() -> None:
    """German language analysis prompt contains German instruction text."""
    prompt = build_analysis_prompt("de", "Testdokument")
    assert "Analysiere" in prompt
    assert "reines JSON" in prompt


def test_build_analysis_prompt_english() -> None:
    """English language analysis prompt contains English instruction text."""
    prompt = build_analysis_prompt("en", "Test document")
    assert "Analyze" in prompt
    assert "pure JSON" in prompt


def test_summary_doc_type_hint_german() -> None:
    """German doc-type hint references the suggested type."""
    hint = _summary_doc_type_hint("de", "Rechnung")
    assert "Rechnung" in hint
    assert "heuristisch" in hint


# ---------------------------------------------------------------------------
# 4. llm_schema.py
# ---------------------------------------------------------------------------


def test_validate_result_string_tokens() -> None:
    """Comma-separated string for final_summary_tokens is split into a list."""
    parsed: dict[str, object] = {
        "summary": "A summary",
        "keywords": ["a", "b"],
        "category": "invoice",
        "final_summary_tokens": "token1, token2, token3",
    }
    result = validate_llm_document_result(parsed)
    assert result.final_summary_tokens == ["token1", "token2", "token3"]


def test_validate_result_category_na() -> None:
    """Category 'NA' is normalised to the default."""
    parsed: dict[str, object] = {
        "summary": "Some summary",
        "keywords": [],
        "category": "NA",
    }
    result = validate_llm_document_result(parsed)
    assert result.category == DEFAULT_LLM_CATEGORY


def test_validate_result_empty_summary() -> None:
    """Empty summary string falls back to default."""
    parsed: dict[str, object] = {
        "summary": "",
        "keywords": ["x"],
        "category": "invoice",
    }
    result = validate_llm_document_result(parsed)
    assert result.summary == DEFAULT_LLM_SUMMARY


def test_validate_result_valid() -> None:
    """All valid fields are preserved as-is."""
    parsed: dict[str, object] = {
        "summary": "An invoice from Acme Corp",
        "keywords": ["invoice", "acme"],
        "category": "invoice",
        "final_summary_tokens": ["token1", "token2"],
    }
    result = validate_llm_document_result(parsed)
    assert result.summary == "An invoice from Acme Corp"
    assert result.keywords == ["invoice", "acme"]
    assert result.category == "invoice"
    assert result.final_summary_tokens == ["token1", "token2"]
