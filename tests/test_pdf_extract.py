from __future__ import annotations

import base64
import logging
import sys
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from ai_pdf_renamer import pdf_extract

# ---------------------------------------------------------------------------
# Existing tests
# ---------------------------------------------------------------------------


def test_pdf_to_text_raises_on_open_error(monkeypatch) -> None:
    class DummyFitz:
        def open(self, path):
            raise RuntimeError("boom")

    monkeypatch.setitem(sys.modules, "fitz", DummyFitz())

    with pytest.raises(OSError, match="Could not open PDF file"):
        pdf_extract.pdf_to_text("missing.pdf")


def test_pdf_to_text_returns_empty_when_no_pages(monkeypatch, tmp_path) -> None:
    class DummyDoc:
        page_count = 0

    class DummyFitz:
        def open(self, path):
            return DummyDoc()

    monkeypatch.setitem(sys.modules, "fitz", DummyFitz())

    pdf_path = tmp_path / "empty.pdf"
    pdf_path.write_bytes(b"")

    assert pdf_extract.pdf_to_text(pdf_path) == ""


def test_shrink_to_token_limit_reduces_text(monkeypatch) -> None:
    monkeypatch.setattr(pdf_extract, "_token_count", lambda _t: 10_000)

    text = "a" * 500
    shrunk = pdf_extract._shrink_to_token_limit(text, max_tokens=10)

    assert len(shrunk) < len(text)
    assert len(shrunk) <= 200


# ---------------------------------------------------------------------------
# Token counting (_token_count, _shrink_to_token_limit)
# ---------------------------------------------------------------------------


def test_token_count_without_tiktoken(monkeypatch) -> None:
    """When tiktoken is unavailable, _token_count falls back to len//4."""
    # Force the cached encoding to None so the import path is re-entered.
    monkeypatch.setattr(pdf_extract, "_tiktoken_encoding", None)

    # Make 'import tiktoken' raise ImportError inside _token_count.
    original_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__  # type: ignore[union-attr]

    def _fake_import(name, *args, **kwargs):  # type: ignore[no-untyped-def]
        if name == "tiktoken":
            raise ImportError("no tiktoken")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", _fake_import)

    text = "a" * 400  # len=400, expected fallback = 400//4 = 100
    result = pdf_extract._token_count(text)
    assert result == 100

    # Restore to avoid polluting other tests.
    monkeypatch.setattr(pdf_extract, "_tiktoken_encoding", None)


def test_shrink_to_token_limit_already_under(monkeypatch) -> None:
    """Text already under the token limit is returned as-is."""
    monkeypatch.setattr(pdf_extract, "_token_count", lambda _t: 5)

    text = "Hello world"
    result = pdf_extract._shrink_to_token_limit(text, max_tokens=100)
    assert result == text


def test_shrink_to_token_limit_shrinks() -> None:
    """Text over the limit is truncated."""
    long_text = "word " * 20_000  # ~100K chars
    result = pdf_extract._shrink_to_token_limit(long_text, max_tokens=50)
    assert len(result) < len(long_text)


# ---------------------------------------------------------------------------
# PDF text extraction (pdf_to_text) — mock fitz
# ---------------------------------------------------------------------------


def test_pdf_to_text_none_path() -> None:
    """Passing None as filepath returns empty string."""
    assert pdf_extract.pdf_to_text(None) == ""


def test_pdf_to_text_encrypted_pdf(monkeypatch, tmp_path) -> None:
    """Encrypted PDF returns empty string."""
    mock_doc = MagicMock()
    mock_doc.is_encrypted = True
    mock_doc.page_count = 5

    mock_fitz = MagicMock()
    mock_fitz.open.return_value = mock_doc
    monkeypatch.setitem(sys.modules, "fitz", mock_fitz)

    pdf_path = tmp_path / "encrypted.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 encrypted")

    result = pdf_extract.pdf_to_text(pdf_path)
    assert result == ""


def test_pdf_to_text_empty_pages(monkeypatch, tmp_path) -> None:
    """PDF with 0 page_count returns empty string."""
    mock_doc = MagicMock()
    mock_doc.page_count = 0
    mock_doc.is_encrypted = False

    mock_fitz = MagicMock()
    mock_fitz.open.return_value = mock_doc
    monkeypatch.setitem(sys.modules, "fitz", mock_fitz)

    pdf_path = tmp_path / "zero_pages.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")

    result = pdf_extract.pdf_to_text(pdf_path)
    assert result == ""


def test_pdf_to_text_successful(monkeypatch, tmp_path) -> None:
    """PDF with a page returning text gives that text back."""
    expected_text = "This is a test document about machine learning."

    mock_page = MagicMock()
    mock_page.get_text.return_value = expected_text

    mock_doc = MagicMock()
    mock_doc.page_count = 1
    mock_doc.is_encrypted = False
    mock_doc.__getitem__ = MagicMock(return_value=mock_page)
    mock_doc.load_page = MagicMock(return_value=mock_page)

    mock_fitz = MagicMock()
    mock_fitz.open.return_value = mock_doc
    monkeypatch.setitem(sys.modules, "fitz", mock_fitz)

    pdf_path = tmp_path / "good.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")

    result = pdf_extract.pdf_to_text(pdf_path)
    assert expected_text in result


# ---------------------------------------------------------------------------
# Vision render (pdf_first_page_to_image_base64) — mock fitz
# ---------------------------------------------------------------------------


def test_vision_no_fitz(monkeypatch) -> None:
    """When fitz import fails, returns None."""
    # Remove fitz from sys.modules so the import inside the function fails.
    monkeypatch.delitem(sys.modules, "fitz", raising=False)

    original_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__  # type: ignore[union-attr]

    def _fake_import(name, *args, **kwargs):  # type: ignore[no-untyped-def]
        if name == "fitz":
            raise ImportError("no fitz")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", _fake_import)

    result = pdf_extract.pdf_first_page_to_image_base64("/tmp/test.pdf")
    assert result is None


def test_vision_encrypted(monkeypatch, tmp_path) -> None:
    """Encrypted PDF returns None for vision render."""
    mock_doc = MagicMock()
    mock_doc.is_encrypted = True
    mock_doc.page_count = 1

    mock_fitz = MagicMock()
    mock_fitz.open.return_value = mock_doc
    monkeypatch.setitem(sys.modules, "fitz", mock_fitz)

    pdf_path = tmp_path / "enc.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")

    result = pdf_extract.pdf_first_page_to_image_base64(pdf_path)
    assert result is None


def test_vision_success(monkeypatch, tmp_path) -> None:
    """Successful vision render returns base64-encoded string."""
    fake_image_bytes = b"\xff\xd8\xff\xe0JFIF-fake-jpeg-data"

    mock_pix = MagicMock()
    mock_pix.tobytes.return_value = fake_image_bytes

    mock_page = MagicMock()
    mock_page.get_pixmap.return_value = mock_pix

    mock_doc = MagicMock()
    mock_doc.is_encrypted = False
    mock_doc.page_count = 1
    mock_doc.load_page.return_value = mock_page

    mock_fitz = MagicMock()
    mock_fitz.open.return_value = mock_doc
    monkeypatch.setitem(sys.modules, "fitz", mock_fitz)

    pdf_path = tmp_path / "render.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")

    result = pdf_extract.pdf_first_page_to_image_base64(pdf_path)
    assert result is not None
    # Verify it's valid base64 that decodes back to original bytes.
    decoded = base64.b64decode(result)
    assert decoded == fake_image_bytes


# ---------------------------------------------------------------------------
# OCR (pdf_to_text_with_ocr) — mock ocrmypdf
# ---------------------------------------------------------------------------


def test_ocr_no_ocrmypdf(monkeypatch, tmp_path, caplog) -> None:
    """When ocrmypdf is not installed, a warning is logged and original text returned."""
    # Make pdf_to_text return short text (below MIN_CHARS_BEFORE_OCR).
    monkeypatch.setattr(pdf_extract, "pdf_to_text", lambda *a, **kw: "Hi")

    # Make 'import ocrmypdf' fail.
    monkeypatch.delitem(sys.modules, "ocrmypdf", raising=False)
    original_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__  # type: ignore[union-attr]

    def _fake_import(name, *args, **kwargs):  # type: ignore[no-untyped-def]
        if name == "ocrmypdf":
            raise ImportError("no ocrmypdf")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", _fake_import)

    pdf_path = tmp_path / "needs_ocr.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")

    with caplog.at_level(logging.WARNING):
        result = pdf_extract.pdf_to_text_with_ocr(pdf_path)

    assert result == "Hi"
    assert any("ocrmypdf not installed" in r.message for r in caplog.records)


def test_ocr_success(monkeypatch, tmp_path) -> None:
    """Successful OCR produces text from the OCR'd PDF."""
    # Make pdf_to_text return short text first (triggers OCR), then OCR'd text.
    call_count = 0

    def fake_pdf_to_text(*args, **kwargs):  # type: ignore[no-untyped-def]
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return "Hi"  # Too short, triggers OCR.
        return "Full OCR extracted text from the document."

    monkeypatch.setattr(pdf_extract, "pdf_to_text", fake_pdf_to_text)

    # Mock ocrmypdf.ocr to just create the output file.
    mock_ocrmypdf = MagicMock()

    def fake_ocr(input_path, output_path, **kwargs):  # type: ignore[no-untyped-def]
        Path(output_path).write_bytes(b"%PDF-1.4 ocr output")

    mock_ocrmypdf.ocr = fake_ocr
    monkeypatch.setitem(sys.modules, "ocrmypdf", mock_ocrmypdf)

    pdf_path = tmp_path / "image_only.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")

    result = pdf_extract.pdf_to_text_with_ocr(pdf_path)
    assert result == "Full OCR extracted text from the document."


def test_ocr_failure(monkeypatch, tmp_path, caplog) -> None:
    """OCR failure logs a warning and returns original text."""
    monkeypatch.setattr(pdf_extract, "pdf_to_text", lambda *a, **kw: "Hi")

    mock_ocrmypdf = MagicMock()
    mock_ocrmypdf.ocr.side_effect = RuntimeError("Tesseract not found")
    monkeypatch.setitem(sys.modules, "ocrmypdf", mock_ocrmypdf)

    pdf_path = tmp_path / "ocr_fail.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")

    with caplog.at_level(logging.WARNING):
        result = pdf_extract.pdf_to_text_with_ocr(pdf_path)

    assert result == "Hi"
    assert any("OCR failed" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# PDF metadata (_parse_pdf_date, get_pdf_metadata)
# ---------------------------------------------------------------------------


def test_parse_pdf_date_valid() -> None:
    """Valid D:YYYYMMDD string is parsed to a date."""
    result = pdf_extract._parse_pdf_date("D:20250315120000")
    assert result == date(2025, 3, 15)


def test_parse_pdf_date_invalid() -> None:
    """Invalid date values return None."""
    result = pdf_extract._parse_pdf_date("D:99999999")
    assert result is None


def test_parse_pdf_date_none() -> None:
    """None input returns None."""
    result = pdf_extract._parse_pdf_date(None)
    assert result is None


def test_get_pdf_metadata_no_fitz(monkeypatch) -> None:
    """When fitz is not available, returns default metadata dict."""
    monkeypatch.delitem(sys.modules, "fitz", raising=False)

    original_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__  # type: ignore[union-attr]

    def _fake_import(name, *args, **kwargs):  # type: ignore[no-untyped-def]
        if name == "fitz":
            raise ImportError("no fitz")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", _fake_import)

    result = pdf_extract.get_pdf_metadata("/tmp/test.pdf")
    assert result["title"] == ""
    assert result["author"] == ""
    assert result["creation_date"] is None
    assert result["mod_date"] is None


# ---------------------------------------------------------------------------
# _extract_pages — mock fitz doc
# ---------------------------------------------------------------------------


def test_extract_pages_text_mode() -> None:
    """First get_text('text') succeeds — returns that text."""
    expected = "Page one text content."
    mock_page = MagicMock()
    mock_page.get_text.return_value = expected

    mock_doc = MagicMock()
    mock_doc.page_count = 1
    mock_doc.__getitem__ = MagicMock(return_value=mock_page)
    mock_doc.load_page = MagicMock(return_value=mock_page)

    pieces, errors = pdf_extract._extract_pages(mock_doc, Path("/tmp/test.pdf"))
    assert pieces == [expected]
    assert errors == []


def test_extract_pages_blocks_fallback() -> None:
    """When get_text('text') returns empty, blocks fallback is used."""
    mock_page = MagicMock()
    call_count = 0

    def fake_get_text(mode="text"):  # type: ignore[no-untyped-def]
        nonlocal call_count
        call_count += 1
        if mode == "text":
            return ""  # Empty — triggers blocks fallback.
        if mode == "blocks":
            # Blocks format: list of tuples; index 4 is the text.
            return [
                (0, 0, 100, 100, "Block text from fallback.", 0, 0),
            ]
        return ""

    mock_page.get_text = fake_get_text

    mock_doc = MagicMock()
    mock_doc.page_count = 1
    mock_doc.__getitem__ = MagicMock(return_value=mock_page)
    mock_doc.load_page = MagicMock(return_value=mock_page)

    pieces, errors = pdf_extract._extract_pages(mock_doc, Path("/tmp/test.pdf"))
    assert len(pieces) == 1
    assert "Block text from fallback." in pieces[0]
    assert errors == []


def test_extract_pages_all_fail() -> None:
    """When all three extraction methods fail, errors are recorded."""
    mock_page = MagicMock()
    mock_page.get_text.side_effect = RuntimeError("extraction failed")

    mock_doc = MagicMock()
    mock_doc.page_count = 1
    mock_doc.__getitem__ = MagicMock(return_value=mock_page)
    mock_doc.load_page = MagicMock(return_value=mock_page)

    pieces, errors = pdf_extract._extract_pages(mock_doc, Path("/tmp/test.pdf"))
    assert pieces == []
    assert len(errors) == 1
    assert "rawdict" in errors[0]
