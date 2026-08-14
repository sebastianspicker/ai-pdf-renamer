"""Cover PDF extraction, OCR fallbacks, and adjacent pipeline edge cases."""

from __future__ import annotations

import base64
import logging
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from folionym import pdf_extract
from tests.conftest import (
    install_fitz_mock,
    make_fitz_doc,
    make_pdf_to_text_sequence,
)
from tests.helpers import block_fitz_import


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
    shrunk = pdf_extract.shrink_to_token_limit(text, max_tokens=10)

    assert len(shrunk) < len(text)
    assert len(shrunk) <= 200


def test_token_count_without_tiktoken(monkeypatch) -> None:
    """An unavailable tiktoken module falls back and caches the failure sentinel."""
    monkeypatch.setattr(pdf_extract, "_tiktoken_encoding", None)
    monkeypatch.setitem(sys.modules, "tiktoken", None)

    assert pdf_extract.estimate_token_count("a" * 400) == 100
    assert pdf_extract._tiktoken_encoding is pdf_extract._TIKTOKEN_MISSING


def test_token_count_caches_tokenizer_initialization_failure(monkeypatch) -> None:
    """A tokenizer lookup failure is not retried after the failure sentinel is cached."""
    calls = 0

    class FailingTiktoken:
        @staticmethod
        def get_encoding(name: str) -> object:
            nonlocal calls
            assert name == "cl100k_base"
            calls += 1
            raise LookupError("missing encoding")

    monkeypatch.setattr(pdf_extract, "_tiktoken_encoding", None)
    monkeypatch.setitem(sys.modules, "tiktoken", FailingTiktoken())

    assert pdf_extract._token_count("a" * 8) == 2
    assert pdf_extract._token_count("a" * 8) == 2
    assert calls == 1
    assert pdf_extract._tiktoken_encoding is pdf_extract._TIKTOKEN_MISSING


def test_token_count_uses_cached_tokenizer(monkeypatch) -> None:
    """A successfully initialized tokenizer supplies the exact encoded length."""
    calls = 0

    class Encoding:
        def encode(self, text: str) -> list[str]:
            return list(text)

    encoding = Encoding()

    class FakeTiktoken:
        @staticmethod
        def get_encoding(name: str) -> Encoding:
            nonlocal calls
            assert name == "cl100k_base"
            calls += 1
            return encoding

    monkeypatch.setattr(pdf_extract, "_tiktoken_encoding", None)
    monkeypatch.setitem(sys.modules, "tiktoken", FakeTiktoken())

    assert pdf_extract._token_count("hello") == 5
    assert pdf_extract._token_count("bye") == 3
    assert calls == 1
    assert pdf_extract._tiktoken_encoding is encoding


@pytest.mark.parametrize("error", [AttributeError, RuntimeError, ValueError])
def test_token_count_falls_back_when_encoding_fails(monkeypatch, error: type[Exception]) -> None:
    """Encoding errors retain the four-character heuristic."""

    class FailingEncoding:
        def encode(self, text: str) -> list[str]:
            del text
            raise error("cannot encode")

    monkeypatch.setattr(pdf_extract, "_tiktoken_encoding", FailingEncoding())

    assert pdf_extract._token_count("a" * 8) == 2


def test_token_count_fallback_has_a_minimum_of_one(monkeypatch) -> None:
    monkeypatch.setattr(pdf_extract, "_tiktoken_encoding", pdf_extract._TIKTOKEN_MISSING)

    assert pdf_extract._token_count("") == 1
    assert pdf_extract._token_count("abc") == 1
    assert pdf_extract._token_count("abcdefgh") == 2


def test_token_count_initializes_once_across_threads(monkeypatch) -> None:
    """Concurrent first callers share the double-checked initialization result."""
    calls = 0
    calls_lock = threading.Lock()
    start = threading.Barrier(4)

    class Encoding:
        def encode(self, text: str) -> list[str]:
            return list(text)

    class FakeTiktoken:
        @staticmethod
        def get_encoding(name: str) -> Encoding:
            nonlocal calls
            assert name == "cl100k_base"
            with calls_lock:
                calls += 1
            return Encoding()

    def count_after_barrier(_: int) -> int:
        start.wait()
        return pdf_extract._token_count("test")

    monkeypatch.setattr(pdf_extract, "_tiktoken_encoding", None)
    monkeypatch.setitem(sys.modules, "tiktoken", FakeTiktoken())

    with ThreadPoolExecutor(max_workers=4) as executor:
        assert list(executor.map(count_after_barrier, range(4))) == [4] * 4
    assert calls == 1


def test_shrink_to_token_limit_already_under(monkeypatch) -> None:
    """Text already under the token limit is returned as-is."""
    monkeypatch.setattr(pdf_extract, "_token_count", lambda _t: 5)

    text = "Hello world"
    result = pdf_extract.shrink_to_token_limit(text, max_tokens=100)
    assert result == text


def test_shrink_to_token_limit_shrinks() -> None:
    """Text over the limit is truncated."""
    long_text = "word " * 20_000  # ~100K chars
    result = pdf_extract.shrink_to_token_limit(long_text, max_tokens=50)
    assert len(result) < len(long_text)


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

    install_fitz_mock(monkeypatch, make_fitz_doc(mock_page, item_access=True))

    pdf_path = tmp_path / "good.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")

    result = pdf_extract.pdf_to_text(pdf_path)
    assert expected_text in result


def test_vision_no_fitz(monkeypatch) -> None:
    """When fitz import fails, returns None."""
    block_fitz_import(monkeypatch)

    result = pdf_extract.pdf_first_page_to_image_base64("test.pdf")
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

    install_fitz_mock(monkeypatch, make_fitz_doc(mock_page))

    pdf_path = tmp_path / "render.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")

    result = pdf_extract.pdf_first_page_to_image_base64(pdf_path)
    assert result is not None
    # Verify it's valid base64 that decodes back to original bytes.
    decoded = base64.b64decode(result)
    assert decoded == fake_image_bytes


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
    monkeypatch.setattr(
        pdf_extract,
        "pdf_to_text",
        make_pdf_to_text_sequence("Hi", "Full OCR extracted text from the document."),
    )

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


def test_parse_pdf_date_valid() -> None:
    """Valid D:YYYYMMDD string is parsed to a date."""
    result = pdf_extract.parse_pdf_date("D:20250315120000")
    assert result == date(2025, 3, 15)


def test_parse_pdf_date_invalid() -> None:
    """Invalid date values return None."""
    result = pdf_extract.parse_pdf_date("D:99999999")
    assert result is None


def test_parse_pdf_date_none() -> None:
    """None input returns None."""
    result = pdf_extract.parse_pdf_date(None)
    assert result is None


def test_get_pdf_metadata_no_fitz(monkeypatch) -> None:
    """When fitz is not available, returns default metadata dict."""
    block_fitz_import(monkeypatch)

    result = pdf_extract.get_pdf_metadata("test.pdf")
    assert result["title"] == ""
    assert result["author"] == ""
    assert result["creation_date"] is None
    assert result["mod_date"] is None


def test_extract_pages_text_mode() -> None:
    """Return the text when the first get_text('text') call succeeds."""
    expected = "Page one text content."
    mock_page = MagicMock()
    mock_page.get_text.return_value = expected

    mock_doc = MagicMock()
    mock_doc.page_count = 1
    mock_doc.__getitem__ = MagicMock(return_value=mock_page)
    mock_doc.load_page = MagicMock(return_value=mock_page)

    pieces, errors = pdf_extract.extract_pages(mock_doc, Path("test.pdf"))
    assert pieces == [expected]
    assert errors == []


def test_extract_pages_empty_text_no_fallback() -> None:
    """When get_text('text') returns empty, no text is extracted (no blocks/rawdict fallback)."""
    mock_page = MagicMock()
    mock_page.get_text.return_value = ""

    mock_doc = MagicMock()
    mock_doc.page_count = 1
    mock_doc.load_page = MagicMock(return_value=mock_page)

    pieces, errors = pdf_extract.extract_pages(mock_doc, Path("test.pdf"))
    assert pieces == []
    assert errors == []


def test_extract_pages_text_failure() -> None:
    """When text extraction raises, an error is recorded."""
    mock_page = MagicMock()
    mock_page.get_text.side_effect = RuntimeError("extraction failed")

    mock_doc = MagicMock()
    mock_doc.page_count = 1
    mock_doc.load_page = MagicMock(return_value=mock_page)

    pieces, errors = pdf_extract.extract_pages(mock_doc, Path("test.pdf"))
    assert pieces == []
    assert len(errors) == 1
    assert "extraction failed" in errors[0]
