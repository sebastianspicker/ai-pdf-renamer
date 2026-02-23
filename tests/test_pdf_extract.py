from __future__ import annotations

import sys

from ai_pdf_renamer import pdf_extract


import pytest

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
