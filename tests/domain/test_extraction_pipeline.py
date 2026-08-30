"""Focused contracts for the PDF extraction boundary."""

from __future__ import annotations

from pathlib import Path

import pytest

from folionym.extraction import pdf
from folionym.extraction.metadata import normalize_pdf_metadata
from folionym.extraction.models import ExtractionFunctions
from folionym.extraction.pipeline import extract_pdf_content_with
from folionym.settings.models import (
    ExtractionConfig,
    LLMConfig,
    LLMVisionConfig,
    OutputConfig,
    OutputNamingConfig,
    RenamerConfig,
)


class FakeVisionClient:
    """Minimal client fake that records a vision invocation."""

    model = "vision-test"
    base_url = "http://test.invalid/v1"

    def __init__(self, events: list[str], response: str = "vision content") -> None:
        self.events = events
        self.response = response

    def complete(self, *_args: object, **_kwargs: object) -> str:
        raise AssertionError("text completion is not part of extraction")

    def complete_vision(self, *_args: object, **_kwargs: object) -> str:
        self.events.append("vision")
        return self.response

    def close(self) -> None:
        self.events.append("close")


def _config(*, use_ocr: bool = False, vision_first: bool = False, vision_fallback: bool = False) -> RenamerConfig:
    return RenamerConfig(
        extraction=ExtractionConfig(use_ocr=use_ocr),
        llm=LLMConfig(
            vision=LLMVisionConfig(
                vision_first=vision_first,
                use_vision_fallback=vision_fallback,
                vision_fallback_min_text_len=50,
            )
        ),
        output=OutputConfig(naming=OutputNamingConfig(language="en")),
    )


def test_vision_first_short_circuits_primary_extraction() -> None:
    events: list[str] = []

    def image_adapter(_path: Path) -> dict[str, str]:
        events.append("image")
        return {"image_b64": "abc", "mime_type": "image/png"}

    def prompt_adapter(_language: str) -> str:
        events.append("prompt")
        return "prompt"

    def sanitize_adapter(content: str) -> str:
        events.append("sanitize")
        return content.strip()

    functions = ExtractionFunctions(
        image_fn=image_adapter,
        pdf_to_text_fn=lambda *_args, **_kwargs: pytest.fail("text adapter should not run"),
        pdf_to_text_with_ocr_fn=lambda *_args, **_kwargs: pytest.fail("OCR adapter should not run"),
        prompt_fn=prompt_adapter,
        sanitize_fn=sanitize_adapter,
    )

    content, used_vision = extract_pdf_content_with(
        Path("source.pdf"),
        _config(vision_first=True),
        extraction_fns=functions,
        llm_client=FakeVisionClient(events),
    )

    assert (content, used_vision) == ("vision content", True)
    assert events == ["image", "prompt", "vision", "sanitize"]


def test_short_text_uses_vision_fallback_after_primary_text() -> None:
    events: list[str] = []

    def text_adapter(*_args: object, **_kwargs: object) -> str:
        events.append("text")
        return "short"

    def image_adapter(_path: Path) -> str:
        events.append("image")
        return "abc"

    def prompt_adapter(_language: str) -> str:
        events.append("prompt")
        return "prompt"

    def sanitize_adapter(content: str) -> str:
        events.append("sanitize")
        return content

    functions = ExtractionFunctions(
        image_fn=image_adapter,
        pdf_to_text_fn=text_adapter,
        pdf_to_text_with_ocr_fn=lambda *_args, **_kwargs: pytest.fail("OCR adapter should not run"),
        prompt_fn=prompt_adapter,
        sanitize_fn=sanitize_adapter,
    )

    content, used_vision = extract_pdf_content_with(
        Path("source.pdf"),
        _config(vision_fallback=True),
        extraction_fns=functions,
        llm_client=FakeVisionClient(events),
    )

    assert (content, used_vision) == ("vision content", True)
    assert events == ["text", "image", "prompt", "vision", "sanitize"]


def test_ocr_is_the_primary_strategy_when_configured() -> None:
    events: list[str] = []

    def ocr_adapter(_path: Path, **kwargs: object) -> str:
        events.append(f"ocr:{kwargs['language']}")
        assert kwargs["max_tokens"] == pdf.DEFAULT_MAX_CONTENT_TOKENS
        return "OCR text"

    functions = ExtractionFunctions(
        image_fn=lambda _path: pytest.fail("vision adapter should not run"),
        pdf_to_text_fn=lambda *_args, **_kwargs: pytest.fail("text adapter should not run"),
        pdf_to_text_with_ocr_fn=ocr_adapter,
    )

    assert extract_pdf_content_with(Path("source.pdf"), _config(use_ocr=True), extraction_fns=functions) == (
        "OCR text",
        False,
    )
    assert events == ["ocr:en"]


def test_metadata_normalization_preserves_the_stable_shape() -> None:
    class FakeDocument:
        def __init__(self) -> None:
            self.metadata = {
                "title": "  Invoice  ",
                "author": "  Ada  ",
                "creationDate": "D:20260203120000+01'00'",
                "modDate": "not-a-pdf-date",
            }

    assert normalize_pdf_metadata(FakeDocument()) == {
        "title": "Invoice",
        "author": "Ada",
        "creation_date": "2026-02-03",
        "mod_date": None,
    }


def test_text_adapter_honors_page_and_token_limits(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    class FakePage:
        def __init__(self, text: str) -> None:
            self.text = text

        def get_text(self, _kind: str) -> str:
            return self.text

    class FakeDocument:
        page_count = 2
        is_encrypted = False

        def __init__(self) -> None:
            self.closed = False

        def load_page(self, page_number: int) -> FakePage:
            return FakePage(("first page", "second page")[page_number])

        def close(self) -> None:
            self.closed = True

    document = FakeDocument()

    class FakeFitz:
        @staticmethod
        def open(_path: Path) -> FakeDocument:
            return document

    monkeypatch.setattr(pdf, "_required_fitz_module", lambda: FakeFitz)
    assert pdf.pdf_to_text(tmp_path / "source.pdf", max_pages=1) == "first page"
    assert document.closed

    monkeypatch.setattr(pdf, "_token_count", len)
    assert len(pdf.shrink_to_token_limit("abcdefghijklmnop", max_tokens=8)) <= 8
