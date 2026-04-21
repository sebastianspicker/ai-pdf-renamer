"""Tests for renamer_extract: effective_max_tokens, _try_vision_extraction, extract_pdf_content_with."""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from ai_pdf_renamer.config import RenamerConfig
from ai_pdf_renamer.pdf_extract import DEFAULT_MAX_CONTENT_TOKENS
from ai_pdf_renamer.renamer_extract import (
    _log_extraction_strategy,
    _try_vision_extraction,
    effective_max_tokens,
    extract_pdf_content_with,
)

# ---------------------------------------------------------------------------
# effective_max_tokens
# ---------------------------------------------------------------------------


def test_effective_max_tokens_from_config() -> None:
    """Config.max_tokens_for_extraction takes highest priority."""
    config = RenamerConfig(max_tokens_for_extraction=5000)
    assert effective_max_tokens(config) == 5000


def test_effective_max_tokens_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Falls back to AI_PDF_RENAMER_MAX_TOKENS env var when config is None."""
    monkeypatch.setenv("AI_PDF_RENAMER_MAX_TOKENS", "9999")
    config = RenamerConfig(max_tokens_for_extraction=None)
    assert effective_max_tokens(config) == 9999


def test_effective_max_tokens_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Returns DEFAULT_MAX_CONTENT_TOKENS when neither config nor env is set."""
    monkeypatch.delenv("AI_PDF_RENAMER_MAX_TOKENS", raising=False)
    config = RenamerConfig(max_tokens_for_extraction=None)
    assert effective_max_tokens(config) == DEFAULT_MAX_CONTENT_TOKENS


# ---------------------------------------------------------------------------
# _try_vision_extraction
# ---------------------------------------------------------------------------


def test_vision_extraction_success() -> None:
    """Vision extraction succeeds: image_fn returns base64, client.complete_vision returns text."""
    client = MagicMock()
    client.model = "test-model"
    client.complete_vision.return_value = "Invoice_2024_Telekom"

    config = RenamerConfig(language="de", vision_model=None, llm_timeout_s=30.0)
    path = Path("/fake/doc.pdf")

    result = _try_vision_extraction(
        path,
        config,
        client,
        image_fn=lambda _p: "base64data",
        prompt_fn=lambda _lang: "describe this",
        sanitize_fn=lambda text: text,
    )

    assert result == "Invoice_2024_Telekom"
    client.complete_vision.assert_called_once()
    args = client.complete_vision.call_args.args
    kwargs = client.complete_vision.call_args.kwargs
    assert args == ("base64data", "describe this")
    assert kwargs == {
        "model": "test-model",
        "image_mime_type": "image/jpeg",
        "timeout_s": 60.0,
    }


def test_vision_extraction_no_image() -> None:
    """When image_fn returns None, _try_vision_extraction returns None immediately."""
    client = MagicMock()
    client.model = "test-model"

    config = RenamerConfig(language="de")
    path = Path("/fake/doc.pdf")

    result = _try_vision_extraction(
        path,
        config,
        client,
        image_fn=lambda _p: None,
        prompt_fn=lambda _lang: "describe this",
        sanitize_fn=lambda text: text,
    )

    assert result is None
    client.complete_vision.assert_not_called()


def test_vision_extraction_empty_response() -> None:
    """When client.complete_vision returns empty string, returns None."""
    client = MagicMock()
    client.model = "test-model"
    client.complete_vision.return_value = ""

    config = RenamerConfig(language="de", vision_model=None, llm_timeout_s=30.0)
    path = Path("/fake/doc.pdf")

    result = _try_vision_extraction(
        path,
        config,
        client,
        image_fn=lambda _p: "base64data",
        prompt_fn=lambda _lang: "prompt",
        sanitize_fn=lambda text: text,
    )

    assert result is None


# ---------------------------------------------------------------------------
# extract_pdf_content_with
# ---------------------------------------------------------------------------


def test_extract_text_only() -> None:
    """Normal text extraction, no vision needed."""
    client = MagicMock()
    client.model = "test-model"

    config = RenamerConfig(
        use_ocr=False,
        vision_first=False,
        use_vision_fallback=False,
        max_tokens_for_extraction=1000,
    )
    path = Path("/fake/doc.pdf")

    content, used_vision = extract_pdf_content_with(
        path,
        config,
        pdf_first_page_to_image_base64_fn=lambda _p: None,
        pdf_to_text_fn=lambda _p, max_pages=0, max_tokens=0: "Hello world from PDF",
        pdf_to_text_with_ocr_fn=lambda _p, max_pages=0, max_tokens=0, language="de": "",
        llm_client=client,
    )

    assert content == "Hello world from PDF"
    assert used_vision is False


def test_extract_text_only_does_not_create_llm_client_when_vision_unused(monkeypatch: pytest.MonkeyPatch) -> None:
    """Text-only extraction should not require an unavailable vision backend."""
    config = RenamerConfig(
        use_ocr=False,
        vision_first=False,
        use_vision_fallback=False,
        llm_backend="in-process",
        llm_model_path="/missing/model.gguf",
        max_tokens_for_extraction=1000,
    )

    def fail_if_called(_config: RenamerConfig) -> MagicMock:
        raise AssertionError("LLM client should not be created for text-only extraction")

    monkeypatch.setattr("ai_pdf_renamer.renamer_extract.create_llm_client_from_config", fail_if_called)

    content, used_vision = extract_pdf_content_with(
        Path("/fake/doc.pdf"),
        config,
        pdf_to_text_fn=lambda _p, max_pages=0, max_tokens=0: "Hello world from PDF",
        pdf_to_text_with_ocr_fn=lambda _p, max_pages=0, max_tokens=0, language="de": "",
    )

    assert content == "Hello world from PDF"
    assert used_vision is False


def test_extract_vision_first() -> None:
    """config.vision_first=True: vision tried before text extraction."""
    client = MagicMock()
    client.model = "test-model"
    client.complete_vision.return_value = "Vision result"

    config = RenamerConfig(
        vision_first=True,
        use_vision_fallback=False,
        use_ocr=False,
        llm_timeout_s=30.0,
    )
    path = Path("/fake/doc.pdf")

    text_fn = MagicMock(return_value="text content")

    content, used_vision = extract_pdf_content_with(
        path,
        config,
        pdf_first_page_to_image_base64_fn=lambda _p: "base64img",
        pdf_to_text_fn=text_fn,
        pdf_to_text_with_ocr_fn=lambda _p, max_pages=0, max_tokens=0, language="de": "",
        llm_client=client,
    )

    # sanitize_filename_from_llm replaces spaces with underscores
    assert content == "Vision_result"
    assert used_vision is True
    # Text extraction should NOT be called when vision_first succeeds
    text_fn.assert_not_called()


def test_extract_vision_fallback() -> None:
    """Text is short (< vision_fallback_min_text_len), vision fallback is attempted."""
    client = MagicMock()
    client.model = "test-model"
    client.complete_vision.return_value = "Vision fallback result"

    config = RenamerConfig(
        vision_first=False,
        use_vision_fallback=True,
        vision_fallback_min_text_len=100,
        use_ocr=False,
        llm_timeout_s=30.0,
        max_tokens_for_extraction=1000,
    )
    path = Path("/fake/doc.pdf")

    # Return short text (< 100 chars)
    content, used_vision = extract_pdf_content_with(
        path,
        config,
        pdf_first_page_to_image_base64_fn=lambda _p: "base64img",
        pdf_to_text_fn=lambda _p, max_pages=0, max_tokens=0: "short",
        pdf_to_text_with_ocr_fn=lambda _p, max_pages=0, max_tokens=0, language="de": "",
        llm_client=client,
    )

    # sanitize_filename_from_llm replaces spaces with underscores
    assert content == "Vision_fallback_result"
    assert used_vision is True


def test_extract_vision_fallback_preserves_png_mime_type() -> None:
    """PNG image fallbacks should preserve image/png in the vision request."""
    client = MagicMock()
    client.model = "test-model"
    client.complete_vision.return_value = "Vision fallback result"

    config = RenamerConfig(
        vision_first=False,
        use_vision_fallback=True,
        vision_fallback_min_text_len=100,
        use_ocr=False,
        llm_timeout_s=30.0,
        max_tokens_for_extraction=1000,
    )

    content, used_vision = extract_pdf_content_with(
        Path("/fake/doc.pdf"),
        config,
        pdf_first_page_to_image_base64_fn=lambda _p: {
            "image_b64": "pngbase64",
            "mime_type": "image/png",
        },
        pdf_to_text_fn=lambda _p, max_pages=0, max_tokens=0: "short",
        pdf_to_text_with_ocr_fn=lambda _p, max_pages=0, max_tokens=0, language="de": "",
        llm_client=client,
    )

    assert content == "Vision_fallback_result"
    assert used_vision is True
    client.complete_vision.assert_called_once()
    args = client.complete_vision.call_args.args
    kwargs = client.complete_vision.call_args.kwargs
    assert args[0] == "pngbase64"
    assert "Dateinamen" in args[1]
    assert kwargs == {
        "model": "test-model",
        "image_mime_type": "image/png",
        "timeout_s": 60.0,
    }


def test_extract_vision_first_does_not_retry_vision_fallback() -> None:
    """A failed vision-first attempt should not trigger a second identical vision call."""
    client = MagicMock()
    client.model = "test-model"
    client.complete_vision.return_value = ""

    config = RenamerConfig(
        vision_first=True,
        use_vision_fallback=True,
        vision_fallback_min_text_len=100,
        use_ocr=False,
        llm_timeout_s=30.0,
        max_tokens_for_extraction=1000,
    )

    content, used_vision = extract_pdf_content_with(
        Path("/fake/doc.pdf"),
        config,
        pdf_first_page_to_image_base64_fn=lambda _p: "base64img",
        pdf_to_text_fn=lambda _p, max_pages=0, max_tokens=0: "short",
        pdf_to_text_with_ocr_fn=lambda _p, max_pages=0, max_tokens=0, language="de": "",
        llm_client=client,
    )

    assert content == "short"
    assert used_vision is False
    assert client.complete_vision.call_count == 1


def test_extract_with_ocr() -> None:
    """config.use_ocr=True: OCR extraction is used instead of plain text."""
    client = MagicMock()
    client.model = "test-model"

    config = RenamerConfig(
        use_ocr=True,
        vision_first=False,
        use_vision_fallback=False,
        max_tokens_for_extraction=1000,
        language="de",
    )
    path = Path("/fake/doc.pdf")

    text_fn = MagicMock(return_value="should not be called")

    content, used_vision = extract_pdf_content_with(
        path,
        config,
        pdf_first_page_to_image_base64_fn=lambda _p: None,
        pdf_to_text_fn=text_fn,
        pdf_to_text_with_ocr_fn=lambda _p, max_pages=0, max_tokens=0, language="de": "OCR extracted text",
        llm_client=client,
    )

    assert content == "OCR extracted text"
    assert used_vision is False
    # Plain text fn should NOT be called when use_ocr is True
    text_fn.assert_not_called()


def test_extract_no_content() -> None:
    """All extraction returns empty: result is empty string, no vision used."""
    client = MagicMock()
    client.model = "test-model"
    client.complete_vision.return_value = ""

    config = RenamerConfig(
        use_ocr=False,
        vision_first=False,
        use_vision_fallback=True,
        vision_fallback_min_text_len=50,
        llm_timeout_s=30.0,
        max_tokens_for_extraction=1000,
    )
    path = Path("/fake/doc.pdf")

    content, used_vision = extract_pdf_content_with(
        path,
        config,
        pdf_first_page_to_image_base64_fn=lambda _p: "base64img",
        pdf_to_text_fn=lambda _p, max_pages=0, max_tokens=0: "",
        pdf_to_text_with_ocr_fn=lambda _p, max_pages=0, max_tokens=0, language="de": "",
        llm_client=client,
    )

    assert content == ""
    assert used_vision is False


def test_extract_logs_text_strategy(caplog: pytest.LogCaptureFixture) -> None:
    """Selected extraction strategy is logged for plain text extraction."""
    client = MagicMock()
    client.model = "test-model"
    config = RenamerConfig(use_ocr=False, vision_first=False, use_vision_fallback=False, max_tokens_for_extraction=1000)

    with caplog.at_level(logging.INFO, logger="ai_pdf_renamer.renamer_extract"):
        content, used_vision = extract_pdf_content_with(
            Path("/fake/doc.pdf"),
            config,
            pdf_first_page_to_image_base64_fn=lambda _p: None,
            pdf_to_text_fn=lambda _p, max_pages=0, max_tokens=0: "Hello world from PDF",
            pdf_to_text_with_ocr_fn=lambda _p, max_pages=0, max_tokens=0, language="de": "",
            llm_client=client,
        )

    assert content == "Hello world from PDF"
    assert used_vision is False
    assert any("ExtractionStrategy" in record.message and "text" in record.message for record in caplog.records)


def test_extract_logs_vision_fallback_strategy(caplog: pytest.LogCaptureFixture) -> None:
    """Vision fallback decisions are logged with the selected path."""
    client = MagicMock()
    client.model = "test-model"
    client.complete_vision.return_value = "Vision fallback result"
    config = RenamerConfig(
        vision_first=False,
        use_vision_fallback=True,
        vision_fallback_min_text_len=100,
        use_ocr=False,
        llm_timeout_s=30.0,
        max_tokens_for_extraction=1000,
    )

    with caplog.at_level(logging.INFO, logger="ai_pdf_renamer.renamer_extract"):
        content, used_vision = extract_pdf_content_with(
            Path("/fake/doc.pdf"),
            config,
            pdf_first_page_to_image_base64_fn=lambda _p: "base64img",
            pdf_to_text_fn=lambda _p, max_pages=0, max_tokens=0: "short",
            pdf_to_text_with_ocr_fn=lambda _p, max_pages=0, max_tokens=0, language="de": "",
            llm_client=client,
        )

    assert content == "Vision_fallback_result"
    assert used_vision is True
    assert any(
        "ExtractionStrategy" in record.message and "vision_fallback" in record.message for record in caplog.records
    )


def test_log_extraction_strategy_quotes_dynamic_values(caplog: pytest.LogCaptureFixture) -> None:
    """Dynamic values in extraction logs are quoted and escaped for safe parsing."""
    with caplog.at_level(logging.INFO, logger="ai_pdf_renamer.renamer_extract"):
        _log_extraction_strategy(
            Path('/fake/bad "name\n.pdf'),
            "vision_fallback",
            reason='line break\nand "quotes"',
        )

    message = caplog.records[-1].message
    assert "file=" in message and "strategy=" in message and "reason=" in message
    assert 'file="bad \\"name\\n.pdf"' in message
    assert 'reason="line break\\nand \\"quotes\\""' in message
