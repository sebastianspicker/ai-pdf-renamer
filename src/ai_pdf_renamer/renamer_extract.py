"""Extraction helpers used by the renamer orchestration pipeline."""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Callable
from pathlib import Path

from .config import RenamerConfig
from .llm_backend import LLMClient, create_llm_client_from_config
from .llm_prompts import build_vision_filename_prompt
from .pdf_extract import (
    DEFAULT_MAX_CONTENT_TOKENS,
    pdf_first_page_to_image_base64,
    pdf_first_page_to_image_payload,
    pdf_to_text,
    pdf_to_text_with_ocr,
)
from .rename_ops import sanitize_filename_from_llm

logger = logging.getLogger(__name__)


def effective_max_tokens(config: RenamerConfig) -> int:
    """Max tokens for PDF extraction from config or env (AI_PDF_RENAMER_MAX_TOKENS)."""
    max_tok: int | None = config.max_tokens_for_extraction
    if max_tok is not None and max_tok > 0:
        return max_tok
    try:
        v = int(os.environ.get("AI_PDF_RENAMER_MAX_TOKENS", "") or 0)
        if v > 0:
            return v
    except ValueError:
        pass
    return DEFAULT_MAX_CONTENT_TOKENS


def _try_vision_extraction(
    path: Path,
    config: RenamerConfig,
    client: LLMClient,
    *,
    image_fn: Callable[..., str | dict[str, str] | None] = pdf_first_page_to_image_payload,
    prompt_fn: Callable[..., str] = build_vision_filename_prompt,
    sanitize_fn: Callable[..., str] = sanitize_filename_from_llm,
) -> str | None:
    """Try vision extraction on first page. Returns sanitized text or None on failure."""
    image_data = image_fn(path)
    if not image_data:
        return None
    if isinstance(image_data, dict):
        image_b64 = image_data.get("image_b64", "")
        image_mime_type = image_data.get("mime_type", "image/jpeg")
    else:
        image_b64 = image_data
        image_mime_type = "image/jpeg"
    if not image_b64:
        return None
    model = config.vision_model or client.model
    prompt = prompt_fn(config.language)
    timeout = (config.llm_timeout_s or 60.0) * 2
    vision_text = client.complete_vision(
        image_b64,
        prompt,
        model=model,
        image_mime_type=image_mime_type,
        timeout_s=max(60.0, timeout),
    )
    if vision_text:
        return sanitize_fn(vision_text)
    return None


def _log_extraction_strategy(path: Path, strategy: str, **details: object) -> None:
    """Emit a consistent extraction-strategy log entry for user-visible debugging."""
    serialized_details = " ".join(
        f"{key}={json.dumps(str(value), ensure_ascii=False)}" for key, value in details.items()
    )
    message = (
        "ExtractionStrategy "
        f"file={json.dumps(path.name, ensure_ascii=False)} "
        f"strategy={json.dumps(strategy, ensure_ascii=False)}"
    )
    if serialized_details:
        message = f"{message} {serialized_details}"
    logger.info(message)


def _extract_primary_content(
    path: Path,
    config: RenamerConfig,
    *,
    pdf_to_text_fn: Callable[..., str],
    pdf_to_text_with_ocr_fn: Callable[..., str],
) -> tuple[str, str]:
    """Run the primary extraction path before any optional vision fallback."""
    if config.use_ocr:
        _log_extraction_strategy(path, "ocr", reason="config.use_ocr")
        return (
            pdf_to_text_with_ocr_fn(
                path,
                max_pages=config.max_pages_for_extraction or 0,
                max_tokens=effective_max_tokens(config),
                language=config.language,
            ),
            "ocr",
        )

    _log_extraction_strategy(path, "text", reason="default")
    return (
        pdf_to_text_fn(
            path,
            max_pages=config.max_pages_for_extraction or 0,
            max_tokens=effective_max_tokens(config),
        ),
        "text",
    )


def extract_pdf_content(path: Path, config: RenamerConfig) -> tuple[str, bool]:
    return extract_pdf_content_with(
        path,
        config,
        pdf_first_page_to_image_base64_fn=pdf_first_page_to_image_base64,
        pdf_to_text_fn=pdf_to_text,
        pdf_to_text_with_ocr_fn=pdf_to_text_with_ocr,
    )


def extract_pdf_content_with(
    path: Path,
    config: RenamerConfig,
    *,
    pdf_first_page_to_image_base64_fn: Callable[..., str | dict[str, str] | None] = pdf_first_page_to_image_payload,
    pdf_to_text_fn: Callable[..., str] = pdf_to_text,
    pdf_to_text_with_ocr_fn: Callable[..., str] = pdf_to_text_with_ocr,
    llm_client: LLMClient | None = None,
) -> tuple[str, bool]:
    """Extract content using the configured strategy order.

    Strategy order:
    1. `vision_first` if enabled
    2. primary text extraction (`text` or `ocr`)
    3. `vision_fallback` when extracted text is too short

    Returns `(content, used_vision)`.
    `used_vision` is `True` when any vision-based extraction path was selected,
    including both `vision_first` and `vision_fallback`.
    """
    client = llm_client
    tried_vision = False

    if config.vision_first:
        if client is None:
            client = create_llm_client_from_config(config)
        tried_vision = True
        _log_extraction_strategy(path, "vision_first", outcome="attempt")
        result = _try_vision_extraction(
            path,
            config,
            client,
            image_fn=pdf_first_page_to_image_base64_fn,
        )
        if result:
            _log_extraction_strategy(path, "vision_first", outcome="selected")
            return (result, True)
        _log_extraction_strategy(path, "vision_first", outcome="fall_back_to_primary")

    content, primary_strategy = _extract_primary_content(
        path,
        config,
        pdf_to_text_fn=pdf_to_text_fn,
        pdf_to_text_with_ocr_fn=pdf_to_text_with_ocr_fn,
    )

    content_length = len(content.strip())
    if not tried_vision and config.use_vision_fallback and content_length < config.vision_fallback_min_text_len:
        if client is None:
            client = create_llm_client_from_config(config)
        _log_extraction_strategy(
            path,
            "vision_fallback",
            outcome="attempt",
            text_length=content_length,
            threshold=config.vision_fallback_min_text_len,
        )
        result = _try_vision_extraction(
            path,
            config,
            client,
            image_fn=pdf_first_page_to_image_base64_fn,
        )
        if result:
            _log_extraction_strategy(
                path,
                "vision_fallback",
                outcome="selected",
                text_length=content_length,
                threshold=config.vision_fallback_min_text_len,
            )
            return (result, True)
        _log_extraction_strategy(path, primary_strategy, outcome="selected_after_failed_vision_fallback")
        return (content, False)

    if tried_vision and config.use_vision_fallback and content_length < config.vision_fallback_min_text_len:
        _log_extraction_strategy(
            path,
            primary_strategy,
            outcome="selected_after_failed_vision_first",
            text_length=content_length,
            threshold=config.vision_fallback_min_text_len,
        )
        return (content, False)

    _log_extraction_strategy(path, primary_strategy, outcome="selected", text_length=content_length)
    return (content, False)
