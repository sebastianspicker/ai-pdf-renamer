"""Extraction helpers used by the renamer orchestration pipeline."""

from __future__ import annotations

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
    pdf_to_text,
    pdf_to_text_with_ocr,
)
from .rename_ops import sanitize_filename_from_llm

logger = logging.getLogger(__name__)


def effective_max_tokens(config: RenamerConfig) -> int:
    """Max tokens for PDF extraction from config or env (AI_PDF_RENAMER_MAX_TOKENS)."""
    if config.max_tokens_for_extraction is not None and config.max_tokens_for_extraction > 0:
        return config.max_tokens_for_extraction
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
    image_fn: Callable[..., str | None] = pdf_first_page_to_image_base64,
    prompt_fn: Callable[..., str] = build_vision_filename_prompt,
    sanitize_fn: Callable[..., str] = sanitize_filename_from_llm,
) -> str | None:
    """Try vision extraction on first page. Returns sanitized text or None on failure."""
    image_b64 = image_fn(path)
    if not image_b64:
        return None
    model = config.vision_model or client.model
    prompt = prompt_fn(config.language)
    timeout = (config.llm_timeout_s or 60.0) * 2
    vision_text = client.complete_vision(
        image_b64,
        prompt,
        model=model,
        timeout_s=max(60.0, timeout),
    )
    if vision_text:
        return sanitize_fn(vision_text)
    return None


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
    pdf_first_page_to_image_base64_fn: Callable[..., str | None] = pdf_first_page_to_image_base64,
    pdf_to_text_fn: Callable[..., str] = pdf_to_text,
    pdf_to_text_with_ocr_fn: Callable[..., str] = pdf_to_text_with_ocr,
    llm_client: LLMClient | None = None,
) -> tuple[str, bool]:
    """Extract text or optional vision fallback output. Returns (content, used_vision_fallback)."""
    client = llm_client or create_llm_client_from_config(config)

    if config.vision_first:
        result = _try_vision_extraction(
            path,
            config,
            client,
            image_fn=pdf_first_page_to_image_base64_fn,
        )
        if result:
            return (result, True)

    if config.use_ocr:
        content = pdf_to_text_with_ocr_fn(
            path,
            max_pages=config.max_pages_for_extraction or 0,
            max_tokens=effective_max_tokens(config),
            language=config.language,
        )
    else:
        content = pdf_to_text_fn(
            path,
            max_pages=config.max_pages_for_extraction or 0,
            max_tokens=effective_max_tokens(config),
        )

    if config.use_vision_fallback and len(content.strip()) < config.vision_fallback_min_text_len:
        result = _try_vision_extraction(
            path,
            config,
            client,
            image_fn=pdf_first_page_to_image_base64_fn,
        )
        if result:
            logger.info(
                "Used vision fallback for %s (text length %d < %d)",
                path.name,
                len(content.strip()),
                config.vision_fallback_min_text_len,
            )
            return (result, True)

    return (content, False)
