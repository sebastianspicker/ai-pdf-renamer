"""Extraction helpers used by the renamer orchestration pipeline."""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from pathlib import Path

from .config import RenamerConfig
from .filename import _llm_client_from_config
from .llm import complete_vision
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


def extract_pdf_content(path: Path, config: RenamerConfig) -> tuple[str, bool]:
    return extract_pdf_content_with(
        path,
        config,
        pdf_first_page_to_image_base64_fn=pdf_first_page_to_image_base64,
        llm_client_from_config_fn=_llm_client_from_config,
        build_vision_prompt_fn=build_vision_filename_prompt,
        complete_vision_fn=complete_vision,
        sanitize_filename_from_llm_fn=sanitize_filename_from_llm,
        pdf_to_text_fn=pdf_to_text,
        pdf_to_text_with_ocr_fn=pdf_to_text_with_ocr,
    )


def extract_pdf_content_with(
    path: Path,
    config: RenamerConfig,
    *,
    pdf_first_page_to_image_base64_fn: Callable[..., str | None],
    llm_client_from_config_fn: Callable[..., object],
    build_vision_prompt_fn: Callable[..., str],
    complete_vision_fn: Callable[..., str],
    sanitize_filename_from_llm_fn: Callable[..., str],
    pdf_to_text_fn: Callable[..., str],
    pdf_to_text_with_ocr_fn: Callable[..., str],
) -> tuple[str, bool]:
    """Extract text or optional vision fallback output. Returns (content, used_vision_fallback)."""
    if getattr(config, "vision_first", False):
        image_b64 = pdf_first_page_to_image_base64_fn(path)
        if image_b64:
            client = llm_client_from_config_fn(config)
            model = getattr(config, "vision_model", None) or client.model
            prompt = build_vision_prompt_fn(config.language)
            timeout = getattr(config, "llm_timeout_s", None) or 60.0
            vision_text = complete_vision_fn(
                client.base_url,
                model,
                image_b64,
                prompt,
                timeout_s=max(60.0, timeout * 2),
            )
            if vision_text:
                vision_text = sanitize_filename_from_llm_fn(vision_text)
                return (vision_text, True)

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

    min_len = getattr(config, "vision_fallback_min_text_len", 50)
    if getattr(config, "use_vision_fallback", False) and len(content.strip()) < min_len:
        image_b64 = pdf_first_page_to_image_base64_fn(path)
        if image_b64:
            client = llm_client_from_config_fn(config)
            model = getattr(config, "vision_model", None) or client.model
            prompt = build_vision_prompt_fn(config.language)
            timeout = getattr(config, "llm_timeout_s", None) or 60.0
            vision_text = complete_vision_fn(
                client.base_url,
                model,
                image_b64,
                prompt,
                timeout_s=max(60.0, timeout * 2),
            )
            if vision_text:
                vision_text = sanitize_filename_from_llm_fn(vision_text)
                logger.info(
                    "Used vision fallback for %s (text length %d < %d)",
                    path.name,
                    len(content.strip()),
                    min_len,
                )
                return (vision_text, True)

    return (content, False)
