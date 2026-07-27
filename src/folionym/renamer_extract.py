"""Extraction helpers used by the renamer orchestration pipeline."""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .config import RenamerConfig
from .llm_backend import LLMClient, VisionCompletionOptions, create_llm_client_from_config
from .llm_prompts import build_vision_filename_prompt
from .pdf_extract import (
    DEFAULT_MAX_CONTENT_TOKENS,
    pdf_first_page_to_image_base64,
    pdf_first_page_to_image_payload,
    pdf_to_text,
    pdf_to_text_with_ocr,
)
from .rename_ops import sanitize_filename_from_llm
from .renamer_files import reject_source_symlink

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExtractionFunctions:
    """Injectable extraction functions used by strategy orchestration and tests."""

    image_fn: Callable[..., str | dict[str, str] | None]
    pdf_to_text_fn: Callable[..., str]
    pdf_to_text_with_ocr_fn: Callable[..., str]
    prompt_fn: Callable[..., str] = build_vision_filename_prompt
    sanitize_fn: Callable[..., str] = sanitize_filename_from_llm


@dataclass(frozen=True)
class VisionExtractionRequest:
    """Inputs required for one first-page vision extraction request."""

    path: Path
    config: RenamerConfig
    client: LLMClient
    extraction_fns: ExtractionFunctions


@dataclass(frozen=True)
class VisionAttempt:
    """Outcome of vision-first extraction: content, reusable client, and attempted state."""

    content: str | None
    client: LLMClient | None
    attempted: bool


def effective_max_tokens(config: RenamerConfig) -> int:
    """Max tokens for PDF extraction from config or env (FOLIONYM_MAX_TOKENS)."""
    max_tok = config.extraction.max_tokens_for_extraction
    if max_tok is not None and max_tok > 0:
        return max_tok
    try:
        v = int(os.environ.get("FOLIONYM_MAX_TOKENS", "") or 0)
        if v > 0:
            return v
    except ValueError:
        pass
    return DEFAULT_MAX_CONTENT_TOKENS


def _try_vision_extraction(request: VisionExtractionRequest) -> str | None:
    """Try vision extraction on first page. Returns sanitized text or None on failure."""
    image_data = request.extraction_fns.image_fn(request.path)
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
    model = request.config.llm.vision.vision_model or request.client.model
    prompt = request.extraction_fns.prompt_fn(request.config.output.naming.language)
    timeout = (request.config.llm.backend.llm_timeout_s or 60.0) * 2
    vision_text = request.client.complete_vision(
        image_b64,
        prompt,
        VisionCompletionOptions(
            model=model,
            image_mime_type=image_mime_type,
            timeout_s=max(60.0, timeout),
        ),
    )
    if vision_text:
        return request.extraction_fns.sanitize_fn(vision_text)
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
    extraction = config.extraction
    if extraction.use_ocr:
        _log_extraction_strategy(path, "ocr", reason="config.use_ocr")
        return (
            pdf_to_text_with_ocr_fn(
                path,
                max_pages=extraction.max_pages_for_extraction or 0,
                max_tokens=effective_max_tokens(config),
                language=config.output.naming.language,
            ),
            "ocr",
        )

    _log_extraction_strategy(path, "text", reason="default")
    return (
        pdf_to_text_fn(
            path,
            max_pages=extraction.max_pages_for_extraction or 0,
            max_tokens=effective_max_tokens(config),
        ),
        "text",
    )


def extract_pdf_content(path: Path, config: RenamerConfig) -> tuple[str, bool]:
    """Extract PDF content using the production text, OCR, and vision adapters."""
    return extract_pdf_content_with(
        path,
        config,
        extraction_fns=ExtractionFunctions(
            image_fn=pdf_first_page_to_image_base64,
            pdf_to_text_fn=pdf_to_text,
            pdf_to_text_with_ocr_fn=pdf_to_text_with_ocr,
        ),
    )


def extract_pdf_content_with(
    path: Path,
    config: RenamerConfig,
    *,
    extraction_fns: ExtractionFunctions | None = None,
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
    reject_source_symlink(path)
    extraction_fns = extraction_fns or ExtractionFunctions(
        image_fn=pdf_first_page_to_image_payload,
        pdf_to_text_fn=pdf_to_text,
        pdf_to_text_with_ocr_fn=pdf_to_text_with_ocr,
    )
    vision_first_result = _run_vision_first(path, config, llm_client, extraction_fns)
    if vision_first_result.content is not None:
        return (vision_first_result.content, True)

    return _extract_primary_or_fallback(
        path,
        config,
        extraction_fns,
        client=vision_first_result.client,
        tried_vision=vision_first_result.attempted,
    )


def _run_vision_first(
    path: Path,
    config: RenamerConfig,
    client: LLMClient | None,
    extraction_fns: ExtractionFunctions,
) -> VisionAttempt:
    """Run vision-first when enabled and return its content, client, and attempted state."""
    if not config.llm.vision.vision_first:
        return VisionAttempt(None, client, False)
    client = _ensure_llm_client(config, client)
    _log_extraction_strategy(path, "vision_first", outcome="attempt")
    content = _try_vision_extraction(VisionExtractionRequest(path, config, client, extraction_fns))
    if content:
        _log_extraction_strategy(path, "vision_first", outcome="selected")
        return VisionAttempt(content, client, True)
    _log_extraction_strategy(path, "vision_first", outcome="fall_back_to_primary")
    return VisionAttempt(None, client, True)


def _extract_primary_or_fallback(
    path: Path,
    config: RenamerConfig,
    extraction_fns: ExtractionFunctions,
    *,
    client: LLMClient | None,
    tried_vision: bool,
) -> tuple[str, bool]:
    """Run primary text or OCR extraction, then optional vision fallback."""
    content, primary_strategy = _extract_primary_content(
        path,
        config,
        pdf_to_text_fn=extraction_fns.pdf_to_text_fn,
        pdf_to_text_with_ocr_fn=extraction_fns.pdf_to_text_with_ocr_fn,
    )
    content_length = len(content.strip())
    if _should_try_vision_fallback(config, tried_vision, content_length):
        fallback = _run_vision_fallback(path, config, extraction_fns, client, content_length=content_length)
        if fallback is not None:
            return (fallback, True)
        _log_extraction_strategy(path, primary_strategy, outcome="selected_after_failed_vision_fallback")
        return (content, False)
    _log_primary_selection(path, config, primary_strategy, content_length=content_length, tried_vision=tried_vision)
    return (content, False)


def _ensure_llm_client(config: RenamerConfig, client: LLMClient | None) -> LLMClient:
    """Reuse the supplied LLM client or create one; this helper does not close a new client."""
    if client is not None:
        return client
    return create_llm_client_from_config(config)


def _should_try_vision_fallback(config: RenamerConfig, tried_vision: bool, content_length: int) -> bool:
    """Return whether vision fallback is enabled, untried, and below the text threshold."""
    vision = config.llm.vision
    return not tried_vision and vision.use_vision_fallback and content_length < vision.vision_fallback_min_text_len


def _run_vision_fallback(
    path: Path,
    config: RenamerConfig,
    extraction_fns: ExtractionFunctions,
    client: LLMClient | None,
    *,
    content_length: int,
) -> str | None:
    """Attempt vision fallback and log its attempt and selection; propagate request failures."""
    client = _ensure_llm_client(config, client)
    _log_vision_fallback(path, config, outcome="attempt", text_length=content_length)
    content = _try_vision_extraction(VisionExtractionRequest(path, config, client, extraction_fns))
    if content:
        _log_vision_fallback(path, config, outcome="selected", text_length=content_length)
        return content
    return None


def _log_vision_fallback(path: Path, config: RenamerConfig, *, outcome: str, text_length: int) -> None:
    """Log vision fallback with the context needed for operational diagnosis."""
    threshold = config.llm.vision.vision_fallback_min_text_len
    _log_extraction_strategy(
        path,
        "vision_fallback",
        outcome=outcome,
        text_length=text_length,
        threshold=threshold,
    )


def _log_primary_selection(
    path: Path,
    config: RenamerConfig,
    primary_strategy: str,
    *,
    content_length: int,
    tried_vision: bool,
) -> None:
    """Log primary selection with the context needed for operational diagnosis."""
    vision = config.llm.vision
    if tried_vision and vision.use_vision_fallback and content_length < vision.vision_fallback_min_text_len:
        _log_extraction_strategy(
            path,
            primary_strategy,
            outcome="selected_after_failed_vision_first",
            text_length=content_length,
            threshold=vision.vision_fallback_min_text_len,
        )
        return
    _log_extraction_strategy(path, primary_strategy, outcome="selected", text_length=content_length)
