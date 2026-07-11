"""Filename generation pipeline public orchestration.

This module keeps the public filename API stable while the detailed category,
LLM, template, and string-building helpers live in smaller focused modules.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

from .cache import ResponseCache, get_shared_response_cache
from .config import RenamerConfig
from .filename_builders import _apply_filename_template as _apply_filename_template
from .filename_builders import _build_filename_str as _build_filename_str
from .filename_builders import _build_timestamp_fallback_filename as _build_timestamp_fallback_filename
from .filename_builders import _filename_sep as _filename_sep
from .filename_builders import (
    _final_generated_filename,
    _generate_simple_filename,
    _with_structured_metadata,
)
from .filename_builders import _should_use_timestamp_fallback as _should_use_timestamp_fallback
from .filename_builders import _truncate_filename_to_max_chars as _truncate_filename_to_max_chars
from .filename_llm_metadata import _get_llm_summary_and_keywords as _get_llm_summary_and_keywords
from .filename_metadata import _build_metadata_tokens as _build_metadata_tokens
from .filename_metadata import (
    _get_category_summary_keywords_metadata,
)
from .filename_metadata import _heuristic_text_for_category as _heuristic_text_for_category
from .filename_metadata import _resolve_category_with_llm as _resolve_category_with_llm
from .filename_models import (
    FilenameGenerationRequest,
    _CategoryResolutionInput,
    _FilenameDependencies,
    _FilenameMetadataParts,
    _FilenameTemplateInput,
    _MetadataResolutionInput,
)
from .llm_backend import create_llm_client_from_config
from .loaders import default_heuristic_scorer, default_stopwords
from .text_utils import extract_date_from_content

__all__ = [
    "FilenameGenerationRequest",
    "_CategoryResolutionInput",
    "_FilenameTemplateInput",
    "_apply_filename_template",
    "_build_filename_str",
    "_build_metadata_tokens",
    "_build_timestamp_fallback_filename",
    "_filename_sep",
    "_get_date_str",
    "_get_llm_summary_and_keywords",
    "_heuristic_text_for_category",
    "_resolve_category_with_llm",
    "_should_use_timestamp_fallback",
    "_truncate_filename_to_max_chars",
    "generate_filename",
]

_FILENAME_REQUEST_LEGACY_KEYS = {
    "llm_client",
    "heuristic_scorer",
    "stopwords",
    "override_category",
    "today",
    "pdf_metadata",
    "rules",
    "source_path",
}


def _get_date_str(
    pdf_content: str,
    config: RenamerConfig,
    today: date | None = None,
    pdf_metadata: dict[str, object] | None = None,
) -> str:
    """Extract date from content and optional PDF metadata fallback as YYYYMMDD."""
    content_date = extract_date_from_content(
        pdf_content,
        today=today,
        date_locale=config.date_locale,
        prefer_leading_chars=config.date_prefer_leading_chars or 0,
        pdf_metadata=pdf_metadata if config.use_pdf_metadata_for_date else None,
    )
    return content_date.replace("-", "")


def _cache_context(config: RenamerConfig, source_path: Path | None) -> tuple[ResponseCache | None, str | None]:
    response_cache = get_shared_response_cache(config.cache_dir) if config.use_cache else None
    cache_key_base = None
    if response_cache is not None and source_path is not None and source_path.exists():
        cache_key_base = ResponseCache.build_file_key(source_path)
    return response_cache, cache_key_base


def _coerce_generation_request(
    request: FilenameGenerationRequest | None,
    legacy_request: dict[str, Any],
) -> FilenameGenerationRequest:
    if request is not None:
        if legacy_request:
            raise TypeError("generate_filename accepts either a request object or legacy keyword arguments, not both")
        return request
    config = legacy_request.pop("config", None)
    if not isinstance(config, RenamerConfig):
        raise TypeError("generate_filename requires FilenameGenerationRequest or config=RenamerConfig")
    unexpected = set(legacy_request) - _FILENAME_REQUEST_LEGACY_KEYS
    if unexpected:
        unexpected_names = ", ".join(sorted(unexpected))
        raise TypeError(f"generate_filename got unexpected keyword argument(s): {unexpected_names}")
    return FilenameGenerationRequest(config=config, **legacy_request)


def _filename_dependencies(request: FilenameGenerationRequest) -> _FilenameDependencies:
    config = request.config
    response_cache, cache_key_base = _cache_context(config, request.source_path)
    return _FilenameDependencies(
        llm_client=request.llm_client or create_llm_client_from_config(config),
        heuristic_scorer=request.heuristic_scorer or default_heuristic_scorer(config.language),
        stopwords=request.stopwords or default_stopwords(),
        response_cache=response_cache,
        cache_key_base=cache_key_base,
    )


def _filename_metadata_parts(
    pdf_content: str,
    request: FilenameGenerationRequest,
    dependencies: _FilenameDependencies,
) -> _FilenameMetadataParts:
    category_for_filename, category_clean, keyword_clean, summary_clean, metadata = (
        _get_category_summary_keywords_metadata(
            _MetadataResolutionInput(
                pdf_content=pdf_content,
                override_category=request.override_category,
                rules=request.rules,
                response_cache=dependencies.response_cache,
                cache_key_base=dependencies.cache_key_base,
            ),
            request.config,
            dependencies.llm_client,
            dependencies.heuristic_scorer,
            dependencies.stopwords,
        )
    )
    return _FilenameMetadataParts(category_for_filename, category_clean, keyword_clean, summary_clean, metadata)


def generate_filename(
    pdf_content: str,
    request: FilenameGenerationRequest | None = None,
    **legacy_request: Any,
) -> tuple[str, dict[str, object]]:
    """
    Constructs the final filename and metadata:
    - date (YYYYMMDD), optionally from PDF metadata when content has no date
    - optional project
    - category (heuristic + optional LLM)
    - keywords (<=3)
    - short summary tokens (<=5)
    - optional version
    """
    if pdf_content is None or not isinstance(pdf_content, str):
        raise ValueError("pdf_content must be a non-None string")
    request = _coerce_generation_request(request, legacy_request)
    config = request.config
    dependencies = _filename_dependencies(request)

    date_str = _get_date_str(pdf_content, config, request.today, request.pdf_metadata)
    if config.simple_naming_mode:
        return _generate_simple_filename(
            pdf_content,
            request,
            date_str,
            dependencies,
        )
    parts = _filename_metadata_parts(pdf_content, request, dependencies)
    structured_fields = _with_structured_metadata(parts.metadata, pdf_content, config)
    filename = _final_generated_filename(date_str, parts, structured_fields, config)
    return filename, parts.metadata
