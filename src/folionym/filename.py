"""Filename generation pipeline orchestration."""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

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
from .llm_backend import LLMClient, VisionCompletionOptions, create_llm_client_from_config
from .loaders import default_heuristic_scorer, default_stopwords
from .text_utils import extract_date_from_content

logger = logging.getLogger(__name__)

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


class _DisabledLLMClient:
    """Non-networking placeholder used when the LLM features are disabled."""

    model = "disabled"
    base_url = ""

    def complete(
        self,
        prompt: str,
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        response_format: dict[str, str] | None = None,
    ) -> str:
        """Return an empty completion without contacting an LLM backend."""
        return ""

    def complete_vision(
        self,
        image_b64: str,
        prompt: str,
        options: VisionCompletionOptions | None = None,
    ) -> str:
        """Return an empty vision completion without contacting an LLM backend."""
        return ""

    def close(self) -> None:
        """No-op; this placeholder owns no resources."""
        return None


_DISABLED_LLM_CLIENT = _DisabledLLMClient()


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
        date_locale=config.output.naming.date_locale,
        prefer_leading_chars=config.output.naming.date_prefer_leading_chars or 0,
        pdf_metadata=pdf_metadata if config.extraction.use_pdf_metadata_for_date else None,
    )
    return content_date.replace("-", "")


def _cache_context(config: RenamerConfig, source_path: Path | None) -> tuple[ResponseCache | None, str | None]:
    """Resolve the response cache and source-file cache key."""
    response_cache = get_shared_response_cache(config.llm.content.cache_dir) if config.llm.content.use_cache else None
    cache_key_base = None
    if response_cache is not None and source_path is not None and source_path.exists():
        try:
            cache_key_base = ResponseCache.build_file_key(source_path)
        except OSError as exc:
            # A file changed while its cache key was being derived. Continue
            # without persistent response caching rather than couple naming to
            # a stale identity.
            response_cache = None
            logger.warning("Skipping persistent response cache for %s: %s", source_path, exc)
    return response_cache, cache_key_base


def _filename_dependencies(request: FilenameGenerationRequest) -> tuple[_FilenameDependencies, LLMClient | None]:
    """Resolve generation dependencies and return any newly owned LLM client."""
    config = request.config
    response_cache, cache_key_base = _cache_context(config, request.source_path)
    heuristic_scorer = request.heuristic_scorer or default_heuristic_scorer(config.output.naming.language)
    stopwords = request.stopwords or default_stopwords()
    owned_llm_client: LLMClient | None = None
    llm_client = request.llm_client
    if llm_client is None:
        if config.llm.runtime.use_llm:
            owned_llm_client = create_llm_client_from_config(config)
            llm_client = owned_llm_client
        else:
            # Do not construct an HTTP session for the heuristic-only path.
            # The placeholder is never invoked while
            # ``use_llm`` is false, but keeps the dependency contract explicit.
            llm_client = _DISABLED_LLM_CLIENT
    return (
        _FilenameDependencies(
            llm_client=llm_client,
            heuristic_scorer=heuristic_scorer,
            stopwords=stopwords,
            response_cache=response_cache,
            cache_key_base=cache_key_base,
        ),
        owned_llm_client,
    )


def _filename_metadata_parts(
    pdf_content: str,
    request: FilenameGenerationRequest,
    dependencies: _FilenameDependencies,
) -> _FilenameMetadataParts:
    """Resolve and clean category, keyword, summary, and metadata parts."""
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
    request: FilenameGenerationRequest,
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
    config = request.config
    dependencies, owned_llm_client = _filename_dependencies(request)
    try:
        date_str = _get_date_str(pdf_content, config, request.today, request.pdf_metadata)
        if config.llm.runtime.simple_naming_mode:
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
    finally:
        if owned_llm_client is not None:
            owned_llm_client.close()
