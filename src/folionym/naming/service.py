"""Filename generation pipeline orchestration."""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

from ..llm.cache import ResponseCache, get_shared_response_cache
from ..llm.models import VisionCompletionOptions
from ..settings import RenamerConfig
from .dates import extract_date_from_content
from .loaders import default_heuristic_scorer, default_stopwords
from .metadata import (
    _get_category_summary_keywords_metadata,
)
from .models import (
    FilenameGenerationRequest,
    _FilenameDependencies,
    _FilenameMetadataParts,
    _MetadataResolutionInput,
)
from .templates import (
    _final_generated_filename,
    _generate_simple_filename,
    _with_structured_metadata,
)

logger = logging.getLogger(__name__)

__all__ = ["FilenameGenerationRequest", "generate_filename"]


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


def _filename_dependencies(request: FilenameGenerationRequest) -> _FilenameDependencies:
    """Resolve caller-supplied generation dependencies without constructing transport."""
    config = request.config
    response_cache, cache_key_base = _cache_context(config, request.source_path)
    heuristic_scorer = request.heuristic_scorer or default_heuristic_scorer(config.output.naming.language)
    stopwords = request.stopwords or default_stopwords()
    llm_client = request.llm_client
    if llm_client is None:
        if config.llm.runtime.use_llm:
            raise RuntimeError("LLM-enabled filename generation requires a caller-supplied LLM client")
        else:
            # Do not construct an HTTP session for the heuristic-only path.
            # The placeholder is never invoked while
            # ``use_llm`` is false, but keeps the dependency contract explicit.
            llm_client = _DISABLED_LLM_CLIENT
    return _FilenameDependencies(
        llm_client=llm_client,
        heuristic_scorer=heuristic_scorer,
        stopwords=stopwords,
        response_cache=response_cache,
        cache_key_base=cache_key_base,
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
    dependencies = _filename_dependencies(request)
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
