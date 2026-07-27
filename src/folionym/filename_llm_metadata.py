"""LLM summary, keyword, category-analysis, and summary-token calls for filenames."""

from __future__ import annotations

from dataclasses import dataclass

from .cache import ResponseCache
from .config import RenamerConfig
from .filename_models import _LlmResult, _LlmSummaryInput
from .heuristic_scoring import TopNOptions
from .heuristics import HeuristicScorer
from .llm import get_document_analysis, get_document_keywords, get_document_summary, get_final_summary_tokens
from .llm_backend import LLMClient
from .llm_options import (
    AnalysisGuidance,
    AnalysisOptions,
    FinalSummaryOptions,
    KeywordsOptions,
    LlmCacheOptions,
    LlmContentLimits,
    LlmPromptOptions,
    SummaryOptions,
)
from .text_utils import split_to_tokens


@dataclass(frozen=True)
class _FinalSummaryTokenInput:
    """Inputs used to derive final summary tokens."""

    summary: str
    keywords: list[str]
    category: str
    precomputed_summary_tokens: list[str] | tuple[str, ...] | None = None
    response_cache: ResponseCache | None = None
    cache_key_base: str | None = None


def _suggested_categories_for_llm(
    heuristic_text: str,
    config: RenamerConfig,
    heuristic_scorer: HeuristicScorer,
) -> list[str]:
    """Return the highest-scoring heuristic categories as LLM suggestions."""
    top_n = heuristic_scorer.top_n_categories(
        heuristic_text,
        TopNOptions(
            n=config.heuristic.skip.heuristic_suggestions_top_n,
            language=config.output.naming.language,
            max_score_per_category=config.heuristic.scoring.max_score_per_category,
            title_weight_region=config.heuristic.scoring.title_weight_region,
            title_weight_factor=config.heuristic.scoring.title_weight_factor,
        ),
    )
    return [c for c in top_n if c and c != "unknown"]


def _allowed_and_suggested_for_analysis(
    request: _LlmSummaryInput,
    config: RenamerConfig,
    heuristic_scorer: HeuristicScorer,
) -> tuple[list[str] | None, list[str] | None]:
    """Select allowed or suggested categories for single-call analysis."""
    if request.override_category is not None:
        return (None, None)
    if request.rules is not None and request.rules.allowed_categories:
        return ([c for c in request.rules.allowed_categories if c and c.strip()], None)
    if config.heuristic.category.use_constrained_llm_category:
        return (list(heuristic_scorer.all_categories()), None)
    return (None, _suggested_categories_for_llm(request.heuristic_text, config, heuristic_scorer))


def _get_single_call_analysis(
    request: _LlmSummaryInput,
    config: RenamerConfig,
    llm_client: LLMClient,
    heuristic_scorer: HeuristicScorer,
    effective_max_content_chars: int | None,
) -> _LlmResult:
    """Run one LLM call for summary, keywords, category, and summary tokens."""
    allowed, suggested = _allowed_and_suggested_for_analysis(request, config, heuristic_scorer)
    analysis = get_document_analysis(
        llm_client,
        request.pdf_content,
        AnalysisOptions(
            prompt=LlmPromptOptions(
                language=config.output.naming.language,
                lenient_json=config.llm.runtime.lenient_llm_json,
            ),
            limits=LlmContentLimits(
                max_content_chars=effective_max_content_chars,
                max_content_tokens=config.llm.content.max_content_tokens,
            ),
            guidance=AnalysisGuidance(
                suggested_doc_type=request.suggested_doc_type,
                allowed_categories=allowed,
                suggested_categories=suggested if not allowed else None,
            ),
            json_mode=config.llm.runtime.llm_json_mode,
            cache_options=LlmCacheOptions(request.response_cache, request.cache_key_base),
        ),
    )
    return _LlmResult(analysis.summary, analysis.keywords or [], analysis.category, analysis.final_summary_tokens)


def _get_multi_call_summary_keywords(
    request: _LlmSummaryInput,
    config: RenamerConfig,
    llm_client: LLMClient,
    effective_max_content_chars: int | None,
) -> _LlmResult:
    """Run separate LLM calls for summary and keywords."""
    summary = get_document_summary(
        llm_client,
        request.pdf_content,
        SummaryOptions(
            prompt=LlmPromptOptions(
                language=config.output.naming.language,
                lenient_json=config.llm.runtime.lenient_llm_json,
            ),
            limits=LlmContentLimits(
                max_content_chars=effective_max_content_chars,
                max_content_tokens=config.llm.content.max_content_tokens,
            ),
            suggested_doc_type=request.suggested_doc_type,
            cache_options=LlmCacheOptions(request.response_cache, request.cache_key_base),
        ),
    )
    raw_keywords: tuple[str, ...] | list[str] = (
        get_document_keywords(
            llm_client,
            summary,
            KeywordsOptions(
                language=config.output.naming.language,
                suggested_category=request.cat_heur if request.cat_heur != "unknown" else None,
                lenient_json=config.llm.runtime.lenient_llm_json,
                cache=request.response_cache,
                cache_key_base=request.cache_key_base,
            ),
        )
        or []
    )
    return _LlmResult(summary, raw_keywords)


def _get_llm_summary_and_keywords(
    request: _LlmSummaryInput,
    config: RenamerConfig,
    llm_client: LLMClient,
    heuristic_scorer: HeuristicScorer,
) -> tuple[str, list[str] | tuple[str, ...], str | None, list[str] | tuple[str, ...] | None]:
    """Run LLM to get summary, keywords, and optionally a precomputed category."""
    effective_max_content_chars = config.llm.content.max_content_chars or config.llm.content.max_context_chars
    if config.llm.runtime.use_single_llm_call:
        result = _get_single_call_analysis(
            request,
            config,
            llm_client,
            heuristic_scorer,
            effective_max_content_chars,
        )
    else:
        result = _get_multi_call_summary_keywords(request, config, llm_client, effective_max_content_chars)
    return (
        result.summary,
        result.raw_keywords,
        result.precomputed_category,
        result.precomputed_summary_tokens,
    )


def _resolve_final_summary_tokens(
    config: RenamerConfig,
    llm_client: LLMClient,
    request: _FinalSummaryTokenInput,
) -> list[str]:
    """Resolve final summary tokens from LLM analysis or a separate LLM call."""
    if config.llm.runtime.use_single_llm_call:
        if request.precomputed_summary_tokens:
            return list(request.precomputed_summary_tokens)
        if request.summary and request.summary != "na":
            return split_to_tokens(request.summary)[:5]
        return []
    return (
        get_final_summary_tokens(
            llm_client,
            summary=request.summary,
            keywords=request.keywords,
            category=request.category,
            options=FinalSummaryOptions(
                language=config.output.naming.language,
                lenient_json=config.llm.runtime.lenient_llm_json,
                cache=request.response_cache,
                cache_key_base=request.cache_key_base,
            ),
        )
        or []
    )
