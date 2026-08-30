"""Category, LLM summary, keyword, and metadata resolution for filenames."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from ..llm import get_document_category
from ..llm.models import CategoryOptions
from ..llm.protocol import LLMClient
from ..settings import RenamerConfig
from .heuristics import (
    CategoryCombineOptions,
    CategoryCombineParams,
    HeuristicScorer,
    combine_categories,
    normalize_llm_category,
)
from .llm_metadata import (
    _FinalSummaryTokenInput,
    _get_llm_summary_and_keywords,
    _resolve_final_summary_tokens,
    _suggested_categories_for_llm,
)
from .models import (
    _CategoryContext,
    _CategoryResolutionInput,
    _FilenameCacheContext,
    _HeuristicCategorySignal,
    _LlmResult,
    _LlmSummaryInput,
    _MetadataResolutionInput,
)
from .rules import ProcessingRules
from .scoring import ConfidenceOptions, TopNOptions
from .tokens import Stopwords, clean_token, normalize_keywords, split_to_tokens, subtract_tokens

logger = logging.getLogger(__name__)

_HEURISTIC_SUGGESTED_DOC_TYPE_MIN_SCORE = 0.25


@dataclass(frozen=True)
class _MetadataRuntime:
    """Shared services used while resolving filename metadata."""

    config: RenamerConfig
    llm_client: LLMClient
    heuristic_scorer: HeuristicScorer


def _heuristic_text_for_category(pdf_content: str, config: RenamerConfig) -> str:
    """Return the slice of PDF content used for heuristic category scoring."""
    window = config.heuristic.window
    if window.heuristic_leading_chars > 0:
        return pdf_content[: window.heuristic_leading_chars]
    if window.heuristic_long_doc_chars_threshold > 0 and len(pdf_content) >= window.heuristic_long_doc_chars_threshold:
        return pdf_content[: window.heuristic_long_doc_leading_chars]
    return pdf_content


def _resolve_heuristic_category(
    heuristic_text: str,
    config: RenamerConfig,
    heuristic_scorer: HeuristicScorer,
) -> tuple[str, float, str, float, float, str | None]:
    """Run heuristic scoring; return (cat_heur, score, runner_up_cat, runner_up_score, gap, suggested_doc_type)."""
    scoring = config.heuristic.scoring
    cat_heur, heuristic_score, runner_up_cat, runner_up_score = heuristic_scorer.best_category_with_confidence(
        heuristic_text,
        ConfidenceOptions(
            language=config.output.naming.language,
            min_score_gap=scoring.min_heuristic_score_gap,
            max_score_per_category=scoring.max_score_per_category,
            title_weight_region=scoring.title_weight_region,
            title_weight_factor=scoring.title_weight_factor,
        ),
    )
    heuristic_gap = (
        heuristic_score - runner_up_score if (cat_heur != "unknown" and runner_up_score is not None) else 0.0
    )
    suggested_doc_type = (
        cat_heur if (cat_heur != "unknown" and heuristic_score >= _HEURISTIC_SUGGESTED_DOC_TYPE_MIN_SCORE) else None
    )
    return (
        cat_heur,
        heuristic_score,
        runner_up_cat,
        runner_up_score,
        heuristic_gap,
        suggested_doc_type,
    )


def _resolve_allowed_categories(
    config: RenamerConfig,
    heuristic_scorer: HeuristicScorer,
    rules: ProcessingRules | None,
) -> list[str] | None:
    """Resolve the allowed category set from rules or config."""
    if rules is not None and rules.allowed_categories:
        return [c for c in rules.allowed_categories if c and c.strip()]
    if config.heuristic.category.use_constrained_llm_category:
        return list(heuristic_scorer.all_categories())
    return None


def _validate_llm_category_against_allowed(cat_llm: str, allowed: list[str] | None) -> str:
    """Check LLM category against allowed set. Returns cat_llm or 'unknown'."""
    if not allowed:
        return cat_llm
    norm = normalize_llm_category(cat_llm).strip().lower().replace(" ", "_")
    allowed_set = frozenset(c.strip().lower().replace(" ", "_") for c in allowed)
    if norm and norm not in allowed_set and norm not in {"unknown", "na", "document"}:
        logger.info("LLM category %r not in allowed set; using heuristic.", cat_llm)
        return "unknown"
    return cat_llm


def _should_skip_llm_category(
    request: _CategoryResolutionInput,
    config: RenamerConfig,
) -> bool:
    """Return whether strong heuristic evidence permits skipping LLM classification."""
    skip = config.heuristic.skip
    return bool(
        skip.skip_llm_category_if_heuristic_score_ge is not None
        and skip.skip_llm_category_if_heuristic_gap_ge is not None
        and request.cat_heur != "unknown"
        and request.heuristic_score >= skip.skip_llm_category_if_heuristic_score_ge
        and request.heuristic_gap >= skip.skip_llm_category_if_heuristic_gap_ge
    )


def _resolve_llm_category_candidate(
    request: _CategoryResolutionInput,
    runtime: _MetadataRuntime,
    *,
    skip_llm: bool,
) -> str:
    """Choose a heuristic, precomputed, or newly requested LLM category."""
    config = runtime.config
    heuristic_scorer = runtime.heuristic_scorer
    allowed = _resolve_allowed_categories(config, heuristic_scorer, request.rules)
    if skip_llm:
        return request.cat_heur
    if request.precomputed_llm_category is not None:
        return _validate_llm_category_against_allowed(request.precomputed_llm_category, allowed)
    cat_llm = get_document_category(
        runtime.llm_client,
        summary=request.summary,
        keywords=request.keywords,
        options=CategoryOptions(
            language=config.output.naming.language,
            suggested_categories=(
                _suggested_categories_for_llm(request.heuristic_text, config, heuristic_scorer) if not allowed else None
            ),
            allowed_categories=allowed,
            lenient_json=config.llm.runtime.lenient_llm_json,
            cache=request.response_cache,
            cache_key_base=request.cache_key_base,
        ),
    )
    return _validate_llm_category_against_allowed(cat_llm, allowed)


def _category_overlap_context(request: _CategoryResolutionInput, config: RenamerConfig) -> str | None:
    """Build summary-and-keyword context for category overlap scoring."""
    if not config.heuristic.category.use_keyword_overlap_for_category:
        return None
    return (request.summary + " " + " ".join(request.keywords)).strip()


def _category_combine_params(config: RenamerConfig) -> CategoryCombineParams:
    """Map category-resolution configuration to combination parameters."""
    category = config.heuristic.category
    scoring = config.heuristic.scoring
    skip = config.heuristic.skip
    return CategoryCombineParams(
        prefer_llm=category.prefer_llm_category,
        min_heuristic_score=scoring.min_heuristic_score,
        heuristic_override_min_score=skip.heuristic_override_min_score,
        heuristic_override_min_gap=skip.heuristic_override_min_gap,
        heuristic_score_weight=scoring.heuristic_score_weight,
        use_keyword_overlap=category.use_keyword_overlap_for_category,
    )


def _category_source(skip_llm: bool, cat_heur: str) -> str:
    """Describe whether the category came from heuristics, LLM, or both."""
    if skip_llm:
        return "heuristic"
    if cat_heur == "unknown":
        return "llm"
    return "combined"


def _log_category_resolution(
    request: _CategoryResolutionInput,
    config: RenamerConfig,
    *,
    category_source: str,
    cat_llm: str,
    category: str,
) -> None:
    """Log category resolution to make the resolution decision observable."""
    logger.info(
        "CategorySource source=%s heuristic=%s llm=%s category=%s",
        category_source,
        request.cat_heur,
        cat_llm,
        category,
    )
    if config.output.progress_options.explain:
        logger.info(
            "Explain ConflictResolution source=%s heuristic=%s heuristic_score=%.2f heuristic_gap=%.2f llm=%s final=%s",
            category_source,
            request.cat_heur,
            request.heuristic_score,
            request.heuristic_gap,
            cat_llm,
            category,
        )


def _resolve_category_with_llm(
    request: _CategoryResolutionInput,
    config: RenamerConfig,
    heuristic_scorer: HeuristicScorer,
    llm_client: LLMClient,
) -> tuple[str, str, str]:
    """Resolve final category (heuristic + optional LLM, combine_categories).
    Returns (category, category_for_filename, category_source)."""
    cat_heur = request.cat_heur
    if not config.llm.runtime.use_llm:
        category_for_filename = heuristic_scorer.get_display_category(
            cat_heur, config.heuristic.category.category_display
        )
        logger.info("CategorySource source=heuristic category=%s (use_llm=False)", cat_heur)
        return (cat_heur, category_for_filename, "heuristic")

    skip_llm = _should_skip_llm_category(request, config)
    cat_llm = _resolve_llm_category_candidate(
        request,
        _MetadataRuntime(config, llm_client, heuristic_scorer),
        skip_llm=skip_llm,
    )
    category = combine_categories(
        cat_llm,
        cat_heur,
        CategoryCombineOptions(
            heuristic_score=request.heuristic_score if cat_heur != "unknown" else None,
            heuristic_gap=request.heuristic_gap if cat_heur != "unknown" else None,
            params=_category_combine_params(config),
            context_for_overlap=_category_overlap_context(request, config),
            category_parent_map=heuristic_scorer.category_parent_map(),
        ),
    )
    category_for_filename = heuristic_scorer.get_display_category(category, config.heuristic.category.category_display)
    category_source = _category_source(skip_llm, cat_heur)
    _log_category_resolution(request, config, category_source=category_source, cat_llm=cat_llm, category=category)
    return (category, category_for_filename, category_source)


def _build_metadata_tokens(
    category_for_filename: str,
    keywords: list[str],
    final_summary_tokens: list[str],
    stopwords: Stopwords,
) -> tuple[list[str], list[str], list[str], dict[str, object]]:
    """Filter, clean, subtract tokens; return (category_clean, keyword_clean, summary_clean, metadata)."""
    category_tokens = stopwords.filter_tokens(split_to_tokens(category_for_filename))
    keyword_tokens = stopwords.filter_tokens(keywords)[:3]
    summary_tokens = stopwords.filter_tokens(final_summary_tokens)[:5]

    category_clean = [clean_token(t) for t in category_tokens]
    keyword_clean = [clean_token(t) for t in keyword_tokens]
    summary_clean = [clean_token(t) for t in summary_tokens]

    keyword_clean = subtract_tokens(keyword_clean, category_clean)
    summary_clean = subtract_tokens(summary_clean, category_clean + keyword_clean)

    metadata: dict[str, object] = {
        "category": " ".join(category_clean) or (category_for_filename or ""),
        "summary": " ".join(summary_clean),
        "keywords": " ".join(keyword_clean),
    }
    return (category_clean, keyword_clean, summary_clean, metadata)


def _override_category_context(
    override_category: str,
    config: RenamerConfig,
    heuristic_scorer: HeuristicScorer,
) -> _CategoryContext:
    """Build category context for an explicit category override."""
    logger.info("CategorySource source=override category=%s", override_category)
    return _CategoryContext(
        category=override_category,
        category_for_filename=heuristic_scorer.get_display_category(
            override_category,
            config.heuristic.category.category_display,
        ),
        category_source="override",
        heuristic=_HeuristicCategorySignal(
            heuristic_text="",
            cat_heur=override_category,
            heuristic_score=0.0,
            heuristic_gap=0.0,
            suggested_doc_type=override_category,
        ),
        skip_llm_by_rule=False,
    )


def _heuristic_category_context(
    request: _MetadataResolutionInput,
    config: RenamerConfig,
    heuristic_scorer: HeuristicScorer,
) -> _CategoryContext:
    """Score the document heuristically and build initial category context."""
    heuristic_text = _heuristic_text_for_category(request.pdf_content, config)
    cat_heur, heuristic_score, _runner_up_cat, _runner_up_score, heuristic_gap, suggested_doc_type = (
        _resolve_heuristic_category(heuristic_text, config, heuristic_scorer)
    )
    if config.output.progress_options.explain:
        scoring = config.heuristic.scoring
        logger.info(
            "Explain Heuristic category=%s score=%.2f gap=%.2f top=%s",
            cat_heur,
            heuristic_score,
            heuristic_gap,
            heuristic_scorer.top_n_categories(
                heuristic_text,
                TopNOptions(
                    n=min(5, config.heuristic.skip.heuristic_suggestions_top_n),
                    language=config.output.naming.language,
                    max_score_per_category=scoring.max_score_per_category,
                    title_weight_region=scoring.title_weight_region,
                    title_weight_factor=scoring.title_weight_factor,
                ),
            ),
        )
    return _CategoryContext(
        category=cat_heur,
        category_for_filename=heuristic_scorer.get_display_category(
            cat_heur, config.heuristic.category.category_display
        ),
        category_source="heuristic",
        heuristic=_HeuristicCategorySignal(
            heuristic_text=heuristic_text,
            cat_heur=cat_heur,
            heuristic_score=heuristic_score,
            heuristic_gap=heuristic_gap,
            suggested_doc_type=suggested_doc_type,
        ),
        skip_llm_by_rule=_skip_llm_by_rule(config, request.rules, cat_heur),
    )


def _skip_llm_by_rule(
    config: RenamerConfig,
    rules: ProcessingRules | None,
    cat_heur: str,
) -> bool:
    """Return whether processing rules suppress LLM metadata calls."""
    return bool(config.llm.runtime.use_llm and rules is not None and cat_heur in rules.skip_llm_if_heuristic_category)


def _initial_category_context(
    request: _MetadataResolutionInput,
    config: RenamerConfig,
    heuristic_scorer: HeuristicScorer,
) -> _CategoryContext:
    """Build the initial category context from an override or heuristic scoring."""
    if request.override_category is not None:
        return _override_category_context(request.override_category, config, heuristic_scorer)
    return _heuristic_category_context(request, config, heuristic_scorer)


def _resolve_summary_and_keywords(
    context: _CategoryContext,
    request: _MetadataResolutionInput,
    config: RenamerConfig,
    llm_client: LLMClient,
    heuristic_scorer: HeuristicScorer,
) -> _LlmResult:
    """Return empty metadata when LLM use is skipped; otherwise call the LLM."""
    if context.skip_llm_by_rule:
        logger.info("CategorySource source=heuristic (rules skip_llm) category=%s", context.category)
        return _LlmResult("", [])
    if not config.llm.runtime.use_llm:
        return _LlmResult("", [])
    return _LlmResult(
        *_get_llm_summary_and_keywords(
            _LlmSummaryInput(
                pdf_content=request.pdf_content,
                heuristic=context.heuristic,
                override_category=request.override_category,
                rules=request.rules,
                cache_context=_FilenameCacheContext(request.response_cache, request.cache_key_base),
            ),
            config,
            llm_client,
            heuristic_scorer,
        )
    )


def _resolve_final_category_context(
    context: _CategoryContext,
    request: _MetadataResolutionInput,
    result: _LlmResult,
    keywords: list[str],
    runtime: _MetadataRuntime,
) -> _CategoryContext:
    """Reconcile the initial category with available LLM metadata."""
    if context.skip_llm_by_rule or request.override_category is not None:
        return context
    category, category_for_filename, category_source = _resolve_category_with_llm(
        _CategoryResolutionInput(
            heuristic=context.heuristic,
            summary=result.summary,
            keywords=keywords,
            rules=request.rules,
            precomputed_llm_category=result.precomputed_category,
            cache_context=_FilenameCacheContext(request.response_cache, request.cache_key_base),
        ),
        runtime.config,
        runtime.heuristic_scorer,
        runtime.llm_client,
    )
    return _CategoryContext(
        category=category,
        category_for_filename=category_for_filename,
        category_source=category_source,
        heuristic=context.heuristic,
        skip_llm_by_rule=context.skip_llm_by_rule,
    )


def _final_summary_tokens_for_context(
    context: _CategoryContext,
    request: _MetadataResolutionInput,
    result: _LlmResult,
    keywords: list[str],
    runtime: _MetadataRuntime,
) -> list[str]:
    """Resolve final summary tokens unless LLM processing was skipped."""
    if not runtime.config.llm.runtime.use_llm or context.skip_llm_by_rule:
        return []
    return _resolve_final_summary_tokens(
        runtime.config,
        runtime.llm_client,
        _FinalSummaryTokenInput(
            summary=result.summary,
            keywords=keywords,
            category=context.category,
            precomputed_summary_tokens=result.precomputed_summary_tokens,
            response_cache=request.response_cache,
            cache_key_base=request.cache_key_base,
        ),
    )


def _log_llm_metadata_result(
    context: _CategoryContext,
    result: _LlmResult,
    keywords: list[str],
    config: RenamerConfig,
) -> None:
    """Log LLM metadata when explanatory logging is enabled."""
    if not (config.output.progress_options.explain and config.llm.runtime.use_llm and not context.skip_llm_by_rule):
        return
    logger.info(
        "Explain LLM summary=%r keywords=%s precomputed_category=%r",
        result.summary,
        keywords,
        result.precomputed_category,
    )


def _llm_failed_for_context(context: _CategoryContext, config: RenamerConfig) -> bool:
    """Flag an unknown category after an attempted LLM resolution."""
    category_unknown = (context.category or "").strip().lower() in ("unknown", "na", "document", "")
    llm_failed = bool(config.llm.runtime.use_llm and not context.skip_llm_by_rule and category_unknown)
    if llm_failed:
        logger.warning("LLM was used but category is unknown; using heuristic or timestamp fallback.")
    return llm_failed


def _get_category_summary_keywords_metadata(
    request: _MetadataResolutionInput,
    config: RenamerConfig,
    llm_client: LLMClient,
    heuristic_scorer: HeuristicScorer,
    stopwords: Stopwords,
) -> tuple[str, list[str], list[str], list[str], dict[str, object]]:
    """Resolve category (override/heuristic/LLM), run LLM summary/keywords, clean tokens, build metadata."""
    runtime = _MetadataRuntime(config, llm_client, heuristic_scorer)
    context = _initial_category_context(request, config, heuristic_scorer)
    result = _resolve_summary_and_keywords(context, request, config, llm_client, heuristic_scorer)
    keywords = normalize_keywords(result.raw_keywords)
    _log_llm_metadata_result(context, result, keywords, config)

    context = _resolve_final_category_context(context, request, result, keywords, runtime)
    llm_failed = _llm_failed_for_context(context, config)
    final_summary_tokens = _final_summary_tokens_for_context(context, request, result, keywords, runtime)

    category_clean, keyword_clean, summary_clean, metadata = _build_metadata_tokens(
        context.category_for_filename, keywords, final_summary_tokens, stopwords
    )
    metadata["category_source"] = context.category_source
    metadata["llm_failed"] = llm_failed
    return (
        context.category_for_filename,
        category_clean,
        keyword_clean,
        summary_clean,
        metadata,
    )
