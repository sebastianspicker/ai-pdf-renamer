"""
Filename generation pipeline: date, category, keywords, summary, template, truncation.

Uses config, heuristics, LLM, text_utils, rules, loaders; re-export generate_filename from renamer.
"""

from __future__ import annotations

import logging
import os
from datetime import date, datetime

from .config import RenamerConfig
from .heuristics import (
    HeuristicScorer,
    combine_categories,
    normalize_llm_category,
)
from .llm import (
    LocalLLMClient,
    get_document_category,
    get_document_filename_simple,
    get_document_keywords,
    get_document_summary,
    get_final_summary_tokens,
)
from .loaders import default_heuristic_scorer, default_stopwords
from .rename_ops import sanitize_filename_base
from .rules import ProcessingRules
from .text_utils import (
    Stopwords,
    clean_token,
    convert_case,
    extract_date_from_content,
    extract_structured_fields,
    normalize_keywords,
    split_to_tokens,
    subtract_tokens,
)

logger = logging.getLogger(__name__)

# Min heuristic score to suggest doc type to LLM summary (otherwise None).
_HEURISTIC_SUGGESTED_DOC_TYPE_MIN_SCORE = 0.25


def _config_or_env(value: str | None, env_key: str, default: str) -> str:
    """Resolve string from config value, then env, then default. Empty string treated as unset."""
    s = (value or "").strip() or (os.environ.get(env_key) or "").strip()
    return s or default


def _llm_client_from_config(config: RenamerConfig) -> LocalLLMClient:
    """Build LocalLLMClient from config and env (AI_PDF_RENAMER_LLM_*)."""
    base_url = _config_or_env(
        config.llm_base_url,
        "AI_PDF_RENAMER_LLM_URL",
        "http://127.0.0.1:11434/v1/completions",
    )
    model = _config_or_env(config.llm_model, "AI_PDF_RENAMER_LLM_MODEL", "qwen3:8b")
    timeout_s = config.llm_timeout_s
    if timeout_s is None or timeout_s <= 0:
        try:
            timeout_s = float(os.environ.get("AI_PDF_RENAMER_LLM_TIMEOUT", "") or 0)
        except ValueError:
            timeout_s = 60.0
        if timeout_s <= 0:
            timeout_s = 60.0
    return LocalLLMClient(base_url=base_url, model=model, timeout_s=timeout_s)


def _get_date_str(
    pdf_content: str,
    config: RenamerConfig,
    today: date | None = None,
    pdf_metadata: dict | None = None,
) -> str:
    """Extract date from content (and optionally PDF metadata fallback) and return YYYYMMDD string."""
    content_date = extract_date_from_content(
        pdf_content,
        today=today,
        date_locale=config.date_locale,
        prefer_leading_chars=config.date_prefer_leading_chars or 0,
    )
    if not getattr(config, "use_pdf_metadata_for_date", True):
        return content_date.replace("-", "")
    if today is None:
        today = date.today()
    today_str = today.strftime("%Y-%m-%d")
    if content_date != today_str and content_date:
        return content_date.replace("-", "")
    if pdf_metadata:
        for key in ("creation_date", "mod_date"):
            meta_date = pdf_metadata.get(key)
            if meta_date and isinstance(meta_date, str):
                return meta_date.replace("-", "")
    return content_date.replace("-", "")


def _heuristic_text_for_category(pdf_content: str, config: RenamerConfig) -> str:
    """Return the slice of PDF content used for heuristic category scoring."""
    if config.heuristic_leading_chars > 0:
        return pdf_content[: config.heuristic_leading_chars]
    if config.heuristic_long_doc_chars_threshold > 0 and len(pdf_content) >= config.heuristic_long_doc_chars_threshold:
        return pdf_content[: config.heuristic_long_doc_leading_chars]
    return pdf_content


def _resolve_heuristic_category(
    heuristic_text: str,
    config: RenamerConfig,
    heuristic_scorer: HeuristicScorer,
) -> tuple[str, float, str, float, float, str | None]:
    """Run heuristic scoring; return (cat_heur, score, runner_up_cat, runner_up_score, gap, suggested_doc_type)."""
    cat_heur, heuristic_score, runner_up_cat, runner_up_score = heuristic_scorer.best_category_with_confidence(
        heuristic_text,
        language=config.language,
        min_score_gap=config.min_heuristic_score_gap,
        max_score_per_category=config.max_score_per_category,
        title_weight_region=config.title_weight_region,
        title_weight_factor=config.title_weight_factor,
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


def _resolve_category_with_llm(
    heuristic_text: str,
    cat_heur: str,
    heuristic_score: float,
    heuristic_gap: float,
    config: RenamerConfig,
    heuristic_scorer: HeuristicScorer,
    llm_client: LocalLLMClient,
    summary: str,
    keywords: list[str],
    rules: ProcessingRules | None = None,
) -> tuple[str, str, str]:
    """Resolve final category (heuristic + optional LLM, combine_categories).
    Returns (category, category_for_filename, category_source)."""
    if not config.use_llm:
        category_for_filename = heuristic_scorer.get_display_category(cat_heur, config.category_display)
        logger.info(
            "CategorySource source=heuristic category=%s (use_llm=False)",
            cat_heur,
        )
        return (cat_heur, category_for_filename, "heuristic")
    skip_llm = False
    if (
        config.skip_llm_category_if_heuristic_score_ge is not None
        and config.skip_llm_category_if_heuristic_gap_ge is not None
        and cat_heur != "unknown"
        and heuristic_score >= config.skip_llm_category_if_heuristic_score_ge
        and heuristic_gap >= config.skip_llm_category_if_heuristic_gap_ge
    ):
        skip_llm = True
    if skip_llm:
        cat_llm = cat_heur
    else:
        top_n = heuristic_scorer.top_n_categories(
            heuristic_text,
            n=config.heuristic_suggestions_top_n,
            language=config.language,
            max_score_per_category=config.max_score_per_category,
            title_weight_region=config.title_weight_region,
            title_weight_factor=config.title_weight_factor,
        )
        suggested = [c for c in top_n if c and c != "unknown"]
        # Rules-level allowlist has precedence over default constrained category set.
        if rules is not None and rules.allowed_categories:
            allowed = [c for c in rules.allowed_categories if c and c.strip()]
        elif config.use_constrained_llm_category:
            allowed = list(heuristic_scorer.all_categories())
        else:
            allowed = None
        cat_llm = get_document_category(
            llm_client,
            summary=summary,
            keywords=keywords,
            language=config.language,
            suggested_categories=suggested if not allowed else None,
            allowed_categories=allowed,
            lenient_json=config.lenient_llm_json,
        )
        if allowed:
            norm = normalize_llm_category(cat_llm).strip().lower().replace(" ", "_")
            allowed_set = frozenset(c.strip().lower().replace(" ", "_") for c in allowed)
            if norm and norm not in allowed_set and norm not in {"unknown", "na", "document"}:
                logger.info(
                    "LLM category %r not in allowed set; using heuristic.",
                    cat_llm,
                )
                cat_llm = "unknown"
    context_for_overlap = None
    if config.use_keyword_overlap_for_category:
        context_for_overlap = (summary + " " + " ".join(keywords)).strip()
    category = combine_categories(
        cat_llm,
        cat_heur,
        prefer_llm=config.prefer_llm_category,
        heuristic_score=heuristic_score if cat_heur != "unknown" else None,
        heuristic_gap=heuristic_gap if cat_heur != "unknown" else None,
        min_heuristic_score=config.min_heuristic_score,
        heuristic_override_min_score=config.heuristic_override_min_score,
        heuristic_override_min_gap=config.heuristic_override_min_gap,
        heuristic_score_weight=config.heuristic_score_weight,
        context_for_overlap=context_for_overlap,
        use_keyword_overlap=config.use_keyword_overlap_for_category,
        use_embeddings_for_conflict=config.use_embeddings_for_conflict,
        category_parent_map=heuristic_scorer._category_to_parent(),
    )
    category_for_filename = heuristic_scorer.get_display_category(category, config.category_display)
    if skip_llm:
        category_source = "heuristic"
    elif cat_heur == "unknown":
        category_source = "llm"
    else:
        category_source = "combined"
    logger.info(
        "CategorySource source=%s heuristic=%s llm=%s category=%s",
        category_source,
        cat_heur,
        cat_llm,
        category,
    )
    return (category, category_for_filename, category_source)


def _build_metadata_tokens(
    category_for_filename: str,
    keywords: list[str],
    final_summary_tokens: list[str],
    stopwords: Stopwords,
) -> tuple[list[str], list[str], list[str], dict]:
    """Filter, clean, subtract tokens; return (category_clean, keyword_clean, summary_clean, metadata)."""
    category_tokens = stopwords.filter_tokens(split_to_tokens(category_for_filename))
    keyword_tokens = stopwords.filter_tokens(keywords)[:3]
    summary_tokens = stopwords.filter_tokens(final_summary_tokens)[:5]

    category_clean = [clean_token(t) for t in category_tokens]
    keyword_clean = [clean_token(t) for t in keyword_tokens]
    summary_clean = [clean_token(t) for t in summary_tokens]

    keyword_clean = subtract_tokens(keyword_clean, category_clean)
    summary_clean = subtract_tokens(summary_clean, category_clean + keyword_clean)

    metadata = {
        "category": " ".join(category_clean) or (category_for_filename or ""),
        "summary": " ".join(summary_clean),
        "keywords": " ".join(keyword_clean),
    }
    return (category_clean, keyword_clean, summary_clean, metadata)


def _get_category_summary_keywords_metadata(
    pdf_content: str,
    config: RenamerConfig,
    llm_client: LocalLLMClient,
    heuristic_scorer: HeuristicScorer,
    stopwords: Stopwords,
    override_category: str | None,
    rules: ProcessingRules | None = None,
) -> tuple[str, list[str], list[str], list[str], dict]:
    """Resolve category (override/heuristic/LLM), run LLM summary/keywords, clean tokens, build metadata.
    Returns (category_for_filename, category_clean, keyword_clean, summary_clean, metadata)."""
    if override_category is not None:
        category = override_category
        cat_heur = override_category
        category_for_filename = heuristic_scorer.get_display_category(override_category, config.category_display)
        category_source = "override"
        logger.info(
            "CategorySource source=override category=%s",
            category,
        )
        suggested_doc_type_for_summary = override_category
        heuristic_text = ""
        heuristic_score = 0.0
        heuristic_gap = 0.0
        skip_llm_by_rule = False
    else:
        heuristic_text = _heuristic_text_for_category(pdf_content, config)
        (
            cat_heur,
            heuristic_score,
            runner_up_cat,
            runner_up_score,
            heuristic_gap,
            suggested_doc_type_for_summary,
        ) = _resolve_heuristic_category(heuristic_text, config, heuristic_scorer)
        skip_llm_by_rule = config.use_llm and rules is not None and cat_heur in rules.skip_llm_if_heuristic_category

    if skip_llm_by_rule:
        summary = ""
        raw_keywords = []
        category = cat_heur
        category_for_filename = heuristic_scorer.get_display_category(cat_heur, config.category_display)
        category_source = "heuristic"
        logger.info("CategorySource source=heuristic (rules skip_llm) category=%s", category)
    elif config.use_llm:
        summary = get_document_summary(
            llm_client,
            pdf_content,
            language=config.language,
            suggested_doc_type=suggested_doc_type_for_summary,
            lenient_json=config.lenient_llm_json,
            max_content_chars=config.max_content_chars,
            max_content_tokens=config.max_content_tokens,
        )
        raw_keywords = (
            get_document_keywords(
                llm_client,
                summary,
                language=config.language,
                suggested_category=cat_heur if cat_heur != "unknown" else None,
                lenient_json=config.lenient_llm_json,
            )
            or []
        )
    else:
        summary = ""
        raw_keywords = []
    keywords = normalize_keywords(raw_keywords)

    if not skip_llm_by_rule and override_category is None:
        category, category_for_filename, category_source = _resolve_category_with_llm(
            heuristic_text,
            cat_heur,
            heuristic_score,
            heuristic_gap,
            config,
            heuristic_scorer,
            llm_client,
            summary,
            keywords,
            rules=rules,
        )

    category_unknown = (category or "").strip().lower() in ("unknown", "na", "document", "")
    if config.use_llm and not skip_llm_by_rule and category_unknown:
        logger.warning("LLM was used but category is unknown; using heuristic or timestamp fallback.")
    llm_failed = bool(config.use_llm and not skip_llm_by_rule and category_unknown)

    if config.use_llm and not skip_llm_by_rule:
        final_summary_tokens = (
            get_final_summary_tokens(
                llm_client,
                summary=summary,
                keywords=keywords,
                category=category,
                language=config.language,
                lenient_json=config.lenient_llm_json,
            )
            or []
        )
    else:
        final_summary_tokens = []

    category_clean, keyword_clean, summary_clean, metadata = _build_metadata_tokens(
        category_for_filename, keywords, final_summary_tokens, stopwords
    )
    metadata["category_source"] = category_source
    metadata["llm_failed"] = llm_failed
    return (
        category_for_filename,
        category_clean,
        keyword_clean,
        summary_clean,
        metadata,
    )


def _filename_sep(config: RenamerConfig) -> str:
    """Return filename part separator: '_' for snakeCase, '-' otherwise."""
    return "_" if config.desired_case == "snakeCase" else "-"


def _should_use_timestamp_fallback(
    category_for_filename: str,
    category_clean: list[str],
    keyword_clean: list[str],
    summary_clean: list[str],
) -> bool:
    """True when category and all content-derived tokens are empty/unknown (full fallback)."""
    empty_cat = (category_for_filename or "").strip().lower() in {"unknown", "document", "na", ""}
    no_tokens = not category_clean and not keyword_clean and not summary_clean
    return bool(empty_cat and no_tokens)


def _build_timestamp_fallback_filename(
    date_str: str,
    config: RenamerConfig,
    *,
    now: datetime | None = None,
) -> str:
    """Build minimal filename: date + segment + HHMMSS when heuristic+LLM both fail."""
    if now is None:
        now = datetime.now()
    time_str = now.strftime("%H%M%S")
    segment = (config.timestamp_fallback_segment or "document").strip() or "document"
    sep = _filename_sep(config)
    filename = sep.join([date_str, segment, time_str])
    filename = _apply_filename_template(
        filename,
        date_str,
        (config.project or "").strip(),
        (config.version or "").strip(),
        segment,
        [segment, time_str],
        [],
        [],
        config,
        structured_fields=None,
    )
    return _truncate_filename_to_max_chars(filename, config)


def _apply_filename_template(
    filename: str,
    date_str: str,
    project: str,
    version: str,
    category_for_filename: str,
    category_clean: list[str],
    keyword_clean: list[str],
    summary_clean: list[str],
    config: RenamerConfig,
    structured_fields: dict[str, str] | None = None,
) -> str:
    """If config has a filename template, format and return; else return filename unchanged."""
    if not config.filename_template or not isinstance(config.filename_template, str):
        return filename
    cat_str = convert_case(category_clean, config.desired_case) if category_clean else (category_for_filename or "")
    kw_str = convert_case(keyword_clean, config.desired_case) if keyword_clean else ""
    sum_str = convert_case(summary_clean, config.desired_case) if summary_clean else ""
    sf = structured_fields or {}
    repl = {
        "date": date_str,
        "project": project or "",
        "category": cat_str,
        "keywords": kw_str,
        "summary": sum_str,
        "version": version or "",
        "invoice_id": sf.get("invoice_id", ""),
        "amount": sf.get("amount", ""),
        "company": sf.get("company", ""),
    }
    try:
        filename = config.filename_template.format(**repl).strip()
    except (KeyError, AttributeError, TypeError) as e:
        logger.warning("Template failed (%s); using default filename", e)
        return filename
    if filename.lower().endswith(".pdf"):
        filename = filename[:-4]
    return sanitize_filename_base(filename) if filename else filename


def _truncate_filename_to_max_chars(filename: str, config: RenamerConfig) -> str:
    """Truncate filename to config.max_filename_chars (at separator if possible)."""
    if not config.max_filename_chars or config.max_filename_chars <= 0 or len(filename) <= config.max_filename_chars:
        return filename
    sep = _filename_sep(config)
    while len(filename) > config.max_filename_chars and sep in filename:
        filename = filename.rsplit(sep, 1)[0]
    if len(filename) > config.max_filename_chars:
        filename = filename[: config.max_filename_chars]
    return filename


def _build_filename_str(
    date_str: str,
    category_for_filename: str,
    category_clean: list[str],
    keyword_clean: list[str],
    summary_clean: list[str],
    config: RenamerConfig,
    structured_fields: dict[str, str] | None = None,
) -> str:
    """Build final filename from date, cleaned tokens, and config (project, version, case, template, max chars)."""
    project = (config.project or "").strip()
    version = (config.version or "").strip()
    if project.lower() == "default":
        project = ""
    if version.lower() == "default":
        version = ""

    if config.desired_case == "camelCase":
        tokens: list[str] = [date_str]
        tokens += split_to_tokens(project) if project else []
        tokens += category_clean
        tokens += keyword_clean
        tokens += summary_clean
        tokens += split_to_tokens(version) if version else []
        filename = convert_case(tokens, "camelCase")
    else:
        parts: list[str] = [date_str]
        if project:
            parts.append(convert_case(split_to_tokens(project), config.desired_case))
        parts.append(convert_case(category_clean, config.desired_case))
        if keyword_clean:
            parts.append(convert_case(keyword_clean, config.desired_case))
        if summary_clean:
            parts.append(convert_case(summary_clean, config.desired_case))
        if version:
            parts.append(convert_case(split_to_tokens(version), config.desired_case))
        sep = _filename_sep(config)
        filename = sep.join(p for p in parts if p)
        # Ensure consistent separators: if we are in snakeCase, no dashes; if kebabCase, no underscores.
        alt_sep = "-" if sep == "_" else "_"
        filename = filename.replace(alt_sep, sep)
        filename = sep.join(x for x in filename.split(sep) if x)

    filename = _apply_filename_template(
        filename,
        date_str,
        project,
        version,
        category_for_filename,
        category_clean,
        keyword_clean,
        summary_clean,
        config,
        structured_fields=structured_fields,
    )
    return _truncate_filename_to_max_chars(filename, config)


def generate_filename(
    pdf_content: str,
    *,
    config: RenamerConfig,
    llm_client: LocalLLMClient | None = None,
    heuristic_scorer: HeuristicScorer | None = None,
    stopwords: Stopwords | None = None,
    override_category: str | None = None,
    today: date | None = None,
    pdf_metadata: dict | None = None,
    rules: ProcessingRules | None = None,
) -> tuple[str, dict]:
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
    llm_client = llm_client or _llm_client_from_config(config)
    stopwords = stopwords or default_stopwords()
    heuristic_scorer = heuristic_scorer or default_heuristic_scorer(config.language)

    date_str = _get_date_str(pdf_content, config, today, pdf_metadata)
    if getattr(config, "simple_naming_mode", False):
        simple_part = get_document_filename_simple(
            llm_client,
            pdf_content,
            language=config.language,
            max_content_chars=config.max_content_chars,
            max_content_tokens=config.max_content_tokens,
        )
        sep = _filename_sep(config)
        filename = sep.join([date_str, simple_part])
        filename = _apply_filename_template(
            filename,
            date_str,
            (config.project or "").strip(),
            (config.version or "").strip(),
            simple_part,
            [simple_part],
            [],
            [],
            config,
            structured_fields=None,
        )
        filename = _truncate_filename_to_max_chars(filename, config)
        metadata = {"category": simple_part, "summary": "", "keywords": ""}
        if getattr(config, "use_structured_fields", True):
            structured_fields = extract_structured_fields(pdf_content)
            metadata["invoice_id"] = structured_fields.get("invoice_id", "")
            metadata["amount"] = structured_fields.get("amount", "")
            metadata["company"] = structured_fields.get("company", "")
        return sanitize_filename_base(filename), metadata
    category_for_filename, category_clean, keyword_clean, summary_clean, metadata = (
        _get_category_summary_keywords_metadata(
            pdf_content,
            config,
            llm_client,
            heuristic_scorer,
            stopwords,
            override_category,
            rules=rules,
        )
    )
    structured_fields: dict[str, str] = {}
    if getattr(config, "use_structured_fields", True):
        structured_fields = extract_structured_fields(pdf_content)
        metadata["invoice_id"] = structured_fields.get("invoice_id", "")
        metadata["amount"] = structured_fields.get("amount", "")
        metadata["company"] = structured_fields.get("company", "")
    if getattr(config, "use_timestamp_fallback", True) and _should_use_timestamp_fallback(
        category_for_filename, category_clean, keyword_clean, summary_clean
    ):
        filename = _build_timestamp_fallback_filename(date_str, config)
        return sanitize_filename_base(filename), metadata
    filename = _build_filename_str(
        date_str,
        category_for_filename,
        category_clean,
        keyword_clean,
        summary_clean,
        config,
        structured_fields=structured_fields,
    )
    return filename, metadata
