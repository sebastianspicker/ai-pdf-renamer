from __future__ import annotations

import csv
import fnmatch
import json
import logging
import os
import re
import signal
import sys
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date
from functools import lru_cache
from pathlib import Path

from .data_paths import data_path
from .heuristics import (
    HeuristicScorer,
    combine_categories,
    load_heuristic_rules_for_language,
    normalize_llm_category,
)
from .llm import (
    LocalLLMClient,
    get_document_category,
    get_document_keywords,
    get_document_summary,
    get_final_summary_tokens,
)
from .pdf_extract import (
    DEFAULT_MAX_CONTENT_TOKENS,
    get_pdf_metadata,
    pdf_to_text,
    pdf_to_text_with_ocr,
)
from .rename_ops import MAX_RENAME_RETRIES, apply_single_rename, sanitize_filename_base
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

# Min heuristic score to suggest doc type to LLM summary (otherwise None).
_HEURISTIC_SUGGESTED_DOC_TYPE_MIN_SCORE = 0.25


def _matches_patterns(name: str, include: list[str] | None, exclude: list[str] | None) -> bool:
    """True if basename matches include (if set) and does not match any exclude."""
    if include is not None and include:
        if not any(fnmatch.fnmatch(name, p) for p in include):
            return False
    if exclude is not None and exclude:
        if any(fnmatch.fnmatch(name, p) for p in exclude):
            return False
    return True


def _collect_pdf_files(
    directory: Path,
    *,
    recursive: bool = False,
    max_depth: int = 0,
    include_patterns: list[str] | None = None,
    exclude_patterns: list[str] | None = None,
    skip_if_already_named: bool = False,
    files_override: list[Path] | None = None,
) -> list[Path]:
    """Collect PDF file paths from directory (or use files_override)."""
    if files_override is not None:
        candidates = [p for p in files_override if p.is_file() and p.suffix.lower() == ".pdf"]
    elif recursive:
        candidates = []
        for p in directory.rglob("*.pdf"):
            if not p.is_file() or p.name.startswith("."):
                continue
            if max_depth > 0:
                try:
                    rel = p.relative_to(directory)
                    if len(rel.parts) > max_depth:
                        continue
                except ValueError:
                    continue
            candidates.append(p)
    else:
        candidates = [
            p for p in directory.iterdir() if p.is_file() and p.suffix.lower() == ".pdf" and not p.name.startswith(".")
        ]
    out = [p for p in candidates if _matches_patterns(p.name, include_patterns, exclude_patterns)]
    if skip_if_already_named:
        already_named = re.compile(r"^\d{8}-.+\.[pP][dD][fF]$")
        out = [p for p in out if not already_named.match(p.name)]
    return out


logger = logging.getLogger(__name__)


def _write_pdf_title_metadata(pdf_path: Path, title: str) -> None:
    """Write /Title metadata to PDF (PyMuPDF). No-op if fitz unavailable or on error."""
    try:
        import fitz

        doc = fitz.open(pdf_path)
        try:
            doc.set_metadata({"title": title or pdf_path.stem})
            try:
                doc.save_incremental()
            except Exception as inc_exc:
                logger.debug("Incremental save failed for %s (%s); falling back to full save.", pdf_path.name, inc_exc)
                doc.save(pdf_path, incremental=False, encryption=fitz.PDF_ENCRYPT_KEEP)
        finally:
            doc.close()
    except Exception as exc:
        logger.warning("Could not write PDF metadata for %s: %s", pdf_path, exc)


def _write_json_or_csv(
    path: Path,
    rows: list[dict],
    csv_fieldnames: list[str] | None,
) -> None:
    """Write rows to path as CSV (if csv_fieldnames and .csv suffix) or JSON. Creates parent dirs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".csv" and csv_fieldnames:
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=csv_fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    else:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)


def _apply_post_rename_actions(
    config: RenamerConfig,
    file_path: Path,
    target: Path,
    current_base: str,
    meta: dict,
    export_rows: list[dict],
) -> None:
    """Write rename log, PDF metadata, and export row after a successful rename. Mutates export_rows."""
    if config.rename_log_path:
        Path(config.rename_log_path).parent.mkdir(parents=True, exist_ok=True)
        with open(config.rename_log_path, "a", encoding="utf-8") as f:
            f.write(f"{file_path}\t{target}\n")
    if config.write_pdf_metadata:
        _write_pdf_title_metadata(target, current_base)
    if config.export_metadata_path:
        export_rows.append(
            {
                "path": str(file_path),
                "new_name": target.name,
                "category": meta.get("category", ""),
                "summary": meta.get("summary", ""),
                "keywords": meta.get("keywords", ""),
                "invoice_id": meta.get("invoice_id", ""),
                "amount": meta.get("amount", ""),
                "company": meta.get("company", ""),
            }
        )


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


def _effective_max_tokens(config: RenamerConfig) -> int:
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


def _extract_pdf_content(path: Path, config: RenamerConfig) -> str:
    """Extract text from PDF (OCR or plain) according to config. Used by _process_one_file and single-worker loop."""
    if config.use_ocr:
        return pdf_to_text_with_ocr(
            path,
            max_pages=config.max_pages_for_extraction or 0,
            max_tokens=_effective_max_tokens(config),
            language=config.language,
        )
    return pdf_to_text(
        path,
        max_pages=config.max_pages_for_extraction or 0,
        max_tokens=_effective_max_tokens(config),
    )


def load_meta_stopwords(path: str | Path) -> Stopwords:
    path_obj = Path(path)
    try:
        raw = path_obj.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"Could not read data file at {path_obj.absolute()}: {exc!s}") from exc
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in data file at {path_obj.absolute()}. {exc!s}") from exc
    raw = data.get("stopwords", [])
    if not isinstance(raw, list):
        raw = []
    words = {str(w).lower() for w in raw if str(w).strip()}
    return Stopwords(words=words)


@lru_cache(maxsize=32)
def _stopwords_cached(path_str: str) -> Stopwords:
    return load_meta_stopwords(Path(path_str))


def default_stopwords() -> Stopwords:
    return _stopwords_cached(str(data_path("meta_stopwords.json")))


@lru_cache(maxsize=32)
def _heuristic_scorer_cached(path_str: str, language: str) -> HeuristicScorer:
    rules = load_heuristic_rules_for_language(Path(path_str), language)
    return HeuristicScorer(rules)


def default_heuristic_scorer(language: str = "de") -> HeuristicScorer:
    return _heuristic_scorer_cached(str(data_path("heuristic_scores.json")), language)


_VALID_DESIRED_CASES = frozenset({"camelCase", "kebabCase", "snakeCase"})
_VALID_DATE_LOCALES = frozenset({"dmy", "mdy"})
_VALID_CATEGORY_DISPLAY = frozenset({"specific", "with_parent", "parent_only"})


@dataclass(frozen=True)
class RenamerConfig:
    language: str = "de"
    desired_case: str = "kebabCase"
    project: str = ""
    version: str = ""
    prefer_llm_category: bool = True  # Prefer LLM; --prefer-heuristic uses heuristic
    date_locale: str = "dmy"
    date_prefer_leading_chars: int = 8000  # Prefer date from first N chars (0 = full text)
    use_pdf_metadata_for_date: bool = True  # Use PDF CreationDate/ModDate when content has no date
    dry_run: bool = False
    min_heuristic_score_gap: float = 0.0
    min_heuristic_score: float = 0.0
    title_weight_region: int = 2000  # Weight first N chars (0 = off)
    title_weight_factor: float = 1.5
    max_score_per_category: float | None = None
    use_keyword_overlap_for_category: bool = True  # On conflict pick by context overlap
    use_embeddings_for_conflict: bool = False  # If True and [embeddings], use embedding similarity instead of overlap
    category_display: str = "specific"
    # Skip LLM category call when heuristic is confident (saves latency/cost)
    skip_llm_category_if_heuristic_score_ge: float | None = None
    skip_llm_category_if_heuristic_gap_ge: float | None = None
    heuristic_suggestions_top_n: int = 5  # Top-N heuristic categories to LLM
    heuristic_score_weight: float = 0.15  # Bonus for heuristic in overlap comparison
    heuristic_override_min_score: float | None = None  # Override LLM when score >= this
    heuristic_override_min_gap: float | None = None  # and gap >= this
    use_constrained_llm_category: bool = True  # Pass full category list to LLM
    heuristic_leading_chars: int = 0  # If > 0, score category only on first N chars
    # When len(content) >= threshold, use first leading_chars for heuristic
    heuristic_long_doc_chars_threshold: int = 40_000
    heuristic_long_doc_leading_chars: int = 12_000
    max_pages_for_extraction: int = 0  # If > 0, only extract text from first N pages
    # LLM (env: AI_PDF_RENAMER_LLM_URL, AI_PDF_RENAMER_LLM_MODEL, AI_PDF_RENAMER_LLM_TIMEOUT)
    llm_base_url: str | None = None
    llm_model: str | None = None
    llm_timeout_s: float | None = None
    # PDF extraction token cap (env: AI_PDF_RENAMER_MAX_TOKENS)
    max_tokens_for_extraction: int | None = None
    # UX: skip PDFs whose name already matches YYYYMMDD-*.pdf
    skip_if_already_named: bool = False
    # Optional backup dir (copy before rename) and/or rename log (old_path -> new_path)
    backup_dir: str | Path | None = None
    rename_log_path: str | Path | None = None
    # Optional export of proposed renames + metadata to CSV/JSON before applying
    export_metadata_path: str | Path | None = None
    # Cap filename length (truncate from right at separator)
    max_filename_chars: int | None = None
    # Per-file category override: filename -> category (from --override-category-file)
    override_category_map: dict[str, str] | None = None
    # If True, run OCR (OCRmyPDF) when extracted text is too short (scanned PDFs). Optional [ocr].
    use_ocr: bool = False
    # Number of parallel workers for extract+generate_filename (default 1). Renames are applied sequentially.
    workers: int = 1
    # Recursive: collect PDFs from subdirectories (default False).
    recursive: bool = False
    # Max depth when recursive (0 = unlimited). Only used when recursive=True.
    max_depth: int = 0
    # Include/exclude fnmatch patterns for basename (e.g. ["*.pdf"], ["draft-*"]). None = no filter.
    include_patterns: list[str] | None = None
    exclude_patterns: list[str] | None = None
    # Custom filename template. Placeholders: date, project, category, keywords, summary, version,
    # invoice_id, amount, company. None = default schema.
    filename_template: str | None = None
    # If True, extract structured fields (invoice_id, amount, company) from content for template placeholders.
    use_structured_fields: bool = True
    # Write rename plan to this file without applying (old_path, new_path). Implies no renames.
    plan_file_path: str | Path | None = None
    # Interactive: prompt for each file (y/n/e=edit) before renaming.
    interactive: bool = False
    # Write new filename as PDF /Title metadata after rename (requires PyMuPDF).
    write_pdf_metadata: bool = False
    # If False, do not call LLM (heuristic-only: category from heuristics, summary/keywords empty).
    use_llm: bool = True
    # If True, try to extract JSON fields from LLM responses that don't start with "{" (regex fallback).
    lenient_llm_json: bool = False

    def __post_init__(self) -> None:
        if self.desired_case not in _VALID_DESIRED_CASES:
            raise ValueError(f"desired_case must be one of {sorted(_VALID_DESIRED_CASES)}, got {self.desired_case!r}")
        loc = (self.date_locale or "dmy").strip().lower()
        if loc not in _VALID_DATE_LOCALES:
            raise ValueError(f"date_locale must be one of {sorted(_VALID_DATE_LOCALES)}, got {self.date_locale!r}")
        disp = (self.category_display or "specific").strip().lower()
        if disp not in _VALID_CATEGORY_DISPLAY:
            raise ValueError(
                f"category_display must be one of {sorted(_VALID_CATEGORY_DISPLAY)}, got {self.category_display!r}"
            )


def build_config_from_flat_dict(data: dict) -> RenamerConfig:
    """Build RenamerConfig from a flat dict of option names -> values. Used by CLI and GUI to avoid duplication."""
    allowed = set(RenamerConfig.__dataclass_fields__)
    kwargs = {k: v for k, v in data.items() if k in allowed}
    return RenamerConfig(**kwargs)


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
) -> tuple[str, str]:
    """Resolve final category (heuristic + optional LLM, combine_categories).
    Returns (category, category_for_filename)."""
    if not config.use_llm:
        category_for_filename = heuristic_scorer.get_display_category(cat_heur, config.category_display)
        logger.info(
            "CategorySource source=heuristic category=%s (use_llm=False)",
            cat_heur,
        )
        return (cat_heur, category_for_filename)
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
        allowed = list(heuristic_scorer.all_categories()) if config.use_constrained_llm_category else None
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
    return (category, category_for_filename)


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
) -> tuple[str, list[str], list[str], list[str], dict]:
    """Resolve category (override/heuristic/LLM), run LLM summary/keywords, clean tokens, build metadata.
    Returns (category_for_filename, category_clean, keyword_clean, summary_clean, metadata)."""
    if override_category is not None:
        category = override_category
        cat_heur = override_category
        category_for_filename = heuristic_scorer.get_display_category(override_category, config.category_display)
        logger.info(
            "CategorySource source=override category=%s",
            category,
        )
        suggested_doc_type_for_summary = override_category
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

    if config.use_llm:
        summary = get_document_summary(
            llm_client,
            pdf_content,
            language=config.language,
            suggested_doc_type=suggested_doc_type_for_summary,
            lenient_json=config.lenient_llm_json,
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

    if override_category is None:
        category, category_for_filename = _resolve_category_with_llm(
            heuristic_text,
            cat_heur,
            heuristic_score,
            heuristic_gap,
            config,
            heuristic_scorer,
            llm_client,
            summary,
            keywords,
        )

    if config.use_llm:
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
    category_for_filename, category_clean, keyword_clean, summary_clean, metadata = (
        _get_category_summary_keywords_metadata(
            pdf_content,
            config,
            llm_client,
            heuristic_scorer,
            stopwords,
            override_category,
        )
    )
    structured_fields: dict[str, str] = {}
    if getattr(config, "use_structured_fields", True):
        structured_fields = extract_structured_fields(pdf_content)
        metadata["invoice_id"] = structured_fields.get("invoice_id", "")
        metadata["amount"] = structured_fields.get("amount", "")
        metadata["company"] = structured_fields.get("company", "")
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


def _process_content_to_result(
    file_path: Path,
    content: str,
    config: RenamerConfig,
) -> tuple[Path, str | None, dict | None, BaseException | None]:
    """
    Generate filename from already-extracted content. Returns (path, new_base, meta, error).
    Caller must ensure content is non-empty if expecting a non-skip result.
    """
    try:
        override_cat = (config.override_category_map or {}).get(file_path.name) or None
        pdf_meta = get_pdf_metadata(file_path) if getattr(config, "use_pdf_metadata_for_date", True) else None
        filename_str, meta = generate_filename(
            content,
            config=config,
            override_category=override_cat,
            pdf_metadata=pdf_meta,
        )
        new_base = sanitize_filename_base(filename_str)
        return (file_path, new_base, meta or {}, None)
    except Exception as exc:
        return (file_path, None, None, exc)


def _interactive_rename_prompt(
    file_path: Path,
    target: Path,
    default_base: str,
) -> tuple[str, str, Path]:
    """Prompt for y/n/e=edit. Returns (reply, base, target); reply in {'y','n'}; base/target may change on edit."""
    reply: str = "y"
    base = default_base
    current_target = target
    while True:
        try:
            reply = (
                input(f"Rename '{file_path.name}' to '{current_target.name}'? (y/n/e=edit, default y): ")
                .strip()
                .lower()
                or "y"
            )
        except (EOFError, KeyboardInterrupt):
            reply = "n"
        if reply == "n":
            return ("n", base, current_target)
        if reply == "e":
            try:
                custom = input("New filename (without path): ").strip()
                if custom:
                    custom_base = sanitize_filename_base(
                        custom.removesuffix(file_path.suffix)
                        if custom.lower().endswith(file_path.suffix.lower())
                        else custom
                    )
                    base = custom_base
                    current_target = file_path.with_name(custom_base + file_path.suffix)
                    return ("y", base, current_target)
            except (EOFError, KeyboardInterrupt):
                pass
            continue
        return ("y", base, current_target)


def _process_one_file(
    file_path: Path,
    config: RenamerConfig,
) -> tuple[Path, str | None, dict | None, BaseException | None]:
    """
    Extract text and generate filename for one file. Returns (path, new_base, meta, error).
    If error is not None or new_base is None, the file should be counted as failed/skipped.
    Empty or unextractable PDF: returns (path, None, None, None); caller logs and skips the file.
    """
    try:
        content = _extract_pdf_content(file_path, config)
    except Exception as exc:
        return (file_path, None, None, exc)
    if not content.strip():
        return (file_path, None, None, None)  # skipped empty
    try:
        return _process_content_to_result(file_path, content, config)
    except Exception as exc:
        return (file_path, None, None, exc)


def _produce_rename_results(
    files: list[Path],
    config: RenamerConfig,
) -> list[tuple[Path, str | None, dict | None, BaseException | None]]:
    """Produce (file_path, new_base, meta, exc) per file; parallel or single-worker with prefetch."""
    workers = max(1, getattr(config, "workers", 1) or 1)
    if config.interactive:
        workers = 1
    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            results = list(
                executor.map(
                    lambda p: _process_one_file(p, config),
                    files,
                )
            )
        order = {f: i for i, f in enumerate(files)}
        results.sort(key=lambda r: order.get(r[0], 0))
        return results
    results = []
    prefetched: Future[str] | None = None
    with ThreadPoolExecutor(max_workers=1) as executor:
        for i, file_path in enumerate(files):
            try:
                content = prefetched.result() if prefetched is not None else _extract_pdf_content(file_path, config)
            except Exception as exc:
                results.append((file_path, None, None, exc))
                prefetched = None
                continue
            prefetched = executor.submit(_extract_pdf_content, files[i + 1], config) if i + 1 < len(files) else None
            if not content.strip():
                results.append((file_path, None, None, None))
                continue
            try:
                results.append(_process_content_to_result(file_path, content, config))
            except Exception as exc:
                results.append((file_path, None, None, exc))
    return results


def rename_pdfs_in_directory(
    directory: str | Path,
    *,
    config: RenamerConfig,
    files_override: list[Path] | None = None,
) -> None:
    dir_str = str(directory).strip()
    if not dir_str:
        raise ValueError("Directory path must be non-empty. Use --dir or provide when prompted.")
    path = Path(directory)
    if not path.exists():
        raise FileNotFoundError(f"Directory does not exist: {path}")
    if files_override is None and not path.is_dir():
        raise NotADirectoryError(f"Not a directory: {path}")
    path = path.resolve()

    files = _collect_pdf_files(
        path,
        recursive=config.recursive,
        max_depth=config.max_depth,
        include_patterns=config.include_patterns,
        exclude_patterns=config.exclude_patterns,
        skip_if_already_named=config.skip_if_already_named,
        files_override=files_override,
    )
    if not files:
        logger.info("No matching PDF files found in %s", path)
        return

    def _mtime_key(p: Path) -> float:
        try:
            return p.stat().st_mtime
        except OSError:
            return 0.0

    files.sort(key=_mtime_key, reverse=True)

    if not config.use_llm:
        logger.info("Heuristic-only mode (LLM disabled). Category from heuristics; summary and keywords will be empty.")

    renamed_count = 0
    skipped_count = 0
    failed_count = 0
    export_rows: list[dict] = []
    plan_entries: list[dict[str, str]] = []

    results = _produce_rename_results(files, config)

    for i, (file_path, new_base, meta, exc) in enumerate(results):
        logger.info("Processing %s/%s: %s", i + 1, len(files), file_path)
        if exc is not None:
            # Data-file/config errors (e.g. invalid JSON) should propagate so CLI can exit with clear message.
            if isinstance(exc, ValueError) and "Invalid JSON in data file" in str(exc):
                raise exc
            if isinstance(exc, ValueError) and "No text extracted from" in str(exc):
                logger.warning("Skipping %s: %s", file_path.name, exc)
                skipped_count += 1
                continue
            logger.exception("Failed to process %s: %s", file_path, exc)
            failed_count += 1
            continue
        if new_base is None:
            # This case usually means truly empty content without causing an error.
            logger.info("PDF content is empty. Skipping %s.", file_path.name)
            skipped_count += 1
            continue
        meta = meta or {}
        base = new_base
        target = file_path.with_name(new_base + file_path.suffix)
        if config.interactive:
            reply, base, target = _interactive_rename_prompt(file_path, target, new_base)
            if reply == "n":
                skipped_count += 1
                continue
        try:

            def _on_rename_success(
                _fp: Path, _target: Path, _current_base: str, _meta: dict = meta, _rows: list = export_rows
            ) -> None:
                _apply_post_rename_actions(config, _fp, _target, _current_base, _meta, _rows)

            success, target = apply_single_rename(
                file_path,
                base,
                plan_file_path=config.plan_file_path,
                plan_entries=plan_entries,
                dry_run=config.dry_run,
                backup_dir=config.backup_dir,
                on_success=_on_rename_success,
                max_filename_chars=config.max_filename_chars,
            )
            if not success:
                logger.error(
                    "Skipping %s: could not rename after %s attempts",
                    file_path.name,
                    MAX_RENAME_RETRIES,
                )
                failed_count += 1
            else:
                if config.dry_run:
                    logger.info(
                        "Dry-run: would rename '%s' to '%s'",
                        file_path.name,
                        target.name,
                    )
                else:
                    logger.info("Renamed '%s' to '%s'", file_path.name, target.name)
                renamed_count += 1
        except Exception as e:
            logger.exception("Failed to process %s: %s", file_path, e)
            failed_count += 1

    if config.export_metadata_path and export_rows:
        _write_json_or_csv(
            Path(config.export_metadata_path),
            export_rows,
            [
                "path",
                "new_name",
                "category",
                "summary",
                "keywords",
                "invoice_id",
                "amount",
                "company",
            ],
        )

    if config.plan_file_path and plan_entries:
        plan_path = Path(config.plan_file_path)
        _write_json_or_csv(plan_path, plan_entries, ["old", "new"])
        logger.info("Wrote rename plan (%s entries) to %s", len(plan_entries), plan_path)

    if not files:
        logger.info("No PDFs found in %s", path)
        print(f"No PDFs found in {path}.", file=sys.stderr)
    else:
        logger.info(
            "Summary: %s file(s) processed, %s renamed, %s skipped, %s failed",
            len(files),
            renamed_count,
            skipped_count,
            failed_count,
        )
        print(
            f"Processed {len(files)}, renamed {renamed_count}, skipped {skipped_count}, failed {failed_count}.",
            file=sys.stderr,
        )


def run_watch_loop(
    directory: str | Path,
    *,
    config: RenamerConfig,
    interval_seconds: float = 60.0,
) -> None:
    """Run rename in a loop, scanning the directory every interval_seconds. Processes new/changed PDFs."""
    path = Path(directory).resolve()
    if not path.is_dir():
        raise NotADirectoryError(f"Not a directory: {path}")

    stop_event = False

    def handle_stop(sig, frame):
        nonlocal stop_event
        logger.info("Watch mode: received signal %s, stopping...", sig)
        stop_event = True

    # Handle SIGTERM (Docker/systemd) and SIGINT (Ctrl+C)
    original_sigterm = signal.signal(signal.SIGTERM, handle_stop)
    original_sigint = signal.signal(signal.SIGINT, handle_stop)

    seen: dict[Path, float] = {}
    logger.info("Watch mode: scanning %s every %.1fs (Ctrl+C or SIGTERM to stop)", path, interval_seconds)
    try:
        while not stop_event:
            try:
                # Cleanup 'seen' map: remove files that no longer exist
                to_remove = [p for p in seen if not p.exists()]
                for p in to_remove:
                    del seen[p]

                files = _collect_pdf_files(
                    path,
                    recursive=config.recursive,
                    max_depth=config.max_depth,
                    include_patterns=config.include_patterns,
                    exclude_patterns=config.exclude_patterns,
                    skip_if_already_named=config.skip_if_already_named,
                    files_override=None,
                )
                to_process: list[Path] = []
                for p in files:
                    try:
                        mtime = p.stat().st_mtime
                    except OSError:
                        continue
                    if p not in seen or seen[p] != mtime:
                        to_process.append(p)
                        seen[p] = mtime
                if to_process:
                    to_process.sort(key=lambda p: seen.get(p, 0), reverse=True)
                    for single in to_process:
                        if stop_event:
                            break
                        rename_pdfs_in_directory(
                            path,
                            config=config,
                            files_override=[single],
                        )
                if not stop_event:
                    time.sleep(interval_seconds)
            except Exception as exc:
                if not stop_event:
                    logger.exception("Watch iteration failed: %s", exc)
                    time.sleep(interval_seconds)
    finally:
        # Restore original handlers
        signal.signal(signal.SIGTERM, original_sigterm)
        signal.signal(signal.SIGINT, original_sigint)
        logger.info("Watch stopped")
