"""
CLI argument parser construction. Used by cli.main(); build_parser() returns the parser.
"""

from __future__ import annotations

import argparse

from ...extraction.pdf import DEFAULT_MAX_CONTENT_TOKENS
from .parser_sections import (
    add_dirs_and_file_args,
    add_language_case_project_args,
    add_output_and_ux_args,
    add_template_and_plan_args,
)


def _add_heuristic_args(p: argparse._ActionsContainer) -> None:
    """Register heuristic command-line flags on the shared parser."""
    _add_llm_heuristic_mode_args(p)
    _add_date_heuristic_args(p)
    _add_score_heuristic_args(p)
    _add_conflict_heuristic_args(p)
    _add_long_doc_heuristic_args(p)


def _add_llm_heuristic_mode_args(p: argparse._ActionsContainer) -> None:
    """Register llm heuristic mode command-line flags on the shared parser."""
    p.add_argument(
        "--no-llm",
        dest="use_llm",
        action="store_false",
        help="Do not call LLM; use heuristics only for category, empty summary/keywords.",
    )
    p.add_argument(
        "--prefer-llm",
        dest="prefer_llm_category",
        action="store_true",
        help="On category conflict, use LLM (default). Heuristic fills gaps only.",
    )
    p.add_argument(
        "--lenient-llm-json",
        dest="lenient_llm_json",
        action="store_true",
        help="Try to extract JSON from LLM responses that don't start with '{' (regex fallback).",
    )
    p.add_argument(
        "--no-timestamp-fallback",
        dest="use_timestamp_fallback",
        action="store_false",
        help="Do not use date+segment+time filename when category/summary/keywords are all empty.",
    )
    p.add_argument(
        "--timestamp-fallback-segment",
        dest="timestamp_fallback_segment",
        default="document",
        metavar="NAME",
        help="Segment name when using timestamp fallback (default: document).",
    )
    p.add_argument(
        "--simple-naming",
        dest="simple_naming_mode",
        action="store_true",
        help="Use single LLM call for short filename only (3-6 words), skip full category/summary pipeline.",
    )
    p.add_argument(
        "--prefer-heuristic",
        dest="prefer_heuristic",
        action="store_true",
        help="On category conflict, use the heuristic result instead of the LLM result.",
    )


def _add_date_heuristic_args(p: argparse._ActionsContainer) -> None:
    """Register date heuristic command-line flags on the shared parser."""
    p.add_argument(
        "--date-format",
        dest="date_locale",
        default=None,
        choices=["dmy", "mdy"],
        help="Date order: dmy (day-month-year) or mdy (month-day-year). Default: dmy",
    )
    p.add_argument(
        "--date-prefer-leading-chars",
        dest="date_prefer_leading_chars",
        type=int,
        default=8000,
        metavar="N",
        help="Prefer date from first N chars of text (default 8000). Use 0 to search full text.",
    )
    p.add_argument(
        "--no-pdf-metadata-date",
        dest="use_pdf_metadata_for_date",
        action="store_false",
        help="Do not use PDF CreationDate/ModDate as date fallback when content has no date.",
    )
    p.add_argument(
        "--dry-run",
        dest="dry_run",
        action="store_true",
        help="Do not rename files; only log what would be done.",
    )


def _add_score_heuristic_args(p: argparse._ActionsContainer) -> None:
    """Register score heuristic command-line flags on the shared parser."""
    p.add_argument(
        "--min-heuristic-gap",
        dest="min_heuristic_score_gap",
        type=float,
        default=0.0,
        metavar="DELTA",
        help="Heuristic best category must lead by DELTA; else 'unknown'. Default: 0",
    )
    p.add_argument(
        "--min-heuristic-score",
        dest="min_heuristic_score",
        type=float,
        default=0.0,
        metavar="T",
        help="If heuristic score < T, prefer LLM category. Default: 0",
    )
    p.add_argument(
        "--title-weight-region",
        dest="title_weight_region",
        type=int,
        default=2000,
        metavar="N",
        help="Weight matches in first N chars by title-weight-factor (default 2000). Use 0 to disable.",
    )
    p.add_argument(
        "--title-weight-factor",
        dest="title_weight_factor",
        type=float,
        default=1.5,
        metavar="F",
        help="Multiplier for matches in title region. Default: 1.5",
    )
    p.add_argument(
        "--max-score-per-category",
        dest="max_score_per_category",
        type=float,
        default=None,
        metavar="M",
        help="Cap heuristic score per category at M. Default: no cap",
    )


def _add_conflict_heuristic_args(p: argparse._ActionsContainer) -> None:
    """Register conflict heuristic command-line flags on the shared parser."""
    _add_conflict_resolution_args(p)
    _add_conflict_threshold_args(p)


def _add_conflict_resolution_args(p: argparse._ActionsContainer) -> None:
    """Register conflict resolution command-line flags on the shared parser."""
    p.add_argument(
        "--no-keyword-overlap",
        dest="use_keyword_overlap_for_category",
        action="store_false",
        help="Disable keyword-overlap on category conflict (default: overlap on)",
    )
    p.add_argument(
        "--category-display",
        dest="category_display",
        default="specific",
        choices=["specific", "with_parent", "parent_only"],
        help="Category in filename: specific | with_parent | parent_only",
    )
    p.add_argument(
        "--skip-llm-if-heuristic-score-ge",
        dest="skip_llm_category_if_heuristic_score_ge",
        type=float,
        default=None,
        metavar="S",
        help="Skip LLM category when heuristic score >= S (use with gap-ge)",
    )
    p.add_argument(
        "--skip-llm-if-heuristic-gap-ge",
        dest="skip_llm_category_if_heuristic_gap_ge",
        type=float,
        default=None,
        metavar="G",
        help="Skip LLM category when heuristic gap >= G (use with score-ge)",
    )
    p.add_argument(
        "--heuristic-suggestions-top-n",
        dest="heuristic_suggestions_top_n",
        type=int,
        default=5,
        metavar="N",
        help="Top-N heuristic categories passed to LLM as suggestions (default 5)",
    )


def _add_conflict_threshold_args(p: argparse._ActionsContainer) -> None:
    """Register conflict threshold command-line flags on the shared parser."""
    p.add_argument(
        "--heuristic-score-weight",
        dest="heuristic_score_weight",
        type=float,
        default=0.15,
        metavar="W",
        help="Weight heuristic score in overlap comparison (default 0.15)",
    )
    p.add_argument(
        "--heuristic-override-min-score",
        dest="heuristic_override_min_score",
        type=float,
        default=None,
        metavar="S",
        help="When heuristic score >= S and gap >= override-min-gap, use heuristic",
    )
    p.add_argument(
        "--heuristic-override-min-gap",
        dest="heuristic_override_min_gap",
        type=float,
        default=None,
        metavar="G",
        help="When heuristic gap >= G and score >= override-min-score, use heuristic",
    )
    p.add_argument(
        "--no-heuristic-override",
        dest="no_heuristic_override",
        action="store_true",
        help="Disable high-confidence heuristic override (default: override at score>=0.55, gap>=0.3)",
    )
    p.add_argument(
        "--no-constrained-llm",
        dest="use_constrained_llm_category",
        action="store_false",
        help="Do not restrict LLM to heuristic category list (default: constrained)",
    )


def _add_long_doc_heuristic_args(p: argparse._ActionsContainer) -> None:
    """Register long doc heuristic command-line flags on the shared parser."""
    p.add_argument(
        "--heuristic-leading-chars",
        dest="heuristic_leading_chars",
        type=int,
        default=0,
        metavar="N",
        help="Use only first N chars of text for category heuristic (0 = full text)",
    )
    p.add_argument(
        "--heuristic-long-doc-threshold",
        dest="heuristic_long_doc_chars_threshold",
        type=int,
        default=40000,
        metavar="N",
        help="When text length >= N, use first --heuristic-long-doc-leading chars (0=off)",
    )
    p.add_argument(
        "--heuristic-long-doc-leading",
        dest="heuristic_long_doc_leading_chars",
        type=int,
        default=12000,
        metavar="N",
        help="For long docs, use first N chars for heuristic (default 12000)",
    )
    p.add_argument(
        "--max-pages-for-extraction",
        dest="max_pages_for_extraction",
        type=int,
        default=0,
        metavar="N",
        help="Extract text only from first N pages of each PDF (0 = all pages)",
    )


def _add_llm_args(p: argparse._ActionsContainer) -> None:
    """Register llm command-line flags on the shared parser."""
    _add_llm_backend_args(p)
    _add_llm_limit_args(p)
    _add_llm_vision_args(p)
    _add_llm_call_mode_args(p)
    _add_llm_preset_args(p)


def _add_llm_backend_args(p: argparse._ActionsContainer) -> None:
    """Register LLM transport and connection flags on the shared parser."""
    _add_llm_transport_args(p)
    _add_llm_connection_args(p)


def _add_llm_transport_args(p: argparse._ActionsContainer) -> None:
    """Register llm transport command-line flags on the shared parser."""
    p.add_argument(
        "--require-https",
        dest="require_https",
        action="store_true",
        default=False,
        help=(
            "Require HTTPS for non-localhost LLM endpoints (raises error instead of warning). "
            "Env: FOLIONYM_REQUIRE_HTTPS=1"
        ),
    )


def _add_llm_connection_args(p: argparse._ActionsContainer) -> None:
    """Register llm connection command-line flags on the shared parser."""
    p.add_argument(
        "--llm-url",
        dest="llm_base_url",
        default=None,
        metavar="URL",
        help=(
            "LLM HTTP endpoint URL "
            "(default: env FOLIONYM_LLM_URL or preset-driven http://127.0.0.1:11434/v1/completions)"
        ),
    )
    p.add_argument(
        "--llm-model",
        dest="llm_model",
        default=None,
        metavar="MODEL",
        help="LLM model name for HTTP backend (default: env FOLIONYM_LLM_MODEL or 'default')",
    )
    p.add_argument(
        "--llm-timeout",
        dest="llm_timeout_s",
        type=float,
        default=None,
        metavar="SEC",
        help="LLM request timeout in seconds (default: env FOLIONYM_LLM_TIMEOUT or 60)",
    )


def _add_llm_limit_args(p: argparse._ActionsContainer) -> None:
    """Register llm limit command-line flags on the shared parser."""
    p.add_argument(
        "--max-tokens",
        dest="max_tokens_for_extraction",
        type=int,
        default=None,
        metavar="N",
        help=(f"Max tokens for PDF text extraction (default: env FOLIONYM_MAX_TOKENS or {DEFAULT_MAX_CONTENT_TOKENS})"),
    )
    p.add_argument(
        "--max-content-chars",
        dest="max_content_chars",
        type=int,
        default=None,
        metavar="N",
        help="Cap chars of text sent to LLM. Default: env FOLIONYM_MAX_CONTENT_CHARS or no cap.",
    )
    p.add_argument(
        "--max-content-tokens",
        dest="max_content_tokens",
        type=int,
        default=None,
        metavar="N",
        help="Cap tokens for LLM when tiktoken installed. Default: env FOLIONYM_MAX_CONTENT_TOKENS.",
    )


def _add_llm_vision_args(p: argparse._ActionsContainer) -> None:
    """Register llm vision command-line flags on the shared parser."""
    p.add_argument(
        "--vision-fallback",
        dest="use_vision_fallback",
        action="store_true",
        help="When text extraction is short, use LLM vision on first page (requires vision-capable model).",
    )
    p.add_argument(
        "--vision-fallback-min-len",
        dest="vision_fallback_min_text_len",
        type=int,
        default=50,
        metavar="N",
        help="Use vision fallback when extracted text length < N (default: 50).",
    )
    p.add_argument(
        "--vision-model",
        dest="vision_model",
        default=None,
        metavar="MODEL",
        help="Model for vision (default: same as --llm-model; e.g. llava for vision-capable models).",
    )
    p.add_argument(
        "--vision-first",
        dest="vision_first",
        action="store_true",
        help="Vision on first page first; else extract text (scan-only; needs vision-capable model).",
    )


def _add_llm_call_mode_args(p: argparse._ActionsContainer) -> None:
    """Register llm call mode command-line flags on the shared parser."""
    p.add_argument(
        "--no-single-llm-call",
        dest="use_single_llm_call",
        action="store_false",
        help="Use separate LLM calls for summary, keywords, category instead of one combined call.",
    )
    p.add_argument(
        "--no-chat-api",
        dest="llm_use_chat_api",
        action="store_false",
        help="Use /v1/completions instead of /v1/chat/completions for LLM text calls.",
    )
    p.add_argument(
        "--no-json-mode",
        dest="llm_json_mode",
        action="store_false",
        help="Do not request JSON mode (response_format) from the LLM server.",
    )


def _add_llm_preset_args(p: argparse._ActionsContainer) -> None:
    """Register llm preset command-line flags on the shared parser."""
    p.add_argument(
        "--preset",
        dest="preset",
        default=None,
        choices=["high-confidence-heuristic", "scanned", "fast", "accurate", "batch"],
        help=(
            "Preset: high-confidence-heuristic, scanned, fast (heuristics-first), "
            "accurate (more LLM analysis), or batch (higher workers + persistent cache)."
        ),
    )
    p.add_argument(
        "--llm-preset",
        dest="llm_preset",
        default=None,
        choices=["apple-silicon", "gpu"],
        help="Hardware profile: apple-silicon (default, Qwen 2.5 3B) or gpu (Qwen 2.5 7B). Sets model, context limits.",
    )


def build_parser() -> argparse.ArgumentParser:
    """Build and return the main CLI argument parser."""
    p = argparse.ArgumentParser(
        prog="folionym",
        description=(
            "Local-first document naming for PDFs. "
            "Extracts text via PyMuPDF, classifies via heuristics or LLM, "
            "and generates structured filenames (date-category-keywords.pdf)."
        ),
        epilog=(
            "Examples:\n"
            "  folionym --dir ./invoices --dry-run\n"
            "  folionym --file report.pdf --manual\n"
            "  folionym --dir ./scans --preset scanned --ocr\n"
            "  folionym --dir ./archive --preset batch --progress\n"
            "  folionym --doctor\n"
            "\n"
            "Config: defaults can be set in a JSON/YAML file via --config.\n"
            "Environment: FOLIONYM_LLM_URL, FOLIONYM_LLM_MODEL, etc."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.color = False
    add_dirs_and_file_args(p.add_argument_group("Input"))
    add_template_and_plan_args(p.add_argument_group("Filename template & plan"))
    add_language_case_project_args(p.add_argument_group("Naming"))
    _add_heuristic_args(p.add_argument_group("Heuristics & classification"))
    _add_llm_args(p.add_argument_group("LLM backend"))
    add_output_and_ux_args(p.add_argument_group("Output & logging"))
    return p
