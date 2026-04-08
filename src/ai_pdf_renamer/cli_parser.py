"""
CLI argument parser construction. Used by cli.main(); build_parser() returns the parser.
"""

from __future__ import annotations

import argparse

from .pdf_extract import DEFAULT_MAX_CONTENT_TOKENS
from .text_utils import VALID_CASE_CHOICES


def _add_dirs_and_file_args(p: argparse._ActionsContainer) -> None:
    p.add_argument(
        "--doctor",
        dest="doctor",
        action="store_true",
        help="Check dependencies, data files, and LLM connectivity, then exit.",
    )
    p.add_argument(
        "--validate-config",
        dest="validate_config",
        action="store_true",
        help="Validate CLI/env/config-file settings and exit without processing files.",
    )
    p.add_argument(
        "--dir",
        dest="dirs",
        nargs="*",
        default=None,
        metavar="DIR",
        help="One or more directories containing PDFs. Interactive prompt if omitted.",
    )
    p.add_argument(
        "--dirs-from-file",
        dest="dirs_from_file",
        default=None,
        metavar="FILE",
        help="Read directory paths from a file (one per line). Combined with --dir.",
    )
    p.add_argument(
        "--file",
        dest="single_file",
        default=None,
        metavar="PATH",
        help="Process a single PDF. Overrides --dir.",
    )
    p.add_argument(
        "--manual",
        dest="manual_file",
        default=None,
        metavar="PATH",
        help="Interactive single-file mode: suggest name, show metadata, prompt to confirm/edit.",
    )
    p.add_argument(
        "--recursive",
        "-r",
        dest="recursive",
        action="store_true",
        help="Recursively collect PDFs from subdirectories.",
    )
    p.add_argument(
        "--max-depth",
        dest="max_depth",
        type=int,
        default=0,
        metavar="N",
        help="Max directory depth when --recursive (0 = unlimited).",
    )
    p.add_argument(
        "--include",
        dest="include_patterns",
        action="append",
        default=None,
        metavar="PATTERN",
        help="Include only files matching fnmatch PATTERN (e.g. *.pdf). Can be repeated.",
    )
    p.add_argument(
        "--exclude",
        dest="exclude_patterns",
        action="append",
        default=None,
        metavar="PATTERN",
        help="Exclude files matching fnmatch PATTERN (e.g. draft-*). Can be repeated.",
    )


def _add_template_and_plan_args(p: argparse._ActionsContainer) -> None:
    p.add_argument(
        "--template",
        dest="filename_template",
        default=None,
        metavar="TEMPLATE",
        help="Template placeholders: {date}, {project}, {category}, {keywords}, {summary}, {version}, "
        "{invoice_id}, {amount}, {company}.",
    )
    p.add_argument(
        "--no-structured-fields",
        dest="use_structured_fields",
        action="store_false",
        help="Do not extract invoice_id, amount, company from content for template placeholders.",
    )
    p.add_argument(
        "--plan-file",
        dest="plan_file_path",
        default=None,
        metavar="FILE",
        help="Write rename plan (old,new) to FILE without applying. JSON or .csv.",
    )
    p.add_argument(
        "--interactive",
        "-i",
        dest="interactive",
        action="store_true",
        help="Prompt for each file (y/n/e=edit) before renaming.",
    )
    p.add_argument(
        "--watch",
        dest="watch",
        action="store_true",
        help="Watch directory and process new PDFs periodically.",
    )
    p.add_argument(
        "--watch-interval",
        dest="watch_interval",
        type=float,
        default=60.0,
        metavar="SEC",
        help="Seconds between watch scans (default 60).",
    )
    p.add_argument(
        "--write-pdf-metadata",
        dest="write_pdf_metadata",
        action="store_true",
        help="Write new filename as PDF /Title metadata after rename.",
    )


def _add_language_case_project_args(p: argparse._ActionsContainer) -> None:
    p.add_argument("--language", default=None, choices=["de", "en"], help="LLM prompt language (default: de).")
    p.add_argument(
        "--case",
        dest="desired_case",
        default=None,
        choices=list(VALID_CASE_CHOICES),
        help="Filename case style (default: kebabCase).",
    )
    p.add_argument("--project", default=None, help="Project prefix in generated filenames.")
    p.add_argument("--version", default=None, help="Version suffix in generated filenames.")


def _add_heuristic_args(p: argparse._ActionsContainer) -> None:
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
        help="On category conflict, use heuristic instead of LLM (legacy override).",
    )
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
    p.add_argument(
        "--no-keyword-overlap",
        dest="use_keyword_overlap_for_category",
        action="store_false",
        help="Disable keyword-overlap on category conflict (default: overlap on)",
    )
    p.add_argument(
        "--embeddings-conflict",
        dest="use_embeddings_for_conflict",
        action="store_true",
        help="Use embedding similarity (sentence-transformers) for conflict resolution. Requires [embeddings].",
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
    p.add_argument(
        "--llm-backend",
        dest="llm_backend",
        default=None,
        choices=["http", "in-process", "auto"],
        help=(
            "LLM backend: http (any OpenAI-compatible server, default), "
            "in-process (llama-cpp-python, requires --llm-model-path), "
            "or auto (use in-process if --llm-model-path set, else http). "
            "Env: AI_PDF_RENAMER_LLM_BACKEND"
        ),
    )
    p.add_argument(
        "--llm-model-path",
        dest="llm_model_path",
        default=None,
        metavar="PATH",
        help="Path to GGUF model file for in-process backend (llama-cpp-python). Env: AI_PDF_RENAMER_LLM_MODEL_PATH",
    )
    p.add_argument(
        "--require-https",
        dest="require_https",
        action="store_true",
        default=False,
        help=(
            "Require HTTPS for non-localhost LLM endpoints (raises error instead of warning). "
            "Env: AI_PDF_RENAMER_REQUIRE_HTTPS=1"
        ),
    )
    p.add_argument(
        "--llm-url",
        dest="llm_base_url",
        default=None,
        metavar="URL",
        help="LLM HTTP endpoint URL (default: env AI_PDF_RENAMER_LLM_URL or http://127.0.0.1:8080/v1/completions)",
    )
    p.add_argument(
        "--llm-model",
        dest="llm_model",
        default=None,
        metavar="MODEL",
        help="LLM model name for HTTP backend (default: env AI_PDF_RENAMER_LLM_MODEL or 'default')",
    )
    p.add_argument(
        "--llm-timeout",
        dest="llm_timeout_s",
        type=float,
        default=None,
        metavar="SEC",
        help="LLM request timeout in seconds (default: env AI_PDF_RENAMER_LLM_TIMEOUT or 60)",
    )
    p.add_argument(
        "--max-tokens",
        dest="max_tokens_for_extraction",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Max tokens for PDF text extraction "
            f"(default: env AI_PDF_RENAMER_MAX_TOKENS or {DEFAULT_MAX_CONTENT_TOKENS})"
        ),
    )
    p.add_argument(
        "--max-content-chars",
        dest="max_content_chars",
        type=int,
        default=None,
        metavar="N",
        help="Cap chars of text sent to LLM. Default: env AI_PDF_RENAMER_MAX_CONTENT_CHARS or no cap.",
    )
    p.add_argument(
        "--max-content-tokens",
        dest="max_content_tokens",
        type=int,
        default=None,
        metavar="N",
        help="Cap tokens for LLM when tiktoken installed. Default: env AI_PDF_RENAMER_MAX_CONTENT_TOKENS.",
    )
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
    p.add_argument(
        "--preset",
        dest="preset",
        default=None,
        choices=["high-confidence-heuristic", "scanned", "fast", "accurate", "batch"],
        help=(
            "Preset: high-confidence-heuristic, scanned, fast (heuristics-first), "
            "accurate (more LLM + embeddings), or batch (higher workers + persistent cache)."
        ),
    )
    p.add_argument(
        "--llm-preset",
        dest="llm_preset",
        default=None,
        choices=["apple-silicon", "gpu"],
        help="Hardware profile: apple-silicon (default, Qwen 2.5 3B) or gpu (Qwen 2.5 7B). Sets model, context limits.",
    )


def _add_output_and_ux_args(p: argparse._ActionsContainer) -> None:
    p.add_argument(
        "--quiet",
        dest="quiet",
        action="store_true",
        help="Less output (log level WARNING). Overridden by --verbose.",
    )
    p.add_argument(
        "--verbose",
        dest="verbose",
        action="store_true",
        help="More output (log level DEBUG). Overrides --quiet.",
    )
    p.add_argument(
        "--log-file",
        dest="log_file",
        default=None,
        metavar="PATH",
        help="Log file path (default: env AI_PDF_RENAMER_LOG_FILE or error.log)",
    )
    p.add_argument(
        "--log-level",
        dest="log_level",
        default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log level (default: env AI_PDF_RENAMER_LOG_LEVEL or INFO). Overridden by --verbose/--quiet.",
    )
    p.add_argument(
        "--ocr",
        dest="use_ocr",
        action="store_true",
        help="Run OCR (OCRmyPDF) when PDF has little/no text (scanned PDFs). Requires [ocr] and Tesseract.",
    )
    p.add_argument(
        "--skip-already-named",
        dest="skip_if_already_named",
        action="store_true",
        help="Skip PDFs whose name already matches YYYYMMDD-*.pdf.",
    )
    p.add_argument(
        "--backup-dir",
        dest="backup_dir",
        default=None,
        metavar="DIR",
        help="Copy each PDF to DIR before renaming (for undo).",
    )
    p.add_argument(
        "--rename-log",
        dest="rename_log_path",
        default=None,
        metavar="FILE",
        help="Append old_path\\tnew_path to FILE after each rename.",
    )
    p.add_argument(
        "--export-metadata",
        dest="export_metadata_path",
        default=None,
        metavar="FILE",
        help="Write proposed renames + category/summary/keywords to CSV or JSON (.csv/.json).",
    )
    p.add_argument(
        "--summary-json",
        dest="summary_json_path",
        default=None,
        metavar="FILE",
        help="Write run summary JSON (processed/renamed/skipped/failed and failure details).",
    )
    p.add_argument(
        "--max-filename-chars",
        dest="max_filename_chars",
        type=int,
        default=None,
        metavar="N",
        help="Truncate generated filenames to N characters (at separator).",
    )
    p.add_argument(
        "--config",
        dest="config",
        default=None,
        metavar="FILE",
        help="Load defaults from JSON or YAML file; CLI options override.",
    )
    p.add_argument(
        "--override-category-file",
        dest="override_category_file",
        default=None,
        metavar="FILE",
        help="CSV with filename,category to force category per file.",
    )
    p.add_argument(
        "--rules-file",
        dest="rules_file",
        default=None,
        metavar="FILE",
        help="JSON rules: skip_llm_if_heuristic_category, force_category_by_pattern, skip_files_by_pattern.",
    )
    p.add_argument(
        "--post-rename-hook",
        dest="post_rename_hook",
        default=None,
        metavar="CMD",
        help="Command run after each successful rename. Receives paths/meta via AI_PDF_RENAMER_* env vars.",
    )
    p.add_argument(
        "--workers",
        dest="workers",
        type=int,
        default=1,
        metavar="N",
        help="Parallel workers for extract+generate (default 1). Renames applied sequentially.",
    )
    p.add_argument(
        "--cache-dir",
        dest="cache_dir",
        default=None,
        metavar="DIR",
        help="Persistent cache directory for LLM responses. Env: AI_PDF_RENAMER_CACHE_DIR",
    )
    p.add_argument(
        "--no-cache",
        dest="use_cache",
        action="store_false",
        help="Disable response caching for LLM calls.",
    )
    p.add_argument(
        "--progress",
        dest="progress",
        action="store_true",
        help="Show a Rich progress bar during processing. Opt-in to keep piped output stable.",
    )
    p.add_argument(
        "--quiet-progress",
        dest="quiet_progress",
        action="store_true",
        help="Show compact percentage-only progress output.",
    )
    p.add_argument(
        "--explain",
        dest="explain",
        action="store_true",
        help="Log detailed classification reasoning: heuristic scores, LLM outputs, and conflict resolution.",
    )


def build_parser() -> argparse.ArgumentParser:
    """Build and return the main CLI argument parser."""
    p = argparse.ArgumentParser(
        prog="ai-pdf-renamer",
        description=(
            "Rename PDFs based on content analysis. "
            "Extracts text via PyMuPDF, classifies via heuristics or LLM, "
            "and generates structured filenames (date-category-keywords.pdf)."
        ),
        epilog=(
            "Examples:\n"
            "  ai-pdf-renamer --dir ./invoices --dry-run\n"
            "  ai-pdf-renamer --file report.pdf --manual\n"
            "  ai-pdf-renamer --dir ./scans --preset scanned --ocr\n"
            "  ai-pdf-renamer --dir ./archive --preset batch --progress\n"
            "  ai-pdf-renamer --doctor\n"
            "\n"
            "Config: defaults can be set in a JSON/YAML file via --config.\n"
            "Environment: AI_PDF_RENAMER_LLM_URL, AI_PDF_RENAMER_LLM_MODEL, etc."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_dirs_and_file_args(p.add_argument_group("Input"))
    _add_template_and_plan_args(p.add_argument_group("Filename template & plan"))
    _add_language_case_project_args(p.add_argument_group("Naming"))
    _add_heuristic_args(p.add_argument_group("Heuristics & classification"))
    _add_llm_args(p.add_argument_group("LLM backend"))
    _add_output_and_ux_args(p.add_argument_group("Output & logging"))
    return p
