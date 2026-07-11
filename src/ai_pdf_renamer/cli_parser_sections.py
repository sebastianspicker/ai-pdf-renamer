"""Reusable CLI parser sections."""

from __future__ import annotations

import argparse

from .text_utils import VALID_CASE_CHOICES


def add_dirs_and_file_args(p: argparse._ActionsContainer) -> None:
    add_input_mode_args(p)
    add_discovery_filter_args(p)


def add_input_mode_args(p: argparse._ActionsContainer) -> None:
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


def add_discovery_filter_args(p: argparse._ActionsContainer) -> None:
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


def add_template_and_plan_args(p: argparse._ActionsContainer) -> None:
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


def add_language_case_project_args(p: argparse._ActionsContainer) -> None:
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


def add_output_and_ux_args(p: argparse._ActionsContainer) -> None:
    add_logging_args(p)
    add_extraction_output_args(p)
    add_file_side_effect_args(p)
    add_runtime_control_args(p)


def add_logging_args(p: argparse._ActionsContainer) -> None:
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
        help="Log file path (default: env AI_PDF_RENAMER_LOG_FILE or ~/.local/share/ai-pdf-renamer/error.log)",
    )
    p.add_argument(
        "--log-level",
        dest="log_level",
        default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log level (default: env AI_PDF_RENAMER_LOG_LEVEL or INFO). Overridden by --verbose/--quiet.",
    )


def add_extraction_output_args(p: argparse._ActionsContainer) -> None:
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


def add_file_side_effect_args(p: argparse._ActionsContainer) -> None:
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
        metavar="URL",
        help="HTTP(S) endpoint called after each successful rename. Local command hooks are not executed.",
    )


def add_runtime_control_args(p: argparse._ActionsContainer) -> None:
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
        help=(
            "Log detailed classification reasoning: heuristic scores, LLM outputs, and conflict resolution. "
            "May include sensitive raw LLM outputs, document excerpts, summaries, or keywords in the "
            "configured log sink."
        ),
    )
