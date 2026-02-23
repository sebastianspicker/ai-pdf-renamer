from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
from collections.abc import Callable
from pathlib import Path

import requests

from .logging_utils import setup_logging
from .renamer import (
    RenamerConfig,
    rename_pdfs_in_directory,
    run_watch_loop,
)
from .text_utils import VALID_CASE_CHOICES

logger = logging.getLogger(__name__)


def _load_config_file(path: str | Path) -> dict:
    """Load JSON or YAML config file. Returns a dict (empty on error or unknown format)."""
    p = Path(path)
    if not p.exists():
        return {}
    try:
        raw = p.read_text(encoding="utf-8")
    except OSError:
        return {}
    suf = p.suffix.lower()
    if suf == ".json":
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {}
    if suf in (".yaml", ".yml"):
        try:
            import yaml

            return yaml.safe_load(raw) or {}
        except Exception:
            return {}
    return {}


def _load_override_category_map(path: str | Path) -> dict[str, str]:
    """Load CSV with columns filename,category (or path,category). Returns dict filename -> category."""
    result: dict[str, str] = {}
    p = Path(path)
    if not p.exists():
        return result
    try:
        with open(p, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                name = (row.get("filename") or row.get("path") or row.get("file") or "").strip()
                cat = (row.get("category") or "").strip()
                if name and cat:
                    result[name] = cat
    except OSError:
        pass
    return result


def _is_interactive() -> bool:
    """True if stdin is a TTY (interactive prompt is safe)."""
    return sys.stdin.isatty()


def _add_dirs_and_file_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--dir",
        dest="dirs",
        nargs="*",
        default=None,
        metavar="DIR",
        help="Directories with PDFs. If omitted: interactive prompts (default ./input_files); "
        "non-interactive needs --dir or --file.",
    )
    p.add_argument(
        "--dirs-from-file",
        dest="dirs_from_file",
        default=None,
        metavar="FILE",
        help="Read directory paths from file (one per line). Combined with --dir.",
    )
    p.add_argument(
        "--file",
        dest="single_file",
        default=None,
        metavar="PATH",
        help="Process a single PDF file (path to file). Overrides --dir.",
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


def _add_template_and_plan_args(p: argparse.ArgumentParser) -> None:
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


def _add_language_case_project_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--language", default=None, choices=["de", "en"], help="Prompt language")
    p.add_argument(
        "--case",
        dest="desired_case",
        default=None,
        choices=list(VALID_CASE_CHOICES),
        help="Filename case format",
    )
    p.add_argument("--project", default=None, help="Optional project name")
    p.add_argument("--version", default=None, help="Optional version")


def _add_heuristic_args(p: argparse.ArgumentParser) -> None:
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


def _add_llm_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--llm-url",
        dest="llm_base_url",
        default=None,
        metavar="URL",
        help="LLM endpoint URL (default: env AI_PDF_RENAMER_LLM_URL or http://127.0.0.1:11434/v1/completions)",
    )
    p.add_argument(
        "--llm-model",
        dest="llm_model",
        default=None,
        metavar="MODEL",
        help="LLM model name (default: env AI_PDF_RENAMER_LLM_MODEL or qwen3:8b)",
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
        help="Max tokens for PDF text extraction (default: env AI_PDF_RENAMER_MAX_TOKENS or 120000)",
    )
    p.add_argument(
        "--preset",
        dest="preset",
        default=None,
        choices=["high-confidence-heuristic"],
        help="Apply preset: high-confidence-heuristic = skip LLM when heuristic score>=0.5, gap>=0.3",
    )


def _add_output_and_ux_args(p: argparse.ArgumentParser) -> None:
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
        "--workers",
        dest="workers",
        type=int,
        default=1,
        metavar="N",
        help="Parallel workers for extract+generate (default 1). Renames applied sequentially.",
    )


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Rename PDFs based on their content.")
    _add_dirs_and_file_args(p.add_argument_group("Directories and files"))
    _add_template_and_plan_args(p.add_argument_group("Template and plan"))
    _add_language_case_project_args(p.add_argument_group("Language, case, project"))
    _add_heuristic_args(p.add_argument_group("Heuristics and behaviour"))
    _add_llm_args(p.add_argument_group("LLM"))
    _add_output_and_ux_args(p.add_argument_group("Output and UX"))
    return p


def _resolve_option(
    args: argparse.Namespace,
    attr: str,
    file_defaults: dict,
    file_key: str,
    default: str,
    *,
    choice_prompt: str | None = None,
    choices: list[str] | None = None,
    choice_normalize: Callable[[str], str] | None = None,
    free_prompt: str | None = None,
) -> str:
    """Resolve option: getattr(args) -> file_defaults -> interactive prompt (if TTY) -> default."""
    value = getattr(args, attr, None)
    if value is None:
        value = file_defaults.get(file_key)
    # Treat empty string as unset when choices are defined (avoid invalid "" for language/case)
    if value == "" and choices is not None:
        value = None
    if value is None:
        if _is_interactive():
            if choice_prompt is not None and choices is not None:
                value = _prompt_choice(
                    choice_prompt,
                    choices=choices,
                    default=default,
                    normalize=choice_normalize,
                )
            elif free_prompt is not None:
                try:
                    value = input(free_prompt).strip() or default
                except EOFError:
                    print("Error: stdin closed.", file=sys.stderr)
                    sys.exit(1)
                except KeyboardInterrupt:
                    print("Interrupted.", file=sys.stderr)
                    sys.exit(130)
            else:
                value = default
        else:
            value = default
    return value if value is not None else default


def _prompt_choice(
    prompt: str,
    *,
    choices: list[str],
    default: str,
    normalize: Callable[[str], str] | None = None,
) -> str:
    # First occurrence wins when normalized keys collide (e.g. different casing).
    mapping: dict[str, str] = {}
    for c in choices:
        key = normalize(c) if normalize else c
        if key not in mapping:
            mapping[key] = c
    default_key = normalize(default) if normalize else default
    if default_key not in mapping:
        mapping[default_key] = default

    while True:
        try:
            value = input(prompt).strip()
        except EOFError:
            print("Error: stdin closed.", file=sys.stderr)
            sys.exit(1)
        except KeyboardInterrupt:
            print("Interrupted.", file=sys.stderr)
            sys.exit(130)
        if not value:
            return mapping[default_key]
        key = normalize(value) if normalize else value
        if key in mapping:
            return mapping[key]
        print(f"Invalid choice: {value}. Valid choices: {', '.join(choices)}")


def _float_opt(args: argparse.Namespace, attr: str, default: float = 0.0) -> float:
    """Get float from args; default if missing or invalid."""
    try:
        v = getattr(args, attr, None)
        if v is None:
            return default
        return float(v)
    except (TypeError, ValueError):
        return default


def _int_opt(args: argparse.Namespace, attr: str, default: int = 0) -> int:
    """Get int from args; default if missing or invalid."""
    try:
        v = getattr(args, attr, None)
        if v is None:
            return default
        return int(v)
    except (TypeError, ValueError):
        return default


def _bool_opt(args: argparse.Namespace, attr: str, default: bool = False) -> bool:
    """Get bool from args."""
    v = getattr(args, attr, None)
    if v is None:
        return default
    return bool(v)


def _optional_float(args: argparse.Namespace, attr: str) -> float | None:
    """Get optional float from args; None if missing or invalid."""
    v = getattr(args, attr, None)
    if v is None:
        return None
    try:
        f = float(v)
        return f
    except (TypeError, ValueError):
        return None


def _str_opt(args: argparse.Namespace, attr: str, default: str = "") -> str:
    """Get string from args; default if missing or empty."""
    v = getattr(args, attr, None)
    if v is None or (isinstance(v, str) and not v.strip()):
        return default
    return str(v).strip()


def _resolve_log_config(args: argparse.Namespace) -> tuple[str, int]:
    """Resolve log file path and log level from args and env. Returns (log_file_path, log_level)."""
    log_file = getattr(args, "log_file", None) or os.environ.get("AI_PDF_RENAMER_LOG_FILE") or "error.log"
    if getattr(args, "verbose", False):
        log_level = logging.DEBUG
    elif getattr(args, "quiet", False):
        log_level = logging.WARNING
    elif getattr(args, "log_level", None):
        log_level = getattr(logging, args.log_level)
    else:
        env_level = os.environ.get("AI_PDF_RENAMER_LOG_LEVEL", "INFO").upper()
        log_level = getattr(logging, env_level, logging.INFO)
    return (log_file, log_level)


def _resolve_dirs(args: argparse.Namespace) -> tuple[list[str], str | None]:
    """Resolve directory list and optional single-file path from args. Raises SystemExit on error."""
    single_file = getattr(args, "single_file", None)
    dirs: list[str] = list(getattr(args, "dirs", None) or [])
    dirs_from_file = getattr(args, "dirs_from_file", None)
    if dirs_from_file:
        try:
            dirs.extend(
                line.strip() for line in Path(dirs_from_file).read_text(encoding="utf-8").splitlines() if line.strip()
            )
        except OSError as e:
            raise SystemExit(f"Error: could not read --dirs-from-file: {e}") from e
    if single_file:
        single_path = Path(single_file).resolve()
        if not single_path.is_file():
            raise SystemExit(f"Error: --file must be an existing file: {single_path}")
        dirs = [str(single_path.parent)]
    if not dirs:
        if _is_interactive():
            try:
                directory = (
                    input("Path to the directory with PDFs (default: ./input_files): ").strip() or "./input_files"
                )
            except EOFError:
                print("Error: stdin closed.", file=sys.stderr)
                sys.exit(1)
            except KeyboardInterrupt:
                print("Interrupted.", file=sys.stderr)
                sys.exit(130)
            dirs = [directory]
        else:
            raise SystemExit(
                "Error: at least one directory or --file is required in non-interactive mode. "
                "Use --dir PATH or --file PATH."
            )
    # Resolve paths and filter out empty strings
    resolved_dirs = [str(Path(d).resolve()) for d in dirs if d.strip()]
    if not resolved_dirs:
        msg = (
            "Error: --dir must be non-empty. Provide a path or set the directory when prompted."
            if (dirs and not single_file) # Use original 'dirs' to check if any were provided before filtering
            else "Error: at least one directory or --file is required. Use --dir or --file."
        )
        raise SystemExit(msg)
    return (dirs, single_file)


def _build_config_from_args(
    args: argparse.Namespace,
    file_defaults: dict,
) -> RenamerConfig:
    """Build RenamerConfig from args and file defaults (with interactive prompts where applicable)."""
    language = _resolve_option(
        args,
        "language",
        file_defaults,
        "language",
        "de",
        choice_prompt="Language (de/en, default: de): ",
        choices=["de", "en"],
        choice_normalize=str.lower,
    )
    desired_case = _resolve_option(
        args,
        "desired_case",
        file_defaults,
        "desired_case",
        "kebabCase",
        choice_prompt="Desired case format (camelCase, kebabCase, snakeCase, default: kebabCase): ",
        choices=list(VALID_CASE_CHOICES),
        choice_normalize=str.lower,
    )
    project = _resolve_option(
        args,
        "project",
        file_defaults,
        "project",
        "",
        free_prompt="Project name (optional): ",
    )
    version = _resolve_option(
        args,
        "version",
        file_defaults,
        "version",
        "",
        free_prompt="Version (optional): ",
    )

    prefer_llm_category = _bool_opt(args, "prefer_llm_category", False) or not _bool_opt(
        args, "prefer_heuristic", False
    )
    skip_llm_score = _optional_float(args, "skip_llm_category_if_heuristic_score_ge")
    skip_llm_gap = _optional_float(args, "skip_llm_category_if_heuristic_gap_ge")
    if getattr(args, "preset", None) == "high-confidence-heuristic":
        if skip_llm_score is None:
            skip_llm_score = 0.5
        if skip_llm_gap is None:
            skip_llm_gap = 0.3
    heuristic_override_min_score = _optional_float(args, "heuristic_override_min_score")
    heuristic_override_min_gap = _optional_float(args, "heuristic_override_min_gap")
    if _bool_opt(args, "no_heuristic_override", False):
        heuristic_override_min_score = None
        heuristic_override_min_gap = None
    elif heuristic_override_min_score is None and heuristic_override_min_gap is None:
        heuristic_override_min_score = 0.55
        heuristic_override_min_gap = 0.3

    x_max_tokens = getattr(args, "max_tokens_for_extraction", None)
    max_tokens_for_extraction = int(x_max_tokens) if x_max_tokens is not None and x_max_tokens > 0 else None
    x_max_filename = getattr(args, "max_filename_chars", None)
    max_filename_chars = int(x_max_filename) if x_max_filename is not None and x_max_filename > 0 else None

    kwargs: dict = {
        "language": language,
        "desired_case": desired_case,
        "project": project,
        "version": version,
        "prefer_llm_category": prefer_llm_category,
        "date_locale": _str_opt(args, "date_locale", "dmy"),
        "date_prefer_leading_chars": _int_opt(args, "date_prefer_leading_chars", 8000),
        "use_pdf_metadata_for_date": _bool_opt(args, "use_pdf_metadata_for_date", True),
        "dry_run": _bool_opt(args, "dry_run", False),
        "min_heuristic_score_gap": _float_opt(args, "min_heuristic_score_gap", 0.0),
        "min_heuristic_score": _float_opt(args, "min_heuristic_score", 0.0),
        "title_weight_region": _int_opt(args, "title_weight_region", 2000),
        "title_weight_factor": _float_opt(args, "title_weight_factor", 1.5),
        "max_score_per_category": _optional_float(args, "max_score_per_category"),
        "use_keyword_overlap_for_category": _bool_opt(args, "use_keyword_overlap_for_category", True),
        "use_embeddings_for_conflict": _bool_opt(args, "use_embeddings_for_conflict", False),
        "category_display": _str_opt(args, "category_display", "specific"),
        "skip_llm_category_if_heuristic_score_ge": skip_llm_score,
        "skip_llm_category_if_heuristic_gap_ge": skip_llm_gap,
        "heuristic_suggestions_top_n": _int_opt(args, "heuristic_suggestions_top_n", 5),
        "heuristic_score_weight": _float_opt(args, "heuristic_score_weight", 0.15),
        "heuristic_override_min_score": heuristic_override_min_score,
        "heuristic_override_min_gap": heuristic_override_min_gap,
        "use_constrained_llm_category": _bool_opt(args, "use_constrained_llm_category", True),
        "heuristic_leading_chars": _int_opt(args, "heuristic_leading_chars", 0),
        "heuristic_long_doc_chars_threshold": _int_opt(args, "heuristic_long_doc_chars_threshold", 40000),
        "heuristic_long_doc_leading_chars": _int_opt(args, "heuristic_long_doc_leading_chars", 12000),
        "max_pages_for_extraction": _int_opt(args, "max_pages_for_extraction", 0),
        "llm_base_url": getattr(args, "llm_base_url", None) or None,
        "llm_model": getattr(args, "llm_model", None) or None,
        "llm_timeout_s": getattr(args, "llm_timeout_s", None),
        "max_tokens_for_extraction": max_tokens_for_extraction,
        "use_ocr": _bool_opt(args, "use_ocr", False),
        "skip_if_already_named": _bool_opt(args, "skip_if_already_named", False),
        "backup_dir": getattr(args, "backup_dir", None) or None,
        "rename_log_path": getattr(args, "rename_log_path", None) or None,
        "export_metadata_path": getattr(args, "export_metadata_path", None) or None,
        "max_filename_chars": max_filename_chars,
        "override_category_map": (
            _load_override_category_map(args.override_category_file)
            if getattr(args, "override_category_file", None)
            else None
        ),
        "workers": max(1, _int_opt(args, "workers", 1)),
        "recursive": _bool_opt(args, "recursive", False),
        "max_depth": _int_opt(args, "max_depth", 0),
        "include_patterns": getattr(args, "include_patterns", None),
        "exclude_patterns": getattr(args, "exclude_patterns", None),
        "filename_template": getattr(args, "filename_template", None) or file_defaults.get("filename_template"),
        "use_structured_fields": _bool_opt(args, "use_structured_fields", True),
        "plan_file_path": getattr(args, "plan_file_path", None) or None,
        "interactive": _bool_opt(args, "interactive", False),
        "write_pdf_metadata": _bool_opt(args, "write_pdf_metadata", False),
        "use_llm": _bool_opt(args, "use_llm", True),
        "lenient_llm_json": _bool_opt(args, "lenient_llm_json", False),
    }
    try:
        return RenamerConfig(**kwargs)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


def _run_renamer_or_watch(
    dirs: list[str],
    config: RenamerConfig,
    args: argparse.Namespace,
    *,
    single_file: str | None = None,
) -> None:
    """Run watch loop or rename each directory. Raises SystemExit on path/data errors."""
    try:
        if getattr(args, "watch", False):
            if len(dirs) > 1:
                raise SystemExit("Error: --watch supports only one directory. Use a single --dir.")
            run_watch_loop(
                dirs[0],
                config=config,
                interval_seconds=float(getattr(args, "watch_interval", 60) or 60),
            )
        else:
            for directory in dirs:
                if not directory:
                    continue
                path_obj = Path(directory)
                files_override = None
                if single_file:
                    single_path = Path(single_file).resolve()
                    if Path(directory).resolve() == single_path.parent:
                        files_override = [single_path]
                rename_pdfs_in_directory(directory, config=config, files_override=files_override)
    except (FileNotFoundError, NotADirectoryError, OSError) as exc:
        raise SystemExit(str(exc)) from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(
            f"Invalid JSON in data file. Check heuristic_scores.json / "
            f"meta_stopwords.json in the data directory. {exc!s}"
        ) from exc
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    except requests.RequestException as exc:
        raise SystemExit(f"LLM or network error: {exc!s}") from exc
    except Exception as exc:
        logger.debug("Unhandled exception", exc_info=True)
        raise SystemExit(f"Error: {exc!s}") from exc


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    log_file, log_level = _resolve_log_config(args)
    setup_logging(log_file=log_file, level=log_level)

    dirs, single_file = _resolve_dirs(args)
    file_defaults = _load_config_file(args.config) if getattr(args, "config", None) else {}
    config = _build_config_from_args(args, file_defaults)
    _run_renamer_or_watch(dirs, config, args, single_file=single_file)
