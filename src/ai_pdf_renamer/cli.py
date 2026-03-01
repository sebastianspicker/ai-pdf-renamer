from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import logging
import os
import sys
from collections.abc import Callable
from pathlib import Path

import requests

from .cli_parser import build_parser
from .config_resolver import build_config
from .data_paths import data_path
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
            data = json.loads(raw)
            return data if isinstance(data, dict) else {}
        except json.JSONDecodeError:
            return {}
    if suf in (".yaml", ".yml"):
        try:
            import yaml

            data = yaml.safe_load(raw)
            return data if isinstance(data, dict) else {}
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
    except OSError as e:
        logger.warning("Could not read override-category file %s: %s. Proceeding with no overrides.", p, e)
    return result


def _is_interactive() -> bool:
    """True if stdin is a TTY (interactive prompt is safe)."""
    return sys.stdin.isatty()


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


def run_doctor_checks(args: argparse.Namespace) -> int:
    """Run preflight diagnostics for local environment and dependencies."""
    ok = True

    print("AI-PDF-Renamer doctor")
    print("=====================")

    for filename in ("heuristic_scores.json", "meta_stopwords.json"):
        try:
            path = data_path(filename)
            raw = path.read_text(encoding="utf-8")
            json.loads(raw)
            print(f"[OK] data file: {filename} -> {path}")
        except Exception as exc:
            ok = False
            print(f"[FAIL] data file: {filename} ({exc})")

    if importlib.util.find_spec("fitz") is not None:
        print("[OK] optional dep: PyMuPDF (fitz)")
    else:
        print("[WARN] optional dep missing: PyMuPDF (fitz) -> install with: pip install -e '.[pdf]'")

    if importlib.util.find_spec("ocrmypdf") is not None:
        print("[OK] optional dep: ocrmypdf")
    else:
        print("[INFO] optional dep missing: ocrmypdf (only needed for --ocr)")

    if importlib.util.find_spec("tiktoken") is not None:
        print("[OK] optional dep: tiktoken")
    else:
        print("[INFO] optional dep missing: tiktoken (only needed for token-based truncation)")

    use_llm = _bool_opt(args, "use_llm", True)
    if use_llm:
        from .filename import _llm_client_from_config

        probe_cfg = RenamerConfig(
            llm_base_url=getattr(args, "llm_base_url", None) or None,
            llm_model=getattr(args, "llm_model", None) or None,
            llm_timeout_s=getattr(args, "llm_timeout_s", None),
            use_llm=True,
        )
        client = _llm_client_from_config(probe_cfg)
        try:
            # Keep check lightweight and bounded, and verify completions shape.
            payload = {
                "model": client.model,
                "prompt": "ping",
                "max_tokens": 1,
                "temperature": 0.0,
            }
            with requests.Session() as session:
                session.trust_env = False
                resp = session.post(
                    client.base_url,
                    json=payload,
                    timeout=min(float(client.timeout_s), 3.0),
                )
                resp.raise_for_status()
                data = resp.json()
            if not isinstance(data, dict) or not isinstance(data.get("choices"), list):
                raise ValueError("Response is not OpenAI-compatible completions JSON.")
            print(f"[OK] LLM endpoint reachable: {client.base_url}")
        except (requests.RequestException, ValueError, json.JSONDecodeError) as exc:
            ok = False
            print(f"[FAIL] LLM endpoint unreachable: {client.base_url} ({exc})")
    else:
        print("[INFO] LLM checks skipped (--no-llm)")

    print("=====================")
    if ok:
        print("Doctor checks passed.")
        return 0
    print("Doctor checks found issues.")
    return 1


def _resolve_dirs(args: argparse.Namespace) -> tuple[list[str], str | None]:
    """Resolve directory list and optional single-file path from args. Raises SystemExit on error."""
    single_file = getattr(args, "single_file", None) or getattr(args, "manual_file", None)
    dirs: list[str] = list(getattr(args, "dirs", None) or [])
    dirs_from_file = getattr(args, "dirs_from_file", None)
    if dirs_from_file:
        try:
            lines = Path(dirs_from_file).read_text(encoding="utf-8").splitlines()
            # Cap to avoid unbounded memory on huge or malicious input (P3).
            _MAX_DIRS_FROM_FILE_LINES = 10_000
            if len(lines) > _MAX_DIRS_FROM_FILE_LINES:
                logger.warning(
                    "--dirs-from-file has %s lines; using first %s only.",
                    len(lines),
                    _MAX_DIRS_FROM_FILE_LINES,
                )
                lines = lines[:_MAX_DIRS_FROM_FILE_LINES]
            dirs.extend(line.strip() for line in lines if line.strip())
        except OSError as e:
            raise SystemExit(f"Error: could not read --dirs-from-file: {e}") from e
    if single_file:
        single_path = Path(single_file).resolve()
        if not single_path.is_file():
            raise SystemExit(f"Error: --file/--manual must be an existing file: {single_path}")
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
            if (dirs and not single_file)  # Use original 'dirs' to check if any were provided before filtering
            else "Error: at least one directory or --file is required. Use --dir or --file."
        )
        raise SystemExit(msg)
    return (resolved_dirs, single_file)


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

    raw: dict = {
        "language": language,
        "desired_case": desired_case,
        "project": project,
        "version": version,
        "prefer_llm_category": _bool_opt(args, "prefer_llm_category", True),
        "prefer_heuristic": _bool_opt(args, "prefer_heuristic", False),
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
        "skip_llm_category_if_heuristic_score_ge": _optional_float(args, "skip_llm_category_if_heuristic_score_ge"),
        "skip_llm_category_if_heuristic_gap_ge": _optional_float(args, "skip_llm_category_if_heuristic_gap_ge"),
        "heuristic_suggestions_top_n": _int_opt(args, "heuristic_suggestions_top_n", 5),
        "heuristic_score_weight": _float_opt(args, "heuristic_score_weight", 0.15),
        "heuristic_override_min_score": _optional_float(args, "heuristic_override_min_score"),
        "heuristic_override_min_gap": _optional_float(args, "heuristic_override_min_gap"),
        "no_heuristic_override": _bool_opt(args, "no_heuristic_override", False),
        "use_constrained_llm_category": _bool_opt(args, "use_constrained_llm_category", True),
        "heuristic_leading_chars": _int_opt(args, "heuristic_leading_chars", 0),
        "heuristic_long_doc_chars_threshold": _int_opt(args, "heuristic_long_doc_chars_threshold", 40000),
        "heuristic_long_doc_leading_chars": _int_opt(args, "heuristic_long_doc_leading_chars", 12000),
        "max_pages_for_extraction": _int_opt(args, "max_pages_for_extraction", 0),
        "llm_base_url": getattr(args, "llm_base_url", None) or None,
        "llm_model": getattr(args, "llm_model", None) or None,
        "llm_timeout_s": getattr(args, "llm_timeout_s", None),
        "max_tokens_for_extraction": getattr(args, "max_tokens_for_extraction", None),
        "max_content_chars": getattr(args, "max_content_chars", None),
        "max_content_tokens": getattr(args, "max_content_tokens", None),
        "use_ocr": _bool_opt(args, "use_ocr", False),
        "skip_if_already_named": _bool_opt(args, "skip_if_already_named", False),
        "backup_dir": getattr(args, "backup_dir", None) or None,
        "rename_log_path": getattr(args, "rename_log_path", None) or None,
        "export_metadata_path": getattr(args, "export_metadata_path", None) or None,
        "summary_json_path": getattr(args, "summary_json_path", None) or None,
        "max_filename_chars": getattr(args, "max_filename_chars", None),
        "override_category_map": (
            _load_override_category_map(args.override_category_file)
            if getattr(args, "override_category_file", None)
            else None
        ),
        "rules_file": getattr(args, "rules_file", None) or None,
        "post_rename_hook": getattr(args, "post_rename_hook", None) or None,
        "workers": _int_opt(args, "workers", 1),
        "recursive": _bool_opt(args, "recursive", False),
        "max_depth": _int_opt(args, "max_depth", 0),
        "include_patterns": getattr(args, "include_patterns", None),
        "exclude_patterns": getattr(args, "exclude_patterns", None),
        "filename_template": getattr(args, "filename_template", None) or file_defaults.get("filename_template"),
        "use_structured_fields": _bool_opt(args, "use_structured_fields", True),
        "plan_file_path": getattr(args, "plan_file_path", None) or None,
        "interactive": _bool_opt(args, "interactive", False) or bool(getattr(args, "manual_file", None)),
        "manual_mode": bool(getattr(args, "manual_file", None)),
        "write_pdf_metadata": _bool_opt(args, "write_pdf_metadata", False),
        "use_llm": _bool_opt(args, "use_llm", True),
        "lenient_llm_json": _bool_opt(args, "lenient_llm_json", False),
        "use_timestamp_fallback": _bool_opt(args, "use_timestamp_fallback", True),
        "timestamp_fallback_segment": _str_opt(args, "timestamp_fallback_segment", "document"),
        "simple_naming_mode": _bool_opt(args, "simple_naming_mode", False),
        "use_vision_fallback": _bool_opt(args, "use_vision_fallback", False),
        "vision_fallback_min_text_len": _int_opt(args, "vision_fallback_min_text_len", 50),
        "vision_model": getattr(args, "vision_model", None) or None,
        "vision_first": _bool_opt(args, "vision_first", False),
        "preset": getattr(args, "preset", None) or "",
    }
    try:
        return build_config(raw, file_defaults=file_defaults)
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
    parser = build_parser()
    args = parser.parse_args(argv)

    log_file, log_level = _resolve_log_config(args)
    setup_logging(log_file=log_file, level=log_level)

    if _bool_opt(args, "doctor", False):
        raise SystemExit(run_doctor_checks(args))

    dirs, single_file = _resolve_dirs(args)
    file_defaults = _load_config_file(args.config) if getattr(args, "config", None) else {}
    config = _build_config_from_args(args, file_defaults)
    _run_renamer_or_watch(dirs, config, args, single_file=single_file)
