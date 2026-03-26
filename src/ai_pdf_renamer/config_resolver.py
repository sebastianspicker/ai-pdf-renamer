"""Shared config normalization for CLI and GUI.

This module centralizes preset/default/env-driven normalization so CLI and GUI
produce consistent RenamerConfig values.
"""

from __future__ import annotations

import copy
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .config import RenamerConfig

_TRUE_VALUES = {"1", "true", "yes"}

_LLM_PRESET_DEFAULTS: dict[str, dict[str, object]] = {
    "apple-silicon": {
        "llm_model": "qwen2.5:3b",
        "llm_base_url": "http://127.0.0.1:11434/v1/completions",
        "max_context_chars": 120_000,
    },
    "gpu": {
        "llm_model": "qwen2.5:7b-instruct",
        "llm_base_url": "http://127.0.0.1:11434/v1/completions",
        "max_context_chars": 480_000,
    },
}


def _optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _positive_int_or_none(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        low = value.strip().lower()
        if low in {"1", "true", "yes", "on"}:
            return True
        if low in {"0", "false", "no", "off", ""}:
            return False
    return bool(value)


def _str(value: Any, default: str = "") -> str:
    if value is None:
        return default
    if isinstance(value, str):
        s = value.strip()
        return s if s else default
    return str(value).strip() or default


def _env_true(env: Mapping[str, str], key: str) -> bool:
    return (env.get(key, "") or "").strip().lower() in _TRUE_VALUES


def _int_with_default(value: Any, default: int) -> int:
    """Convert value to int; use default only when value is None or empty string."""
    if value is None or value == "":
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _float_with_default(value: Any, default: float) -> float:
    """Convert value to float; use default only when value is None or empty string."""
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalize_path_or_none(value: Any) -> str | Path | None:
    if value in (None, ""):
        return None
    if isinstance(value, Path):
        return value
    s = str(value).strip()
    return s or None


def _normalize_str_or_none(value: Any) -> str | None:
    if value in (None, ""):
        return None
    return str(value).strip() or None


def _build_core_options(data: dict[str, Any], file_cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Language, case, project, version, date settings, heuristic tuning, and general flags."""
    preset = _str(data.get("preset"), "")

    # --- skip-LLM-category thresholds (preset may supply defaults) ---
    skip_llm_score = _optional_float(data.get("skip_llm_category_if_heuristic_score_ge"))
    skip_llm_gap = _optional_float(data.get("skip_llm_category_if_heuristic_gap_ge"))
    if preset == "high-confidence-heuristic":
        if skip_llm_score is None:
            skip_llm_score = 0.5
        if skip_llm_gap is None:
            skip_llm_gap = 0.3

    # --- heuristic override ---
    heuristic_override_min_score = _optional_float(data.get("heuristic_override_min_score"))
    heuristic_override_min_gap = _optional_float(data.get("heuristic_override_min_gap"))
    if _bool(data.get("no_heuristic_override"), False):
        heuristic_override_min_score = None
        heuristic_override_min_gap = None
    elif heuristic_override_min_score is None and heuristic_override_min_gap is None:
        heuristic_override_min_score = 0.55
        heuristic_override_min_gap = 0.3

    # --- prefer LLM vs heuristic ---
    prefer_llm_category = _bool(data.get("prefer_llm_category"), True)
    if _bool(data.get("prefer_heuristic"), False):
        prefer_llm_category = False

    return {
        "language": _str(data.get("language"), "de"),
        "desired_case": _str(data.get("desired_case"), "kebabCase"),
        "project": _str(data.get("project"), ""),
        "version": _str(data.get("version"), ""),
        "prefer_llm_category": prefer_llm_category,
        "date_locale": _str(data.get("date_locale"), "dmy"),
        "date_prefer_leading_chars": _int_with_default(data.get("date_prefer_leading_chars"), 8000),
        "use_pdf_metadata_for_date": _bool(data.get("use_pdf_metadata_for_date"), True),
        "min_heuristic_score_gap": _float_with_default(data.get("min_heuristic_score_gap"), 0.0),
        "min_heuristic_score": _float_with_default(data.get("min_heuristic_score"), 0.0),
        "title_weight_region": _int_with_default(data.get("title_weight_region"), 2000),
        "title_weight_factor": _float_with_default(data.get("title_weight_factor"), 1.5),
        "max_score_per_category": _optional_float(data.get("max_score_per_category")),
        "use_keyword_overlap_for_category": _bool(data.get("use_keyword_overlap_for_category"), True),
        "use_embeddings_for_conflict": _bool(data.get("use_embeddings_for_conflict"), False),
        "category_display": _str(data.get("category_display"), "specific"),
        "skip_llm_category_if_heuristic_score_ge": skip_llm_score,
        "skip_llm_category_if_heuristic_gap_ge": skip_llm_gap,
        "heuristic_suggestions_top_n": _int_with_default(data.get("heuristic_suggestions_top_n"), 5),
        "heuristic_score_weight": _float_with_default(data.get("heuristic_score_weight"), 0.15),
        "heuristic_override_min_score": heuristic_override_min_score,
        "heuristic_override_min_gap": heuristic_override_min_gap,
        "use_constrained_llm_category": _bool(data.get("use_constrained_llm_category"), True),
        "heuristic_leading_chars": _int_with_default(data.get("heuristic_leading_chars"), 0),
        "heuristic_long_doc_chars_threshold": _int_with_default(data.get("heuristic_long_doc_chars_threshold"), 40000),
        "heuristic_long_doc_leading_chars": _int_with_default(data.get("heuristic_long_doc_leading_chars"), 12000),
        "skip_if_already_named": _bool(data.get("skip_if_already_named"), False),
        "use_llm": _bool(data.get("use_llm"), True),
        "lenient_llm_json": _bool(data.get("lenient_llm_json"), False),
        "use_timestamp_fallback": _bool(data.get("use_timestamp_fallback"), True),
        "timestamp_fallback_segment": _str(data.get("timestamp_fallback_segment"), "document"),
        "simple_naming_mode": _bool(data.get("simple_naming_mode"), False),
        "use_structured_fields": _bool(data.get("use_structured_fields"), True),
        "write_pdf_metadata": _bool(data.get("write_pdf_metadata"), False),
        "stop_event": data.get("stop_event"),
    }


def _build_llm_options(
    data: dict[str, Any],
    file_cfg: Mapping[str, Any],
    preset_defaults: dict[str, object],
) -> dict[str, Any]:
    """LLM backend, URL, model, timeout, chat API, JSON mode, and preset."""
    llm_preset = _normalize_str_or_none(data.get("llm_preset"))

    # Apply preset defaults only where user didn't set a value
    user_llm_model = _str(data.get("llm_model"), "") or None
    user_llm_base_url = _normalize_str_or_none(data.get("llm_base_url"))
    user_max_context_chars = _positive_int_or_none(data.get("max_context_chars"))

    resolved_llm_model = user_llm_model or preset_defaults["llm_model"]
    resolved_llm_base_url = user_llm_base_url or preset_defaults["llm_base_url"]
    resolved_max_context_chars = user_max_context_chars or preset_defaults["max_context_chars"]

    return {
        "llm_backend": _str(data.get("llm_backend"), "http"),
        "llm_base_url": resolved_llm_base_url,
        "llm_model": resolved_llm_model,
        "llm_timeout_s": _optional_float(data.get("llm_timeout_s")),
        "llm_model_path": _normalize_str_or_none(data.get("llm_model_path")),
        "require_https": _bool(data.get("require_https"), False),
        "use_single_llm_call": _bool(data.get("use_single_llm_call"), True),
        "llm_use_chat_api": _bool(data.get("llm_use_chat_api"), True),
        "llm_json_mode": _bool(data.get("llm_json_mode"), True),
        "llm_preset": llm_preset,
        "max_context_chars": resolved_max_context_chars,
    }


def _build_extraction_options(
    data: dict[str, Any],
    file_cfg: Mapping[str, Any],
    env_map: Mapping[str, str],
) -> dict[str, Any]:
    """OCR, vision, tokens, workers, and max content settings."""
    preset = _str(data.get("preset"), "")

    use_vision_fallback = _bool(data.get("use_vision_fallback"), False) or _env_true(
        env_map, "AI_PDF_RENAMER_USE_VISION_FALLBACK"
    )
    vision_first = _bool(data.get("vision_first"), False) or _env_true(env_map, "AI_PDF_RENAMER_VISION_FIRST")

    if preset == "scanned":
        use_vision_fallback = True
        data["simple_naming_mode"] = True

    max_content_chars = _positive_int_or_none(
        data.get("max_content_chars") or env_map.get("AI_PDF_RENAMER_MAX_CONTENT_CHARS")
    )
    max_content_tokens = _positive_int_or_none(
        data.get("max_content_tokens") or env_map.get("AI_PDF_RENAMER_MAX_CONTENT_TOKENS")
    )

    return {
        "use_ocr": _bool(data.get("use_ocr"), False),
        "use_vision_fallback": use_vision_fallback,
        "vision_fallback_min_text_len": _int_with_default(data.get("vision_fallback_min_text_len"), 50),
        "vision_model": _str(data.get("vision_model"), "") or None,
        "vision_first": vision_first,
        "max_tokens_for_extraction": _positive_int_or_none(data.get("max_tokens_for_extraction")),
        "max_content_chars": max_content_chars,
        "max_content_tokens": max_content_tokens,
        "workers": max(1, _int_with_default(data.get("workers"), 1)),
        "max_pages_for_extraction": _int_with_default(data.get("max_pages_for_extraction"), 0),
    }


def _build_output_options(
    data: dict[str, Any],
    file_cfg: Mapping[str, Any],
    env_map: Mapping[str, str],
) -> dict[str, Any]:
    """Dry-run, backup, hooks, export, logging, interactive, and rules."""
    post_rename_hook = _str(data.get("post_rename_hook"), "") or _str(
        env_map.get("AI_PDF_RENAMER_POST_RENAME_HOOK"), ""
    )

    return {
        "dry_run": _bool(data.get("dry_run"), False),
        "backup_dir": _normalize_path_or_none(data.get("backup_dir")),
        "rename_log_path": _normalize_path_or_none(data.get("rename_log_path")),
        "export_metadata_path": _normalize_path_or_none(data.get("export_metadata_path")),
        "summary_json_path": _normalize_path_or_none(data.get("summary_json_path")),
        "max_filename_chars": _positive_int_or_none(data.get("max_filename_chars")),
        "override_category_map": data.get("override_category_map"),
        "rules_file": _normalize_path_or_none(data.get("rules_file")),
        "post_rename_hook": post_rename_hook or None,
        "recursive": _bool(data.get("recursive"), False),
        "max_depth": _int_with_default(data.get("max_depth"), 0),
        "include_patterns": data.get("include_patterns"),
        "exclude_patterns": data.get("exclude_patterns"),
        "filename_template": (
            _str(data.get("filename_template"), "") or _str(file_cfg.get("filename_template"), "") or None
        ),
        "plan_file_path": _normalize_path_or_none(data.get("plan_file_path")),
        "interactive": _bool(data.get("interactive"), False),
        "manual_mode": _bool(data.get("manual_mode"), False),
    }


def build_config(
    raw: Mapping[str, Any],
    *,
    file_defaults: Mapping[str, Any] | None = None,
    env: Mapping[str, str] | None = None,
) -> RenamerConfig:
    """Build a normalized RenamerConfig from raw values (CLI or GUI)."""
    env_map = env if env is not None else os.environ
    defaults = file_defaults or {}

    # P2: Deep copy to avoid mutating the caller's dict (e.g. scanned preset sets simple_naming_mode)
    data: dict[str, Any] = copy.copy(dict(raw))

    # --- Resolve LLM hardware preset defaults ---
    llm_preset = _normalize_str_or_none(data.get("llm_preset"))
    effective_preset = llm_preset if llm_preset in _LLM_PRESET_DEFAULTS else "apple-silicon"
    preset_defaults = _LLM_PRESET_DEFAULTS[effective_preset]

    # --- Build partial dicts from helpers ---
    # NOTE: _build_extraction_options must run before _build_core_options because the
    # "scanned" preset mutates data["simple_naming_mode"], which _build_core_options reads.
    extraction = _build_extraction_options(data, defaults, env_map)
    core = _build_core_options(data, defaults)
    llm = _build_llm_options(data, defaults, preset_defaults)
    output = _build_output_options(data, defaults, env_map)

    # --- Merge into one dict ---
    kwargs: dict[str, Any] = {}
    kwargs.update(core)
    kwargs.update(llm)
    kwargs.update(extraction)
    kwargs.update(output)

    # Manual mode implies interactive behavior.
    if kwargs["manual_mode"]:
        kwargs["interactive"] = True

    return RenamerConfig(**kwargs)
