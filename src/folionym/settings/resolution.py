"""Precedence, presets, and normalization for flat Folionym settings."""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..infrastructure.http import validate_http_endpoint
from ..infrastructure.resources import default_cache_dir
from .models import (
    _VALID_CATEGORY_DISPLAY,
    _VALID_DATE_LOCALES,
    _VALID_DESIRED_CASES,
    _VALID_LANGUAGES,
    RenamerConfig,
    build_config_from_flat_dict,
)

logger = logging.getLogger(__name__)
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


@dataclass(frozen=True)
class _PresetResolution:
    data: dict[str, Any]
    llm_defaults: dict[str, object]


def _optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except TypeError, ValueError:
        return None


def _positive_int_or_none(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        number = int(value)
    except TypeError, ValueError:
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
        return value.strip() or default
    return str(value).strip() or default


def _env_true(env: Mapping[str, str], key: str) -> bool:
    return (env.get(key, "") or "").strip().lower() in _TRUE_VALUES


def _int_with_default(value: Any, default: int) -> int:
    if value in (None, ""):
        return default
    try:
        return int(value)
    except TypeError, ValueError:
        return default


def _float_with_default(value: Any, default: float) -> float:
    if value in (None, ""):
        return default
    try:
        return float(value)
    except TypeError, ValueError:
        return default


def _path_or_none(value: Any) -> str | Path | None:
    if value in (None, ""):
        return None
    if isinstance(value, Path):
        return value
    return _str(value) or None


def _string_or_none(value: Any) -> str | None:
    return _str(value) or None


def _first_set(*values: Any) -> Any:
    return next((value for value in values if value not in (None, "")), None)


def _set_if_unset(data: dict[str, Any], key: str, value: Any) -> None:
    if data.get(key) in (None, ""):
        data[key] = value


def _apply_named_preset(data: dict[str, Any], preset: str) -> None:
    if preset == "scanned":
        data["use_vision_fallback"] = True
        data["simple_naming_mode"] = True
    elif preset == "high-confidence-heuristic":
        _set_if_unset(data, "skip_llm_category_if_heuristic_score_ge", 0.5)
        _set_if_unset(data, "skip_llm_category_if_heuristic_gap_ge", 0.3)
    elif preset == "fast":
        data["use_llm"] = False
        _set_if_unset(data, "min_heuristic_score", 0.6)
        _set_if_unset(data, "min_heuristic_score_gap", 0.25)
    elif preset == "accurate":
        data["use_llm"] = True
        data["use_single_llm_call"] = False
        _set_if_unset(data, "min_heuristic_score", 0.1)
        _set_if_unset(data, "min_heuristic_score_gap", 0.0)
    elif preset == "batch":
        data["use_cache"] = True
        _set_if_unset(data, "workers", 4)
        _set_if_unset(data, "cache_dir", str(default_cache_dir()))


def _resolve_presets(raw: Mapping[str, Any], defaults: Mapping[str, Any]) -> _PresetResolution:
    data = dict(raw)
    preset = _str(_first_set(data.get("preset"), defaults.get("preset")))
    if preset:
        data.setdefault("preset", preset)
        _apply_named_preset(data, preset)
    llm_preset = _string_or_none(_first_set(data.get("llm_preset"), defaults.get("llm_preset")))
    if llm_preset is not None:
        data.setdefault("llm_preset", llm_preset)
    effective = llm_preset if llm_preset in _LLM_PRESET_DEFAULTS else "apple-silicon"
    if llm_preset is not None and llm_preset != effective:
        logger.warning("Unknown llm_preset=%r; falling back to %r", llm_preset, effective)
    return _PresetResolution(data, _LLM_PRESET_DEFAULTS[effective])


def _heuristic_options(data: Mapping[str, Any]) -> dict[str, Any]:
    override_score = _optional_float(data.get("heuristic_override_min_score"))
    override_gap = _optional_float(data.get("heuristic_override_min_gap"))
    if _bool(data.get("no_heuristic_override")):
        override_score = override_gap = None
    elif override_score is None and override_gap is None:
        override_score, override_gap = 0.55, 0.3
    prefer_llm = _bool(data.get("prefer_llm_category"), True) and not _bool(data.get("prefer_heuristic"))
    return {
        "min_heuristic_score_gap": _float_with_default(data.get("min_heuristic_score_gap"), 0.0),
        "min_heuristic_score": _float_with_default(data.get("min_heuristic_score"), 0.0),
        "title_weight_region": _int_with_default(data.get("title_weight_region"), 2000),
        "title_weight_factor": _float_with_default(data.get("title_weight_factor"), 1.5),
        "max_score_per_category": _optional_float(data.get("max_score_per_category")),
        "use_keyword_overlap_for_category": _bool(data.get("use_keyword_overlap_for_category"), True),
        "category_display": _str(data.get("category_display"), "specific").lower(),
        "skip_llm_category_if_heuristic_score_ge": _optional_float(data.get("skip_llm_category_if_heuristic_score_ge")),
        "skip_llm_category_if_heuristic_gap_ge": _optional_float(data.get("skip_llm_category_if_heuristic_gap_ge")),
        "heuristic_suggestions_top_n": _int_with_default(data.get("heuristic_suggestions_top_n"), 5),
        "heuristic_score_weight": _float_with_default(data.get("heuristic_score_weight"), 0.15),
        "use_constrained_llm_category": _bool(data.get("use_constrained_llm_category"), True),
        "heuristic_leading_chars": _int_with_default(data.get("heuristic_leading_chars"), 0),
        "heuristic_long_doc_chars_threshold": _int_with_default(data.get("heuristic_long_doc_chars_threshold"), 40_000),
        "heuristic_long_doc_leading_chars": _int_with_default(data.get("heuristic_long_doc_leading_chars"), 12_000),
        "heuristic_override_min_score": override_score,
        "heuristic_override_min_gap": override_gap,
        "prefer_llm_category": prefer_llm,
    }


def _config_parts(
    data: dict[str, Any],
    defaults: Mapping[str, Any],
    env: Mapping[str, str],
    preset: dict[str, object],
) -> dict[str, Any]:
    max_content_chars = _positive_int_or_none(
        _first_set(
            data.get("max_content_chars"),
            env.get("FOLIONYM_MAX_CONTENT_CHARS"),
            defaults.get("max_content_chars"),
        )
    )
    return {
        "language": _str(data.get("language"), "de").lower(),
        "desired_case": _str(data.get("desired_case"), "kebabCase"),
        "project": _str(data.get("project")),
        "version": _str(data.get("version")),
        "date_locale": _str(data.get("date_locale"), "dmy").lower(),
        "date_prefer_leading_chars": _int_with_default(data.get("date_prefer_leading_chars"), 8000),
        "use_pdf_metadata_for_date": _bool(data.get("use_pdf_metadata_for_date"), True),
        **_heuristic_options(data),
        "skip_if_already_named": _bool(data.get("skip_if_already_named")),
        "use_llm": _bool(data.get("use_llm"), True),
        "lenient_llm_json": _bool(data.get("lenient_llm_json")),
        "use_timestamp_fallback": _bool(data.get("use_timestamp_fallback"), True),
        "timestamp_fallback_segment": _str(data.get("timestamp_fallback_segment"), "document"),
        "simple_naming_mode": _bool(data.get("simple_naming_mode")),
        "use_structured_fields": _bool(data.get("use_structured_fields"), True),
        "write_pdf_metadata": _bool(data.get("write_pdf_metadata")),
        "stop_event": data.get("stop_event"),
        "llm_base_url": _string_or_none(
            _first_set(
                data.get("llm_base_url"),
                env.get("FOLIONYM_LLM_URL"),
                defaults.get("llm_base_url"),
                preset["llm_base_url"],
            )
        ),
        "llm_model": _string_or_none(
            _first_set(
                data.get("llm_model"),
                env.get("FOLIONYM_LLM_MODEL"),
                defaults.get("llm_model"),
                preset["llm_model"],
            )
        ),
        "llm_timeout_s": _optional_float(
            _first_set(data.get("llm_timeout_s"), env.get("FOLIONYM_LLM_TIMEOUT"), defaults.get("llm_timeout_s"))
        ),
        "require_https": _bool(
            _first_set(data.get("require_https"), env.get("FOLIONYM_REQUIRE_HTTPS"), defaults.get("require_https"))
        ),
        "use_single_llm_call": _bool(data.get("use_single_llm_call"), True),
        "llm_use_chat_api": _bool(data.get("llm_use_chat_api"), True),
        "llm_json_mode": _bool(data.get("llm_json_mode"), True),
        "llm_preset": _string_or_none(data.get("llm_preset")),
        "use_cache": _bool(data.get("use_cache"), True),
        "cache_dir": _path_or_none(_first_set(data.get("cache_dir"), env.get("FOLIONYM_CACHE_DIR"))),
        "max_context_chars": _positive_int_or_none(
            _first_set(
                data.get("max_context_chars"),
                env.get("FOLIONYM_MAX_CONTENT_CHARS"),
                defaults.get("max_context_chars"),
                preset["max_context_chars"],
            )
        ),
        "use_ocr": _bool(data.get("use_ocr")),
        "use_vision_fallback": _bool(data.get("use_vision_fallback")) or _env_true(env, "FOLIONYM_USE_VISION_FALLBACK"),
        "vision_fallback_min_text_len": _int_with_default(data.get("vision_fallback_min_text_len"), 50),
        "vision_model": _string_or_none(data.get("vision_model")),
        "vision_first": _bool(data.get("vision_first")) or _env_true(env, "FOLIONYM_VISION_FIRST"),
        "max_tokens_for_extraction": _positive_int_or_none(data.get("max_tokens_for_extraction")),
        "max_content_chars": max_content_chars,
        "max_content_tokens": _positive_int_or_none(
            _first_set(
                data.get("max_content_tokens"),
                env.get("FOLIONYM_MAX_CONTENT_TOKENS"),
                defaults.get("max_content_tokens"),
            )
        ),
        "workers": max(1, _int_with_default(data.get("workers"), 1)),
        "max_pages_for_extraction": _int_with_default(data.get("max_pages_for_extraction"), 0),
        "dry_run": _bool(data.get("dry_run")),
        "backup_dir": _path_or_none(data.get("backup_dir")),
        "rename_log_path": _path_or_none(data.get("rename_log_path")),
        "export_metadata_path": _path_or_none(data.get("export_metadata_path")),
        "summary_json_path": _path_or_none(data.get("summary_json_path")),
        "max_filename_chars": _positive_int_or_none(data.get("max_filename_chars")),
        "override_category_map": data.get("override_category_map"),
        "rules_file": _path_or_none(data.get("rules_file")),
        "post_rename_hook": _string_or_none(
            _first_set(data.get("post_rename_hook"), env.get("FOLIONYM_POST_RENAME_HOOK"))
        ),
        "recursive": _bool(data.get("recursive")),
        "max_depth": _int_with_default(data.get("max_depth"), 0),
        "include_patterns": data.get("include_patterns"),
        "exclude_patterns": data.get("exclude_patterns"),
        "filename_template": _string_or_none(
            _first_set(data.get("filename_template"), defaults.get("filename_template"))
        ),
        "plan_file_path": _path_or_none(data.get("plan_file_path")),
        "interactive": _bool(data.get("interactive")),
        "manual_mode": _bool(data.get("manual_mode")),
        "progress": _bool(data.get("progress")),
        "quiet_progress": _bool(data.get("quiet_progress")),
        "explain": _bool(data.get("explain")),
    }


def _validation_errors(kwargs: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    llm_base_url = _str(kwargs.get("llm_base_url"))
    if llm_base_url:
        try:
            endpoint = validate_http_endpoint(llm_base_url)
        except ValueError as exc:
            errors.append(f"llm_base_url is invalid: {exc}")
        else:
            if _bool(kwargs.get("require_https")) and endpoint.scheme == "http" and not endpoint.is_loopback:
                errors.append("llm_base_url must use HTTPS for a non-loopback host when require_https is enabled")
    for key, default, choices in (
        ("language", "de", _VALID_LANGUAGES),
        ("desired_case", "kebabCase", _VALID_DESIRED_CASES),
        ("date_locale", "dmy", _VALID_DATE_LOCALES),
        ("category_display", "specific", _VALID_CATEGORY_DISPLAY),
    ):
        value = _str(kwargs.get(key), default).lower() if key != "desired_case" else _str(kwargs.get(key), default)
        if value not in choices:
            errors.append(f"{key}={value!r} must be one of: {', '.join(sorted(choices))}")
    return errors


def build_config(
    raw: Mapping[str, Any], *, file_defaults: Mapping[str, Any] | None = None, env: Mapping[str, str] | None = None
) -> RenamerConfig:
    """Build grouped configuration with CLI > environment > file > preset/default precedence."""
    env_map = env if env is not None else os.environ
    defaults = file_defaults or {}
    resolution = _resolve_presets(raw, defaults)
    kwargs = _config_parts(resolution.data, defaults, env_map, resolution.llm_defaults)
    if kwargs["manual_mode"]:
        kwargs["interactive"] = True
    errors = _validation_errors(kwargs)
    if errors:
        raise ValueError("Invalid configuration:\n- " + "\n- ".join(errors))
    return build_config_from_flat_dict(kwargs)
