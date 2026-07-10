"""
LLM response schema and validation. Used by llm for document analysis result.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from functools import lru_cache

from .data_paths import data_dir, package_data_path

logger = logging.getLogger(__name__)

_SAFE_SCHEMA_PATH_SEGMENTS = {
    "additionalProperties",
    "allOf",
    "anyOf",
    "contains",
    "else",
    "if",
    "items",
    "not",
    "oneOf",
    "patternProperties",
    "prefixItems",
    "properties",
    "then",
    "type",
}
_SAFE_VALIDATORS = {
    "additionalProperties",
    "allOf",
    "anyOf",
    "contains",
    "dependentRequired",
    "enum",
    "exclusiveMaximum",
    "exclusiveMinimum",
    "format",
    "items",
    "maxItems",
    "maxLength",
    "maxProperties",
    "maximum",
    "minItems",
    "minLength",
    "minProperties",
    "minimum",
    "multipleOf",
    "not",
    "oneOf",
    "pattern",
    "required",
    "type",
    "uniqueItems",
}

# Declarative schema for LLM document-analysis response (summary, keywords, category, final_summary_tokens).
# Overridable via AI_PDF_RENAMER_DATA_DIR/llm_response_schema.json.
LLM_RESPONSE_SCHEMA_DEFAULT: dict[str, object] = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "keywords": {"type": "array", "items": {"type": "string"}},
        "category": {"type": "string"},
        "final_summary_tokens": {
            "oneOf": [
                {"type": "array", "items": {"type": "string"}},
                {"type": "null"},
            ]
        },
    },
    "additionalProperties": True,
}


@lru_cache(maxsize=1)
def _load_llm_response_schema() -> dict[str, object]:
    """Load LLM response schema from data dir or package data. Returns default if missing or invalid."""
    for path in [
        data_dir() / "llm_response_schema.json",
        package_data_path("llm_response_schema.json"),
    ]:
        if path.exists():
            try:
                result: dict[str, object] = json.loads(path.read_text(encoding="utf-8"))
                return result
            except (OSError, json.JSONDecodeError) as e:
                logger.debug("Could not load LLM response schema from %s: %s", path, e)
    return LLM_RESPONSE_SCHEMA_DEFAULT


def clear_llm_response_schema_cache() -> None:
    """Clear the cached LLM response schema after tests or data-dir changes."""
    _load_llm_response_schema.cache_clear()


def _safe_validation_location(error: object) -> tuple[str, str]:
    """Describe validator structure without rendering schema or instance values."""
    raw_validator = getattr(error, "validator", None)
    validator = raw_validator if isinstance(raw_validator, str) and raw_validator in _SAFE_VALIDATORS else "other"
    path_parts: list[str] = []
    for part in getattr(error, "absolute_schema_path", ()):
        if isinstance(part, int):
            path_parts.append("[index]")
        elif isinstance(part, str) and part in _SAFE_SCHEMA_PATH_SEGMENTS:
            path_parts.append(part)
        else:
            path_parts.append("<field>")
    return validator, ".".join(path_parts) or "<root>"


# Default values for LLM document analysis when parsing fails or fields are missing.
DEFAULT_LLM_SUMMARY = "na"
DEFAULT_LLM_CATEGORY = "unknown"
DEFAULT_LLM_KEYWORDS: list[str] = []


@dataclass(frozen=True)
class DocumentAnalysisResult:
    """Structured result of LLM document analysis; used for validation and defaults."""

    summary: str = DEFAULT_LLM_SUMMARY
    keywords: tuple[str, ...] = ()
    category: str = DEFAULT_LLM_CATEGORY
    final_summary_tokens: tuple[str, ...] | None = None


def _validate_with_optional_jsonschema(parsed: dict[str, object]) -> None:
    schema = _load_llm_response_schema()
    try:
        import jsonschema
    except ImportError:
        return
    try:
        jsonschema.validate(instance=parsed, schema=schema)
    except jsonschema.ValidationError as error:
        # Schema mismatches are recoverable because callers still apply
        # local normalization/defaults. Never render the invalid instance or
        # ValidationError text because either may contain document content.
        validator, schema_path = _safe_validation_location(error)
        logger.info(
            "LLM response did not match schema (validator=%s, schema_path=%s).",
            validator,
            schema_path,
        )


def _normalized_summary(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        return DEFAULT_LLM_SUMMARY
    summary = value.strip()
    return DEFAULT_LLM_SUMMARY if summary.lower() == "na" else summary


def _normalized_keywords(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(str(item).strip() for item in value if item and str(item).strip())


def _normalized_category(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        return DEFAULT_LLM_CATEGORY
    category = value.strip()
    return DEFAULT_LLM_CATEGORY if category.lower() in ("na", "unknown", "document", "") else category


def _clean_string_items(values: list[object]) -> tuple[str, ...]:
    cleaned: list[str] = []
    for item in values:
        if not item:
            continue
        text = str(item).strip()
        if text:
            cleaned.append(text)
    return tuple(cleaned)


def _normalized_final_summary_tokens(value: object) -> tuple[str, ...] | None:
    if isinstance(value, list):
        return _clean_string_items(value)
    if isinstance(value, str) and value.strip():
        return tuple(token.strip() for token in value.split(",") if token.strip())
    return None


def validate_llm_document_result(parsed: dict[str, object]) -> DocumentAnalysisResult:
    """
    Validate and fill defaults for a parsed LLM document analysis dict.
    Accepts optional keys: summary, keywords, category, final_summary_tokens.
    Uses declarative schema (llm_response_schema.json or default) for optional
    jsonschema validation; keeps existing defaults when validation fails or keys are missing.
    """
    _validate_with_optional_jsonschema(parsed)
    return DocumentAnalysisResult(
        summary=_normalized_summary(parsed.get("summary")),
        keywords=_normalized_keywords(parsed.get("keywords")),
        category=_normalized_category(parsed.get("category")),
        final_summary_tokens=_normalized_final_summary_tokens(parsed.get("final_summary_tokens")),
    )
