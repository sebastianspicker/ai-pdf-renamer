"""
LLM response schema and validation. Used by llm for document analysis result.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

from .data_paths import data_dir, package_data_path

logger = logging.getLogger(__name__)

# Declarative schema for LLM document-analysis response (summary, keywords, category, final_summary_tokens).
# Overridable via AI_PDF_RENAMER_DATA_DIR/llm_response_schema.json.
LLM_RESPONSE_SCHEMA_DEFAULT: dict = {
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


def _load_llm_response_schema() -> dict:
    """Load LLM response schema from data dir or package data. Returns default if missing or invalid."""
    for path in [
        data_dir() / "llm_response_schema.json",
        package_data_path("llm_response_schema.json"),
    ]:
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as e:
                logger.debug("Could not load LLM response schema from %s: %s", path, e)
    return LLM_RESPONSE_SCHEMA_DEFAULT


# Default values for LLM document analysis when parsing fails or fields are missing.
DEFAULT_LLM_SUMMARY = "na"
DEFAULT_LLM_CATEGORY = "unknown"
DEFAULT_LLM_KEYWORDS: list[str] = []


@dataclass(frozen=False)
class DocumentAnalysisResult:
    """Structured result of LLM document analysis; used for validation and defaults."""

    summary: str = DEFAULT_LLM_SUMMARY
    keywords: list[str] = field(default_factory=list)
    category: str = DEFAULT_LLM_CATEGORY
    final_summary_tokens: list[str] | None = None


def validate_llm_document_result(parsed: dict) -> DocumentAnalysisResult:
    """
    Validate and fill defaults for a parsed LLM document analysis dict.
    Accepts optional keys: summary, keywords, category, final_summary_tokens.
    Uses declarative schema (llm_response_schema.json or default) for optional
    jsonschema validation; keeps existing defaults when validation fails or keys are missing.
    """
    schema = _load_llm_response_schema()
    try:
        import jsonschema

        jsonschema.validate(instance=parsed, schema=schema)
    except ImportError:
        pass
    except Exception as e:
        logger.debug("LLM response did not match schema: %s", e)

    summary = parsed.get("summary")
    if isinstance(summary, str) and summary.strip():
        summary = summary.strip()
        if summary.lower() == "na":
            summary = DEFAULT_LLM_SUMMARY
    else:
        summary = DEFAULT_LLM_SUMMARY

    keywords = parsed.get("keywords")
    if isinstance(keywords, list):
        keywords = [str(x).strip() for x in keywords if x and str(x).strip()]
    else:
        keywords = []

    category = parsed.get("category")
    if isinstance(category, str) and category.strip():
        category = category.strip()
        if category.lower() in ("na", "unknown", "document", ""):
            category = DEFAULT_LLM_CATEGORY
    else:
        category = DEFAULT_LLM_CATEGORY

    final_summary_tokens = parsed.get("final_summary_tokens")
    if isinstance(final_summary_tokens, list):
        final_summary_tokens = [str(x).strip() for x in final_summary_tokens if x and str(x).strip()]
    elif isinstance(final_summary_tokens, str) and final_summary_tokens.strip():
        final_summary_tokens = [t.strip() for t in final_summary_tokens.split(",") if t.strip()]
    else:
        final_summary_tokens = None

    return DocumentAnalysisResult(
        summary=summary,
        keywords=keywords,
        category=category,
        final_summary_tokens=final_summary_tokens,
    )
