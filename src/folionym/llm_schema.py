"""
LLM response schema and validation. Used by llm for document analysis result.
"""

from __future__ import annotations

from dataclasses import dataclass

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


def _normalized_summary(value: object) -> str:
    """Normalize a summary or return the default placeholder."""
    if not isinstance(value, str) or not value.strip():
        return DEFAULT_LLM_SUMMARY
    summary = value.strip()
    return DEFAULT_LLM_SUMMARY if summary.lower() == "na" else summary


def _normalized_keywords(value: object) -> tuple[str, ...]:
    """Normalize a list of keyword values to stripped strings."""
    if not isinstance(value, list):
        return ()
    return tuple(str(item).strip() for item in value if item and str(item).strip())


def _normalized_category(value: object) -> str:
    """Normalize a category or return the unknown-category default."""
    if not isinstance(value, str) or not value.strip():
        return DEFAULT_LLM_CATEGORY
    category = value.strip()
    return DEFAULT_LLM_CATEGORY if category.lower() in ("na", "unknown", "document", "") else category


def _clean_string_items(values: list[object]) -> tuple[str, ...]:
    """Clean string items for safe filename metadata."""
    cleaned: list[str] = []
    for item in values:
        if not item:
            continue
        text = str(item).strip()
        if text:
            cleaned.append(text)
    return tuple(cleaned)


def _normalized_final_summary_tokens(value: object) -> tuple[str, ...] | None:
    """Normalize list- or comma-separated final summary tokens."""
    if isinstance(value, list):
        return _clean_string_items(value)
    if isinstance(value, str) and value.strip():
        return tuple(token.strip() for token in value.split(",") if token.strip())
    return None


def validate_llm_document_result(parsed: dict[str, object]) -> DocumentAnalysisResult:
    """
    Validate and fill defaults for a parsed LLM document analysis dict.
    Accepts optional keys: summary, keywords, category, final_summary_tokens.
    Normalizes the API response locally and fills defaults for malformed fields.
    """
    return DocumentAnalysisResult(
        summary=_normalized_summary(parsed.get("summary")),
        keywords=_normalized_keywords(parsed.get("keywords")),
        category=_normalized_category(parsed.get("category")),
        final_summary_tokens=_normalized_final_summary_tokens(parsed.get("final_summary_tokens")),
    )
