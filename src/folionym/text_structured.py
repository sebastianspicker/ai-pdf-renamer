"""Structured-field extraction helpers for document text."""

from __future__ import annotations

import re

from .rename_ops import FILENAME_UNSAFE_RE

_AMOUNT_MAX = 1_000_000.0
_COMPANY_SUFFIXES = r"(?:GmbH|AG|Inc\.?|Ltd\.?|LLC)"
_INVOICE_ID_VALUE_SHORTHAND = r"[A-Z0-9]{2,}(?:[/-][A-Z0-9]{2,})+"
_INVOICE_ID_VALUE_EXPLICIT = r"(?:[A-Z]{2,}\d{2,}|\d{4,})"
_INVOICE_ID_VALUE = f"({_INVOICE_ID_VALUE_SHORTHAND}|{_INVOICE_ID_VALUE_EXPLICIT})"
_INVOICE_ID_PATTERNS = [
    re.compile(
        r"\b(?:rechnungsnummer|rechnung\s*(?:nr\.?|nummer|#)|invoice\s*(?:no\.?|number|#|id)|"
        r"bill\s*(?:no\.?|number|#)|order\s*(?:no\.?|number)|auftragsnummer|bestellnummer|belegnummer|reference\s*no\.?)"
        r"\s*[:\s#-]*\s*" + _INVOICE_ID_VALUE + r"\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b(?:inv|rechnung|rg|ref)\s*[#:-]?\s*(" + _INVOICE_ID_VALUE_SHORTHAND + r")\b", re.IGNORECASE),
    re.compile(r"\b(\d{4,}-\d+)\b"),
]
_AMOUNT_PATTERNS = [
    re.compile(
        r"\b(?:betrag|summe|total|gesamt|amount|invoice\s*total)\s*[:\s]*"
        r"([\d.,]+)\s*(?:€|EUR|eur)?\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b([\d.,]{3,})\s*€\b"),
    re.compile(r"\b(?:EUR|eur)\s*([\d.,]+)\b"),
]
_COMPANY_PATTERNS = [
    re.compile(
        r"\b(?:rechnung\s*von|von\s*[:.]|an\s*[:.]|from\s*[:.]|seller\s*[:.]|lieferant\s*[:.])\s*([^\n\r]{2,50}?)(?:\n|$)",
        re.IGNORECASE,
    ),
    re.compile(r"\b(?:company|firma|an\s*:)\s*([A-Za-z0-9\s&.\-]{2,40})\b", re.IGNORECASE),
    re.compile(r"(?m)^\s*([A-Z][A-Za-z0-9&.,' -]{1,60}\s+" + _COMPANY_SUFFIXES + r")\s*$"),
]
_MAX_STRUCTURED_LEN = 50


def _sanitize_structured_value(value: str) -> str:
    """Remove path/control chars and truncate for use in filenames."""
    if not value or not isinstance(value, str):
        return ""
    s = FILENAME_UNSAFE_RE.sub("", value.strip()).strip()
    s = re.sub(r"\s+", "_", s)
    return s[:_MAX_STRUCTURED_LEN] if len(s) > _MAX_STRUCTURED_LEN else s


def _normalize_amount(raw: str) -> str:
    """Normalize amount for filename: digits and one dot, no spaces."""
    s = re.sub(r"[^\d.,]", "", raw)
    if not s:
        return ""

    if "," in s and "." in s:
        s = _normalize_mixed_separator_amount(s)
    elif "," in s:
        s = _normalize_single_separator_amount(s, separator=",")
    elif "." in s:
        s = _normalize_single_separator_amount(s, separator=".")

    if not re.fullmatch(r"\d+(?:\.\d+)?", s):
        return ""
    return s


def _normalize_mixed_separator_amount(value: str) -> str:
    """Normalize an amount containing both decimal and grouping separators."""
    decimal_separator = "," if value.rfind(",") > value.rfind(".") else "."
    integer_part, decimal_part = value.rsplit(decimal_separator, 1)
    integer_digits = integer_part.replace(".", "").replace(",", "")
    decimal_digits = re.sub(r"[^\d]", "", decimal_part)
    return f"{integer_digits}.{decimal_digits}" if decimal_digits else integer_digits


def _normalize_single_separator_amount(value: str, *, separator: str) -> str:
    """Normalize an amount containing one possible decimal separator."""
    left, right = value.rsplit(separator, 1)
    thousands_free = left.replace(",", "").replace(".", "")
    if len(right) in (1, 2):
        return f"{thousands_free}.{right}"
    return thousands_free + right


def _is_plausible_amount(normalized: str) -> bool:
    """Reject obviously implausible consumer-document amounts."""
    try:
        amount = float(normalized)
    except ValueError:
        return False
    return 0 < amount <= _AMOUNT_MAX


def _first_sanitized_match(patterns: list[re.Pattern[str]], text: str) -> str:
    """Return the first matching capture after filename sanitization."""
    for pattern in patterns:
        match = pattern.search(text)
        if match:
            return _sanitize_structured_value(match.group(1))
    return ""


def _first_plausible_amount(text: str) -> str:
    """Return the first matched amount that passes plausibility checks."""
    for pattern in _AMOUNT_PATTERNS:
        match = pattern.search(text)
        if not match:
            continue
        normalized = _normalize_amount(match.group(1))
        if normalized and _is_plausible_amount(normalized):
            return _sanitize_structured_value(normalized)
    return ""


def extract_structured_fields(
    content: str | None,
    *,
    max_chars: int = 5000,
) -> dict[str, str]:
    """
    Extract invoice_id, amount, and company from document text using heuristics.

    Searches the first max_chars characters. Returns dict with keys invoice_id,
    amount, company; empty strings are used when fields are not found.
    """
    result: dict[str, str] = {"invoice_id": "", "amount": "", "company": ""}
    if not content or not isinstance(content, str):
        return result
    text = content[:max_chars] if len(content) > max_chars else content
    result["invoice_id"] = _first_sanitized_match(_INVOICE_ID_PATTERNS, text)
    result["amount"] = _first_plausible_amount(text)
    result["company"] = _first_sanitized_match(_COMPANY_PATTERNS, text)
    return result
