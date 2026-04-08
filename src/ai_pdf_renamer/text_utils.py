from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date

from .rename_ops import FILENAME_RESERVED_WIN, FILENAME_UNSAFE_RE

# Maximum keywords returned from normalize_keywords.
MAX_NORMALIZED_KEYWORDS = 7

_DATE_RE_YMD = re.compile(r"\b(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})\b")
_DATE_RE_DMY = re.compile(r"\b(\d{1,2})[./-](\d{1,2})[./-](\d{4})\b")

# Long-form: "18. Februar 2025" (DE) or "February 18, 2025" / "18 February 2025" (EN)
_DE_MONTHS = "januar|februar|märz|maerz|april|mai|juni|juli|august|september|oktober|november|dezember"
_EN_MONTHS = "january|february|march|april|may|june|july|august|september|october|november|december"
_MONTH_TO_NUM = {
    "januar": 1,
    "january": 1,
    "februar": 2,
    "february": 2,
    "märz": 3,
    "maerz": 3,
    "march": 3,
    "april": 4,
    "mai": 5,
    "may": 5,
    "juni": 6,
    "june": 6,
    "juli": 7,
    "july": 7,
    "august": 8,
    "september": 9,
    "oktober": 10,
    "october": 10,
    "november": 11,
    "dezember": 12,
    "december": 12,
}
_DATE_RE_DE_LONG = re.compile(
    r"\b(\d{1,2})\.\s*(" + _DE_MONTHS + r")\s+(\d{4})\b",
    re.IGNORECASE,
)
_DATE_RE_EN_LONG = re.compile(
    r"\b(" + _EN_MONTHS + r")\s+(\d{1,2}),?\s+(\d{4})\b",
    re.IGNORECASE,
)
_DATE_RE_EN_LONG_DD = re.compile(
    r"\b(\d{1,2})\s+(" + _EN_MONTHS + r")\s+(\d{4})\b",
    re.IGNORECASE,
)
# "Stand: 18.02.2025", "Datum: 18.02.2025", "Rechnungsdatum:", "Erstellt:", "Invoice date:", etc.
_DATE_RE_PREFIX_DMY = re.compile(
    r"\b(?:stand|datum|rechnungsdatum|datum\s+des\s+dokuments?|erstellt|datum\s+rechnung|"
    r"invoice\s+date|document\s+date|rechnungsdatum\s+des\s+dokuments?)\s*:\s*"
    r"(\d{1,2})[./-](\d{1,2})[./-](\d{4})\b",
    re.IGNORECASE,
)
# "January 2025", "Februar 2025" -> use 1st of month
_DATE_RE_MONTH_YEAR_DE = re.compile(
    r"\b(" + _DE_MONTHS + r")\s+(\d{4})\b",
    re.IGNORECASE,
)
_DATE_RE_MONTH_YEAR_EN = re.compile(
    r"\b(" + _EN_MONTHS + r")\s+(\d{4})\b",
    re.IGNORECASE,
)


def _find_date_in_text(
    content: str,
    *,
    date_locale: str,
    today: date,
) -> str | None:
    """Try all date patterns on content; return YYYY-MM-DD or None."""

    def make_ymd(y: str, m: str, d: str) -> str | None:
        try:
            year_int = int(y)
            if year_int < 1900 or year_int > 2100:
                return None
            dte = date(year_int, int(m), int(d))
            return dte.strftime("%Y-%m-%d")
        except (ValueError, TypeError):
            return None

    match = _DATE_RE_YMD.search(content)
    if match:
        y, m, d = match.groups()
        if (p := make_ymd(y, m, d)) is not None:
            return p
    match = _DATE_RE_DMY.search(content)
    if match:
        g1, g2, year = match.groups()
        month, day = (g1, g2) if (date_locale or "dmy").lower() == "mdy" else (g2, g1)
        if (p := make_ymd(year, month, day)) is not None:
            return p
    match = _DATE_RE_PREFIX_DMY.search(content)
    if match:
        g1, g2, year = match.groups()
        # P1: Check if the matched label is German; if so, always parse as DMY
        matched_text = match.group(0).lower()
        is_german_label = any(
            lbl in matched_text for lbl in ("rechnungsdatum", "datum", "stand", "erstellt", "rechnung")
        )
        if is_german_label or (date_locale or "dmy").lower() == "dmy":
            day, month = g1, g2
        else:
            day, month = g2, g1
        if (p := make_ymd(year, month, day)) is not None:
            return p
    match = _DATE_RE_DE_LONG.search(content)
    if match:
        day, month_name, year = match.groups()
        m = str(_MONTH_TO_NUM.get(month_name.lower(), 0))
        if m != "0" and (p := make_ymd(year, m, day)) is not None:
            return p
    match = _DATE_RE_EN_LONG.search(content)
    if match:
        month_name, day, year = match.groups()
        m = str(_MONTH_TO_NUM.get(month_name.lower(), 0))
        if m != "0" and (p := make_ymd(year, m, day)) is not None:
            return p
    match = _DATE_RE_EN_LONG_DD.search(content)
    if match:
        day, month_name, year = match.groups()
        m = str(_MONTH_TO_NUM.get(month_name.lower(), 0))
        if m != "0" and (p := make_ymd(year, m, day)) is not None:
            return p
    match = _DATE_RE_MONTH_YEAR_DE.search(content)
    if match:
        month_name, year = match.groups()
        m = str(_MONTH_TO_NUM.get(month_name.lower(), 0))
        if m != "0" and (p := make_ymd(year, m, "1")) is not None:
            return p
    match = _DATE_RE_MONTH_YEAR_EN.search(content)
    if match:
        month_name, year = match.groups()
        m = str(_MONTH_TO_NUM.get(month_name.lower(), 0))
        if m != "0" and (p := make_ymd(year, m, "1")) is not None:
            return p
    return None


def extract_date_from_content(
    content: str | None,
    *,
    today: date | None = None,
    date_locale: str = "dmy",
    prefer_leading_chars: int = 0,
) -> str:
    """
    Search text for date formats; return 'YYYY-MM-DD'. Uses first valid date.
    When prefer_leading_chars > 0, search in first N chars first (document date
    often in header). Supports YYYY-MM-DD, DD.MM.YYYY, long forms, Stand:/Datum:,
    and month-year (e.g. January 2025 -> 2025-01-01).
    If no date is found, returns today's date (Fallback: heute) so the filename always has a date.
    """
    if content is None or not isinstance(content, str):
        content = ""
    if today is None:
        today = date.today()
    loc = (date_locale or "dmy").lower()

    if prefer_leading_chars > 0 and len(content) > prefer_leading_chars:
        leading = content[:prefer_leading_chars]
        if parsed := _find_date_in_text(leading, date_locale=loc, today=today):
            return parsed
    if parsed := _find_date_in_text(content, date_locale=loc, today=today):
        return parsed
    return today.strftime("%Y-%m-%d")


def chunk_text(text: str, *, chunk_size: int = 8000, overlap: int = 1000) -> list[str]:
    """Split text into overlapping chunks of chunk_size characters with overlap for LLM context windows."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    if overlap >= chunk_size:
        raise ValueError("overlap must be < chunk_size")
    if not (text and text.strip()):
        return []

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks


def tokens_similar(token_a: str, token_b: str) -> bool:
    """Check if two tokens are similar (case-insensitive equal, or one is a prefix of the other within 2 chars)."""
    a = token_a.lower()
    b = token_b.lower()
    if a == b:
        return True
    return bool((a.startswith(b) or b.startswith(a)) and abs(len(a) - len(b)) <= 2)


def subtract_tokens(main_tokens: Iterable[str], remove_tokens: Iterable[str]) -> list[str]:
    """Remove tokens from main_tokens that are similar to any token in remove_tokens."""
    remove = [t for t in (rt.strip() for rt in remove_tokens) if t]
    result: list[str] = []
    for token in (t.strip() for t in main_tokens):
        if not token:
            continue
        if any(tokens_similar(token, rt) for rt in remove):
            continue
        result.append(token)
    return result


def normalize_keywords(raw: str | list[str] | tuple[str, ...] | None) -> list[str]:
    """Clean, deduplicate, and filter placeholder tokens from keywords, capped at MAX_NORMALIZED_KEYWORDS."""
    if raw is None:
        return []
    tokens = (
        [str(x).strip() for x in raw] if isinstance(raw, (list, tuple)) else [t.strip() for t in str(raw).split(",")]
    )

    filtered: list[str] = []
    for token in tokens:
        t = token.strip()
        if not t:
            continue
        low = t.lower()
        if low in {
            "...",
            "…",
            "na",
            "n/a",
            "xxx",
            "w1",
            "w2",
            "tbd",
            "tba",
            "etc",
            "etc.",
            "tbd.",
            "tba.",
        }:
            continue
        filtered.append(t)
    return filtered[:MAX_NORMALIZED_KEYWORDS]


# Windows reserved names (device names); avoid creating filenames that match on Windows.
_FILENAME_RESERVED_WIN = frozenset(name.lower() for name in FILENAME_RESERVED_WIN)


def clean_token(text: str) -> str:
    """
    Normalizes a token for filenames:
    - trims whitespace and trailing dots
    - replaces German umlauts
    - removes forbidden characters
    - converts whitespace to underscores
    - lowercases
    - avoids Windows reserved names (CON, PRN, AUX, NUL, COM1-9, LPT1-9) by appending _
    """
    text = text.strip().rstrip(".")
    if not text:
        return "na"
    text = (
        text.replace("Ä", "Ae")
        .replace("Ö", "Oe")
        .replace("Ü", "Ue")
        .replace("ä", "ae")
        .replace("ö", "oe")
        .replace("ü", "ue")
        .replace("ß", "ss")
        .replace("ẞ", "SS")  # P3: Uppercase Eszett (U+1E9E)
    )
    text = re.sub(r'[\\/:*?"<>|]', "", text)
    text = re.sub(r"\s+", "_", text)
    text = text.lower()
    if text in _FILENAME_RESERVED_WIN:
        text = text + "_"
    return text


_VALID_CASES = frozenset({"camelCase", "kebabCase", "snakeCase"})
# Single source of truth for CLI/GUI choices (avoid duplication).
VALID_CASE_CHOICES: tuple[str, ...] = tuple(sorted(_VALID_CASES))


def convert_case(tokens: Iterable[str], desired_case: str) -> str:
    """Convert a token list to a single string in camelCase, snakeCase, or kebabCase."""
    if desired_case not in _VALID_CASES:
        raise ValueError(f"Unknown desired_case: {desired_case!r}. Use one of: {sorted(_VALID_CASES)}")
    words = [w for w in (clean_token(t) for t in tokens) if w and w != "na"]
    if desired_case == "camelCase":
        split_words: list[str] = []
        for word in words:
            split_words.extend([w for w in word.split("_") if w])
        words = split_words
    if not words:
        return ""

    if desired_case == "camelCase":
        first = words[0].lower()
        rest = "".join(w.capitalize() for w in words[1:])
        return first + rest
    if desired_case == "snakeCase":
        return "_".join(w.lower() for w in words)
    return "-".join(w.lower() for w in words)


@dataclass(frozen=True)
class Stopwords:
    words: set[str]

    def filter_tokens(self, tokens: Iterable[str]) -> list[str]:
        out: list[str] = []
        for token in tokens:
            t = token.strip()
            if not t:
                continue
            if t.lower() in self.words:
                continue
            out.append(t)
        return out


def split_to_tokens(text: str | None) -> list[str]:
    """Split text on whitespace, commas, underscores, and hyphens into non-empty lowercase tokens."""
    if text is None or not isinstance(text, str):
        return []
    return [t for t in re.split(r"[\s,_-]+", text) if t]


# Structured fields: invoice number, amount, company (for template placeholders)
_INVOICE_ID_PATTERNS = [
    re.compile(
        r"\b(?:rechnungsnummer|rechnung\s*nr\.?|rechnung\s*#?|invoice\s*no\.?|invoice\s*#?|"
        r"bill\s*no\.?|order\s*no\.?|auftragsnummer|bestellnummer)\s*[:\s#]*\s*([A-Z0-9][A-Z0-9\-/]+)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b(?:inv|rechnung)\s*[#:]?\s*([A-Z0-9][A-Z0-9\-/]{2,})\b", re.IGNORECASE),
    re.compile(r"\b(\d{4,}-\d+)\b"),  # e.g. 2025-001234
]
_AMOUNT_PATTERNS = [
    re.compile(
        r"\b(?:betrag|summe|total|gesamt|amount|invoice\s*total)\s*[:\s]*"
        r"([\d.,]+)\s*(?:€|EUR|eur)?\b",
        re.IGNORECASE,
    ),
    # P2: Removed \s from character class to prevent overly broad matching
    re.compile(r"\b([\d.,]{3,})\s*€\b"),
    re.compile(r"\b(?:EUR|eur)\s*([\d.,]+)\b"),
]
# Company: same line after label or next non-empty line (simplified: first line after "Rechnung von" etc.)
_COMPANY_PATTERNS = [
    # P2: Limit capture group to prevent greedy matching to end of line
    re.compile(
        r"\b(?:rechnung\s*von|von\s*[:.]|an\s*[:.]|from\s*[:.]|seller\s*[:.]|lieferant\s*[:.])\s*([^\n\r]{2,50}?)(?:\n|$)",
        re.IGNORECASE,
    ),
    re.compile(r"\b(?:company|firma|an\s*:)\s*([A-Za-z0-9\s&.\-]{2,40})\b", re.IGNORECASE),
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
    """Normalize amount for filename: digits and one dot, no spaces.

    Handles European format (1.234,56 or 1 234,56) and US format (1,234.56).
    """
    s = re.sub(r"[^\d.,]", "", raw)
    # P2: Handle European thousand separators
    # European: 1.234,56 -> dots are thousand separators, comma is decimal
    # US: 1,234.56 -> commas are thousand separators, dot is decimal
    if "," in s and "." in s:
        last_comma = s.rfind(",")
        last_dot = s.rfind(".")
        # European (1.234,56) vs US (1,234.56): comma after dot means European
        s = s.replace(".", "").replace(",", ".") if last_comma > last_dot else s.replace(",", "")
    elif "," in s:
        # Could be European decimal (3,50) or thousand separator (1,000)
        # If comma has exactly 3 digits after it and it's the only comma, treat as thousand sep
        # Exactly 3 digits after a single comma = thousand separator; otherwise decimal
        parts = s.split(",")
        s = s.replace(",", "") if len(parts) == 2 and len(parts[1]) == 3 else s.replace(",", ".")
    # Dots only: could be thousand separator (1.000) or decimal (3.50)
    # Multiple dots -> thousand separators
    elif s.count(".") > 1:
        # Multiple dots are thousand separators, no decimal
        s = s.replace(".", "")
    if re.match(r"^[\d.]+$", s):
        return s
    return ""


def extract_structured_fields(
    content: str | None,
    *,
    max_chars: int = 5000,
) -> dict[str, str]:
    """
    Extract invoice_id, amount, and company from document text using heuristics.
    Searches the first max_chars characters. Returns dict with keys invoice_id, amount, company
    (empty string if not found). Values are sanitized for use in filenames.
    """
    result: dict[str, str] = {"invoice_id": "", "amount": "", "company": ""}
    if not content or not isinstance(content, str):
        return result
    text = content[:max_chars] if len(content) > max_chars else content
    for pat in _INVOICE_ID_PATTERNS:
        m = pat.search(text)
        if m:
            result["invoice_id"] = _sanitize_structured_value(m.group(1))
            break
    for pat in _AMOUNT_PATTERNS:
        m = pat.search(text)
        if m:
            normalized = _normalize_amount(m.group(1))
            if normalized:
                result["amount"] = _sanitize_structured_value(normalized)
            break
    for pat in _COMPANY_PATTERNS:
        m = pat.search(text)
        if m:
            result["company"] = _sanitize_structured_value(m.group(1))
            break
    return result
