from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, timedelta

from .rename_ops import FILENAME_RESERVED_WIN, FILENAME_UNSAFE_RE

# Maximum keywords returned from normalize_keywords.
MAX_NORMALIZED_KEYWORDS = 7

_DATE_RE_YMD = re.compile(
    r"\b(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})(?:[T\s]\d{1,2}:\d{2}(?::\d{2})?(?:[.,]\d+)?(?:Z|[+-]\d{2}:?\d{2})?)?\b"
)
_DATE_RE_YMD_COMPACT = re.compile(r"\b(\d{4})(\d{2})(\d{2})(?:T\d{6}(?:Z|[+-]\d{2}:?\d{2})?)?\b")
_DATE_RE_DMY = re.compile(r"\b(\d{1,2})[./-](\d{1,2})[./-](\d{4})\b")

# Long-form: "18. Februar 2025" (DE) or "February 18, 2025" / "18 February 2025" (EN)
_DE_MONTHS = (
    "januar|jan\\.?|februar|feb\\.?|märz|mär\\.?|mrz\\.?|maerz|april|apr\\.?|mai|"
    "juni|jun\\.?|juli|jul\\.?|august|aug\\.?|september|sept?\\.?|oktober|okt\\.?|november|nov\\.?|dezember|dez\\.?"
)
_EN_MONTHS = (
    "january|jan\\.?|february|feb\\.?|march|mar\\.?|april|apr\\.?|may|"
    "june|jun\\.?|july|jul\\.?|august|aug\\.?|september|sept?\\.?|october|oct\\.?|november|nov\\.?|december|dec\\.?"
)
_MONTH_TO_NUM = {
    "januar": 1,
    "jan": 1,
    "january": 1,
    "jan.": 1,
    "februar": 2,
    "feb": 2,
    "february": 2,
    "feb.": 2,
    "märz": 3,
    "mär": 3,
    "mrz": 3,
    "maerz": 3,
    "march": 3,
    "mar": 3,
    "mär.": 3,
    "mrz.": 3,
    "mar.": 3,
    "april": 4,
    "apr": 4,
    "apr.": 4,
    "mai": 5,
    "may": 5,
    "juni": 6,
    "june": 6,
    "jun": 6,
    "jun.": 6,
    "juli": 7,
    "july": 7,
    "jul": 7,
    "jul.": 7,
    "august": 8,
    "aug": 8,
    "aug.": 8,
    "september": 9,
    "sep": 9,
    "sep.": 9,
    "sept": 9,
    "sept.": 9,
    "oktober": 10,
    "october": 10,
    "okt": 10,
    "okt.": 10,
    "oct": 10,
    "oct.": 10,
    "november": 11,
    "nov": 11,
    "nov.": 11,
    "dezember": 12,
    "december": 12,
    "dez": 12,
    "dez.": 12,
    "dec": 12,
    "dec.": 12,
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
_DATE_KEYWORD_RE = re.compile(
    r"(?i)\b(datum|date|rechnungsdatum|invoice\s+date|document\s+date|issue\s+date|issued|statement\s+date|"
    r"billing\s+date|stand|erstellt|ausgestellt|belegdatum)\b"
)
_DATE_MIN = date(1990, 1, 1)
_DATE_MAX_FUTURE_DELTA = timedelta(days=366)
_AMOUNT_MAX = 1_000_000.0
_COMPANY_SUFFIXES = r"(?:GmbH|AG|Inc\.?|Ltd\.?|LLC)"
_INVOICE_ID_VALUE_SHORTHAND = r"[A-Z0-9]{2,}(?:[/-][A-Z0-9]{2,})+"
_INVOICE_ID_VALUE_EXPLICIT = r"(?:[A-Z]{2,}\d{2,}|\d{4,})"
_INVOICE_ID_VALUE = f"({_INVOICE_ID_VALUE_SHORTHAND}|{_INVOICE_ID_VALUE_EXPLICIT})"


@dataclass(frozen=True)
class _DateCandidate:
    value: date
    start: int
    score: int


def _normalize_month_token(month_name: str) -> str:
    """Normalize month names and abbreviations for lookup."""
    return month_name.strip().lower()


def _validation_reference_date(today: date) -> date:
    """Use the later of the injected fallback date and the real current date for plausibility checks."""
    return max(today, date.today())


def _make_date_candidate(
    y: str,
    m: str,
    d: str,
    *,
    today: date,
    start: int,
    content: str,
    prefer_leading_chars: int,
    base_score: int,
) -> _DateCandidate | None:
    """Build a validated candidate with positional weighting."""
    try:
        candidate = date(int(y), int(m), int(d))
    except (TypeError, ValueError):
        return None
    validation_today = _validation_reference_date(today)
    if candidate < _DATE_MIN or candidate > validation_today + _DATE_MAX_FUTURE_DELTA:
        return None
    score = base_score
    if prefer_leading_chars > 0 and start < prefer_leading_chars:
        score += 30
        prefix = content[max(0, start - 40) : start]
        if _DATE_KEYWORD_RE.search(prefix):
            score += 60
    score -= min(start // 2000, 12)
    return _DateCandidate(value=candidate, start=start, score=score)


def _best_date_candidate(candidates: list[_DateCandidate]) -> str | None:
    """Return the highest-scoring candidate as YYYY-MM-DD."""
    if not candidates:
        return None
    best = max(candidates, key=lambda candidate: (candidate.score, -candidate.start))
    return best.value.strftime("%Y-%m-%d")


def _find_date_candidates(
    content: str,
    *,
    date_locale: str,
    today: date,
    prefer_leading_chars: int = 0,
) -> list[_DateCandidate]:
    """Collect validated date candidates from document text."""
    candidates: list[_DateCandidate] = []

    for match in _DATE_RE_PREFIX_DMY.finditer(content):
        g1, g2, year = match.groups()
        matched_text = match.group(0).lower()
        is_german_label = any(
            lbl in matched_text for lbl in ("rechnungsdatum", "datum", "stand", "erstellt", "rechnung")
        )
        day, month = (g1, g2) if is_german_label or date_locale == "dmy" else (g2, g1)
        if candidate := _make_date_candidate(
            year,
            month,
            day,
            today=today,
            start=match.start(),
            content=content,
            prefer_leading_chars=prefer_leading_chars,
            base_score=100,
        ):
            candidates.append(candidate)

    for match in _DATE_RE_YMD.finditer(content):
        year, month, day = match.groups()
        if candidate := _make_date_candidate(
            year,
            month,
            day,
            today=today,
            start=match.start(),
            content=content,
            prefer_leading_chars=prefer_leading_chars,
            base_score=120,
        ):
            candidates.append(candidate)

    for match in _DATE_RE_YMD_COMPACT.finditer(content):
        year, month, day = match.groups()
        if candidate := _make_date_candidate(
            year,
            month,
            day,
            today=today,
            start=match.start(),
            content=content,
            prefer_leading_chars=prefer_leading_chars,
            base_score=115,
        ):
            candidates.append(candidate)

    for match in _DATE_RE_DMY.finditer(content):
        g1, g2, year = match.groups()
        month, day = (g1, g2) if date_locale == "mdy" else (g2, g1)
        if candidate := _make_date_candidate(
            year,
            month,
            day,
            today=today,
            start=match.start(),
            content=content,
            prefer_leading_chars=prefer_leading_chars,
            base_score=100,
        ):
            candidates.append(candidate)

    for match in _DATE_RE_DE_LONG.finditer(content):
        day, month_name, year = match.groups()
        month = str(_MONTH_TO_NUM.get(_normalize_month_token(month_name), 0))
        if month != "0" and (
            candidate := _make_date_candidate(
                year,
                month,
                day,
                today=today,
                start=match.start(),
                content=content,
                prefer_leading_chars=prefer_leading_chars,
                base_score=110,
            )
        ):
            candidates.append(candidate)

    for match in _DATE_RE_EN_LONG.finditer(content):
        month_name, day, year = match.groups()
        month = str(_MONTH_TO_NUM.get(_normalize_month_token(month_name), 0))
        if month != "0" and (
            candidate := _make_date_candidate(
                year,
                month,
                day,
                today=today,
                start=match.start(),
                content=content,
                prefer_leading_chars=prefer_leading_chars,
                base_score=110,
            )
        ):
            candidates.append(candidate)

    for match in _DATE_RE_EN_LONG_DD.finditer(content):
        day, month_name, year = match.groups()
        month = str(_MONTH_TO_NUM.get(_normalize_month_token(month_name), 0))
        if month != "0" and (
            candidate := _make_date_candidate(
                year,
                month,
                day,
                today=today,
                start=match.start(),
                content=content,
                prefer_leading_chars=prefer_leading_chars,
                base_score=110,
            )
        ):
            candidates.append(candidate)

    for match in _DATE_RE_MONTH_YEAR_DE.finditer(content):
        month_name, year = match.groups()
        month = str(_MONTH_TO_NUM.get(_normalize_month_token(month_name), 0))
        if month != "0" and (
            candidate := _make_date_candidate(
                year,
                month,
                "1",
                today=today,
                start=match.start(),
                content=content,
                prefer_leading_chars=prefer_leading_chars,
                base_score=70,
            )
        ):
            candidates.append(candidate)

    for match in _DATE_RE_MONTH_YEAR_EN.finditer(content):
        month_name, year = match.groups()
        month = str(_MONTH_TO_NUM.get(_normalize_month_token(month_name), 0))
        if month != "0" and (
            candidate := _make_date_candidate(
                year,
                month,
                "1",
                today=today,
                start=match.start(),
                content=content,
                prefer_leading_chars=prefer_leading_chars,
                base_score=70,
            )
        ):
            candidates.append(candidate)

    return candidates


def _find_date_in_text(
    content: str,
    *,
    date_locale: str,
    today: date,
    prefer_leading_chars: int = 0,
) -> str | None:
    """Try all date patterns on content; return YYYY-MM-DD or None."""
    return _best_date_candidate(
        _find_date_candidates(
            content,
            date_locale=(date_locale or "dmy").lower(),
            today=today,
            prefer_leading_chars=prefer_leading_chars,
        )
    )


def _best_metadata_date(pdf_metadata: dict[str, object] | None, *, today: date) -> str | None:
    """Return the best valid PDF metadata date."""
    if not isinstance(pdf_metadata, dict):
        return None
    best: _DateCandidate | None = None
    for index, key in enumerate(("creation_date", "mod_date")):
        raw_value = pdf_metadata.get(key)
        if not isinstance(raw_value, str) or not raw_value.strip():
            continue
        match = _DATE_RE_YMD.search(raw_value) or _DATE_RE_YMD_COMPACT.search(raw_value)
        if not match:
            continue
        year, month, day = match.groups()
        candidate = _make_date_candidate(
            year,
            month,
            day,
            today=today,
            start=index,
            content=key,
            prefer_leading_chars=0,
            base_score=25 - index,
        )
        if candidate is None:
            continue
        if best is None or candidate.score > best.score:
            best = candidate
    return best.value.strftime("%Y-%m-%d") if best is not None else None


def extract_date_from_content(
    content: str | None,
    *,
    today: date | None = None,
    date_locale: str = "dmy",
    prefer_leading_chars: int = 0,
    pdf_metadata: dict[str, object] | None = None,
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

    if parsed := _find_date_in_text(content, date_locale=loc, today=today, prefer_leading_chars=prefer_leading_chars):
        return parsed
    if parsed := _best_metadata_date(pdf_metadata, today=today):
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
        r"\b(?:rechnungsnummer|rechnung\s*(?:nr\.?|nummer|#)|invoice\s*(?:no\.?|number|#|id)|"
        r"bill\s*(?:no\.?|number|#)|order\s*(?:no\.?|number)|auftragsnummer|bestellnummer|belegnummer|reference\s*no\.?)"
        r"\s*[:\s#-]*\s*" + _INVOICE_ID_VALUE + r"\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b(?:inv|rechnung|rg|ref)\s*[#:-]?\s*(" + _INVOICE_ID_VALUE_SHORTHAND + r")\b", re.IGNORECASE),
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
    """Normalize amount for filename: digits and one dot, no spaces.

    Handles European format (1.234,56 or 1 234,56) and US format (1,234.56).
    """
    s = re.sub(r"[^\d.,]", "", raw)
    if not s:
        return ""

    if "," in s and "." in s:
        last_comma = s.rfind(",")
        last_dot = s.rfind(".")
        if last_comma > last_dot:
            integer_part = s[:last_comma].replace(".", "").replace(",", "")
            decimal_part = re.sub(r"[^\d]", "", s[last_comma + 1 :])
            s = f"{integer_part}.{decimal_part}" if decimal_part else integer_part
        else:
            integer_part = s[:last_dot].replace(",", "").replace(".", "")
            decimal_part = re.sub(r"[^\d]", "", s[last_dot + 1 :])
            s = f"{integer_part}.{decimal_part}" if decimal_part else integer_part
    elif "," in s:
        left, right = s.rsplit(",", 1)
        s = f"{left.replace(',', '').replace('.', '')}.{right}" if len(right) == 2 else left.replace(",", "") + right
    elif "." in s:
        left, right = s.rsplit(".", 1)
        s = f"{left.replace('.', '').replace(',', '')}.{right}" if len(right) == 2 else left.replace(".", "") + right

    if not re.fullmatch(r"\d+(?:\.\d+)?", s):
        return ""
    return s


def _is_plausible_amount(normalized: str) -> bool:
    """Reject obviously implausible consumer-document amounts."""
    try:
        amount = float(normalized)
    except ValueError:
        return False
    return 0 < amount <= _AMOUNT_MAX


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
            if normalized and _is_plausible_amount(normalized):
                result["amount"] = _sanitize_structured_value(normalized)
                break
    for pat in _COMPANY_PATTERNS:
        m = pat.search(text)
        if m:
            result["company"] = _sanitize_structured_value(m.group(1))
            break
    return result
