"""Date extraction helpers for document text and PDF metadata."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, timedelta

_DATE_RE_YMD = re.compile(
    r"\b(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})(?:[T\s]\d{1,2}:\d{2}(?::\d{2})?(?:[.,]\d+)?(?:Z|[+-]\d{2}:?\d{2})?)?\b"
)
_DATE_RE_YMD_COMPACT = re.compile(r"\b(\d{4})(\d{2})(\d{2})(?:T\d{6}(?:Z|[+-]\d{2}:?\d{2})?)?\b")
_DATE_RE_DMY = re.compile(r"\b(\d{1,2})[./-](\d{1,2})[./-](\d{4})\b")

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
_DATE_RE_PREFIX_DMY = re.compile(
    r"\b(?:stand|datum|rechnungsdatum|datum\s+des\s+dokuments?|erstellt|datum\s+rechnung|"
    r"invoice\s+date|document\s+date|rechnungsdatum\s+des\s+dokuments?)\s*:\s*"
    r"(\d{1,2})[./-](\d{1,2})[./-](\d{4})\b",
    re.IGNORECASE,
)
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


@dataclass(frozen=True)
class _DateCandidate:
    """A validated date candidate with source position and ranking score."""

    value: date
    start: int
    score: int


@dataclass(frozen=True)
class _CandidateContext:
    """Shared document and reference-date context for candidate scoring."""

    today: date
    content: str
    prefer_leading_chars: int


@dataclass(frozen=True)
class _DateParts:
    """Raw year, month, and day components for a date candidate."""

    year: str
    month: str
    day: str


@dataclass(frozen=True)
class _OrderedDateParts:
    """All inputs needed to append a candidate whose first two parts may swap."""

    first: str
    second: str
    year: str
    month_first: bool
    month_is_name: bool
    context: _CandidateContext
    start: int
    base_score: int


def _normalize_month_token(month_name: str) -> str:
    """Normalize month names and abbreviations for lookup."""
    return month_name.strip().lower()


def _validation_reference_date(today: date) -> date:
    """Use the injected fallback date as the sole reference for plausibility checks."""
    return today


def _make_date_candidate(
    parts: _DateParts,
    *,
    context: _CandidateContext,
    start: int,
    base_score: int,
) -> _DateCandidate | None:
    """Build a validated candidate with positional weighting."""
    try:
        candidate = date(int(parts.year), int(parts.month), int(parts.day))
    except (
        TypeError,
        ValueError,
    ):
        return None
    validation_today = _validation_reference_date(context.today)
    if candidate < _DATE_MIN or candidate > validation_today + _DATE_MAX_FUTURE_DELTA:
        return None
    score = base_score
    if context.prefer_leading_chars > 0 and start < context.prefer_leading_chars:
        score += 30
        prefix = context.content[max(0, start - 40) : start]
        if _DATE_KEYWORD_RE.search(prefix):
            score += 60
    score -= min(start // 2000, 12)
    return _DateCandidate(value=candidate, start=start, score=score)


def _best_date_candidate(candidates: list[_DateCandidate]) -> str | None:
    """Return the highest-scoring candidate as YYYY-MM-DD."""
    if not candidates:
        return None
    best = max(candidates, key=lambda candidate: (candidate.score, -candidate.start))
    return best.value.isoformat()


def _append_date_candidate(
    candidates: list[_DateCandidate],
    *,
    parts: _DateParts,
    context: _CandidateContext,
    start: int,
    base_score: int,
) -> None:
    """Validate and append one numeric date candidate."""
    candidate = _make_date_candidate(
        parts,
        context=context,
        start=start,
        base_score=base_score,
    )
    if candidate is not None:
        candidates.append(candidate)


def _append_month_name_candidate(
    candidates: list[_DateCandidate],
    *,
    parts: _DateParts,
    context: _CandidateContext,
    start: int,
    base_score: int,
) -> None:
    """Resolve a month name, then validate and append the candidate."""
    month = str(_MONTH_TO_NUM.get(_normalize_month_token(parts.month), 0))
    if month == "0":
        return
    _append_date_candidate(
        candidates,
        parts=_DateParts(year=parts.year, month=month, day=parts.day),
        context=context,
        start=start,
        base_score=base_score,
    )


def _append_ordered_date_candidate(
    candidates: list[_DateCandidate],
    ordered: _OrderedDateParts,
) -> None:
    """Append one ordered numeric or month-name date candidate."""
    month, day = (ordered.first, ordered.second) if ordered.month_first else (ordered.second, ordered.first)
    parts = _DateParts(year=ordered.year, month=month, day=day)
    if ordered.month_is_name:
        _append_month_name_candidate(
            candidates,
            parts=parts,
            context=ordered.context,
            start=ordered.start,
            base_score=ordered.base_score,
        )
        return
    _append_date_candidate(
        candidates,
        parts=parts,
        context=ordered.context,
        start=ordered.start,
        base_score=ordered.base_score,
    )


def _append_prefixed_dmy_candidates(
    candidates: list[_DateCandidate],
    context: _CandidateContext,
    *,
    date_locale: str,
) -> None:
    """Collect labeled numeric dates using the configured locale order."""
    for match in _DATE_RE_PREFIX_DMY.finditer(context.content):
        g1, g2, year = match.groups()
        matched_text = match.group(0).lower()
        is_german_label = any(
            label in matched_text for label in ("rechnungsdatum", "datum", "stand", "erstellt", "rechnung")
        )
        _append_ordered_date_candidate(
            candidates,
            _OrderedDateParts(
                first=g1,
                second=g2,
                year=year,
                month_first=not (is_german_label or date_locale == "dmy"),
                month_is_name=False,
                context=context,
                start=match.start(),
                base_score=100,
            ),
        )


def _append_ymd_candidates(candidates: list[_DateCandidate], context: _CandidateContext) -> None:
    """Collect year-month-day dates in separated and compact forms."""
    for regex, score in ((_DATE_RE_YMD, 120), (_DATE_RE_YMD_COMPACT, 115)):
        for match in regex.finditer(context.content):
            year, month, day = match.groups()
            _append_date_candidate(
                candidates,
                parts=_DateParts(year=year, month=month, day=day),
                context=context,
                start=match.start(),
                base_score=score,
            )


def _append_dmy_candidates(
    candidates: list[_DateCandidate],
    context: _CandidateContext,
    *,
    date_locale: str,
) -> None:
    """Collect ambiguous numeric dates using the configured locale order."""
    for match in _DATE_RE_DMY.finditer(context.content):
        g1, g2, year = match.groups()
        _append_ordered_date_candidate(
            candidates,
            _OrderedDateParts(
                first=g1,
                second=g2,
                year=year,
                month_first=date_locale == "mdy",
                month_is_name=False,
                context=context,
                start=match.start(),
                base_score=100,
            ),
        )


def _append_long_month_candidates(candidates: list[_DateCandidate], context: _CandidateContext) -> None:
    """Collect German and English dates containing month names."""
    for regex, month_first in (
        (_DATE_RE_DE_LONG, False),
        (_DATE_RE_EN_LONG, True),
        (_DATE_RE_EN_LONG_DD, False),
    ):
        for match in regex.finditer(context.content):
            first, second, year = match.groups()
            _append_ordered_date_candidate(
                candidates,
                _OrderedDateParts(
                    first=first,
                    second=second,
                    year=year,
                    month_first=month_first,
                    month_is_name=True,
                    context=context,
                    start=match.start(),
                    base_score=110,
                ),
            )


def _append_month_year_candidates(candidates: list[_DateCandidate], context: _CandidateContext) -> None:
    """Collect month-year dates using the first day of the month."""
    for regex in (_DATE_RE_MONTH_YEAR_DE, _DATE_RE_MONTH_YEAR_EN):
        for match in regex.finditer(context.content):
            month_name, year = match.groups()
            _append_month_name_candidate(
                candidates,
                parts=_DateParts(year=year, month=month_name, day="1"),
                context=context,
                start=match.start(),
                base_score=70,
            )


def _find_date_candidates(
    content: str,
    *,
    date_locale: str,
    today: date,
    prefer_leading_chars: int = 0,
) -> list[_DateCandidate]:
    """Collect validated date candidates from document text."""
    candidates: list[_DateCandidate] = []
    context = _CandidateContext(today=today, content=content, prefer_leading_chars=prefer_leading_chars)
    _append_prefixed_dmy_candidates(candidates, context, date_locale=date_locale)
    _append_ymd_candidates(candidates, context)
    _append_dmy_candidates(candidates, context, date_locale=date_locale)
    _append_long_month_candidates(candidates, context)
    _append_month_year_candidates(candidates, context)
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


def _metadata_date_candidate(
    pdf_metadata: dict[str, object],
    key: str,
    *,
    index: int,
    context: _CandidateContext,
) -> _DateCandidate | None:
    """Parse and score one date field from PDF metadata."""
    raw_value = pdf_metadata.get(key)
    if not isinstance(raw_value, str) or not raw_value.strip():
        return None
    match = _DATE_RE_YMD.search(raw_value) or _DATE_RE_YMD_COMPACT.search(raw_value)
    if not match:
        return None
    year, month, day = match.groups()
    return _make_date_candidate(
        _DateParts(year=year, month=month, day=day),
        context=context,
        start=index,
        base_score=25 - index,
    )


def _best_metadata_date(pdf_metadata: dict[str, object] | None, *, today: date) -> str | None:
    """Return the best valid PDF metadata date."""
    if not isinstance(pdf_metadata, dict):
        return None
    best: _DateCandidate | None = None
    context = _CandidateContext(today=today, content="", prefer_leading_chars=0)
    for index, key in enumerate(("creation_date", "mod_date")):
        candidate = _metadata_date_candidate(pdf_metadata, key, index=index, context=context)
        if candidate is None:
            continue
        if best is None or candidate.score > best.score:
            best = candidate
    return best.value.isoformat() if best is not None else None


def extract_date_from_content(
    content: str | None,
    *,
    today: date | None = None,
    date_locale: str = "dmy",
    prefer_leading_chars: int = 0,
    pdf_metadata: dict[str, object] | None = None,
) -> str:
    """
    Search text for date formats; return 'YYYY-MM-DD'.

    Supports YYYY-MM-DD, DD.MM.YYYY, long forms, Stand:/Datum:, and
    month-year. If no date is found, returns today's date so the filename
    always has a date.
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
    return today.isoformat()
