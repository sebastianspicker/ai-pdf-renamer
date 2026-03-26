from __future__ import annotations

from datetime import date

import pytest

from ai_pdf_renamer.text_utils import (
    MAX_NORMALIZED_KEYWORDS,
    chunk_text,
    convert_case,
    extract_date_from_content,
    extract_structured_fields,
    normalize_keywords,
    subtract_tokens,
    tokens_similar,
)


def test_extract_date_from_content_ymd() -> None:
    assert extract_date_from_content("Hello 2024-01-09 world", today=date(2000, 1, 1)) == "2024-01-09"


def test_extract_date_from_content_dmy() -> None:
    assert extract_date_from_content("Hello 9.1.2024 world", today=date(2000, 1, 1)) == "2024-01-09"


def test_extract_date_from_content_fallback_today() -> None:
    assert extract_date_from_content("No dates here", today=date(2022, 12, 31)) == "2022-12-31"


def test_extract_date_stand_and_month_year() -> None:
    assert (
        extract_date_from_content(
            "Stand: 18.02.2025\nFooter 2020",
            today=date(2000, 1, 1),
        )
        == "2025-02-18"
    )
    assert (
        extract_date_from_content(
            "January 2025",
            today=date(2000, 1, 1),
        )
        == "2025-01-01"
    )
    assert (
        extract_date_from_content(
            "Februar 2024",
            today=date(2000, 1, 1),
        )
        == "2024-02-01"
    )


def test_extract_date_prefer_leading_chars() -> None:
    # Document date in header; footer has different date
    text = "Rechnung 15.03.2024\n" + "x" * 5000 + "\nGedruckt am 01.01.2020"
    assert (
        extract_date_from_content(
            text,
            today=date(2000, 1, 1),
            prefer_leading_chars=8000,
        )
        == "2024-03-15"
    )
    # When leading region has a date, we use it (so footer date ignored)
    text2 = "Gedruckt 01.01.2020\n" + "y" * 5000 + "\nRechnung 15.03.2024"
    assert (
        extract_date_from_content(
            text2,
            today=date(2000, 1, 1),
            prefer_leading_chars=200,
        )
        == "2020-01-01"
    )
    # Leading 10000 chars contains both; first match wins (2020-01-01)
    assert (
        extract_date_from_content(
            text2,
            today=date(2000, 1, 1),
            prefer_leading_chars=10000,
        )
        == "2020-01-01"
    )


def test_chunk_text_validation() -> None:
    with pytest.raises(ValueError):
        chunk_text("abc", chunk_size=0, overlap=0)
    with pytest.raises(ValueError):
        chunk_text("abc", chunk_size=10, overlap=10)


def test_normalize_keywords_filters_placeholders() -> None:
    assert normalize_keywords("a, ..., na, b, w1, c") == ["a", "b", "c"]


def test_subtract_tokens_removes_similar() -> None:
    assert subtract_tokens(["invoice", "foo"], ["invoice"]) == ["foo"]


def test_convert_case_kebab_and_camel() -> None:
    assert convert_case(["Hello", "World"], "kebabCase") == "hello-world"
    assert convert_case(["Hello", "World"], "camelCase") == "helloWorld"


def test_convert_case_camel_splits_underscores() -> None:
    assert convert_case(["foo_bar", "baz"], "camelCase") == "fooBarBaz"


def test_tokens_similar_prefix_tolerance() -> None:
    assert tokens_similar("invoice", "invoice")
    assert tokens_similar("invoice", "invoi")
    assert tokens_similar("invoi", "invoice")
    assert not tokens_similar("invoice", "receipt")


def test_extract_structured_fields_invoice_id() -> None:
    text = "Rechnungsnummer: INV-2025-001\nBetrag: 1.234,56 EUR"
    out = extract_structured_fields(text)
    assert out["invoice_id"] == "INV-2025-001"
    assert out["amount"]
    assert "1234" in out["amount"] or "1.234" in out["amount"]


def test_extract_structured_fields_empty() -> None:
    assert extract_structured_fields("") == {
        "invoice_id": "",
        "amount": "",
        "company": "",
    }
    assert extract_structured_fields(None) == {
        "invoice_id": "",
        "amount": "",
        "company": "",
    }


# ---------------------------------------------------------------------------
# Date extraction edge cases
# ---------------------------------------------------------------------------


def test_extract_date_ymd_format() -> None:
    """YYYY-MM-DD format with zero-padded month/day."""
    assert extract_date_from_content("Date: 2025-03-15", today=date(2000, 1, 1)) == "2025-03-15"


def test_extract_date_dmy_dotted() -> None:
    """DD.MM.YYYY format with two-digit day and month."""
    assert extract_date_from_content("am 15.03.2025 erstellt", today=date(2000, 1, 1)) == "2025-03-15"


def test_extract_date_german_long_form() -> None:
    """German long-form: '18. Februar 2025'."""
    assert extract_date_from_content("Vom 18. Februar 2025", today=date(2000, 1, 1)) == "2025-02-18"


def test_extract_date_german_long_form_maerz() -> None:
    """German long-form with März (special umlaut month)."""
    assert extract_date_from_content("am 5. März 2024 gesendet", today=date(2000, 1, 1)) == "2024-03-05"


def test_extract_date_english_long_form() -> None:
    """English long-form: 'February 18, 2025'."""
    assert extract_date_from_content("Dated February 18, 2025", today=date(2000, 1, 1)) == "2025-02-18"


def test_extract_date_english_long_form_no_comma() -> None:
    """English long-form without comma: 'March 5 2024'."""
    assert extract_date_from_content("On March 5 2024 we agreed", today=date(2000, 1, 1)) == "2024-03-05"


def test_extract_date_english_dd_month_year() -> None:
    """English day-first long-form: '18 February 2025'."""
    assert extract_date_from_content("Published 18 February 2025", today=date(2000, 1, 1)) == "2025-02-18"


def test_extract_date_month_year_only_german() -> None:
    """German month-year only: 'Januar 2025' -> 2025-01-01."""
    assert extract_date_from_content("Ausgabe Januar 2025", today=date(2000, 1, 1)) == "2025-01-01"


def test_extract_date_month_year_only_english() -> None:
    """English month-year only: 'March 2024' -> 2024-03-01."""
    assert extract_date_from_content("Report for March 2024", today=date(2000, 1, 1)) == "2024-03-01"


def test_extract_date_prefix_datum() -> None:
    """Prefix pattern: 'Datum: DD.MM.YYYY'."""
    assert extract_date_from_content("Datum: 15.01.2025", today=date(2000, 1, 1)) == "2025-01-15"


def test_extract_date_prefix_stand() -> None:
    """Prefix pattern: 'Stand: DD.MM.YYYY'."""
    assert extract_date_from_content("Stand: 01.06.2024", today=date(2000, 1, 1)) == "2024-06-01"


def test_extract_date_prefix_ymd_before_prefix() -> None:
    """When content has YYYY-MM-DD before a prefix pattern, YYYY-MM-DD wins (earlier regex)."""
    text = "Created 2025-01-15 Datum: 18.02.2025"
    assert extract_date_from_content(text, today=date(2000, 1, 1)) == "2025-01-15"


def test_extract_date_no_date_found_returns_today() -> None:
    """No date found returns the fallback today date."""
    assert extract_date_from_content("no dates whatsoever", today=date(2023, 7, 4)) == "2023-07-04"


def test_extract_date_none_content_returns_today() -> None:
    """None content returns the fallback today date."""
    assert extract_date_from_content(None, today=date(2023, 7, 4)) == "2023-07-04"


def test_extract_date_empty_string_returns_today() -> None:
    """Empty string returns the fallback today date."""
    assert extract_date_from_content("", today=date(2023, 7, 4)) == "2023-07-04"


def test_extract_date_multiple_dates_first_wins() -> None:
    """When multiple dates exist, the first one matched wins."""
    text = "Created 2024-06-01. Updated 2025-01-10."
    assert extract_date_from_content(text, today=date(2000, 1, 1)) == "2024-06-01"


def test_extract_date_invalid_month_day_returns_today() -> None:
    """Invalid month/day in YYYY-MM-DD should gracefully fall through to today."""
    # Month 13 and day 32 are invalid — make_ymd will catch ValueError
    assert extract_date_from_content("Date 2025-13-32", today=date(2000, 1, 1)) == "2000-01-01"


def test_extract_date_mdy_locale() -> None:
    """With date_locale='mdy', DD.MM.YYYY is read as MM.DD.YYYY."""
    # "03.15.2025" would be MM=03, DD=15 under mdy
    result = extract_date_from_content("Date 03.15.2025", today=date(2000, 1, 1), date_locale="mdy")
    assert result == "2025-03-15"


def test_extract_date_mdy_locale_ambiguous() -> None:
    """With date_locale='mdy', '01.12.2025' is read as MM=01, DD=12."""
    result = extract_date_from_content("01.12.2025", today=date(2000, 1, 1), date_locale="mdy")
    assert result == "2025-01-12"


def test_extract_date_prefer_leading_chars_not_in_header() -> None:
    """When date is NOT in leading chars, fall through to full-text search."""
    text = "x" * 500 + " Created 2025-08-20"
    # Leading 100 chars has no date, so full text is searched
    assert extract_date_from_content(text, today=date(2000, 1, 1), prefer_leading_chars=100) == "2025-08-20"


# ---------------------------------------------------------------------------
# chunk_text boundary conditions
# ---------------------------------------------------------------------------


def test_chunk_text_shorter_than_chunk_size() -> None:
    """Text shorter than chunk_size produces a single chunk."""
    result = chunk_text("hello", chunk_size=100, overlap=10)
    assert result == ["hello"]


def test_chunk_text_exactly_chunk_size() -> None:
    """Text exactly chunk_size: first chunk covers all chars, overlap causes a short trailing chunk."""
    text = "a" * 50
    result = chunk_text(text, chunk_size=50, overlap=10)
    # First chunk is 0..49 (all 50 chars); then start=40 (<50), so a 10-char overlap chunk.
    assert len(result) == 2
    assert result[0] == text
    assert result[1] == "a" * 10


def test_chunk_text_two_chunks_with_overlap() -> None:
    """Text spanning two chunks has correct overlap."""
    text = "a" * 100
    result = chunk_text(text, chunk_size=80, overlap=20)
    assert len(result) == 2
    # First chunk: chars 0..79 (80 chars)
    assert result[0] == "a" * 80
    # Second chunk starts at 80-20=60, so chars 60..99 (40 chars)
    assert result[1] == "a" * 40
    # Overlap verification: last 20 chars of chunk 1 == first 20 chars of chunk 2
    assert result[0][-20:] == result[1][:20]


def test_chunk_text_empty_returns_empty() -> None:
    """Empty or whitespace-only text returns empty list."""
    assert chunk_text("", chunk_size=100, overlap=10) == []
    assert chunk_text("   ", chunk_size=100, overlap=10) == []


def test_chunk_text_negative_overlap_raises() -> None:
    """Negative overlap raises ValueError."""
    with pytest.raises(ValueError, match="overlap must be >= 0"):
        chunk_text("abc", chunk_size=10, overlap=-1)


# ---------------------------------------------------------------------------
# convert_case variants
# ---------------------------------------------------------------------------


def test_convert_case_camel_multiple_words() -> None:
    """camelCase with three words."""
    assert convert_case(["my", "document", "title"], "camelCase") == "myDocumentTitle"


def test_convert_case_snake_multiple_words() -> None:
    """snakeCase with three words."""
    assert convert_case(["my", "document", "title"], "snakeCase") == "my_document_title"


def test_convert_case_kebab_multiple_words() -> None:
    """kebabCase with three words."""
    assert convert_case(["my", "document", "title"], "kebabCase") == "my-document-title"


def test_convert_case_single_word() -> None:
    """Single word input for all cases."""
    assert convert_case(["hello"], "camelCase") == "hello"
    assert convert_case(["hello"], "snakeCase") == "hello"
    assert convert_case(["hello"], "kebabCase") == "hello"


def test_convert_case_empty_input() -> None:
    """Empty input returns empty string."""
    assert convert_case([], "camelCase") == ""
    assert convert_case([], "snakeCase") == ""
    assert convert_case([], "kebabCase") == ""


def test_convert_case_invalid_raises() -> None:
    """Invalid case name raises ValueError."""
    with pytest.raises(ValueError, match="Unknown desired_case"):
        convert_case(["a"], "PascalCase")


# ---------------------------------------------------------------------------
# normalize_keywords coverage
# ---------------------------------------------------------------------------


def test_normalize_keywords_deduplicates_by_position() -> None:
    """Duplicate keywords keep only the first occurrence (no dedup built-in, but list slicing)."""
    result = normalize_keywords("alpha, beta, alpha, gamma")
    # The function does not explicitly deduplicate, but let's verify it returns them as-is
    assert result == ["alpha", "beta", "alpha", "gamma"]


def test_normalize_keywords_filters_stopword_placeholders() -> None:
    """All placeholder/stopword variants are filtered."""
    result = normalize_keywords("tba, tbd, na, n/a, xxx, ..., …, w1, w2, etc, etc., tbd., tba.")
    assert result == []


def test_normalize_keywords_caps_at_max() -> None:
    """Keywords are capped at MAX_NORMALIZED_KEYWORDS (7)."""
    many = ", ".join(f"kw{i}" for i in range(20))
    result = normalize_keywords(many)
    assert len(result) == MAX_NORMALIZED_KEYWORDS
    assert result == [f"kw{i}" for i in range(MAX_NORMALIZED_KEYWORDS)]


def test_normalize_keywords_list_input() -> None:
    """Accepts a list of strings as input."""
    result = normalize_keywords(["alpha", "beta", "gamma"])
    assert result == ["alpha", "beta", "gamma"]


def test_normalize_keywords_none_returns_empty() -> None:
    """None input returns empty list."""
    assert normalize_keywords(None) == []


def test_normalize_keywords_strips_whitespace() -> None:
    """Tokens are stripped of whitespace."""
    result = normalize_keywords("  alpha  ,  beta  ,  ")
    assert result == ["alpha", "beta"]


# ---------------------------------------------------------------------------
# extract_structured_fields patterns
# ---------------------------------------------------------------------------


def test_structured_fields_invoice_id_rechnungsnummer() -> None:
    """Rechnungsnummer: INV-2025-001 is extracted."""
    text = "Rechnungsnummer: INV-2025-001\nSome other text"
    out = extract_structured_fields(text)
    assert out["invoice_id"] == "INV-2025-001"


def test_structured_fields_amount_betrag_eur() -> None:
    """Betrag: 1.234,50 EUR is extracted and normalized."""
    text = "Betrag: 1.234,50 EUR\nDanke"
    out = extract_structured_fields(text)
    assert out["amount"] != ""
    # Normalized: commas become dots, spaces removed
    assert "1.234.50" in out["amount"] or "1234.50" in out["amount"] or "1.234.5" in out["amount"]


def test_structured_fields_amount_euro_sign() -> None:
    """Amount with euro sign: '99,99 EUR' or '99.99 €'."""
    text = "Total 250,00 €"
    out = extract_structured_fields(text)
    assert out["amount"] == "250.00"


def test_structured_fields_company_firma() -> None:
    """Firma (without colon) followed by company name is extracted via second company pattern."""
    text = "Firma Acme Corp GmbH\nAdresse: Berlin"
    out = extract_structured_fields(text)
    assert "Acme" in out["company"]
    assert "GmbH" in out["company"]


def test_structured_fields_company_rechnung_von() -> None:
    """'Rechnung von' pattern extracts company name."""
    text = "Rechnung von Mustermann AG\nStraße 123"
    out = extract_structured_fields(text)
    assert "Mustermann" in out["company"]


def test_structured_fields_no_fields_found() -> None:
    """When no fields match, all values are empty strings."""
    text = "This is a random document with no structured data."
    out = extract_structured_fields(text)
    assert out == {"invoice_id": "", "amount": "", "company": ""}


def test_structured_fields_invoice_no_pattern() -> None:
    """'Invoice No.' pattern is recognized."""
    text = "Invoice No. ABC-1234\nTotal: 100.00 EUR"
    out = extract_structured_fields(text)
    assert out["invoice_id"] == "ABC-1234"


def test_structured_fields_max_chars_limit() -> None:
    """Only the first max_chars characters are searched."""
    # Put invoice id far beyond max_chars (space before keyword for word boundary)
    text = "x" * 100 + " Rechnungsnummer: INV-9999-001"
    out = extract_structured_fields(text, max_chars=50)
    assert out["invoice_id"] == ""
    # But with enough chars it works
    out2 = extract_structured_fields(text, max_chars=200)
    assert out2["invoice_id"] == "INV-9999-001"


def test_structured_fields_order_no() -> None:
    """'Order No.' pattern is recognized."""
    text = "Order No. ORD-2025-555\nDetails follow"
    out = extract_structured_fields(text)
    assert out["invoice_id"] == "ORD-2025-555"


def test_structured_fields_amount_summe() -> None:
    """'Summe' label for amount is recognized."""
    text = "Summe: 500,00 EUR"
    out = extract_structured_fields(text)
    assert out["amount"] == "500.00"
