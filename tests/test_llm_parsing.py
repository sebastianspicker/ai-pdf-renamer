from __future__ import annotations

from ai_pdf_renamer.llm import parse_json_field, truncate_for_llm


def test_parse_json_field_string() -> None:
    assert parse_json_field('{"summary":"Hello"}', key="summary") == "Hello"


def test_parse_json_field_list() -> None:
    assert parse_json_field('{"keywords":["A","B"]}', key="keywords") == ["A", "B"]


def test_parse_json_field_invalid_returns_none() -> None:
    assert parse_json_field("not json", key="summary") is None


def test_parse_json_field_sanitizes_quotes() -> None:
    # Unescaped quotes inside value should be fixed by sanitizer.
    raw = '{"summary":"He said "hello" today"}'
    assert parse_json_field(raw, key="summary") == 'He said "hello" today'


def test_parse_json_field_lenient_extracts_without_brace() -> None:
    # Lenient: response that does not start with "{" but contains "key":"value".
    assert (
        parse_json_field(
            'Here is the result. "summary":"A short summary."',
            key="summary",
            lenient=True,
        )
        == "A short summary."
    )
    assert parse_json_field("No JSON here", key="summary", lenient=True) is None
    assert parse_json_field('"category":"invoice"', key="category", lenient=True) == "invoice"


def test_truncate_for_llm() -> None:
    assert truncate_for_llm("short", None) == "short"
    assert truncate_for_llm("short", 100) == "short"
    text_200 = "x" * 200
    out = truncate_for_llm(text_200, 50)
    assert len(out) == 50
    assert out.endswith("\n[...]")
    assert out == "x" * (50 - len("\n[...]")) + "\n[...]"
