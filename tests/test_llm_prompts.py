"""Tests for LLM prompt builders (e.g. build_vision_filename_prompt)."""

from __future__ import annotations

from ai_pdf_renamer.llm_prompts import PROMPT_STRINGS, _language_code, build_analysis_prompt, build_vision_filename_prompt


def test_prompt_strings_has_german_and_english() -> None:
    assert "de" in PROMPT_STRINGS
    assert "en" in PROMPT_STRINGS
    assert "analysis_intro" in PROMPT_STRINGS["de"]
    assert "analysis_intro" in PROMPT_STRINGS["en"]


def test_build_analysis_prompt_uses_language_table() -> None:
    prompt = build_analysis_prompt("en", "A test document")
    assert PROMPT_STRINGS["en"]["analysis_intro"] in prompt
    assert PROMPT_STRINGS["en"]["analysis_rules_heading"] in prompt


def test_build_vision_filename_prompt_de() -> None:
    prompt = build_vision_filename_prompt("de")
    assert "Deutsch" in prompt
    assert "3" in prompt and "6" in prompt
    assert "RECHNUNG_AMAZON_MAX_2023-11-15" in prompt
    assert "NUR dem Dateinamen" in prompt or "nur dem Dateinamen" in prompt
    assert "sonst nichts" in prompt


def test_build_vision_filename_prompt_en() -> None:
    prompt = build_vision_filename_prompt("en")
    assert "English" in prompt
    assert "3" in prompt and "6" in prompt
    assert "INVOICE_AMAZON_JOHN_2023-11-15" in prompt
    assert "ONLY the filename" in prompt or "only the filename" in prompt
    assert "nothing else" in prompt


def test_language_code_accepts_common_german_variants() -> None:
    assert _language_code("de") == "de"
    assert _language_code("DE") == "de"
    assert _language_code(" de-DE ") == "de"
    assert _language_code("de_DE") == "de"
    assert _language_code("en") == "en"
