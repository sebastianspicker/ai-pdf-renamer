"""Tests for LLM prompt builders (e.g. build_vision_filename_prompt)."""

from __future__ import annotations

from ai_pdf_renamer.llm_prompts import build_vision_filename_prompt


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
