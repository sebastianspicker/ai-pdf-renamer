"""Tests for processing rules (load, force_category_by_pattern, skip_files_by_pattern)."""

from __future__ import annotations

from pathlib import Path

import pytest

from ai_pdf_renamer.rules import (
    ProcessingRules,
    force_category_for_basename,
    load_processing_rules,
    should_skip_file_by_rules,
)


def test_load_processing_rules_missing_returns_none(tmp_path: Path) -> None:
    assert load_processing_rules(tmp_path / "nonexistent.json") is None


def test_load_processing_rules_invalid_json_returns_none(tmp_path: Path) -> None:
    p = tmp_path / "rules.json"
    p.write_text("not json")
    assert load_processing_rules(p) is None


def test_load_processing_rules_invalid_json_raises_when_requested(tmp_path: Path) -> None:
    p = tmp_path / "rules.json"
    p.write_text("not json")
    with pytest.raises(ValueError, match="Could not load processing rules"):
        load_processing_rules(p, raise_on_error=True)


def test_load_processing_rules_valid(tmp_path: Path) -> None:
    p = tmp_path / "rules.json"
    p.write_text(
        """{
        "skip_llm_if_heuristic_category": ["invoice", "receipt"],
        "force_category_by_pattern": [{"pattern": "draft-*.pdf", "category": "draft"}],
        "skip_files_by_pattern": ["*.tmp.pdf"]
    }"""
    )
    rules = load_processing_rules(p)
    assert rules is not None
    assert rules.skip_llm_if_heuristic_category == ["invoice", "receipt"]
    assert len(rules.force_category_by_pattern) == 1
    assert rules.force_category_by_pattern[0]["pattern"] == "draft-*.pdf"
    assert rules.force_category_by_pattern[0]["category"] == "draft"
    assert rules.skip_files_by_pattern == ["*.tmp.pdf"]
    assert rules.allowed_categories == []


def test_load_processing_rules_with_allowed_categories(tmp_path: Path) -> None:
    p = tmp_path / "rules.json"
    p.write_text(
        """{
        "skip_llm_if_heuristic_category": [],
        "force_category_by_pattern": [],
        "skip_files_by_pattern": [],
        "allowed_categories": ["Invoice", "Contract", "Receipt"]
    }"""
    )
    rules = load_processing_rules(p)
    assert rules is not None
    assert rules.allowed_categories == ["Invoice", "Contract", "Receipt"]


def test_force_category_for_basename_none_rules() -> None:
    assert force_category_for_basename(None, "draft-1.pdf") is None


def test_force_category_for_basename_match() -> None:
    rules = ProcessingRules(
        skip_llm_if_heuristic_category=[],
        force_category_by_pattern=[{"pattern": "draft-*.pdf", "category": "draft"}],
        skip_files_by_pattern=[],
        allowed_categories=[],
    )
    assert force_category_for_basename(rules, "draft-1.pdf") == "draft"
    assert force_category_for_basename(rules, "other.pdf") is None


def test_should_skip_file_by_rules_none_rules() -> None:
    assert should_skip_file_by_rules(None, "x.pdf") is False


def test_should_skip_file_by_rules_match() -> None:
    rules = ProcessingRules(
        skip_llm_if_heuristic_category=[],
        force_category_by_pattern=[],
        skip_files_by_pattern=["*.tmp.pdf", "*-old.pdf"],
        allowed_categories=[],
    )
    assert should_skip_file_by_rules(rules, "file.tmp.pdf") is True
    assert should_skip_file_by_rules(rules, "doc-old.pdf") is True
    assert should_skip_file_by_rules(rules, "normal.pdf") is False
