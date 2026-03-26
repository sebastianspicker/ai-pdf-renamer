"""Targeted coverage tests for config_resolver.py, heuristics.py, and pdf_extract.py.

Focuses on uncovered edge cases: helper functions, preset defaults, rule loading
errors, embedding fallback, vision format fallbacks, and shrink loop iterations.
"""

from __future__ import annotations

import base64
import json
import re
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ai_pdf_renamer.config_resolver import (
    _optional_float,
    _positive_int_or_none,
    build_config,
)
from ai_pdf_renamer.heuristics import (
    HeuristicRule,
    _combine_resolve_conflict,
    _embedding_conflict_pick,
    _score_text,
    load_heuristic_rules,
)
from ai_pdf_renamer.pdf_extract import (
    _ocr_language_code,
    _parse_pdf_date,
    _shrink_to_token_limit,
    pdf_first_page_to_image_base64,
)


class TestOptionalFloat:
    def test_optional_float_invalid(self) -> None:
        """_optional_float('abc') returns None for non-numeric strings."""
        assert _optional_float("abc") is None

    def test_optional_float_none(self) -> None:
        """_optional_float(None) returns None."""
        assert _optional_float(None) is None

    def test_optional_float_empty_string(self) -> None:
        """_optional_float('') returns None."""
        assert _optional_float("") is None

    def test_optional_float_valid(self) -> None:
        """_optional_float('3.14') returns 3.14."""
        assert _optional_float("3.14") == pytest.approx(3.14)


class TestPositiveIntOrNone:
    def test_positive_int_or_none_negative(self) -> None:
        """_positive_int_or_none(-5) returns None for negative values."""
        assert _positive_int_or_none(-5) is None

    def test_positive_int_or_none_zero(self) -> None:
        """_positive_int_or_none(0) returns None for zero."""
        assert _positive_int_or_none(0) is None

    def test_positive_int_or_none_positive(self) -> None:
        """_positive_int_or_none(10) returns 10."""
        assert _positive_int_or_none(10) == 10


class TestBuildConfigPresets:
    def test_build_config_high_confidence_preset(self) -> None:
        """preset='high-confidence-heuristic' sets skip_llm thresholds."""
        cfg = build_config({"preset": "high-confidence-heuristic"}, env={})
        assert cfg.skip_llm_category_if_heuristic_score_ge == pytest.approx(0.5)
        assert cfg.skip_llm_category_if_heuristic_gap_ge == pytest.approx(0.3)

    def test_build_config_high_confidence_preset_no_overwrite(self) -> None:
        """preset='high-confidence-heuristic' does NOT overwrite user-set values."""
        cfg = build_config(
            {
                "preset": "high-confidence-heuristic",
                "skip_llm_category_if_heuristic_score_ge": 0.9,
                "skip_llm_category_if_heuristic_gap_ge": 0.8,
            },
            env={},
        )
        assert cfg.skip_llm_category_if_heuristic_score_ge == pytest.approx(0.9)
        assert cfg.skip_llm_category_if_heuristic_gap_ge == pytest.approx(0.8)

    def test_build_config_no_heuristic_override(self) -> None:
        """no_heuristic_override=True clears override scores to None."""
        cfg = build_config({"no_heuristic_override": True}, env={})
        assert cfg.heuristic_override_min_score is None
        assert cfg.heuristic_override_min_gap is None

    def test_build_config_prefer_heuristic(self) -> None:
        """prefer_heuristic=True overrides prefer_llm_category to False."""
        cfg = build_config({"prefer_heuristic": True}, env={})
        assert cfg.prefer_llm_category is False

    def test_build_config_prefer_heuristic_false_default(self) -> None:
        """Without prefer_heuristic, prefer_llm_category defaults to True."""
        cfg = build_config({}, env={})
        assert cfg.prefer_llm_category is True

    def test_build_config_manual_mode(self) -> None:
        """manual_mode=True forces interactive=True."""
        cfg = build_config({"manual_mode": True}, env={})
        assert cfg.manual_mode is True
        assert cfg.interactive is True

    def test_build_config_scanned_preset(self) -> None:
        """preset='scanned' enables vision fallback and simple naming mode."""
        cfg = build_config({"preset": "scanned"}, env={})
        assert cfg.use_vision_fallback is True
        assert cfg.simple_naming_mode is True

    def test_build_config_default_heuristic_override(self) -> None:
        """Without no_heuristic_override, default override scores are set."""
        cfg = build_config({}, env={})
        assert cfg.heuristic_override_min_score == pytest.approx(0.55)
        assert cfg.heuristic_override_min_gap == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# 2. heuristics.py tests
# ---------------------------------------------------------------------------


class TestLoadRulesEdgeCases:
    def test_load_rules_invalid_regex(self, tmp_path: Path) -> None:
        """Rule with invalid regex pattern is skipped and a warning is logged."""
        data = {
            "patterns": [
                {"regex": "[invalid(", "category": "bad", "score": 1.0},
                {"regex": "good", "category": "good", "score": 2.0},
            ]
        }
        rules_file = tmp_path / "rules.json"
        rules_file.write_text(json.dumps(data))
        rules = load_heuristic_rules(rules_file)
        assert len(rules) == 1
        assert rules[0].category == "good"

    def test_load_rules_missing_regex_key(self, tmp_path: Path) -> None:
        """Rule entry without 'regex' key is skipped."""
        data = {
            "patterns": [
                {"category": "orphan", "score": 1.0},
                {"regex": "present", "category": "found", "score": 3.0},
            ]
        }
        rules_file = tmp_path / "rules.json"
        rules_file.write_text(json.dumps(data))
        rules = load_heuristic_rules(rules_file)
        assert len(rules) == 1
        assert rules[0].category == "found"

    def test_load_rules_non_numeric_score(self, tmp_path: Path) -> None:
        """Non-numeric score string defaults to 0.0."""
        data = {
            "patterns": [
                {"regex": "test", "category": "cat", "score": "not_a_number"},
            ]
        }
        rules_file = tmp_path / "rules.json"
        rules_file.write_text(json.dumps(data))
        rules = load_heuristic_rules(rules_file)
        assert len(rules) == 1
        assert rules[0].score == pytest.approx(0.0)

    def test_load_rules_none_score(self, tmp_path: Path) -> None:
        """None score defaults to 0.0."""
        data = {
            "patterns": [
                {"regex": "test", "category": "cat", "score": None},
            ]
        }
        rules_file = tmp_path / "rules.json"
        rules_file.write_text(json.dumps(data))
        rules = load_heuristic_rules(rules_file)
        assert len(rules) == 1
        assert rules[0].score == pytest.approx(0.0)


class TestMaxScorePerCategory:
    def test_max_score_per_category_caps(self) -> None:
        """max_score_per_category=5.0 caps scores that would exceed it."""
        rules = [
            HeuristicRule(pattern=re.compile("invoice"), category="invoice", score=4.0),
            HeuristicRule(pattern=re.compile("INVOICE"), category="invoice", score=4.0),
        ]
        # Both rules match "invoice INVOICE" -> sum = 8.0, but capped at 5.0
        scores = _score_text(
            "invoice INVOICE",
            rules,
            None,
            max_score_per_category=5.0,
        )
        assert scores["invoice"] == pytest.approx(5.0)

    def test_max_score_per_category_no_cap(self) -> None:
        """Without max_score_per_category, score sums freely."""
        rules = [
            HeuristicRule(pattern=re.compile("invoice"), category="invoice", score=4.0),
            HeuristicRule(pattern=re.compile("INVOICE"), category="invoice", score=4.0),
        ]
        scores = _score_text(
            "invoice INVOICE",
            rules,
            None,
            max_score_per_category=None,
        )
        assert scores["invoice"] == pytest.approx(8.0)


class TestEmbeddingConflict:
    def test_embedding_conflict_no_sentence_transformers(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When sentence_transformers import fails, _embedding_conflict_pick returns None."""
        import ai_pdf_renamer.heuristics as hmod

        # Reset the global model cache to ensure fresh import attempt
        monkeypatch.setattr(hmod, "_embedding_model", None)

        import builtins

        real_import = builtins.__import__

        def fake_import(name: str, *args: object, **kwargs: object) -> object:
            if name == "sentence_transformers":
                raise ImportError("No module named 'sentence_transformers'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        result = _embedding_conflict_pick("some context text", "invoice", "receipt")
        assert result is None


class TestCombineResolveConflictKeywordOverlap:
    def test_keyword_overlap_llm_wins(self) -> None:
        """use_keyword_overlap=True with LLM category tokens overlapping more with context."""
        # context has "insurance" which overlaps with LLM category "insurance"
        result = _combine_resolve_conflict(
            "insurance",
            "invoice",
            prefer_llm=False,
            context_for_overlap="this is about insurance policy coverage",
            use_embeddings_for_conflict=False,
            use_keyword_overlap=True,
            heuristic_score=0.1,
            heuristic_score_weight=0.0,
        )
        assert result == "insurance"

    def test_keyword_overlap_heuristic_wins(self) -> None:
        """use_keyword_overlap=True with heuristic category tokens overlapping more."""
        result = _combine_resolve_conflict(
            "insurance",
            "invoice",
            prefer_llm=False,
            context_for_overlap="this is an invoice for payment services",
            use_embeddings_for_conflict=False,
            use_keyword_overlap=True,
            heuristic_score=0.1,
            heuristic_score_weight=0.0,
        )
        assert result == "invoice"

    def test_keyword_overlap_tie_returns_heuristic(self) -> None:
        """On keyword overlap tie, heuristic wins."""
        result = _combine_resolve_conflict(
            "alpha",
            "beta",
            prefer_llm=True,
            context_for_overlap="neither alpha nor beta appears here gamma delta",
            use_embeddings_for_conflict=False,
            use_keyword_overlap=True,
            heuristic_score=None,
            heuristic_score_weight=0.0,
        )
        # Both have 0 overlap -> tie -> heuristic wins
        assert result == "beta"

    def test_keyword_overlap_with_score_weight_bonus(self) -> None:
        """Heuristic score weight bonus can tip the overlap in favor of heuristic."""
        # Both categories have 0 overlap with context, but heuristic_score_weight
        # gives heuristic a bonus, so heuristic wins even with equal overlap.
        result = _combine_resolve_conflict(
            "alpha",
            "beta",
            prefer_llm=True,
            context_for_overlap="unrelated context",
            use_embeddings_for_conflict=False,
            use_keyword_overlap=True,
            heuristic_score=5.0,
            heuristic_score_weight=0.5,
        )
        # overlap_llm=0, overlap_heur_weighted=0+2.5=2.5 -> heuristic wins
        assert result == "beta"


# ---------------------------------------------------------------------------
# 3. pdf_extract.py tests
# ---------------------------------------------------------------------------


class TestShrinkToTokenLimit:
    def test_shrink_to_token_limit_multi_iteration(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Text needs >1 shrink iteration when density estimate is off."""
        import ai_pdf_renamer.pdf_extract as pmod

        text = "word " * 2000  # 10000 chars

        # First call: initial count (over limit) -> 500 tokens
        # After density jump, text is shortened. We want the fine-tuning loop
        # to iterate multiple times. We fake _token_count so the initial jump
        # leaves the text still over limit, requiring iterations.
        call_count = 0

        def fake_token_count(t: str) -> int:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # Initial check: report way over limit
                return 500
            if call_count == 2:
                # After density jump: still over limit to force loop iterations
                return 120
            if call_count == 3:
                # Second fine-tuning iteration: still slightly over
                return 105
            # Third+ iteration: under limit
            return 95

        monkeypatch.setattr(pmod, "_token_count", fake_token_count)
        result = _shrink_to_token_limit(text, max_tokens=100)
        assert len(result) < len(text)
        # At least 3 calls to _token_count (initial + density jump check + loop iterations)
        assert call_count >= 3


class TestVisionFormatFallbacks:
    def test_vision_tobytes_jpeg_fails_png_fallback(self) -> None:
        """When tobytes(jpeg) raises TypeError, falls back to PNG."""
        mock_pix = MagicMock()
        mock_pix.tobytes = MagicMock(side_effect=[TypeError("jpeg not supported"), b"PNG_DATA"])

        mock_page = MagicMock()
        mock_page.get_pixmap.return_value = mock_pix

        mock_doc = MagicMock()
        mock_doc.is_encrypted = False
        mock_doc.page_count = 1
        mock_doc.load_page.return_value = mock_page
        mock_doc.close = MagicMock()

        with patch("ai_pdf_renamer.pdf_extract.fitz", create=True) as mock_fitz:
            mock_fitz.open.return_value = mock_doc
            with patch.dict("sys.modules", {"fitz": mock_fitz}):
                result = pdf_first_page_to_image_base64("/fake/path.pdf")

        assert result is not None

        assert base64.b64decode(result) == b"PNG_DATA"

    def test_vision_getimagedata_fallback(self) -> None:
        """When tobytes is missing, getImageData is used."""
        mock_pix = MagicMock(spec=[])  # no attributes by default
        mock_pix.getImageData = MagicMock(return_value=b"JPEG_VIA_OLD_API")
        # Remove tobytes and getPNGData
        assert not hasattr(mock_pix, "tobytes")

        mock_page = MagicMock()
        mock_page.get_pixmap.return_value = mock_pix

        mock_doc = MagicMock()
        mock_doc.is_encrypted = False
        mock_doc.page_count = 1
        mock_doc.load_page.return_value = mock_page
        mock_doc.close = MagicMock()

        with patch("ai_pdf_renamer.pdf_extract.fitz", create=True) as mock_fitz:
            mock_fitz.open.return_value = mock_doc
            with patch.dict("sys.modules", {"fitz": mock_fitz}):
                result = pdf_first_page_to_image_base64("/fake/path.pdf")

        assert result is not None

        assert base64.b64decode(result) == b"JPEG_VIA_OLD_API"


class TestOcrLanguageCode:
    def test_ocr_language_code_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """AI_PDF_RENAMER_OCR_LANG env var overrides default language mapping."""
        monkeypatch.setenv("AI_PDF_RENAMER_OCR_LANG", "fra")
        assert _ocr_language_code("de") == "fra"
        assert _ocr_language_code("en") == "fra"

    def test_ocr_language_code_default_de(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Default German maps to 'deu'."""
        monkeypatch.delenv("AI_PDF_RENAMER_OCR_LANG", raising=False)
        assert _ocr_language_code("de") == "deu"

    def test_ocr_language_code_en(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """English maps to 'eng'."""
        monkeypatch.delenv("AI_PDF_RENAMER_OCR_LANG", raising=False)
        assert _ocr_language_code("en") == "eng"


class TestParsePdfDate:
    def test_parse_pdf_date_boundary_non_leap_year(self) -> None:
        """'D:20250229' (Feb 29, non-leap year 2025) returns None."""
        result = _parse_pdf_date("D:20250229")
        assert result is None

    def test_parse_pdf_date_valid(self) -> None:
        """Valid PDF date parses correctly."""
        result = _parse_pdf_date("D:20240229")
        assert result == date(2024, 2, 29)

    def test_parse_pdf_date_none_input(self) -> None:
        """None input returns None."""
        assert _parse_pdf_date(None) is None

    def test_parse_pdf_date_no_prefix(self) -> None:
        """String without D: prefix returns None."""
        assert _parse_pdf_date("20250101") is None

    def test_parse_pdf_date_month_13(self) -> None:
        """Invalid month 13 returns None."""
        assert _parse_pdf_date("D:20251301") is None
