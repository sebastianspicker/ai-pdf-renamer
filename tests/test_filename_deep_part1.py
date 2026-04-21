# ruff: noqa: F401

"""Deep coverage tests for filename.py — targeting uncovered branches."""

from __future__ import annotations

import argparse
import json
import re
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ai_pdf_renamer.config import RenamerConfig
from ai_pdf_renamer.heuristics import (
    CategoryCombineParams,
    HeuristicRule,
    HeuristicScorer,
    _combine_resolve_conflict,
    combine_categories,
)
from ai_pdf_renamer.text_utils import Stopwords


def _make_scorer(
    categories: list[tuple[str, str, float]] | None = None,
) -> HeuristicScorer:
    """Build a HeuristicScorer from (regex, category, score) triples."""
    if categories is None:
        categories = [
            (r"invoice", "invoice", 10.0),
            (r"contract", "contract", 5.0),
        ]
    rules = [
        HeuristicRule(pattern=re.compile(regex, re.IGNORECASE), category=cat, score=sc) for regex, cat, sc in categories
    ]
    return HeuristicScorer(rules=rules)


def _make_llm_client() -> MagicMock:
    """Return a MagicMock that satisfies LLMClient protocol."""
    client = MagicMock()
    client.model = "test-model"
    client.base_url = "http://localhost:8080"
    client.complete.return_value = '{"summary": "test"}'
    client.complete_vision.return_value = '{"summary": "test"}'
    return client


def _empty_stopwords() -> Stopwords:
    return Stopwords(words=set())


REFERENCE_TODAY = date(2026, 4, 8)


class TestHeuristicTextForCategory:
    """Tests for _heuristic_text_for_category."""

    def test_heuristic_text_short_content(self) -> None:
        """Short content (below all thresholds) is returned as-is."""
        from ai_pdf_renamer.filename import _heuristic_text_for_category

        content = "Short PDF content"
        config = RenamerConfig(
            heuristic_leading_chars=0,
            heuristic_long_doc_chars_threshold=40_000,
            heuristic_long_doc_leading_chars=12_000,
        )
        result = _heuristic_text_for_category(content, config)
        assert result == content

    def test_heuristic_text_long_content_truncated(self) -> None:
        """Content >= heuristic_long_doc_chars_threshold is truncated to heuristic_long_doc_leading_chars."""
        from ai_pdf_renamer.filename import _heuristic_text_for_category

        threshold = 100
        leading = 30
        content = "A" * 150  # above threshold
        config = RenamerConfig(
            heuristic_leading_chars=0,
            heuristic_long_doc_chars_threshold=threshold,
            heuristic_long_doc_leading_chars=leading,
        )
        result = _heuristic_text_for_category(content, config)
        assert len(result) == leading
        assert result == "A" * leading

    def test_heuristic_text_with_leading_chars(self) -> None:
        """When heuristic_leading_chars > 0, only first N chars are used (takes priority)."""
        from ai_pdf_renamer.filename import _heuristic_text_for_category

        content = "ABCDEFGHIJ" * 100  # 1000 chars
        config = RenamerConfig(
            heuristic_leading_chars=20,
            heuristic_long_doc_chars_threshold=100,
            heuristic_long_doc_leading_chars=50,
        )
        result = _heuristic_text_for_category(content, config)
        assert len(result) == 20
        assert result == content[:20]


class TestGetDateStr:
    """Tests for _get_date_str."""

    def test_get_date_str_prefers_pdf_metadata_after_invalid_text_date(self) -> None:
        """Invalid text dates fall back to PDF metadata when enabled."""
        from ai_pdf_renamer.filename import _get_date_str

        config = RenamerConfig(use_pdf_metadata_for_date=True)
        result = _get_date_str(
            "Printed 2099-12-31",
            config,
            today=date(2026, 4, 8),
            pdf_metadata={"creation_date": "2024-11-02", "mod_date": "2024-11-05"},
        )
        assert result == "20241102"


class TestResolveCategoryWithLlm:
    """Tests for _resolve_category_with_llm."""

    def test_use_llm_false_returns_heuristic(self) -> None:
        """When config.use_llm=False, category_source is 'heuristic' and LLM is not called."""
        from ai_pdf_renamer.filename import _resolve_category_with_llm

        scorer = _make_scorer()
        client = _make_llm_client()
        config = RenamerConfig(use_llm=False)

        category, _cat_display, source = _resolve_category_with_llm(
            heuristic_text="invoice text",
            cat_heur="invoice",
            heuristic_score=5.0,
            heuristic_gap=3.0,
            config=config,
            heuristic_scorer=scorer,
            llm_client=client,
            summary="test summary",
            keywords=["test"],
        )
        assert source == "heuristic"
        assert category == "invoice"
        client.complete.assert_not_called()

    def test_skip_llm_high_score(self) -> None:
        """When heuristic score and gap exceed skip thresholds, LLM is not called."""
        from ai_pdf_renamer.filename import _resolve_category_with_llm

        scorer = _make_scorer()
        client = _make_llm_client()
        config = RenamerConfig(
            use_llm=True,
            skip_llm_category_if_heuristic_score_ge=3.0,
            skip_llm_category_if_heuristic_gap_ge=1.0,
            use_single_llm_call=False,
        )

        _category, _cat_display, source = _resolve_category_with_llm(
            heuristic_text="invoice text",
            cat_heur="invoice",
            heuristic_score=5.0,
            heuristic_gap=3.0,
            config=config,
            heuristic_scorer=scorer,
            llm_client=client,
            summary="test summary",
            keywords=["test"],
        )
        assert source == "heuristic"
        client.complete.assert_not_called()

    def test_precomputed_llm_category(self) -> None:
        """When precomputed_llm_category is provided, it is used without calling LLM."""
        from ai_pdf_renamer.filename import _resolve_category_with_llm

        scorer = _make_scorer()
        client = _make_llm_client()
        config = RenamerConfig(
            use_llm=True,
            use_constrained_llm_category=False,
        )

        category, _cat_display, source = _resolve_category_with_llm(
            heuristic_text="some text",
            cat_heur="unknown",
            heuristic_score=0.0,
            heuristic_gap=0.0,
            config=config,
            heuristic_scorer=scorer,
            llm_client=client,
            summary="test summary",
            keywords=["test"],
            precomputed_llm_category="invoice",
        )
        # cat_heur is "unknown", so source should be "llm"
        assert source == "llm"
        assert "invoice" in category
        client.complete.assert_not_called()

    def test_precomputed_category_not_in_allowed(self) -> None:
        """Precomputed category not in allowed set falls back to 'unknown'."""
        from ai_pdf_renamer.filename import _resolve_category_with_llm
        from ai_pdf_renamer.rules import ProcessingRules

        scorer = _make_scorer()
        client = _make_llm_client()
        rules = ProcessingRules(
            skip_llm_if_heuristic_category=[],
            force_category_by_pattern=[],
            skip_files_by_pattern=[],
            allowed_categories=["contract", "receipt"],
        )
        config = RenamerConfig(
            use_llm=True,
            use_constrained_llm_category=False,
        )

        _category, _cat_display, source = _resolve_category_with_llm(
            heuristic_text="some text",
            cat_heur="unknown",
            heuristic_score=0.0,
            heuristic_gap=0.0,
            config=config,
            heuristic_scorer=scorer,
            llm_client=client,
            summary="test summary",
            keywords=["test"],
            rules=rules,
            precomputed_llm_category="totally_bogus_category",
        )
        # The precomputed category is not in allowed set, so _validate falls back to "unknown"
        client.complete.assert_not_called()
        # Since cat_llm became "unknown" and cat_heur is also "unknown",
        # combine_categories returns the raw LLM value when both are unknown
        assert source == "llm"

    def test_category_source_combined(self) -> None:
        """When both heuristic and LLM contribute (no skip, heuristic not unknown), source is 'combined'."""
        from ai_pdf_renamer.filename import _resolve_category_with_llm

        scorer = _make_scorer()
        client = _make_llm_client()
        config = RenamerConfig(
            use_llm=True,
            skip_llm_category_if_heuristic_score_ge=None,
            skip_llm_category_if_heuristic_gap_ge=None,
            use_constrained_llm_category=False,
        )

        _category, _cat_display, source = _resolve_category_with_llm(
            heuristic_text="invoice text",
            cat_heur="invoice",
            heuristic_score=3.0,
            heuristic_gap=1.0,
            config=config,
            heuristic_scorer=scorer,
            llm_client=client,
            summary="test summary",
            keywords=["test"],
            precomputed_llm_category="contract",
        )
        assert source == "combined"


class TestBuildFilenameStr:
    """Tests for _build_filename_str."""

    def test_build_filename_camel_case(self) -> None:
        """desired_case='camelCase' produces camelCase output."""
        from ai_pdf_renamer.filename import _build_filename_str

        config = RenamerConfig(desired_case="camelCase")
        result = _build_filename_str(
            date_str="20240101",
            category_for_filename="invoice",
            category_clean=["invoice"],
            keyword_clean=["tax"],
            summary_clean=["payment"],
            config=config,
        )
        # camelCase: first token lowercase, rest capitalized, no separators
        assert "Invoice" in result
        assert "Tax" in result
        assert "Payment" in result
        assert result.startswith("20240101")
        # Should not contain dashes or underscores (pure camelCase)
        assert "-" not in result
        assert "_" not in result

    def test_build_filename_snake_case(self) -> None:
        """desired_case='snakeCase' uses underscores."""
        from ai_pdf_renamer.filename import _build_filename_str

        config = RenamerConfig(desired_case="snakeCase")
        result = _build_filename_str(
            date_str="20240101",
            category_for_filename="invoice",
            category_clean=["invoice"],
            keyword_clean=["tax"],
            summary_clean=["payment"],
            config=config,
        )
        assert "_" in result
        assert "-" not in result
        assert "20240101" in result
        assert "invoice" in result
        assert "tax" in result
        assert "payment" in result

    def test_build_filename_kebab_case(self) -> None:
        """desired_case='kebabCase' uses dashes."""
        from ai_pdf_renamer.filename import _build_filename_str

        config = RenamerConfig(desired_case="kebabCase")
        result = _build_filename_str(
            date_str="20240101",
            category_for_filename="invoice",
            category_clean=["invoice"],
            keyword_clean=["tax"],
            summary_clean=["payment"],
            config=config,
        )
        assert "-" in result
        assert "_" not in result
        assert "20240101" in result
        assert "invoice" in result
        assert "tax" in result
        assert "payment" in result

    def test_build_filename_with_template(self) -> None:
        """When filename_template is set, template is applied to produce the filename."""
        from ai_pdf_renamer.filename import _build_filename_str

        config = RenamerConfig(
            desired_case="kebabCase",
            filename_template="{date}-{category}-{keywords}",
        )
        result = _build_filename_str(
            date_str="20240101",
            category_for_filename="invoice",
            category_clean=["invoice"],
            keyword_clean=["tax", "vat"],
            summary_clean=["payment"],
            config=config,
        )
        assert result.startswith("20240101")
        assert "invoice" in result
        assert "tax" in result


class TestGenerateFilename:
    """Tests for generate_filename top-level function."""

    def test_generate_filename_simple_naming_mode(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """simple_naming_mode=True uses get_document_filename_simple for a short name."""
        import ai_pdf_renamer.filename as filename_mod

        monkeypatch.setattr(
            filename_mod,
            "get_document_filename_simple",
            lambda *a, **k: "short-filename",
        )

        config = RenamerConfig(
            simple_naming_mode=True,
            use_structured_fields=False,
        )
        client = _make_llm_client()
        scorer = _make_scorer()

        name, metadata = filename_mod.generate_filename(
            "Invoice 2024-03-15 content",
            config=config,
            llm_client=client,
            heuristic_scorer=scorer,
            stopwords=_empty_stopwords(),
            today=REFERENCE_TODAY,
        )
        assert "short-filename" in name
        assert "20240315" in name
        assert metadata["category"] == "short-filename"

    def test_generate_filename_structured_fields(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """use_structured_fields=True extracts invoice_id/amount/company into metadata."""
        import ai_pdf_renamer.filename as filename_mod

        monkeypatch.setattr(filename_mod, "get_document_summary", lambda *a, **k: "Invoice summary")
        monkeypatch.setattr(filename_mod, "get_document_keywords", lambda *a, **k: ["invoice"])
        monkeypatch.setattr(filename_mod, "get_document_category", lambda *a, **k: "invoice")
        monkeypatch.setattr(filename_mod, "get_final_summary_tokens", lambda *a, **k: ["payment"])

        config = RenamerConfig(
            use_structured_fields=True,
            use_single_llm_call=False,
        )
        client = _make_llm_client()
        scorer = _make_scorer()

        pdf_content = "Rechnungsnummer: INV-12345\nBetrag: 1.234,56 EUR\n2024-05-20"
        _name, metadata = filename_mod.generate_filename(
            pdf_content,
            config=config,
            llm_client=client,
            heuristic_scorer=scorer,
            stopwords=_empty_stopwords(),
            today=REFERENCE_TODAY,
        )
        # Structured fields should be present in metadata
        assert "invoice_id" in metadata
        assert "amount" in metadata
        assert "company" in metadata

    def test_generate_filename_timestamp_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When category/summary/keywords all empty, timestamp fallback is used."""
        import ai_pdf_renamer.filename as filename_mod

        # Make LLM return nothing useful
        monkeypatch.setattr(filename_mod, "get_document_summary", lambda *a, **k: "")
        monkeypatch.setattr(filename_mod, "get_document_keywords", lambda *a, **k: [])
        monkeypatch.setattr(filename_mod, "get_document_category", lambda *a, **k: "unknown")
        monkeypatch.setattr(filename_mod, "get_final_summary_tokens", lambda *a, **k: [])

        config = RenamerConfig(
            use_timestamp_fallback=True,
            timestamp_fallback_segment="document",
            use_single_llm_call=False,
        )
        client = _make_llm_client()
        # scorer with no matching rules -> "unknown" category
        scorer = _make_scorer(categories=[])

        # Add "unknown" to stopwords so that the "unknown" category token is
        # filtered out, making category_clean empty and triggering the fallback.
        stopwords = Stopwords(words={"unknown"})

        with patch.object(
            filename_mod, "_build_timestamp_fallback_filename", wraps=filename_mod._build_timestamp_fallback_filename
        ) as wrapped:
            name, _metadata = filename_mod.generate_filename(
                "Some content with no useful data",
                config=config,
                llm_client=client,
                heuristic_scorer=scorer,
                stopwords=stopwords,
                today=date(2024, 6, 15),
            )
            # The timestamp fallback should have been called
            wrapped.assert_called_once()

        # The filename should contain "document" (the fallback segment)
        assert "document" in name


class TestFilenameTemplateMissingVariable:
    """Test that a template with {undefined_var} triggers the warning fallback."""

    def test_template_with_undefined_var_falls_back(self) -> None:
        """Template with {undefined_var} causes KeyError, so the default filename is used."""
        from ai_pdf_renamer.filename import _apply_filename_template

        config = RenamerConfig(filename_template="{date}-{undefined_var}-{category}")
        original = "20260101-invoice"
        result = _apply_filename_template(
            original,
            date_str="20260101",
            project="",
            version="",
            category_for_filename="invoice",
            category_clean=["invoice"],
            keyword_clean=[],
            summary_clean=[],
            config=config,
        )
        # On KeyError the function should return the original filename unchanged
        assert result == original

    def test_template_with_all_known_vars(self) -> None:
        """Template with all known placeholders works correctly."""
        from ai_pdf_renamer.filename import _apply_filename_template

        config = RenamerConfig(filename_template="{date}_{category}_{keywords}")
        result = _apply_filename_template(
            "fallback",
            date_str="20260322",
            project="",
            version="",
            category_for_filename="invoice",
            category_clean=["invoice"],
            keyword_clean=["payment"],
            summary_clean=[],
            config=config,
        )
        assert "20260322" in result
        assert "invoice" in result.lower()


class TestFilenameMaxCharsTruncation:
    """Test that filename is truncated to max_filename_chars."""

    def test_truncation_at_separator(self) -> None:
        from ai_pdf_renamer.filename import _truncate_filename_to_max_chars

        config = RenamerConfig(max_filename_chars=30)
        filename = "20260101-invoice-payment-reminder-final"
        result = _truncate_filename_to_max_chars(filename, config)
        assert len(result) <= 30
        # When truncated, should end at a separator boundary (not mid-word)
        if len(result) < len(filename):
            assert filename.startswith(result)
            assert not result.endswith("-")

    def test_truncation_no_separator_hard_cut(self) -> None:
        from ai_pdf_renamer.filename import _truncate_filename_to_max_chars

        config = RenamerConfig(max_filename_chars=10)
        filename = "abcdefghijklmnopqrstuvwxyz"  # no separator
        result = _truncate_filename_to_max_chars(filename, config)
        assert len(result) == 10
        assert result == "abcdefghij"

    def test_no_truncation_when_under_limit(self) -> None:
        from ai_pdf_renamer.filename import _truncate_filename_to_max_chars

        config = RenamerConfig(max_filename_chars=100)
        filename = "20260101-invoice"
        result = _truncate_filename_to_max_chars(filename, config)
        assert result == filename

    def test_no_truncation_when_limit_is_none(self) -> None:
        from ai_pdf_renamer.filename import _truncate_filename_to_max_chars

        config = RenamerConfig(max_filename_chars=None)
        filename = "20260101-invoice-very-long-name"
        result = _truncate_filename_to_max_chars(filename, config)
        assert result == filename

    def test_generate_filename_with_max_chars(self) -> None:
        """End-to-end: generate_filename respects max_filename_chars=30."""
        from ai_pdf_renamer.filename import generate_filename

        config = RenamerConfig(
            use_llm=False,
            max_filename_chars=30,
            use_timestamp_fallback=False,
        )
        scorer = _make_scorer(
            [
                (r"invoice", "invoice", 10.0),
            ]
        )
        content = "This is an invoice document with a very long description that should be truncated"
        filename, _meta = generate_filename(
            content,
            config=config,
            heuristic_scorer=scorer,
            stopwords=_empty_stopwords(),
            today=date(2026, 1, 1),
        )
        assert len(filename) <= 30


class TestFilenameDeduplication:
    """Test that keywords overlapping with category tokens are deduplicated."""

    def test_keyword_tokens_subtract_category_tokens(self) -> None:
        from ai_pdf_renamer.filename import _build_metadata_tokens

        stopwords = _empty_stopwords()
        # category_for_filename contains "invoice", keywords also contain "invoice"
        _cat_clean, kw_clean, sum_clean, _meta = _build_metadata_tokens(
            category_for_filename="invoice",
            keywords=["invoice", "payment", "reminder"],
            final_summary_tokens=["invoice", "total", "amount"],
            stopwords=stopwords,
        )
        # "invoice" should be removed from keyword_clean since it's in category_clean
        assert "invoice" not in kw_clean
        # "payment" should remain
        assert "payment" in kw_clean
        # "invoice" should also be removed from summary_clean
        assert "invoice" not in sum_clean

    def test_empty_keywords_still_returns_category(self) -> None:
        from ai_pdf_renamer.filename import _build_metadata_tokens

        stopwords = _empty_stopwords()
        cat_clean, kw_clean, sum_clean, meta = _build_metadata_tokens(
            category_for_filename="invoice",
            keywords=[],
            final_summary_tokens=[],
            stopwords=stopwords,
        )
        assert cat_clean == ["invoice"]
        assert kw_clean == []
        assert sum_clean == []
        assert meta["category"] == "invoice"


class TestFilenameEmptyKeywordsAndSummary:
    """When keywords and summary are both empty, filename still has date + category."""

    def test_heuristic_only_empty_kw_summary(self) -> None:
        from ai_pdf_renamer.filename import generate_filename

        config = RenamerConfig(
            use_llm=False,
            use_timestamp_fallback=False,
        )
        scorer = _make_scorer(
            [
                (r"contract", "contract", 10.0),
            ]
        )
        content = "This is a contract between parties"
        filename, _meta = generate_filename(
            content,
            config=config,
            heuristic_scorer=scorer,
            stopwords=_empty_stopwords(),
            today=date(2026, 3, 22),
        )
        assert "20260322" in filename
        assert "contract" in filename.lower()


class TestCombineResolveConflictAllModes:
    """Test _combine_resolve_conflict with both overlap and embeddings disabled."""

    def test_prefer_llm_fallback(self) -> None:
        """When overlap and embeddings are both disabled and prefer_llm=True, LLM wins."""
        result = _combine_resolve_conflict(
            "report",
            "invoice",
            prefer_llm=True,
            context_for_overlap=None,
            use_embeddings_for_conflict=False,
            use_keyword_overlap=False,
            heuristic_score=5.0,
            heuristic_score_weight=1.0,
        )
        assert result == "report"

    def test_prefer_heuristic_fallback(self) -> None:
        """When overlap and embeddings are both disabled and prefer_llm=False, heuristic wins."""
        result = _combine_resolve_conflict(
            "report",
            "invoice",
            prefer_llm=False,
            context_for_overlap=None,
            use_embeddings_for_conflict=False,
            use_keyword_overlap=False,
            heuristic_score=5.0,
            heuristic_score_weight=1.0,
        )
        assert result == "invoice"

    def test_combine_categories_with_no_overlap_no_embeddings_prefer_llm(self) -> None:
        """Full combine_categories path: LLM and heuristic disagree, no overlap/embeddings, prefer LLM."""
        params = CategoryCombineParams(
            prefer_llm=True,
            use_keyword_overlap=False,
            use_embeddings_for_conflict=False,
        )
        result = combine_categories(
            "report",
            "invoice",
            heuristic_score=5.0,
            heuristic_gap=3.0,
            params=params,
            context_for_overlap=None,
        )
        assert result == "report"

    def test_combine_categories_with_no_overlap_no_embeddings_prefer_heuristic(self) -> None:
        """Full combine_categories path: LLM and heuristic disagree, no overlap/embeddings, prefer heuristic."""
        params = CategoryCombineParams(
            prefer_llm=False,
            use_keyword_overlap=False,
            use_embeddings_for_conflict=False,
        )
        result = combine_categories(
            "report",
            "invoice",
            heuristic_score=5.0,
            heuristic_gap=3.0,
            params=params,
            context_for_overlap=None,
        )
        assert result == "invoice"


class TestScorerNoMatches:
    """Text with no rule matches returns ('unknown', 0.0, 'unknown', 0.0)."""

    def test_no_matches(self) -> None:
        scorer = _make_scorer(
            [
                (r"invoice", "invoice", 10.0),
                (r"contract", "contract", 5.0),
            ]
        )
        cat, score, runner_cat, runner_score = scorer.best_category_with_confidence("this text matches nothing at all")
        assert cat == "unknown"
        assert score == 0.0
        assert runner_cat == "unknown"
        assert runner_score == 0.0

    def test_empty_text(self) -> None:
        scorer = _make_scorer()
        cat, score, _runner_cat, _runner_score = scorer.best_category_with_confidence("")
        assert cat == "unknown"
        assert score == 0.0

    def test_none_text(self) -> None:
        scorer = _make_scorer()
        cat, score, _runner_cat, _runner_score = scorer.best_category_with_confidence(None)  # type: ignore[arg-type]
        assert cat == "unknown"
        assert score == 0.0


class TestScorerSingleMatch:
    """Text matching exactly one rule returns that category, runner_up is 'unknown'."""

    def test_single_match(self) -> None:
        scorer = _make_scorer(
            [
                (r"invoice", "invoice", 10.0),
                (r"contract", "contract", 5.0),
            ]
        )
        cat, score, runner_cat, runner_score = scorer.best_category_with_confidence(
            "This is an invoice for services rendered"
        )
        assert cat == "invoice"
        assert score == 10.0
        # Only one category matched, so runner_up should be 'unknown' with 0.0
        assert runner_cat == "unknown"
        assert runner_score == 0.0
