"""Deep coverage tests for filename.py — targeting uncovered branches."""

from __future__ import annotations

import re
from datetime import date
from unittest.mock import MagicMock, patch

import pytest

from ai_pdf_renamer.config import RenamerConfig
from ai_pdf_renamer.heuristics import HeuristicRule, HeuristicScorer
from ai_pdf_renamer.text_utils import Stopwords

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# _heuristic_text_for_category
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# _resolve_category_with_llm
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# _build_filename_str
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# generate_filename
# ---------------------------------------------------------------------------


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
            today=date(2000, 1, 1),
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
            today=date(2000, 1, 1),
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
