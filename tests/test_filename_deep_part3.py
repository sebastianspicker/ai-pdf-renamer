"""Additional deep coverage tests for filename and heuristic helpers."""

from __future__ import annotations

from datetime import date

from ai_pdf_renamer.config import RenamerConfig
from ai_pdf_renamer.heuristics import (
    CategoryCombineParams,
    _combine_resolve_conflict,
    combine_categories,
)
from tests.conftest import empty_stopwords as _empty_stopwords
from tests.conftest import make_heuristic_scorer as _make_scorer


class TestFilenameTemplateMissingVariable:
    """Test that a template with {undefined_var} triggers the warning fallback."""

    def test_template_with_undefined_var_falls_back(self) -> None:
        """Template with {undefined_var} causes KeyError, so the default filename is used."""
        from ai_pdf_renamer.filename import _apply_filename_template, _FilenameTemplateInput
        from ai_pdf_renamer.filename_models import _FilenameTemplateTokens

        config = RenamerConfig(filename_template="{date}-{undefined_var}-{category}")
        original = "20260101-invoice"
        result = _apply_filename_template(
            _FilenameTemplateInput(
                filename=original,
                date_str="20260101",
                project="",
                version="",
                tokens=_FilenameTemplateTokens(
                    category_for_filename="invoice",
                    category_clean=["invoice"],
                    keyword_clean=[],
                    summary_clean=[],
                ),
            ),
            config=config,
        )
        # On KeyError the function should return the original filename unchanged
        assert result == original

    def test_template_with_all_known_vars(self) -> None:
        """Template with all known placeholders works correctly."""
        from ai_pdf_renamer.filename import _apply_filename_template, _FilenameTemplateInput
        from ai_pdf_renamer.filename_models import _FilenameTemplateTokens

        config = RenamerConfig(filename_template="{date}_{category}_{keywords}")
        result = _apply_filename_template(
            _FilenameTemplateInput(
                filename="fallback",
                date_str="20260322",
                project="",
                version="",
                tokens=_FilenameTemplateTokens(
                    category_for_filename="invoice",
                    category_clean=["invoice"],
                    keyword_clean=["payment"],
                    summary_clean=[],
                ),
            ),
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
        from ai_pdf_renamer.filename import FilenameGenerationRequest, generate_filename

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
            FilenameGenerationRequest(
                config=config,
                heuristic_scorer=scorer,
                stopwords=_empty_stopwords(),
                today=date(2026, 1, 1),
            ),
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
        from ai_pdf_renamer.filename import FilenameGenerationRequest, generate_filename

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
            FilenameGenerationRequest(
                config=config,
                heuristic_scorer=scorer,
                stopwords=_empty_stopwords(),
                today=date(2026, 3, 22),
            ),
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
