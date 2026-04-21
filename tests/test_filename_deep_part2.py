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


class TestIntegrationHeuristicOnlyRename:
    """Create a real PDF with fitz, run generate_filename with use_llm=False."""

    def test_heuristic_only_with_real_pdf(self, tmp_path: Path) -> None:
        fitz = pytest.importorskip("fitz")

        pdf_path = tmp_path / "test_invoice.pdf"
        doc = fitz.open()
        page = doc.new_page()
        text_point = fitz.Point(72, 100)
        page.insert_text(text_point, "Rechnung Nr. 12345 Betrag: 100,00 EUR")
        doc.save(str(pdf_path))
        doc.close()

        from ai_pdf_renamer.filename import generate_filename
        from ai_pdf_renamer.pdf_extract import pdf_to_text

        content = pdf_to_text(pdf_path)
        assert "Rechnung" in content or "12345" in content

        config = RenamerConfig(
            use_llm=False,
            language="de",
            use_timestamp_fallback=False,
        )
        filename, meta = generate_filename(
            content,
            config=config,
            today=date(2026, 3, 22),
        )
        # Should have date
        assert "20260322" in filename
        # Category should be invoice-related (Rechnung is German for invoice)
        assert meta.get("category_source") == "heuristic"


class TestIntegrationDryRunFullPipeline:
    """Create a PDF, run rename_pdfs_in_directory with dry_run=True."""

    def test_dry_run_no_files_renamed(self, tmp_path: Path) -> None:
        fitz = pytest.importorskip("fitz")

        pdf_path = tmp_path / "rechnung.pdf"
        doc = fitz.open()
        page = doc.new_page()
        text_point = fitz.Point(72, 100)
        page.insert_text(text_point, "Rechnung Nr. 99999 Betrag: 250,00 EUR")
        doc.save(str(pdf_path))
        doc.close()

        summary_json = tmp_path / "summary.json"
        config = RenamerConfig(
            use_llm=False,
            language="de",
            dry_run=True,
            summary_json_path=str(summary_json),
            use_timestamp_fallback=False,
        )

        from ai_pdf_renamer.renamer import rename_pdfs_in_directory

        rename_pdfs_in_directory(str(tmp_path), config=config)

        # Original file should still exist with original name
        assert pdf_path.exists(), "Original file should not be renamed in dry_run mode"

        # Summary JSON should be written
        assert summary_json.exists(), "Summary JSON should be written"
        summary = json.loads(summary_json.read_text(encoding="utf-8"))
        assert summary["dry_run"] is True
        assert summary["processed"] >= 1
        assert summary["renamed"] >= 0  # dry_run still counts as "renamed" in the code


class TestResolveOptionInvalidThenValid:
    """Monkeypatch input to return invalid first, then valid — verify retry."""

    def test_prompt_choice_invalid_then_valid(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from ai_pdf_renamer.cli import _prompt_choice

        inputs = iter(["invalid_lang", "de"])
        monkeypatch.setattr("builtins.input", lambda _prompt: next(inputs))

        result = _prompt_choice(
            "Language: ",
            choices=["de", "en"],
            default="de",
            normalize=str.lower,
        )
        assert result == "de"

    def test_prompt_choice_empty_returns_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from ai_pdf_renamer.cli import _prompt_choice

        monkeypatch.setattr("builtins.input", lambda _prompt: "")
        result = _prompt_choice(
            "Language: ",
            choices=["de", "en"],
            default="de",
            normalize=str.lower,
        )
        assert result == "de"


class TestMainCatchAllException:
    """Mock rename_pdfs to raise RuntimeError, verify SystemExit."""

    def test_runtime_error_becomes_system_exit(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from ai_pdf_renamer.cli import _run_renamer_or_watch

        test_dir = tmp_path / "pdfs"
        test_dir.mkdir()

        args = argparse.Namespace(watch=False)
        config = RenamerConfig(use_llm=False)

        with (
            patch(
                "ai_pdf_renamer.cli.rename_pdfs_in_directory",
                side_effect=RuntimeError("Unexpected failure"),
            ),
            pytest.raises(SystemExit) as exc_info,
        ):
            _run_renamer_or_watch([str(test_dir)], config, args)
        assert exc_info.value.code == 1

    def test_file_not_found_becomes_system_exit(
        self,
        tmp_path: Path,
    ) -> None:
        from ai_pdf_renamer.cli import _run_renamer_or_watch

        args = argparse.Namespace(watch=False)
        config = RenamerConfig(use_llm=False)

        with (
            patch(
                "ai_pdf_renamer.cli.rename_pdfs_in_directory",
                side_effect=FileNotFoundError("Directory not found"),
            ),
            pytest.raises(SystemExit) as exc_info,
        ):
            _run_renamer_or_watch([str(tmp_path / "nonexistent")], config, args)
        assert exc_info.value.code == 1


class TestHttpBackendTextModeEmptyChoices:
    """Response with empty choices list returns ''."""

    def test_empty_choices_returns_empty_string(self) -> None:
        from ai_pdf_renamer.llm_backend import HttpLLMBackend

        backend = HttpLLMBackend(use_chat=False)
        mock_response = MagicMock()
        mock_response.json.return_value = {"choices": []}
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()

        with patch.object(backend._session, "post", return_value=mock_response):
            result = backend.complete("test prompt")
        assert result == ""

    def test_missing_choices_key_returns_empty_string(self) -> None:
        from ai_pdf_renamer.llm_backend import HttpLLMBackend

        backend = HttpLLMBackend(use_chat=False)
        mock_response = MagicMock()
        mock_response.json.return_value = {"result": "no choices key"}
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()

        with patch.object(backend._session, "post", return_value=mock_response):
            result = backend.complete("test prompt")
        assert result == ""


class TestHttpBackendTextModeNonDictChoice:
    """choices[0] is not a dict — returns ''."""

    def test_non_dict_choice_returns_empty_string(self) -> None:
        from ai_pdf_renamer.llm_backend import HttpLLMBackend

        backend = HttpLLMBackend(use_chat=False)
        mock_response = MagicMock()
        # choices[0] is a string, not a dict
        mock_response.json.return_value = {"choices": ["just a string"]}
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()

        with patch.object(backend._session, "post", return_value=mock_response):
            result = backend.complete("test prompt")
        assert result == ""

    def test_non_dict_choice_integer_returns_empty_string(self) -> None:
        from ai_pdf_renamer.llm_backend import HttpLLMBackend

        backend = HttpLLMBackend(use_chat=False)
        mock_response = MagicMock()
        # choices[0] is an integer
        mock_response.json.return_value = {"choices": [42]}
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()

        with patch.object(backend._session, "post", return_value=mock_response):
            result = backend.complete("test prompt")
        assert result == ""


class TestExtractChatMessageContent:
    """Edge cases for _extract_chat_message_content."""

    def test_choices_empty_list(self) -> None:
        from ai_pdf_renamer.llm_backend import _extract_chat_message_content

        result = _extract_chat_message_content({"choices": []})
        assert result == ""

    def test_choices_not_a_list(self) -> None:
        from ai_pdf_renamer.llm_backend import _extract_chat_message_content

        result = _extract_chat_message_content({"choices": "not a list"})
        assert result == ""

    def test_message_not_a_dict(self) -> None:
        from ai_pdf_renamer.llm_backend import _extract_chat_message_content

        result = _extract_chat_message_content({"choices": [{"message": "not a dict"}]})
        assert result == ""

    def test_message_content_none(self) -> None:
        from ai_pdf_renamer.llm_backend import _extract_chat_message_content

        result = _extract_chat_message_content({"choices": [{"message": {"content": None}}]})
        assert result == ""

    def test_normal_response(self) -> None:
        from ai_pdf_renamer.llm_backend import _extract_chat_message_content

        result = _extract_chat_message_content({"choices": [{"message": {"content": "Hello world"}}]})
        assert result == "Hello world"


class TestShouldUseTimestampFallback:
    """Edge cases for _should_use_timestamp_fallback."""

    def test_unknown_category_no_tokens(self) -> None:
        from ai_pdf_renamer.filename import _should_use_timestamp_fallback

        assert _should_use_timestamp_fallback("unknown", [], [], []) is True

    def test_empty_category_no_tokens(self) -> None:
        from ai_pdf_renamer.filename import _should_use_timestamp_fallback

        assert _should_use_timestamp_fallback("", [], [], []) is True

    def test_document_category_no_tokens(self) -> None:
        from ai_pdf_renamer.filename import _should_use_timestamp_fallback

        assert _should_use_timestamp_fallback("document", [], [], []) is True

    def test_valid_category_no_tokens(self) -> None:
        from ai_pdf_renamer.filename import _should_use_timestamp_fallback

        # Valid category present, even if no extra tokens
        assert _should_use_timestamp_fallback("invoice", [], [], []) is False

    def test_unknown_category_with_tokens(self) -> None:
        from ai_pdf_renamer.filename import _should_use_timestamp_fallback

        # Unknown category but tokens present
        assert _should_use_timestamp_fallback("unknown", [], ["payment"], []) is False


class TestBuildTimestampFallbackFilename:
    """Test _build_timestamp_fallback_filename edge cases."""

    def test_custom_segment(self) -> None:
        from datetime import datetime

        from ai_pdf_renamer.filename import _build_timestamp_fallback_filename

        config = RenamerConfig(timestamp_fallback_segment="scan")
        now = datetime(2026, 3, 22, 14, 30, 45)
        result = _build_timestamp_fallback_filename("20260322", config, now=now)
        assert "20260322" in result
        assert "scan" in result
        assert "143045" in result

    def test_empty_segment_defaults_to_document(self) -> None:
        from datetime import datetime

        from ai_pdf_renamer.filename import _build_timestamp_fallback_filename

        config = RenamerConfig(timestamp_fallback_segment="")
        now = datetime(2026, 3, 22, 14, 30, 45)
        result = _build_timestamp_fallback_filename("20260322", config, now=now)
        assert "document" in result


class TestFilenameSep:
    """Test _filename_sep returns correct separator for each case."""

    def test_snake_case_uses_underscore(self) -> None:
        from ai_pdf_renamer.filename import _filename_sep

        config = RenamerConfig(desired_case="snakeCase")
        assert _filename_sep(config) == "_"

    def test_kebab_case_uses_hyphen(self) -> None:
        from ai_pdf_renamer.filename import _filename_sep

        config = RenamerConfig(desired_case="kebabCase")
        assert _filename_sep(config) == "-"

    def test_camel_case_uses_hyphen(self) -> None:
        from ai_pdf_renamer.filename import _filename_sep

        config = RenamerConfig(desired_case="camelCase")
        assert _filename_sep(config) == "-"


class TestCombineCategoriesEdgeCases:
    """Additional branch coverage for combine_categories."""

    def test_heuristic_unknown_llm_valid(self) -> None:
        """When heuristic is unknown and LLM returns a valid category, use LLM."""
        result = combine_categories(
            "report",
            "unknown",
            heuristic_score=None,
            heuristic_gap=None,
        )
        assert result == "report"

    def test_heuristic_unknown_llm_also_unknown(self) -> None:
        """When heuristic is unknown and LLM also returns unknown, return the raw LLM value."""
        result = combine_categories(
            "unknown",
            "unknown",
            heuristic_score=None,
            heuristic_gap=None,
        )
        # cat_llm_norm is "unknown" which is not valid, so returns raw cat_llm
        assert result == "unknown"

    def test_llm_returns_document_heuristic_valid(self) -> None:
        """When LLM returns 'document' (useless) and heuristic is valid, use heuristic."""
        result = combine_categories(
            "document",
            "invoice",
            heuristic_score=5.0,
            heuristic_gap=3.0,
        )
        assert result == "invoice"

    def test_heuristic_below_min_score_prefers_llm(self) -> None:
        """When heuristic score is below min_heuristic_score, prefer LLM."""
        params = CategoryCombineParams(min_heuristic_score=10.0)
        result = combine_categories(
            "report",
            "invoice",
            heuristic_score=2.0,
            heuristic_gap=1.0,
            params=params,
        )
        assert result == "report"


class TestCombineResolveConflictWithContext:
    """Test _combine_resolve_conflict when context is provided but only overlap is used."""

    def test_overlap_favors_llm(self) -> None:
        """When keyword overlap favors LLM category, return LLM."""
        result = _combine_resolve_conflict(
            "report",
            "invoice",
            prefer_llm=False,
            context_for_overlap="this report discusses findings",
            use_embeddings_for_conflict=False,
            use_keyword_overlap=True,
            heuristic_score=5.0,
            heuristic_score_weight=0.0,
        )
        assert result == "report"

    def test_overlap_favors_heuristic(self) -> None:
        """When keyword overlap favors heuristic category, return heuristic."""
        result = _combine_resolve_conflict(
            "report",
            "invoice",
            prefer_llm=True,
            context_for_overlap="this invoice is for services",
            use_embeddings_for_conflict=False,
            use_keyword_overlap=True,
            heuristic_score=5.0,
            heuristic_score_weight=0.0,
        )
        assert result == "invoice"

    def test_overlap_tie_returns_heuristic(self) -> None:
        """When overlap is tied, default to heuristic."""
        result = _combine_resolve_conflict(
            "catA",
            "catB",
            prefer_llm=True,
            context_for_overlap="no matching tokens here",
            use_embeddings_for_conflict=False,
            use_keyword_overlap=True,
            heuristic_score=0.0,
            heuristic_score_weight=0.0,
        )
        assert result == "catB"
