# ruff: noqa: F401,F811

from __future__ import annotations

import argparse
import base64
import contextlib
import json
import logging
import os
import re
import sys
import time
from datetime import date
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from ai_pdf_renamer import pdf_extract
from ai_pdf_renamer.config import RenamerConfig
from ai_pdf_renamer.heuristics import (
    HeuristicRule,
    HeuristicScorer,
    _combine_resolve_conflict,
    _embedding_conflict_pick,
    _load_category_aliases,
    load_heuristic_rules,
    load_heuristic_rules_for_language,
)
from ai_pdf_renamer.renamer import (
    _apply_post_rename_actions,
    _produce_rename_results,
    _run_post_rename_hook,
    _write_json_or_csv,
    rename_pdfs_in_directory,
    run_watch_loop,
)


def _cfg(**overrides: object) -> RenamerConfig:
    """Build a RenamerConfig with sensible test defaults and overrides."""
    defaults: dict[str, object] = {
        "use_llm": False,
        "use_single_llm_call": False,
    }
    defaults.update(overrides)
    return RenamerConfig(**defaults)  # type: ignore[arg-type]


def _make_fake_pdf(tmp_path: Path, name: str = "test.pdf", mtime: float | None = None) -> Path:
    """Create a minimal PDF in tmp_path and optionally set its mtime."""
    p = tmp_path / name
    # Minimal valid PDF (enough to be treated as a file with .pdf extension)
    p.write_bytes(b"%PDF-1.0\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n%%EOF\n")
    if mtime is not None:
        os.utime(p, (mtime, mtime))
    return p


class TestWatchLoopMtimeTracking:
    """Test 10: verify second iteration skips unchanged files."""

    def test_watch_loop_skips_unchanged(self, tmp_path: Path) -> None:
        """Second iteration with no mtime change should not process any files."""
        pdf = _make_fake_pdf(tmp_path, "test.pdf")

        config = _cfg()

        iteration = 0
        processed_counts: list[int] = []

        def fake_rename(
            directory: object,
            *,
            config: RenamerConfig,
            files_override: list[Path] | None = None,
            rules_override: object | None = None,
        ) -> None:
            processed_counts.append(len(files_override or []))

        def fake_collect(directory: object, **kwargs: object) -> list[Path]:
            return [pdf]

        def fake_sleep(secs: float) -> None:
            nonlocal iteration
            iteration += 1
            if iteration >= 2:
                # Trigger stop by raising KeyboardInterrupt
                raise KeyboardInterrupt()

        with (
            patch("ai_pdf_renamer.renamer._collect_pdf_files", side_effect=fake_collect),
            patch("ai_pdf_renamer.renamer.rename_pdfs_in_directory", side_effect=fake_rename),
            patch("ai_pdf_renamer.renamer.time.sleep", side_effect=fake_sleep),
            patch("ai_pdf_renamer.renamer.signal.signal"),
            contextlib.suppress(KeyboardInterrupt),
        ):
            run_watch_loop(tmp_path, config=config, interval_seconds=0.1)

        # First iteration: file processed (new mtime)
        # Second iteration: file skipped (same mtime) -> sleep -> KeyboardInterrupt
        assert len(processed_counts) == 1  # Only first iteration processed a file


class TestDryRunAndRenameFailureReporting:
    """Cover dry-run logging and rename failure reporting branches."""

    def test_dry_run_logging(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        """Dry-run logs 'would rename' message."""
        pdf = _make_fake_pdf(tmp_path, "doc.pdf")
        config = _cfg(dry_run=True)

        results = [(pdf, "new_name", {"category": "test"}, None)]

        with (
            patch("ai_pdf_renamer.renamer._produce_rename_results", return_value=results),
            patch("ai_pdf_renamer.renamer.load_processing_rules", return_value=None),
            patch("ai_pdf_renamer.renamer._collect_pdf_files", return_value=[pdf]),
            patch("ai_pdf_renamer.renamer.apply_single_rename", return_value=(True, pdf.with_name("new_name.pdf"))),
        ):
            import logging

            with caplog.at_level(logging.INFO):
                rename_pdfs_in_directory(tmp_path, config=config)

        assert any("Dry-run" in r.message or "would rename" in r.message for r in caplog.records)

    def test_rename_failure_reporting(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        """When apply_single_rename returns success=False, error is logged."""
        pdf = _make_fake_pdf(tmp_path, "doc.pdf")
        config = _cfg()

        results = [(pdf, "new_name", {"category": "test"}, None)]

        with (
            patch("ai_pdf_renamer.renamer._produce_rename_results", return_value=results),
            patch("ai_pdf_renamer.renamer.load_processing_rules", return_value=None),
            patch("ai_pdf_renamer.renamer._collect_pdf_files", return_value=[pdf]),
            patch("ai_pdf_renamer.renamer.apply_single_rename", return_value=(False, pdf)),
        ):
            import logging

            with caplog.at_level(logging.ERROR):
                rename_pdfs_in_directory(tmp_path, config=config)

        assert any("could not rename" in r.message.lower() for r in caplog.records)


class TestRenameApplyException:
    """Cover exception path in apply_single_rename wrapper."""

    def test_apply_raises_exception(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        """When apply_single_rename raises, failure is recorded."""
        pdf = _make_fake_pdf(tmp_path, "doc.pdf")
        config = _cfg()

        results = [(pdf, "new_name", {"category": "test"}, None)]

        with (
            patch("ai_pdf_renamer.renamer._produce_rename_results", return_value=results),
            patch("ai_pdf_renamer.renamer.load_processing_rules", return_value=None),
            patch("ai_pdf_renamer.renamer._collect_pdf_files", return_value=[pdf]),
            patch("ai_pdf_renamer.renamer.apply_single_rename", side_effect=PermissionError("denied")),
        ):
            import logging

            with caplog.at_level(logging.ERROR):
                rename_pdfs_in_directory(tmp_path, config=config)

        assert any("denied" in r.message for r in caplog.records)


class TestLoadRulesLanguageField:
    """Test 11: rule with language='en', verify stored."""

    def test_language_field_stored(self, tmp_path: Path) -> None:
        data = {
            "patterns": [
                {"regex": "invoice", "category": "invoice", "score": 10, "language": "en"},
                {"regex": "rechnung", "category": "rechnung", "score": 10, "language": "de"},
            ]
        }
        f = tmp_path / "rules.json"
        f.write_text(json.dumps(data))
        rules = load_heuristic_rules(f)
        assert len(rules) == 2
        assert rules[0].language == "en"
        assert rules[1].language == "de"

    def test_language_field_invalid_type(self, tmp_path: Path) -> None:
        """Non-string language is set to None."""
        data = {"patterns": [{"regex": "test", "category": "cat", "score": 1, "language": 123}]}
        f = tmp_path / "rules.json"
        f.write_text(json.dumps(data))
        rules = load_heuristic_rules(f)
        assert rules[0].language is None

    def test_language_field_unsupported_value(self, tmp_path: Path) -> None:
        """Language value not in ('de', 'en') is set to None."""
        data = {"patterns": [{"regex": "test", "category": "cat", "score": 1, "language": "fr"}]}
        f = tmp_path / "rules.json"
        f.write_text(json.dumps(data))
        rules = load_heuristic_rules(f)
        assert rules[0].language is None

    def test_language_field_empty_string(self, tmp_path: Path) -> None:
        """Empty string language is set to None."""
        data = {"patterns": [{"regex": "test", "category": "cat", "score": 1, "language": "  "}]}
        f = tmp_path / "rules.json"
        f.write_text(json.dumps(data))
        rules = load_heuristic_rules(f)
        assert rules[0].language is None


class TestLoadRulesParentField:
    """Test 12: rule with parent, verify stored."""

    def test_parent_field_stored(self, tmp_path: Path) -> None:
        data = {"patterns": [{"regex": "test", "category": "sub_cat", "score": 5, "parent": "main_cat"}]}
        f = tmp_path / "rules.json"
        f.write_text(json.dumps(data))
        rules = load_heuristic_rules(f)
        assert rules[0].parent == "main_cat"

    def test_parent_field_invalid_type(self, tmp_path: Path) -> None:
        """Non-string parent is set to None."""
        data = {"patterns": [{"regex": "test", "category": "cat", "score": 1, "parent": 42}]}
        f = tmp_path / "rules.json"
        f.write_text(json.dumps(data))
        rules = load_heuristic_rules(f)
        assert rules[0].parent is None

    def test_parent_field_empty_string(self, tmp_path: Path) -> None:
        """Empty string parent is set to None."""
        data = {"patterns": [{"regex": "test", "category": "cat", "score": 1, "parent": "  "}]}
        f = tmp_path / "rules.json"
        f.write_text(json.dumps(data))
        rules = load_heuristic_rules(f)
        assert rules[0].parent is None


class TestLoadRulesPatternNotList:
    """patterns key that is not a list is treated as empty."""

    def test_patterns_not_list(self, tmp_path: Path) -> None:
        data = {"patterns": "not a list"}
        f = tmp_path / "rules.json"
        f.write_text(json.dumps(data))
        rules = load_heuristic_rules(f)
        assert rules == []


class TestLoadRulesForLanguage:
    """Tests 13-14: locale file loading."""

    def test_locale_file_exists_and_merges(self, tmp_path: Path) -> None:
        """Mock locale file exists, verify merged rules."""
        base_data = {"patterns": [{"regex": "base", "category": "base_cat", "score": 1}]}
        locale_data = {"patterns": [{"regex": "locale", "category": "locale_cat", "score": 2}]}
        base_file = tmp_path / "heuristic_scores.json"
        locale_file = tmp_path / "heuristic_scores_de.json"
        base_file.write_text(json.dumps(base_data))
        locale_file.write_text(json.dumps(locale_data))
        rules = load_heuristic_rules_for_language(base_file, "de")
        assert len(rules) == 2
        assert rules[0].category == "base_cat"
        assert rules[1].category == "locale_cat"

    def test_no_locale_file_returns_base_only(self, tmp_path: Path) -> None:
        """Locale file missing, verify base only returned."""
        base_data = {"patterns": [{"regex": "base", "category": "base_cat", "score": 1}]}
        base_file = tmp_path / "heuristic_scores.json"
        base_file.write_text(json.dumps(base_data))
        rules = load_heuristic_rules_for_language(base_file, "de")
        assert len(rules) == 1
        assert rules[0].category == "base_cat"

    def test_locale_file_invalid_json(self, tmp_path: Path) -> None:
        """Invalid locale file falls back to base rules with warning."""
        base_data = {"patterns": [{"regex": "base", "category": "base_cat", "score": 1}]}
        base_file = tmp_path / "heuristic_scores.json"
        base_file.write_text(json.dumps(base_data))
        locale_file = tmp_path / "heuristic_scores_en.json"
        locale_file.write_text("NOT VALID JSON")
        rules = load_heuristic_rules_for_language(base_file, "en")
        assert len(rules) == 1
        assert rules[0].category == "base_cat"

    def test_unsupported_language_defaults_to_de(self, tmp_path: Path) -> None:
        """Unsupported language falls back to 'de'."""
        base_data = {"patterns": [{"regex": "base", "category": "base_cat", "score": 1}]}
        base_file = tmp_path / "heuristic_scores.json"
        base_file.write_text(json.dumps(base_data))
        # No de locale file, so just base
        rules = load_heuristic_rules_for_language(base_file, "fr")
        assert len(rules) == 1


class TestCategoryAliasesErrorPaths:
    """Tests 15-16: category aliases file missing / invalid JSON."""

    def test_aliases_file_missing(self, tmp_path: Path) -> None:
        """Data file missing, verify empty aliases returned."""
        import ai_pdf_renamer.heuristics as hmod

        old_val = hmod._CATEGORY_ALIASES
        try:
            hmod._CATEGORY_ALIASES = None
            with patch("ai_pdf_renamer.data_paths.category_aliases_path", return_value=tmp_path / "nonexistent.json"):
                result = _load_category_aliases()
            assert result == {}
        finally:
            hmod._CATEGORY_ALIASES = old_val

    def test_aliases_invalid_json(self, tmp_path: Path) -> None:
        """Invalid JSON in aliases file, verify fallback to empty."""
        import ai_pdf_renamer.heuristics as hmod

        old_val = hmod._CATEGORY_ALIASES
        try:
            hmod._CATEGORY_ALIASES = None
            bad_file = tmp_path / "category_aliases.json"
            bad_file.write_text("NOT JSON")
            with patch("ai_pdf_renamer.data_paths.category_aliases_path", return_value=bad_file):
                result = _load_category_aliases()
            assert result == {}
        finally:
            hmod._CATEGORY_ALIASES = old_val

    def test_aliases_not_dict(self, tmp_path: Path) -> None:
        """aliases key is not a dict, verify empty."""
        import ai_pdf_renamer.heuristics as hmod

        old_val = hmod._CATEGORY_ALIASES
        try:
            hmod._CATEGORY_ALIASES = None
            bad_file = tmp_path / "category_aliases.json"
            bad_file.write_text(json.dumps({"aliases": "not a dict"}))
            with patch("ai_pdf_renamer.data_paths.category_aliases_path", return_value=bad_file):
                result = _load_category_aliases()
            assert result == {}
        finally:
            hmod._CATEGORY_ALIASES = old_val


class TestEmbeddingConflictNoModule:
    """Test 17: sentence_transformers unavailable returns None."""

    def test_no_sentence_transformers(self) -> None:
        """When sentence_transformers import fails, _embedding_conflict_pick returns None."""
        import ai_pdf_renamer.heuristics as hmod

        old_model = hmod._embedding_model
        try:
            hmod._embedding_model = None
            with patch.dict("sys.modules", {"sentence_transformers": None}):
                result = _embedding_conflict_pick("some context text", "invoice", "receipt")
            assert result is None
        finally:
            hmod._embedding_model = old_model

    def test_empty_context_returns_none(self) -> None:
        """Empty context string returns None without trying embeddings."""
        result = _embedding_conflict_pick("", "invoice", "receipt")
        assert result is None


class TestKeywordOverlapWithScoreWeight:
    """Test 18: verify heuristic bonus from score in keyword overlap."""

    def test_score_weight_favors_heuristic(self) -> None:
        """With score weight bonus, heuristic wins even if LLM has more token overlap."""
        # LLM category has 1 overlap token, heuristic has 0 but gets score bonus
        result = _combine_resolve_conflict(
            "auto_insurance",  # tokens: auto, insurance -> 1 overlap with context
            "contract",  # tokens: contract -> 0 overlap with context
            prefer_llm=False,
            context_for_overlap="auto insurance policy details",
            use_embeddings_for_conflict=False,
            use_keyword_overlap=True,
            heuristic_score=20.0,
            heuristic_score_weight=1.0,  # 1.0 * 20.0 = 20.0 bonus
        )
        # heuristic_weighted = 0 + 20.0 = 20.0 > llm overlap of 2
        assert result == "contract"

    def test_no_score_weight_llm_wins(self) -> None:
        """Without score weight, LLM with more overlap wins."""
        result = _combine_resolve_conflict(
            "auto_insurance",
            "contract",
            prefer_llm=False,
            context_for_overlap="auto insurance policy",
            use_embeddings_for_conflict=False,
            use_keyword_overlap=True,
            heuristic_score=0.0,
            heuristic_score_weight=0.0,
        )
        # LLM tokens {auto, insurance} overlap 2 vs heuristic {contract} overlap 0
        assert result == "auto_insurance"

    def test_overlap_tie_returns_heuristic(self) -> None:
        """When overlap is a tie, heuristic is returned."""
        result = _combine_resolve_conflict(
            "letter",
            "brief",
            prefer_llm=False,
            context_for_overlap="something unrelated",
            use_embeddings_for_conflict=False,
            use_keyword_overlap=True,
            heuristic_score=0.0,
            heuristic_score_weight=0.0,
        )
        # Neither has overlap with context -> tie -> heuristic
        assert result == "brief"


class TestHeuristicDebugLogging:
    """Cover the debug logging of top-3 categories (lines 224-229)."""

    def test_debug_top3_categories(self, caplog: pytest.LogCaptureFixture) -> None:
        """With DEBUG logging enabled, top-3 categories are logged."""
        import logging

        rules = [
            HeuristicRule(pattern=re.compile(r"invoice", re.I), category="invoice", score=10.0),
            HeuristicRule(pattern=re.compile(r"contract", re.I), category="contract", score=5.0),
            HeuristicRule(pattern=re.compile(r"letter", re.I), category="letter", score=3.0),
        ]
        scorer = HeuristicScorer(rules=rules)
        with caplog.at_level(logging.DEBUG, logger="ai_pdf_renamer.heuristics"):
            result = scorer.best_category_with_confidence("This is an invoice about a contract and a letter")
        assert result[0] == "invoice"
        assert any("top-3" in r.message.lower() for r in caplog.records)


class TestCompleteJsonRetryJsonMode:
    """Test 19: json_mode=True limits to 1 retry and passes response_format."""

    def test_json_mode_single_retry(self) -> None:
        """json_mode=True: only 1 call, response_format passed."""
        from ai_pdf_renamer.llm import complete_json_with_retry

        client = MagicMock()
        client.complete.return_value = '{"result": "ok"}'
        result = complete_json_with_retry(client, "test prompt", json_mode=True, max_retries=5)
        # Should only call once (effective_retries=1 when json_mode=True)
        assert client.complete.call_count == 1
        # response_format should be {"type": "json_object"}
        _, kwargs = client.complete.call_args
        assert kwargs["response_format"] == {"type": "json_object"}
        assert '{"result": "ok"}' in result

    def test_json_mode_false_normal_retries(self) -> None:
        """json_mode=False: normal max_retries applies."""
        from ai_pdf_renamer.llm import complete_json_with_retry

        client = MagicMock()
        # Return non-JSON every time to trigger all retries
        client.complete.return_value = "not json at all"
        complete_json_with_retry(client, "test", json_mode=False, max_retries=3)
        assert client.complete.call_count == 3
        _, kwargs = client.complete.call_args
        assert kwargs["response_format"] is None


class TestDocumentAnalysisWithAllowedCategories:
    """Test 20: verify allowed_categories appears in prompt."""

    def test_allowed_categories_in_prompt(self) -> None:
        from ai_pdf_renamer.llm import get_document_analysis

        client = MagicMock()
        client.complete.return_value = '{"summary":"test","keywords":["a"],"category":"invoice"}'

        content = "This is a long enough document content for the test " * 5

        result = get_document_analysis(
            client,
            content,
            language="en",
            allowed_categories=["invoice", "contract", "letter"],
        )
        # The prompt should contain the allowed categories
        prompt_used = client.complete.call_args[0][0]
        assert "invoice" in prompt_used
        assert "contract" in prompt_used
        assert "letter" in prompt_used
        assert result.category == "invoice"


class TestDocumentAnalysisFallbackPaths:
    """Cover get_document_analysis fallback paths when JSON parsing fails."""

    def test_empty_response_returns_defaults(self) -> None:
        from ai_pdf_renamer.llm import get_document_analysis

        client = MagicMock()
        client.complete.return_value = ""
        content = "This is test content that is long enough to pass the minimum length check " * 3
        result = get_document_analysis(client, content, language="en")
        # Default summary is "na" (from DocumentAnalysisResult defaults)
        assert result.summary == "na"

    def test_lenient_json_fallback(self) -> None:
        from ai_pdf_renamer.llm import get_document_analysis

        client = MagicMock()
        # Return something that is not valid JSON but has extractable key-value pairs
        client.complete.return_value = 'Here is the result: "summary": "a nice doc", "category": "invoice"'
        content = "This is test content that is long enough to pass the minimum length check " * 3
        result = get_document_analysis(client, content, language="en", lenient_json=True)
        assert result.summary == "a nice doc"

    def test_short_content_returns_empty(self) -> None:
        from ai_pdf_renamer.llm import get_document_analysis

        client = MagicMock()
        result = get_document_analysis(client, "short", language="en")
        # Default summary is "na" (from DocumentAnalysisResult defaults)
        assert result.summary == "na"
        client.complete.assert_not_called()


class TestDocumentSummaryMultiChunk:
    """Test 21: text longer than max_chars_single triggers chunked path."""

    def test_multi_chunk_summary(self) -> None:
        from ai_pdf_renamer.llm import get_document_summary

        client = MagicMock()
        # Return valid JSON for each chunk and the final combine
        client.complete.return_value = '{"summary": "chunk summary"}'

        # Create text that is longer than max_chars_single
        # Use a small max_chars_single to trigger the multi-chunk path
        long_text = "A" * 200
        result = get_document_summary(
            client,
            long_text,
            language="en",
            max_chars_single=100,  # Force chunking
        )
        # Should have multiple calls: one per chunk + one combine
        assert client.complete.call_count >= 2
        assert result == "chunk summary"

    def test_multi_chunk_empty_partials(self) -> None:
        """When all chunk summaries are empty, returns 'na'."""
        from ai_pdf_renamer.llm import get_document_summary

        client = MagicMock()
        # Return JSON with empty summary for all chunks
        client.complete.return_value = '{"summary": ""}'

        long_text = "B" * 200
        result = get_document_summary(
            client,
            long_text,
            language="en",
            max_chars_single=100,
        )
        assert result == "na"


class TestDocumentSummaryMaxContentChars:
    """Test 22: override max_content_chars, verify it is used."""

    def test_max_content_chars_override(self) -> None:
        from ai_pdf_renamer.llm import get_document_summary

        client = MagicMock()
        client.complete.return_value = '{"summary": "short doc"}'

        # Use a large text
        long_text = "C" * 10000

        result = get_document_summary(
            client,
            long_text,
            language="en",
            max_content_chars=500,
        )
        # The prompt sent to the client should have truncated content
        prompt_used = client.complete.call_args[0][0]
        # The truncated text should be much shorter than 10000 chars
        assert len(prompt_used) < 10000
        assert result == "short doc"

    def test_max_content_chars_none_uses_default(self) -> None:
        from ai_pdf_renamer.llm import get_document_summary

        client = MagicMock()
        client.complete.return_value = '{"summary": "full doc"}'

        text = "D" * 1000
        result = get_document_summary(client, text, language="en", max_content_chars=None)
        assert result == "full doc"


class TestHookHTTPPost:
    """Cover HTTP POST hook path (lines 172-173, 180, 188-190)."""

    def test_hook_http_post(self, tmp_path: Path) -> None:
        """HTTP hook sends JSON payload with old_path, new_path, meta."""
        old = tmp_path / "old.pdf"
        new = tmp_path / "new.pdf"
        old.touch()
        new.touch()

        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)
        mock_session.post.return_value = mock_response

        with patch("ai_pdf_renamer.renamer.requests.Session", return_value=mock_session):
            _run_post_rename_hook("https://example.com/hook", old, new, {"k": "v"})

        mock_session.post.assert_called_once()
        call_kwargs = mock_session.post.call_args
        assert call_kwargs[1]["json"]["old_path"] == str(old)
        assert call_kwargs[1]["json"]["new_path"] == str(new)
        assert call_kwargs[1]["json"]["meta"]["k"] == "v"

    def test_hook_http_non_loopback_warning(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        """Plain HTTP to non-loopback host logs a warning."""
        import logging

        old = tmp_path / "old.pdf"
        new = tmp_path / "new.pdf"
        old.touch()
        new.touch()

        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)
        mock_session.post.return_value = mock_response

        with (
            patch("ai_pdf_renamer.renamer.requests.Session", return_value=mock_session),
            caplog.at_level(logging.WARNING),
        ):
            _run_post_rename_hook("http://remote.example.com/hook", old, new, {})

        assert any("plain http" in r.message.lower() or "unencrypted" in r.message.lower() for r in caplog.records)

    def test_hook_http_request_exception(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        """HTTP hook failure is logged, not raised."""
        import logging

        old = tmp_path / "old.pdf"
        new = tmp_path / "new.pdf"
        old.touch()
        new.touch()

        import requests as req

        mock_session = MagicMock()
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)
        mock_session.post.side_effect = req.ConnectionError("refused")

        with (
            patch("ai_pdf_renamer.renamer.requests.Session", return_value=mock_session),
            caplog.at_level(logging.WARNING),
        ):
            _run_post_rename_hook("https://example.com/hook", old, new, {})

        assert any("hook" in r.message.lower() and "failed" in r.message.lower() for r in caplog.records)

    def test_hook_general_exception(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        """General exception in hook is logged (lines 188-190)."""
        import logging

        old = tmp_path / "old.pdf"
        new = tmp_path / "new.pdf"
        old.touch()
        new.touch()

        with (
            patch("ai_pdf_renamer.renamer.subprocess.run", side_effect=RuntimeError("unexpected")),
            caplog.at_level(logging.WARNING),
        ):
            _run_post_rename_hook("some_command", old, new, {})

        assert any("hook failed" in r.message.lower() for r in caplog.records)


class TestShrinkDensityJump:
    """Test _shrink_to_token_limit density calculation + fine-tuning loop (lines 50, 54-57, 72-85)."""

    def test_shrink_density_jump(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Text exceeding token limit triggers density-based jump, yielding shorter output."""
        from ai_pdf_renamer import pdf_extract

        # Simulate tiktoken: 1 token per 4 chars (realistic density).
        # First call (full text): 250 tokens; subsequent calls: proportional to len.
        def fake_token_count(text: str) -> int:
            return len(text) // 4

        monkeypatch.setattr(pdf_extract, "_tiktoken_encoding", None)
        monkeypatch.setattr(pdf_extract, "_token_count", fake_token_count)

        text = "word " * 200  # 1000 chars -> 250 tokens
        result = pdf_extract._shrink_to_token_limit(text, max_tokens=50)
        # 50 tokens * 4 chars/token = ~200 chars target (with buffer)
        assert len(result) < len(text)
        assert len(result) <= 300  # density jump should get close to target


class TestPdfToTextMaxPages:
    """Test pdf_to_text max_pages limiting (line 114)."""

    def test_pdf_to_text_max_pages(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """With max_pages=2 and a 5-page doc, only 2 pages are extracted."""
        from ai_pdf_renamer import pdf_extract

        pages_accessed: list[int] = []

        mock_page = MagicMock()
        mock_page.get_text.return_value = "Page text content here."

        mock_doc = MagicMock()
        mock_doc.page_count = 5
        mock_doc.is_encrypted = False

        def getitem(self: Any, idx: int) -> Any:
            pages_accessed.append(idx)
            return mock_page

        mock_doc.__getitem__ = getitem
        mock_doc.load_page = lambda idx: (pages_accessed.append(idx), mock_page)[1]

        mock_fitz = MagicMock()
        mock_fitz.open.return_value = mock_doc
        monkeypatch.setitem(sys.modules, "fitz", mock_fitz)

        pdf_path = tmp_path / "five_pages.pdf"
        pdf_path.write_bytes(b"%PDF-1.4")

        result = pdf_extract.pdf_to_text(pdf_path, max_pages=2)
        assert "Page text content here." in result
        assert pages_accessed == [0, 1]
