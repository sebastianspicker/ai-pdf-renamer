"""Cover extraction helpers, heuristic loading, and metadata edge cases."""

from __future__ import annotations

import json
import logging
import re
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from folionym import pdf_extract
from folionym.heuristics import (
    ConflictResolutionOptions,
    HeuristicRule,
    HeuristicScorer,
    _load_category_aliases,
    clear_category_aliases_cache,
    load_heuristic_rules,
    load_heuristic_rules_for_language,
)
from folionym.heuristics import (
    _combine_resolve_conflict as _resolve_conflict,
)
from folionym.renamer import rename_pdfs_in_directory
from folionym.renamer_hooks import run_post_rename_hook
from tests.conftest import make_config as _cfg
from tests.conftest import make_fake_pdf as _make_fake_pdf
from tests.conftest import make_hook_paths, make_http_hook_session


def _combine_resolve_conflict(cat_llm: str, cat_heuristic: str, **values: object) -> str:
    return _resolve_conflict(cat_llm, cat_heuristic, ConflictResolutionOptions(**values))


def _run_rename_scenario(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    *,
    dry_run: bool = False,
    rename_result: tuple[bool, Path] | BaseException,
) -> None:
    """Run a one-file rename result through the reporting path."""
    pdf = _make_fake_pdf(tmp_path, "doc.pdf")
    results = [(pdf, "new_name", {"category": "test"}, None)]
    patch_kwargs = (
        {"side_effect": rename_result} if isinstance(rename_result, BaseException) else {"return_value": rename_result}
    )
    with (
        patch("folionym.renamer._produce_rename_results", return_value=results),
        patch("folionym.renamer.load_processing_rules", return_value=None),
        patch("folionym.renamer._collect_pdf_files", return_value=[pdf]),
        patch("folionym.renamer.apply_single_rename", **patch_kwargs),
        caplog.at_level(logging.INFO if dry_run else logging.ERROR),
    ):
        rename_pdfs_in_directory(tmp_path, config=_cfg(dry_run=dry_run))


class TestDryRunAndRenameFailureReporting:
    """Cover dry-run logging and rename failure reporting branches."""

    def test_dry_run_logging(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        """Dry-run logs 'would rename' message."""
        _run_rename_scenario(tmp_path, caplog, dry_run=True, rename_result=(True, tmp_path / "new_name.pdf"))

        assert any("Dry-run" in r.message or "would rename" in r.message for r in caplog.records)

    def test_rename_failure_reporting(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        """When apply_single_rename returns success=False, error is logged."""
        _run_rename_scenario(tmp_path, caplog, rename_result=(False, tmp_path / "doc.pdf"))

        assert any("could not rename" in r.message.lower() for r in caplog.records)


class TestRenameApplyException:
    """Cover exception path in apply_single_rename wrapper."""

    def test_apply_raises_exception(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        """When apply_single_rename raises, failure is recorded."""
        _run_rename_scenario(tmp_path, caplog, rename_result=PermissionError("denied"))

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
        try:
            clear_category_aliases_cache()
            with patch("folionym.data_paths.category_aliases_path", return_value=tmp_path / "nonexistent.json"):
                result = _load_category_aliases()
            assert result == {}
        finally:
            clear_category_aliases_cache()

    def test_aliases_invalid_json(self, tmp_path: Path) -> None:
        """Invalid JSON in aliases file, verify fallback to empty."""
        try:
            clear_category_aliases_cache()
            bad_file = tmp_path / "category_aliases.json"
            bad_file.write_text("NOT JSON")
            with patch("folionym.data_paths.category_aliases_path", return_value=bad_file):
                result = _load_category_aliases()
            assert result == {}
        finally:
            clear_category_aliases_cache()

    def test_aliases_not_dict(self, tmp_path: Path) -> None:
        """aliases key is not a dict, verify empty."""
        try:
            clear_category_aliases_cache()
            bad_file = tmp_path / "category_aliases.json"
            bad_file.write_text(json.dumps({"aliases": "not a dict"}))
            with patch("folionym.data_paths.category_aliases_path", return_value=bad_file):
                result = _load_category_aliases()
            assert result == {}
        finally:
            clear_category_aliases_cache()


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

        rules = [
            HeuristicRule(pattern=re.compile(r"invoice", re.I), category="invoice", score=10.0),
            HeuristicRule(pattern=re.compile(r"contract", re.I), category="contract", score=5.0),
            HeuristicRule(pattern=re.compile(r"letter", re.I), category="letter", score=3.0),
        ]
        scorer = HeuristicScorer(rules=rules)
        with caplog.at_level(logging.DEBUG, logger="folionym.heuristics"):
            result = scorer.best_category_with_confidence("This is an invoice about a contract and a letter")
        assert result[0] == "invoice"
        assert any("top-3" in r.message.lower() for r in caplog.records)


class TestHookHTTPPost:
    """Cover HTTP POST hook path (lines 172-173, 180, 188-190)."""

    def test_hook_http_post(self, tmp_path: Path) -> None:
        """HTTP hook sends JSON payload with old_path, new_path, meta."""
        old, new = make_hook_paths(tmp_path)
        mock_session = make_http_hook_session()

        with patch("folionym.renamer_hooks.requests.Session", return_value=mock_session):
            run_post_rename_hook("https://example.com/hook", old, new, {"k": "v"})

        mock_session.post.assert_called_once()
        call_kwargs = mock_session.post.call_args
        assert call_kwargs[1]["json"]["old_path"] == str(old)
        assert call_kwargs[1]["json"]["new_path"] == str(new)
        assert call_kwargs[1]["json"]["meta"]["k"] == "v"

    def test_hook_http_non_loopback_warning(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        """Plain HTTP to non-loopback host logs a warning."""

        old, new = make_hook_paths(tmp_path)
        mock_session = make_http_hook_session()

        with (
            patch("folionym.renamer_hooks.requests.Session", return_value=mock_session),
            caplog.at_level(logging.WARNING),
        ):
            run_post_rename_hook("http://remote.example.com/hook", old, new, {})

        assert any("plain http" in r.message.lower() or "unencrypted" in r.message.lower() for r in caplog.records)

    def test_hook_http_request_exception(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        """HTTP hook failure is logged, not raised."""

        import requests as req

        old, new = make_hook_paths(tmp_path)
        mock_session = make_http_hook_session(post_side_effect=req.ConnectionError("refused"))

        with (
            patch("folionym.renamer_hooks.requests.Session", return_value=mock_session),
            caplog.at_level(logging.WARNING),
        ):
            run_post_rename_hook("https://example.com/hook", old, new, {})

        assert any("hook" in r.message.lower() and "failed" in r.message.lower() for r in caplog.records)

    def test_local_command_hook_is_rejected(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        """Local command hook values are rejected."""

        old, new = make_hook_paths(tmp_path)

        with caplog.at_level(logging.WARNING, logger="folionym.renamer"):
            run_post_rename_hook("some_command", old, new, {})

        assert any("commands are disabled" in r.message.lower() for r in caplog.records)


class TestShrinkDensityJump:
    """Test _shrink_to_token_limit density calculation + fine-tuning loop (lines 50, 54-57, 72-85)."""

    def test_shrink_density_jump(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Text exceeding token limit triggers density-based jump, yielding shorter output."""

        # Simulate tiktoken: 1 token per 4 chars (realistic density).
        # First call (full text): 250 tokens; subsequent calls: proportional to len.
        def fake_token_count(text: str) -> int:
            return len(text) // 4

        monkeypatch.setattr(pdf_extract, "_tiktoken_encoding", None)
        monkeypatch.setattr(pdf_extract, "_token_count", fake_token_count)

        text = "word " * 200  # 1000 chars -> 250 tokens
        result = pdf_extract.shrink_to_token_limit(text, max_tokens=50)
        # 50 tokens * 4 chars/token = ~200 chars target (with buffer)
        assert len(result) < len(text)
        assert len(result) <= 300  # density jump should get close to target


class TestPdfToTextMaxPages:
    """Test pdf_to_text max_pages limiting (line 114)."""

    def test_pdf_to_text_max_pages(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """With max_pages=2 and a 5-page doc, only 2 pages are extracted."""

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
