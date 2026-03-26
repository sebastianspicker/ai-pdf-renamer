"""Round 8: Targeted coverage tests for renamer.py, heuristics.py, and llm.py.

Pushes coverage for these three core modules toward 88-90%+ by exercising
previously-uncovered branches: hook shell detection, export row building,
prefetch exception handling, directory validation, mtime sorting, interactive
manual_mode, watch loop mtime tracking, heuristic rule loading (language/parent),
locale file loading, category alias error paths, embedding conflict fallback,
keyword overlap scoring, LLM json_mode retry, allowed_categories prompt,
and multi-chunk summary paths.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

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

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


# ===========================================================================
# renamer.py tests
# ===========================================================================


class TestHookShellDetection:
    """Tests 1-3: _run_post_rename_hook shell metachar detection and env vars."""

    def test_hook_shell_detection_pipe(self, tmp_path: Path) -> None:
        """Command with '|' is detected as needing a shell."""
        old = tmp_path / "old.pdf"
        new = tmp_path / "new.pdf"
        old.touch()
        new.touch()
        with patch("ai_pdf_renamer.renamer.subprocess.run") as mock_run:
            _run_post_rename_hook("echo hello | cat", old, new, {"k": "v"})
            mock_run.assert_called_once()
            args = mock_run.call_args
            cmd_list = args[0][0]
            # On POSIX, shell metachar triggers [shell, "-c", cmd]
            if os.name != "nt":
                shell = os.environ.get("SHELL", "/bin/sh")
                assert cmd_list[0] == shell
                assert cmd_list[1] == "-c"
                assert cmd_list[2] == "echo hello | cat"
            else:
                # On Windows, COMSPEC is used
                assert "/c" in cmd_list

    def test_hook_shell_detection_redirect(self, tmp_path: Path) -> None:
        """Command with '>' is detected as needing a shell."""
        old = tmp_path / "old.pdf"
        new = tmp_path / "new.pdf"
        old.touch()
        new.touch()
        with patch("ai_pdf_renamer.renamer.subprocess.run") as mock_run:
            _run_post_rename_hook("echo hello > /dev/null", old, new, {})
            mock_run.assert_called_once()
            args = mock_run.call_args
            cmd_list = args[0][0]
            if os.name != "nt":
                shell = os.environ.get("SHELL", "/bin/sh")
                assert cmd_list[0] == shell
                assert "-c" in cmd_list

    def test_hook_env_vars_set(self, tmp_path: Path) -> None:
        """Verify OLD_PATH and NEW_PATH env vars are passed to subprocess."""
        old = tmp_path / "old.pdf"
        new = tmp_path / "new.pdf"
        old.touch()
        new.touch()
        with patch("ai_pdf_renamer.renamer.subprocess.run") as mock_run:
            _run_post_rename_hook("echo test", old, new, {"foo": "bar"})
            mock_run.assert_called_once()
            env = mock_run.call_args[1]["env"]
            assert env["AI_PDF_RENAMER_OLD_PATH"] == str(old)
            assert env["AI_PDF_RENAMER_NEW_PATH"] == str(new)
            assert "AI_PDF_RENAMER_META" in env
            meta = json.loads(env["AI_PDF_RENAMER_META"])
            assert meta["foo"] == "bar"


class TestHookMetaJsonFallback:
    """Edge case: meta with un-serializable values falls back to '{}'."""

    def test_hook_meta_unserializable(self, tmp_path: Path) -> None:
        old = tmp_path / "old.pdf"
        new = tmp_path / "new.pdf"
        old.touch()
        new.touch()
        # An object that cannot be serialized even with default=str
        bad_obj = object()
        # default=str handles arbitrary objects, but let's patch json.dumps to raise
        with (
            patch("ai_pdf_renamer.renamer.json.dumps", side_effect=[TypeError("test"), None]),
            patch("ai_pdf_renamer.renamer.subprocess.run"),
        ):
            _run_post_rename_hook("echo test", old, new, {"bad": bad_obj})


class TestHookEmptyCmd:
    """Empty or whitespace-only hook command is a no-op."""

    def test_hook_empty_string(self, tmp_path: Path) -> None:
        old = tmp_path / "old.pdf"
        new = tmp_path / "new.pdf"
        old.touch()
        new.touch()
        with patch("ai_pdf_renamer.renamer.subprocess.run") as mock_run:
            _run_post_rename_hook("", old, new, {})
            mock_run.assert_not_called()

    def test_hook_whitespace_only(self, tmp_path: Path) -> None:
        old = tmp_path / "old.pdf"
        new = tmp_path / "new.pdf"
        old.touch()
        new.touch()
        with patch("ai_pdf_renamer.renamer.subprocess.run") as mock_run:
            _run_post_rename_hook("   ", old, new, {})
            mock_run.assert_not_called()


class TestApplyPostRenameActions:
    """Test 4: _apply_post_rename_actions builds export row."""

    def test_builds_export_row(self, tmp_path: Path) -> None:
        """Verify export_rows list gets a new entry with expected fields."""
        config = _cfg(export_metadata_path=str(tmp_path / "export.json"))
        file_path = tmp_path / "doc.pdf"
        target = tmp_path / "renamed.pdf"
        file_path.touch()
        target.touch()
        meta: dict[str, object] = {
            "category": "invoice",
            "summary": "An invoice",
            "keywords": "money,pay",
            "category_source": "heuristic",
            "llm_failed": False,
            "used_vision_fallback": False,
            "invoice_id": "INV-001",
            "amount": "100.00",
            "company": "ACME",
        }
        export_rows: list[dict[str, object]] = []
        _apply_post_rename_actions(config, file_path, target, "renamed", meta, export_rows)
        assert len(export_rows) == 1
        row = export_rows[0]
        assert row["path"] == str(file_path)
        assert row["new_name"] == target.name
        assert row["category"] == "invoice"
        assert row["invoice_id"] == "INV-001"

    def test_builds_export_row_with_missing_meta_keys(self, tmp_path: Path) -> None:
        """Meta dict with no keys still creates row with empty defaults."""
        config = _cfg(export_metadata_path=str(tmp_path / "export.json"))
        file_path = tmp_path / "doc.pdf"
        target = tmp_path / "renamed.pdf"
        file_path.touch()
        target.touch()
        export_rows: list[dict[str, object]] = []
        _apply_post_rename_actions(config, file_path, target, "renamed", {}, export_rows)
        assert len(export_rows) == 1
        row = export_rows[0]
        assert row["category"] == ""
        assert row["invoice_id"] == ""


class TestProduceResultsPrefetchException:
    """Test 5: prefetch raises exception, verify processing continues to next file."""

    def test_prefetch_exception_continues(self, tmp_path: Path) -> None:
        f1 = _make_fake_pdf(tmp_path, "a.pdf")
        f2 = _make_fake_pdf(tmp_path, "b.pdf")
        config = _cfg(workers=1, interactive=False)

        call_count = 0

        def mock_extract(path: Path, cfg: RenamerConfig) -> tuple[str, bool]:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # First file extraction succeeds
                return ("some content for testing the pipeline " * 5, False)
            # Second file (prefetched) raises
            raise OSError("Disk error on prefetch")

        with (
            patch("ai_pdf_renamer.renamer._extract_pdf_content", side_effect=mock_extract),
            patch("ai_pdf_renamer.renamer._process_content_to_result") as mock_process,
        ):
            mock_process.return_value = (f1, "new_name", {"category": "test"}, None)
            results = _produce_rename_results([f1, f2], config)

        # Should have results for both files (second one with the exception)
        assert len(results) == 2
        # First result should be processed OK
        assert results[0][1] == "new_name"
        # Second result should have an exception from the prefetch
        assert results[1][3] is not None


class TestRenamePdfsDirectoryValidation:
    """Tests 6-7: rename_pdfs_in_directory raises for nonexistent/not-a-dir paths."""

    def test_nonexistent_dir(self, tmp_path: Path) -> None:
        """Verify FileNotFoundError for nonexistent directory."""
        config = _cfg()
        with pytest.raises(FileNotFoundError, match="does not exist"):
            rename_pdfs_in_directory(tmp_path / "no_such_dir", config=config)

    def test_not_a_dir(self, tmp_path: Path) -> None:
        """Verify NotADirectoryError when path is a file."""
        f = tmp_path / "file.txt"
        f.write_text("not a dir")
        config = _cfg()
        with pytest.raises(NotADirectoryError, match="Not a directory"):
            rename_pdfs_in_directory(f, config=config)

    def test_empty_dir_string(self) -> None:
        """Verify ValueError for empty dir string."""
        config = _cfg()
        with pytest.raises(ValueError, match="non-empty"):
            rename_pdfs_in_directory("", config=config)


class TestRenamePdfsMtimeSort:
    """Test 8: verify files are sorted by mtime (newest first)."""

    def test_mtime_sort(self, tmp_path: Path) -> None:
        """Files should be sorted newest first by mtime."""
        now = time.time()
        # Create PDFs with different mtimes
        old_pdf = _make_fake_pdf(tmp_path, "old.pdf", mtime=now - 100)
        new_pdf = _make_fake_pdf(tmp_path, "new.pdf", mtime=now)
        mid_pdf = _make_fake_pdf(tmp_path, "mid.pdf", mtime=now - 50)

        config = _cfg()

        # Track which files get passed to _produce_rename_results
        captured_files: list[list[Path]] = []

        def fake_produce(
            files: list[Path], config: RenamerConfig, rules: object = None
        ) -> list[tuple[Path, str | None, dict[str, object] | None, BaseException | None]]:
            captured_files.append(list(files))
            return [(f, None, None, None) for f in files]

        with (
            patch("ai_pdf_renamer.renamer._produce_rename_results", side_effect=fake_produce),
            patch("ai_pdf_renamer.renamer.load_processing_rules", return_value=None),
            patch("ai_pdf_renamer.renamer._collect_pdf_files", return_value=[old_pdf, new_pdf, mid_pdf]),
        ):
            rename_pdfs_in_directory(tmp_path, config=config)

        assert len(captured_files) == 1
        order = captured_files[0]
        # Newest first
        assert order[0] == new_pdf
        assert order[1] == mid_pdf
        assert order[2] == old_pdf

    def test_mtime_sort_oserror(self, tmp_path: Path) -> None:
        """Files whose stat() raises OSError get mtime 0.0 (sorted last)."""
        now = time.time()
        good_pdf = _make_fake_pdf(tmp_path, "good.pdf", mtime=now)
        bad_pdf = _make_fake_pdf(tmp_path, "bad.pdf", mtime=now - 10)

        config = _cfg()

        captured_files: list[list[Path]] = []

        def fake_produce(
            files: list[Path], config: RenamerConfig, rules: object = None
        ) -> list[tuple[Path, str | None, dict[str, object] | None, BaseException | None]]:
            captured_files.append(list(files))
            return [(f, None, None, None) for f in files]

        original_stat = Path.stat

        def patched_stat(self_path: Path, *a: object, **kw: object) -> os.stat_result:
            if self_path.name == "bad.pdf":
                raise OSError("stat failed")
            return original_stat(self_path, *a, **kw)  # type: ignore[arg-type]

        with (
            patch("ai_pdf_renamer.renamer._produce_rename_results", side_effect=fake_produce),
            patch("ai_pdf_renamer.renamer.load_processing_rules", return_value=None),
            patch("ai_pdf_renamer.renamer._collect_pdf_files", return_value=[bad_pdf, good_pdf]),
            patch.object(Path, "stat", patched_stat),
        ):
            rename_pdfs_in_directory(tmp_path, config=config)

        assert len(captured_files) == 1
        # good.pdf (has mtime) should be before bad.pdf (mtime=0.0)
        assert captured_files[0][0] == good_pdf


class TestInteractiveModeManualPrints:
    """Test 9: interactive + manual_mode prints 'Suggested:'."""

    def test_manual_mode_prints_suggested(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        """Verify 'Suggested:' is printed when interactive+manual_mode are set."""
        pdf = _make_fake_pdf(tmp_path, "doc.pdf")
        config = _cfg(interactive=True, manual_mode=True)

        meta = {"category": "invoice", "summary": "A test invoice", "keywords": "test", "category_source": "heuristic"}

        results = [(pdf, "new_name", meta, None)]

        with (
            patch("ai_pdf_renamer.renamer._produce_rename_results", return_value=results),
            patch("ai_pdf_renamer.renamer.load_processing_rules", return_value=None),
            patch("ai_pdf_renamer.renamer._collect_pdf_files", return_value=[pdf]),
            patch("ai_pdf_renamer.renamer._interactive_rename_prompt", return_value=("n", "new_name", pdf)),
        ):
            rename_pdfs_in_directory(tmp_path, config=config)

        captured = capsys.readouterr()
        assert "Suggested: new_name.pdf" in captured.out
        assert "category: invoice" in captured.out


class TestWriteJsonOrCsvSanitization:
    """_write_json_or_csv CSV sanitization during write."""

    def test_csv_sanitize_formula_injection(self, tmp_path: Path) -> None:
        """CSV cells starting with = are prefixed with '."""
        out = tmp_path / "out.csv"
        rows = [{"a": "=cmd()", "b": "normal"}]
        _write_json_or_csv(out, rows, ["a", "b"])
        content = out.read_text()
        assert "'=cmd()" in content
        assert "normal" in content

    def test_json_fallback(self, tmp_path: Path) -> None:
        """Non-CSV suffix writes JSON."""
        out = tmp_path / "out.json"
        rows = [{"key": "value"}]
        _write_json_or_csv(out, rows, None)
        data = json.loads(out.read_text())
        assert data[0]["key"] == "value"


class TestWatchLoopMtimeTracking:
    """Test 10: verify second iteration skips unchanged files."""

    def test_watch_loop_skips_unchanged(self, tmp_path: Path) -> None:
        """Second iteration with no mtime change should not process any files."""
        pdf = _make_fake_pdf(tmp_path, "test.pdf")

        config = _cfg()

        iteration = 0
        processed_counts: list[int] = []

        def fake_rename(directory: object, *, config: RenamerConfig, files_override: list[Path] | None = None) -> None:
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


# ===========================================================================
# heuristics.py tests
# ===========================================================================


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


# ===========================================================================
# llm.py tests
# ===========================================================================


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

        assert any("plain HTTP" in r.message.lower() or "unencrypted" in r.message.lower() for r in caplog.records)

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
