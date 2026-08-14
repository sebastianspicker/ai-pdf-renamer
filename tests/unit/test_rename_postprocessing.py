"""Cover post-rename hooks, rename workflows, and output postprocessing."""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from folionym.config import RenamerConfig
from folionym.renamer import _produce_rename_results, rename_pdfs_in_directory
from folionym.renamer_hooks import PostRenameAction, _apply_post_rename_actions, _run_post_rename_hook
from folionym.renamer_output import _write_json_or_csv
from tests.conftest import make_config as _cfg
from tests.conftest import make_fake_pdf as _make_fake_pdf
from tests.conftest import make_hook_paths, make_http_hook_session


def _capture_processed_files() -> tuple[list[list[Path]], object]:
    """Return a result producer that records the ordering passed to the pipeline."""
    captured_files: list[list[Path]] = []

    def fake_produce(
        files: list[Path],
        config: RenamerConfig,
        rules: object = None,
        progress_callback: object | None = None,
    ) -> list[tuple[Path, str | None, dict[str, object] | None, BaseException | None]]:
        del config, rules, progress_callback
        captured_files.append(list(files))
        return [(file_path, None, None, None) for file_path in files]

    return captured_files, fake_produce


class TestHookShellDetection:
    """Tests 1-3: _run_post_rename_hook shell metachar detection and env vars."""

    def test_hook_shell_detection_pipe(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        """Command with '|' is rejected because local command hooks are disabled."""
        old, new = make_hook_paths(tmp_path)
        with caplog.at_level(logging.WARNING, logger="folionym.renamer"):
            _run_post_rename_hook("echo hello | cat", old, new, {"k": "v"})
        assert any("Local post-rename hook commands are disabled" in record.message for record in caplog.records)

    def test_hook_shell_detection_redirect(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        """Command with '>' is rejected because local command hooks are disabled."""
        old, new = make_hook_paths(tmp_path)
        with caplog.at_level(logging.WARNING, logger="folionym.renamer"):
            _run_post_rename_hook("echo hello > /dev/null", old, new, {})
        assert any("Local post-rename hook commands are disabled" in record.message for record in caplog.records)

    def test_hook_env_vars_set(self, tmp_path: Path) -> None:
        """Verify old_path and new_path are passed to HTTP hook payload."""
        old, new = make_hook_paths(tmp_path)
        mock_session = make_http_hook_session()

        with patch("folionym.renamer_hooks.requests.Session", return_value=mock_session):
            _run_post_rename_hook("https://example.invalid/hook", old, new, {"foo": "bar"})

        payload = mock_session.post.call_args.kwargs["json"]
        assert payload["old_path"] == str(old)
        assert payload["new_path"] == str(new)
        assert payload["meta"]["foo"] == "bar"


class TestHookMetaJsonFallback:
    """Edge case: meta with un-serializable values falls back to '{}'."""

    def test_hook_meta_unserializable(self, tmp_path: Path) -> None:
        old, new = make_hook_paths(tmp_path)
        # An object that cannot be serialized even with default=str
        bad_obj = object()
        # default=str handles arbitrary objects, but let's patch json.dumps to raise
        with patch("folionym.renamer_hooks.json.dumps", side_effect=[TypeError("test"), None]):
            _run_post_rename_hook("https://example.invalid/hook", old, new, {"bad": bad_obj})


class TestHookEmptyCmd:
    """Empty or whitespace-only hook command is a no-op."""

    def test_hook_empty_string(self, tmp_path: Path) -> None:
        old, new = make_hook_paths(tmp_path)
        _run_post_rename_hook("", old, new, {})

    def test_hook_whitespace_only(self, tmp_path: Path) -> None:
        old, new = make_hook_paths(tmp_path)
        _run_post_rename_hook("   ", old, new, {})


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
        _apply_post_rename_actions(config, PostRenameAction(file_path, target, "renamed", meta, export_rows))
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
        _apply_post_rename_actions(config, PostRenameAction(file_path, target, "renamed", {}, export_rows))
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
            patch("folionym.renamer._extract_pdf_content", side_effect=mock_extract),
            patch("folionym.renamer._process_content_to_result") as mock_process,
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

        captured_files, fake_produce = _capture_processed_files()

        with (
            patch("folionym.renamer.produce_rename_results", side_effect=fake_produce),
            patch("folionym.renamer.load_processing_rules", return_value=None),
            patch("folionym.renamer._collect_pdf_files", return_value=[old_pdf, new_pdf, mid_pdf]),
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

        captured_files, fake_produce = _capture_processed_files()

        original_stat = Path.stat

        def patched_stat(self_path: Path, *a: object, **kw: object) -> os.stat_result:
            if self_path.name == "bad.pdf":
                raise OSError("stat failed")
            return original_stat(self_path, *a, **kw)  # type: ignore[arg-type]

        with (
            patch("folionym.renamer.produce_rename_results", side_effect=fake_produce),
            patch("folionym.renamer.load_processing_rules", return_value=None),
            patch("folionym.renamer._collect_pdf_files", return_value=[bad_pdf, good_pdf]),
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
            patch("folionym.renamer._produce_rename_results", return_value=results),
            patch("folionym.renamer.load_processing_rules", return_value=None),
            patch("folionym.renamer._collect_pdf_files", return_value=[pdf]),
            patch("folionym.renamer._interactive_rename_prompt", return_value=("n", "new_name", pdf)),
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
