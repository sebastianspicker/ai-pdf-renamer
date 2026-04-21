# ruff: noqa: F401

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


def test_pdf_to_text_raises_on_open_error(monkeypatch) -> None:
    class DummyFitz:
        def open(self, path):
            raise RuntimeError("boom")

    monkeypatch.setitem(sys.modules, "fitz", DummyFitz())

    with pytest.raises(OSError, match="Could not open PDF file"):
        pdf_extract.pdf_to_text("missing.pdf")


def test_pdf_to_text_returns_empty_when_no_pages(monkeypatch, tmp_path) -> None:
    class DummyDoc:
        page_count = 0

    class DummyFitz:
        def open(self, path):
            return DummyDoc()

    monkeypatch.setitem(sys.modules, "fitz", DummyFitz())

    pdf_path = tmp_path / "empty.pdf"
    pdf_path.write_bytes(b"")

    assert pdf_extract.pdf_to_text(pdf_path) == ""


def test_shrink_to_token_limit_reduces_text(monkeypatch) -> None:
    monkeypatch.setattr(pdf_extract, "_token_count", lambda _t: 10_000)

    text = "a" * 500
    shrunk = pdf_extract._shrink_to_token_limit(text, max_tokens=10)

    assert len(shrunk) < len(text)
    assert len(shrunk) <= 200


def test_token_count_without_tiktoken(monkeypatch) -> None:
    """When tiktoken is unavailable, _token_count falls back to len//4."""
    # Force the cached encoding to None so the import path is re-entered.
    monkeypatch.setattr(pdf_extract, "_tiktoken_encoding", None)

    # Make 'import tiktoken' raise ImportError inside _token_count.
    original_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__  # type: ignore[union-attr]

    def _fake_import(name, *args, **kwargs):  # type: ignore[no-untyped-def]
        if name == "tiktoken":
            raise ImportError("no tiktoken")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", _fake_import)

    text = "a" * 400  # len=400, expected fallback = 400//4 = 100
    result = pdf_extract._token_count(text)
    assert result == 100

    # Restore to avoid polluting other tests.
    monkeypatch.setattr(pdf_extract, "_tiktoken_encoding", None)


def test_shrink_to_token_limit_already_under(monkeypatch) -> None:
    """Text already under the token limit is returned as-is."""
    monkeypatch.setattr(pdf_extract, "_token_count", lambda _t: 5)

    text = "Hello world"
    result = pdf_extract._shrink_to_token_limit(text, max_tokens=100)
    assert result == text


def test_shrink_to_token_limit_shrinks() -> None:
    """Text over the limit is truncated."""
    long_text = "word " * 20_000  # ~100K chars
    result = pdf_extract._shrink_to_token_limit(long_text, max_tokens=50)
    assert len(result) < len(long_text)


def test_pdf_to_text_none_path() -> None:
    """Passing None as filepath returns empty string."""
    assert pdf_extract.pdf_to_text(None) == ""


def test_pdf_to_text_encrypted_pdf(monkeypatch, tmp_path) -> None:
    """Encrypted PDF returns empty string."""
    mock_doc = MagicMock()
    mock_doc.is_encrypted = True
    mock_doc.page_count = 5

    mock_fitz = MagicMock()
    mock_fitz.open.return_value = mock_doc
    monkeypatch.setitem(sys.modules, "fitz", mock_fitz)

    pdf_path = tmp_path / "encrypted.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 encrypted")

    result = pdf_extract.pdf_to_text(pdf_path)
    assert result == ""


def test_pdf_to_text_empty_pages(monkeypatch, tmp_path) -> None:
    """PDF with 0 page_count returns empty string."""
    mock_doc = MagicMock()
    mock_doc.page_count = 0
    mock_doc.is_encrypted = False

    mock_fitz = MagicMock()
    mock_fitz.open.return_value = mock_doc
    monkeypatch.setitem(sys.modules, "fitz", mock_fitz)

    pdf_path = tmp_path / "zero_pages.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")

    result = pdf_extract.pdf_to_text(pdf_path)
    assert result == ""


def test_pdf_to_text_successful(monkeypatch, tmp_path) -> None:
    """PDF with a page returning text gives that text back."""
    expected_text = "This is a test document about machine learning."

    mock_page = MagicMock()
    mock_page.get_text.return_value = expected_text

    mock_doc = MagicMock()
    mock_doc.page_count = 1
    mock_doc.is_encrypted = False
    mock_doc.__getitem__ = MagicMock(return_value=mock_page)
    mock_doc.load_page = MagicMock(return_value=mock_page)

    mock_fitz = MagicMock()
    mock_fitz.open.return_value = mock_doc
    monkeypatch.setitem(sys.modules, "fitz", mock_fitz)

    pdf_path = tmp_path / "good.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")

    result = pdf_extract.pdf_to_text(pdf_path)
    assert expected_text in result


def test_vision_no_fitz(monkeypatch) -> None:
    """When fitz import fails, returns None."""
    # Remove fitz from sys.modules so the import inside the function fails.
    monkeypatch.delitem(sys.modules, "fitz", raising=False)

    original_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__  # type: ignore[union-attr]

    def _fake_import(name, *args, **kwargs):  # type: ignore[no-untyped-def]
        if name == "fitz":
            raise ImportError("no fitz")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", _fake_import)

    result = pdf_extract.pdf_first_page_to_image_base64("/tmp/test.pdf")
    assert result is None


def test_vision_encrypted(monkeypatch, tmp_path) -> None:
    """Encrypted PDF returns None for vision render."""
    mock_doc = MagicMock()
    mock_doc.is_encrypted = True
    mock_doc.page_count = 1

    mock_fitz = MagicMock()
    mock_fitz.open.return_value = mock_doc
    monkeypatch.setitem(sys.modules, "fitz", mock_fitz)

    pdf_path = tmp_path / "enc.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")

    result = pdf_extract.pdf_first_page_to_image_base64(pdf_path)
    assert result is None


def test_vision_success(monkeypatch, tmp_path) -> None:
    """Successful vision render returns base64-encoded string."""
    fake_image_bytes = b"\xff\xd8\xff\xe0JFIF-fake-jpeg-data"

    mock_pix = MagicMock()
    mock_pix.tobytes.return_value = fake_image_bytes

    mock_page = MagicMock()
    mock_page.get_pixmap.return_value = mock_pix

    mock_doc = MagicMock()
    mock_doc.is_encrypted = False
    mock_doc.page_count = 1
    mock_doc.load_page.return_value = mock_page

    mock_fitz = MagicMock()
    mock_fitz.open.return_value = mock_doc
    monkeypatch.setitem(sys.modules, "fitz", mock_fitz)

    pdf_path = tmp_path / "render.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")

    result = pdf_extract.pdf_first_page_to_image_base64(pdf_path)
    assert result is not None
    # Verify it's valid base64 that decodes back to original bytes.
    decoded = base64.b64decode(result)
    assert decoded == fake_image_bytes


def test_ocr_no_ocrmypdf(monkeypatch, tmp_path, caplog) -> None:
    """When ocrmypdf is not installed, a warning is logged and original text returned."""
    # Make pdf_to_text return short text (below MIN_CHARS_BEFORE_OCR).
    monkeypatch.setattr(pdf_extract, "pdf_to_text", lambda *a, **kw: "Hi")

    # Make 'import ocrmypdf' fail.
    monkeypatch.delitem(sys.modules, "ocrmypdf", raising=False)
    original_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__  # type: ignore[union-attr]

    def _fake_import(name, *args, **kwargs):  # type: ignore[no-untyped-def]
        if name == "ocrmypdf":
            raise ImportError("no ocrmypdf")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", _fake_import)

    pdf_path = tmp_path / "needs_ocr.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")

    with caplog.at_level(logging.WARNING):
        result = pdf_extract.pdf_to_text_with_ocr(pdf_path)

    assert result == "Hi"
    assert any("ocrmypdf not installed" in r.message for r in caplog.records)


def test_ocr_success(monkeypatch, tmp_path) -> None:
    """Successful OCR produces text from the OCR'd PDF."""
    # Make pdf_to_text return short text first (triggers OCR), then OCR'd text.
    call_count = 0

    def fake_pdf_to_text(*args, **kwargs):  # type: ignore[no-untyped-def]
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return "Hi"  # Too short, triggers OCR.
        return "Full OCR extracted text from the document."

    monkeypatch.setattr(pdf_extract, "pdf_to_text", fake_pdf_to_text)

    # Mock ocrmypdf.ocr to just create the output file.
    mock_ocrmypdf = MagicMock()

    def fake_ocr(input_path, output_path, **kwargs):  # type: ignore[no-untyped-def]
        Path(output_path).write_bytes(b"%PDF-1.4 ocr output")

    mock_ocrmypdf.ocr = fake_ocr
    monkeypatch.setitem(sys.modules, "ocrmypdf", mock_ocrmypdf)

    pdf_path = tmp_path / "image_only.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")

    result = pdf_extract.pdf_to_text_with_ocr(pdf_path)
    assert result == "Full OCR extracted text from the document."


def test_ocr_failure(monkeypatch, tmp_path, caplog) -> None:
    """OCR failure logs a warning and returns original text."""
    monkeypatch.setattr(pdf_extract, "pdf_to_text", lambda *a, **kw: "Hi")

    mock_ocrmypdf = MagicMock()
    mock_ocrmypdf.ocr.side_effect = RuntimeError("Tesseract not found")
    monkeypatch.setitem(sys.modules, "ocrmypdf", mock_ocrmypdf)

    pdf_path = tmp_path / "ocr_fail.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")

    with caplog.at_level(logging.WARNING):
        result = pdf_extract.pdf_to_text_with_ocr(pdf_path)

    assert result == "Hi"
    assert any("OCR failed" in r.message for r in caplog.records)


def test_parse_pdf_date_valid() -> None:
    """Valid D:YYYYMMDD string is parsed to a date."""
    result = pdf_extract._parse_pdf_date("D:20250315120000")
    assert result == date(2025, 3, 15)


def test_parse_pdf_date_invalid() -> None:
    """Invalid date values return None."""
    result = pdf_extract._parse_pdf_date("D:99999999")
    assert result is None


def test_parse_pdf_date_none() -> None:
    """None input returns None."""
    result = pdf_extract._parse_pdf_date(None)
    assert result is None


def test_get_pdf_metadata_no_fitz(monkeypatch) -> None:
    """When fitz is not available, returns default metadata dict."""
    monkeypatch.delitem(sys.modules, "fitz", raising=False)

    original_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__  # type: ignore[union-attr]

    def _fake_import(name, *args, **kwargs):  # type: ignore[no-untyped-def]
        if name == "fitz":
            raise ImportError("no fitz")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", _fake_import)

    result = pdf_extract.get_pdf_metadata("/tmp/test.pdf")
    assert result["title"] == ""
    assert result["author"] == ""
    assert result["creation_date"] is None
    assert result["mod_date"] is None


def test_extract_pages_text_mode() -> None:
    """First get_text('text') succeeds — returns that text."""
    expected = "Page one text content."
    mock_page = MagicMock()
    mock_page.get_text.return_value = expected

    mock_doc = MagicMock()
    mock_doc.page_count = 1
    mock_doc.__getitem__ = MagicMock(return_value=mock_page)
    mock_doc.load_page = MagicMock(return_value=mock_page)

    pieces, errors = pdf_extract._extract_pages(mock_doc, Path("/tmp/test.pdf"))
    assert pieces == [expected]
    assert errors == []


def test_extract_pages_empty_text_no_fallback() -> None:
    """When get_text('text') returns empty, no text is extracted (no blocks/rawdict fallback)."""
    mock_page = MagicMock()
    mock_page.get_text.return_value = ""

    mock_doc = MagicMock()
    mock_doc.page_count = 1
    mock_doc.load_page = MagicMock(return_value=mock_page)

    pieces, errors = pdf_extract._extract_pages(mock_doc, Path("/tmp/test.pdf"))
    assert pieces == []
    assert errors == []


def test_extract_pages_text_failure() -> None:
    """When text extraction raises, an error is recorded."""
    mock_page = MagicMock()
    mock_page.get_text.side_effect = RuntimeError("extraction failed")

    mock_doc = MagicMock()
    mock_doc.page_count = 1
    mock_doc.load_page = MagicMock(return_value=mock_page)

    pieces, errors = pdf_extract._extract_pages(mock_doc, Path("/tmp/test.pdf"))
    assert pieces == []
    assert len(errors) == 1
    assert "extraction failed" in errors[0]


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
