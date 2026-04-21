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


class TestPdfToTextRaisesOnExtractionError:
    """Test that RuntimeError is raised when all pages fail (line 128)."""

    def test_pdf_to_text_raises_on_extraction_error(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """All pages fail extraction -> RuntimeError with error details."""
        from ai_pdf_renamer import pdf_extract

        mock_page = MagicMock()
        # All get_text calls raise, triggering error recording on all methods.
        mock_page.get_text.side_effect = RuntimeError("extraction failed")

        mock_doc = MagicMock()
        mock_doc.page_count = 1
        mock_doc.is_encrypted = False
        mock_doc.__getitem__ = MagicMock(return_value=mock_page)
        mock_doc.load_page = MagicMock(return_value=mock_page)

        mock_fitz = MagicMock()
        mock_fitz.open.return_value = mock_doc
        monkeypatch.setitem(sys.modules, "fitz", mock_fitz)

        pdf_path = tmp_path / "broken.pdf"
        pdf_path.write_bytes(b"%PDF-1.4")

        with pytest.raises(RuntimeError, match="Extraction failed"):
            pdf_extract.pdf_to_text(pdf_path)


class TestVisionRenderJpegSuccess:
    """Test vision render JPEG success path (lines 183-184)."""

    def test_vision_render_jpeg_success(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """tobytes(output='jpeg') succeeds -> base64-encoded result."""
        from ai_pdf_renamer import pdf_extract

        fake_jpeg = b"\xff\xd8\xff\xe0JFIF-test-data"

        mock_pix = MagicMock(spec=["tobytes"])
        mock_pix.tobytes.return_value = fake_jpeg

        mock_page = MagicMock()
        mock_page.get_pixmap.return_value = mock_pix

        mock_doc = MagicMock()
        mock_doc.is_encrypted = False
        mock_doc.page_count = 1
        mock_doc.load_page.return_value = mock_page

        mock_fitz = MagicMock()
        mock_fitz.open.return_value = mock_doc
        monkeypatch.setitem(sys.modules, "fitz", mock_fitz)

        pdf_path = tmp_path / "render_jpeg.pdf"
        pdf_path.write_bytes(b"%PDF-1.4")

        result = pdf_extract.pdf_first_page_to_image_base64(pdf_path)
        assert result is not None
        assert base64.b64decode(result) == fake_jpeg
        mock_pix.tobytes.assert_called_with(output="jpeg", jpg_quality=85)


class TestVisionRenderJpegTypeErrorPngFallback:
    """Test tobytes('jpeg') raises TypeError, falls to PNG (lines 185-186)."""

    def test_vision_render_jpeg_type_error_png_fallback(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """tobytes('jpeg') raises TypeError -> falls back to tobytes('png')."""
        from ai_pdf_renamer import pdf_extract

        fake_png = b"\x89PNG-test-data"

        mock_pix = MagicMock(spec=["tobytes"])

        def tobytes_side_effect(output: str = "png", **kwargs: Any) -> bytes:
            if output == "jpeg":
                raise TypeError("JPEG not supported")
            return fake_png

        mock_pix.tobytes.side_effect = tobytes_side_effect

        mock_page = MagicMock()
        mock_page.get_pixmap.return_value = mock_pix

        mock_doc = MagicMock()
        mock_doc.is_encrypted = False
        mock_doc.page_count = 1
        mock_doc.load_page.return_value = mock_page

        mock_fitz = MagicMock()
        mock_fitz.open.return_value = mock_doc
        monkeypatch.setitem(sys.modules, "fitz", mock_fitz)

        pdf_path = tmp_path / "render_png_fallback.pdf"
        pdf_path.write_bytes(b"%PDF-1.4")

        result = pdf_extract.pdf_first_page_to_image_base64(pdf_path)
        assert result is not None
        assert base64.b64decode(result) == fake_png


class TestVisionRenderGetPNGDataFallback:
    """Test getPNGData fallback when tobytes and getImageData are absent (lines 189-190)."""

    def test_vision_render_getpngdata_fallback(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """No tobytes, no getImageData -> getPNGData is used."""
        from ai_pdf_renamer import pdf_extract

        fake_png = b"\x89PNG-via-getPNGData"

        # Create a pix object with only getPNGData
        mock_pix = MagicMock(spec=["getPNGData"])
        mock_pix.getPNGData.return_value = fake_png

        mock_page = MagicMock()
        mock_page.get_pixmap.return_value = mock_pix

        mock_doc = MagicMock()
        mock_doc.is_encrypted = False
        mock_doc.page_count = 1
        mock_doc.load_page.return_value = mock_page

        mock_fitz = MagicMock()
        mock_fitz.open.return_value = mock_doc
        monkeypatch.setitem(sys.modules, "fitz", mock_fitz)

        pdf_path = tmp_path / "render_getpngdata.pdf"
        pdf_path.write_bytes(b"%PDF-1.4")

        result = pdf_extract.pdf_first_page_to_image_base64(pdf_path)
        assert result is not None
        assert base64.b64decode(result) == fake_png
        mock_pix.getPNGData.assert_called_once()


class TestVisionRenderNoMethods:
    """Test vision render returns None when no rendering methods are available (lines 191-192)."""

    def test_vision_render_no_methods(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Pix object has no tobytes/getImageData/getPNGData -> returns None."""
        from ai_pdf_renamer import pdf_extract

        # spec=[] means no attributes at all -> hasattr checks all return False
        mock_pix = MagicMock(spec=[])

        mock_page = MagicMock()
        mock_page.get_pixmap.return_value = mock_pix

        mock_doc = MagicMock()
        mock_doc.is_encrypted = False
        mock_doc.page_count = 1
        mock_doc.load_page.return_value = mock_page

        mock_fitz = MagicMock()
        mock_fitz.open.return_value = mock_doc
        monkeypatch.setitem(sys.modules, "fitz", mock_fitz)

        pdf_path = tmp_path / "render_no_methods.pdf"
        pdf_path.write_bytes(b"%PDF-1.4")

        result = pdf_extract.pdf_first_page_to_image_base64(pdf_path)
        assert result is None


class TestVisionRenderEmptyBytes:
    """Test vision render returns None when tobytes returns b'' (line 193-194)."""

    def test_vision_render_empty_bytes(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """tobytes returns b'' -> returns None."""
        from ai_pdf_renamer import pdf_extract

        mock_pix = MagicMock(spec=["tobytes"])
        mock_pix.tobytes.return_value = b""

        mock_page = MagicMock()
        mock_page.get_pixmap.return_value = mock_pix

        mock_doc = MagicMock()
        mock_doc.is_encrypted = False
        mock_doc.page_count = 1
        mock_doc.load_page.return_value = mock_page

        mock_fitz = MagicMock()
        mock_fitz.open.return_value = mock_doc
        monkeypatch.setitem(sys.modules, "fitz", mock_fitz)

        pdf_path = tmp_path / "render_empty.pdf"
        pdf_path.write_bytes(b"%PDF-1.4")

        result = pdf_extract.pdf_first_page_to_image_base64(pdf_path)
        assert result is None


class TestExtractPagesEmptyTextResult:
    """Test _extract_pages when text mode returns empty (no fallback since S3 simplification)."""

    def test_extract_pages_empty_text_yields_nothing(self) -> None:
        """Text mode empty -> no pieces extracted, no errors."""
        from ai_pdf_renamer import pdf_extract

        mock_page = MagicMock()
        mock_page.get_text.return_value = ""

        mock_doc = MagicMock()
        mock_doc.page_count = 1
        mock_doc.load_page = MagicMock(return_value=mock_page)

        pieces, errors = pdf_extract._extract_pages(mock_doc, Path("/tmp/test.pdf"))
        assert pieces == []
        assert errors == []


class TestOcrTempFileCleanup:
    """Test OCR temp file is cleaned up after OCR (lines 237, 249, 253-274)."""

    def test_ocr_temp_file_cleanup(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """After OCR, temp file is deleted regardless of outcome."""
        from ai_pdf_renamer import pdf_extract

        call_count = 0

        def fake_pdf_to_text(*args: Any, **kwargs: Any) -> str:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return "Hi"  # Short, triggers OCR
            return "OCR result text with enough characters to pass."

        monkeypatch.setattr(pdf_extract, "pdf_to_text", fake_pdf_to_text)

        temp_files_created: list[Path] = []

        mock_ocrmypdf = MagicMock()

        def fake_ocr(input_path: str, output_path: str, **kwargs: Any) -> None:
            temp_files_created.append(Path(output_path))
            Path(output_path).write_bytes(b"%PDF-1.4 ocr output")

        mock_ocrmypdf.ocr = fake_ocr
        monkeypatch.setitem(sys.modules, "ocrmypdf", mock_ocrmypdf)

        pdf_path = tmp_path / "ocr_cleanup.pdf"
        pdf_path.write_bytes(b"%PDF-1.4")

        result = pdf_extract.pdf_to_text_with_ocr(pdf_path)
        assert "OCR result" in result

        # Verify temp file was cleaned up
        assert len(temp_files_created) == 1
        assert not temp_files_created[0].exists()


class TestVisionRenderOpenError:
    """Test vision render when fitz.open raises error (lines 168-170)."""

    def test_vision_render_open_error(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """fitz.open raises RuntimeError -> returns None."""
        from ai_pdf_renamer import pdf_extract

        mock_fitz = MagicMock()
        mock_fitz.open.side_effect = RuntimeError("Cannot open file")
        monkeypatch.setitem(sys.modules, "fitz", mock_fitz)

        pdf_path = tmp_path / "bad_open.pdf"
        pdf_path.write_bytes(b"%PDF-1.4")

        result = pdf_extract.pdf_first_page_to_image_base64(pdf_path)
        assert result is None


class TestVisionRenderNoneFilepath:
    """Test vision render with None filepath (line 159-160)."""

    def test_vision_render_none_filepath(self) -> None:
        """None filepath -> returns None."""
        from ai_pdf_renamer import pdf_extract

        result = pdf_extract.pdf_first_page_to_image_base64(None)
        assert result is None


class TestVisionRenderZeroPages:
    """Test vision render with 0 pages (lines 176-177)."""

    def test_vision_render_zero_pages(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Doc with 0 page_count -> returns None."""
        from ai_pdf_renamer import pdf_extract

        mock_doc = MagicMock()
        mock_doc.is_encrypted = False
        mock_doc.page_count = 0

        mock_fitz = MagicMock()
        mock_fitz.open.return_value = mock_doc
        monkeypatch.setitem(sys.modules, "fitz", mock_fitz)

        pdf_path = tmp_path / "no_pages.pdf"
        pdf_path.write_bytes(b"%PDF-1.4")

        result = pdf_extract.pdf_first_page_to_image_base64(pdf_path)
        assert result is None


class TestVisionRenderExceptionInBody:
    """Test vision render exception during pixmap/encode (lines 196-198)."""

    def test_vision_render_runtime_error(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """RuntimeError during rendering -> returns None."""
        from ai_pdf_renamer import pdf_extract

        mock_page = MagicMock()
        mock_page.get_pixmap.side_effect = RuntimeError("render error")

        mock_doc = MagicMock()
        mock_doc.is_encrypted = False
        mock_doc.page_count = 1
        mock_doc.load_page.return_value = mock_page

        mock_fitz = MagicMock()
        mock_fitz.open.return_value = mock_doc
        monkeypatch.setitem(sys.modules, "fitz", mock_fitz)

        pdf_path = tmp_path / "render_error.pdf"
        pdf_path.write_bytes(b"%PDF-1.4")

        result = pdf_extract.pdf_first_page_to_image_base64(pdf_path)
        assert result is None


class TestExtractPagesAccessError:
    """Test _extract_pages with page access error (lines 343-347)."""

    def test_extract_pages_access_error(self) -> None:
        """Page access raises IndexError -> error recorded, continues."""
        from ai_pdf_renamer import pdf_extract

        mock_doc = MagicMock()
        mock_doc.page_count = 2

        call_count = 0

        def getitem(self: Any, idx: int) -> Any:
            nonlocal call_count
            call_count += 1
            if idx == 0:
                raise IndexError("Page 0 corrupt")
            page = MagicMock()
            page.get_text.return_value = "Page 1 text."
            return page

        mock_doc.__getitem__ = getitem

        def load_page_fn(idx: int) -> Any:
            nonlocal call_count
            call_count += 1
            if idx == 0:
                raise IndexError("Page 0 corrupt")
            page = MagicMock()
            page.get_text.return_value = "Page 1 text."
            return page

        mock_doc.load_page = load_page_fn

        pieces, errors = pdf_extract._extract_pages(mock_doc, Path("/tmp/test.pdf"))
        assert len(pieces) == 1
        assert "Page 1 text." in pieces[0]
        assert len(errors) == 1
        assert "page 0" in errors[0].lower()


class TestExtractPagesTextExtractionError:
    """Test _extract_pages records error when text extraction fails."""

    def test_extract_pages_text_error_recorded(self) -> None:
        """Text extraction raises -> error recorded with OCR suggestion."""
        from ai_pdf_renamer import pdf_extract

        mock_page = MagicMock()
        mock_page.get_text.side_effect = RuntimeError("corrupt page")

        mock_doc = MagicMock()
        mock_doc.page_count = 1
        mock_doc.load_page = MagicMock(return_value=mock_page)

        pieces, errors = pdf_extract._extract_pages(mock_doc, Path("/tmp/test.pdf"))
        assert pieces == []
        assert len(errors) == 1
        assert "corrupt page" in errors[0]


class TestPdfToTextEmptyContentLargeFile:
    """Test pdf_to_text with no text but large file size triggers ValueError (lines 133-143)."""

    def test_pdf_to_text_empty_content_large_file(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """No text, page_count > 0, file > 1024 bytes -> ValueError with OCR suggestion."""
        from ai_pdf_renamer import pdf_extract

        # Page returns empty text from all methods
        mock_page = MagicMock()
        mock_page.get_text.return_value = ""

        mock_doc = MagicMock()
        mock_doc.page_count = 1
        mock_doc.is_encrypted = False
        mock_doc.__getitem__ = MagicMock(return_value=mock_page)
        mock_doc.load_page = MagicMock(return_value=mock_page)

        mock_fitz = MagicMock()
        mock_fitz.open.return_value = mock_doc
        monkeypatch.setitem(sys.modules, "fitz", mock_fitz)

        # Create a file > 1024 bytes
        pdf_path = tmp_path / "image_only.pdf"
        pdf_path.write_bytes(b"%PDF-1.4" + b"\x00" * 2000)

        with pytest.raises(ValueError, match="Consider using --ocr"):
            pdf_extract.pdf_to_text(pdf_path)


class TestVisionRenderGetImageDataFallback:
    """Test getImageData fallback (lines 187-188)."""

    def test_vision_render_getimagedata_fallback(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """No tobytes, but getImageData present -> uses getImageData('jpeg')."""
        from ai_pdf_renamer import pdf_extract

        fake_jpeg = b"\xff\xd8\xff\xe0JFIF-via-getImageData"

        # Create pix with only getImageData (no tobytes, no getPNGData)
        mock_pix = MagicMock(spec=["getImageData"])
        mock_pix.getImageData.return_value = fake_jpeg

        mock_page = MagicMock()
        mock_page.get_pixmap.return_value = mock_pix

        mock_doc = MagicMock()
        mock_doc.is_encrypted = False
        mock_doc.page_count = 1
        mock_doc.load_page.return_value = mock_page

        mock_fitz = MagicMock()
        mock_fitz.open.return_value = mock_doc
        monkeypatch.setitem(sys.modules, "fitz", mock_fitz)

        pdf_path = tmp_path / "render_getimagedata.pdf"
        pdf_path.write_bytes(b"%PDF-1.4")

        result = pdf_extract.pdf_first_page_to_image_base64(pdf_path)
        assert result is not None
        assert base64.b64decode(result) == fake_jpeg
        mock_pix.getImageData.assert_called_once_with("jpeg")


class TestLoadConfigJsonNonDict:
    """Test _load_config_file with JSON array at top level (line 43)."""

    def test_load_config_json_non_dict(self, tmp_path: Path) -> None:
        """JSON array at top level -> returns {}."""
        from ai_pdf_renamer.cli import _load_config_file

        p = tmp_path / "config.json"
        p.write_text("[1, 2, 3]", encoding="utf-8")

        result = _load_config_file(p)
        assert result == {}


class TestResolveLogConfig:
    """Test _resolve_log_config (lines 161-171)."""

    def test_resolve_log_config_defaults(self) -> None:
        """No args set -> defaults to XDG-style log path and INFO."""
        from ai_pdf_renamer.cli import _resolve_log_config

        args = argparse.Namespace()
        log_file, log_level = _resolve_log_config(args)
        assert log_file.endswith("ai-pdf-renamer/error.log")
        assert log_level == logging.INFO

    def test_resolve_log_config_verbose(self) -> None:
        """--verbose -> DEBUG level."""
        from ai_pdf_renamer.cli import _resolve_log_config

        args = argparse.Namespace(verbose=True, quiet=False, log_file=None, log_level=None)
        _log_file, log_level = _resolve_log_config(args)
        assert log_level == logging.DEBUG

    def test_resolve_log_config_quiet(self) -> None:
        """--quiet -> WARNING level."""
        from ai_pdf_renamer.cli import _resolve_log_config

        args = argparse.Namespace(verbose=False, quiet=True, log_file=None, log_level=None)
        _log_file, log_level = _resolve_log_config(args)
        assert log_level == logging.WARNING

    def test_resolve_log_config_explicit_level(self) -> None:
        """--log-level ERROR -> ERROR level."""
        from ai_pdf_renamer.cli import _resolve_log_config

        args = argparse.Namespace(verbose=False, quiet=False, log_file=None, log_level="ERROR")
        _log_file, log_level = _resolve_log_config(args)
        assert log_level == logging.ERROR

    def test_resolve_log_config_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """AI_PDF_RENAMER_LOG_LEVEL env var -> that level."""
        from ai_pdf_renamer.cli import _resolve_log_config

        monkeypatch.setenv("AI_PDF_RENAMER_LOG_LEVEL", "DEBUG")
        args = argparse.Namespace(verbose=False, quiet=False, log_file=None, log_level=None)
        _log_file, log_level = _resolve_log_config(args)
        assert log_level == logging.DEBUG

    def test_resolve_log_config_log_file_from_args(self) -> None:
        """--log-file custom.log -> custom.log."""
        from ai_pdf_renamer.cli import _resolve_log_config

        args = argparse.Namespace(verbose=False, quiet=False, log_file="custom.log", log_level=None)
        log_file, _log_level = _resolve_log_config(args)
        assert log_file == "custom.log"

    def test_resolve_log_config_log_file_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """AI_PDF_RENAMER_LOG_FILE env var -> that file."""
        from ai_pdf_renamer.cli import _resolve_log_config

        monkeypatch.setenv("AI_PDF_RENAMER_LOG_FILE", "env_log.log")
        args = argparse.Namespace(verbose=False, quiet=False, log_file=None, log_level=None)
        log_file, _log_level = _resolve_log_config(args)
        assert log_file == "env_log.log"


class TestResolveDirsInteractivePrompt:
    """Test _resolve_dirs interactive prompt (lines 295-306)."""

    def test_resolve_dirs_interactive_prompt(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Interactive mode with no --dir prompts user; input is used."""
        import ai_pdf_renamer.cli as cli

        monkeypatch.setattr(cli, "_is_interactive", lambda: True)
        monkeypatch.setattr("builtins.input", lambda _prompt: str(tmp_path))

        args = argparse.Namespace(dirs=None, single_file=None, manual_file=None, dirs_from_file=None)
        dirs, single_file = cli._resolve_dirs(args)
        assert dirs == [str(tmp_path.resolve())]
        assert single_file is None


class TestResolveDirsNoTtyNoDir:
    """Test _resolve_dirs non-interactive with no --dir (lines 307-311)."""

    def test_resolve_dirs_no_tty_no_dir(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Non-interactive, no --dir -> SystemExit."""
        import ai_pdf_renamer.cli as cli

        monkeypatch.setattr(cli, "_is_interactive", lambda: False)

        args = argparse.Namespace(dirs=None, single_file=None, manual_file=None, dirs_from_file=None)
        with pytest.raises(SystemExit) as exc_info:
            cli._resolve_dirs(args)
        assert exc_info.value.code == 1


class TestMainConfigFileLoaded:
    """Test main() with --config loading a JSON file (line 437)."""

    def test_main_config_file_loaded(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Pass --config with valid JSON file, verify config values are used."""
        import ai_pdf_renamer.cli as cli

        monkeypatch.setattr(cli, "setup_logging", lambda **k: None)
        monkeypatch.setattr(cli, "_is_interactive", lambda: False)

        config_data = {"language": "en", "desired_case": "snakeCase"}
        config_file = tmp_path / "myconfig.json"
        config_file.write_text(json.dumps(config_data), encoding="utf-8")

        captured: dict[str, Any] = {}

        def fake_rename(directory: str, *, config: Any, files_override: Any = None) -> None:
            captured["config"] = config

        monkeypatch.setattr(cli, "rename_pdfs_in_directory", fake_rename)

        cli.main(
            [
                "--dir",
                str(tmp_path),
                "--config",
                str(config_file),
                "--project",
                "",
                "--version",
                "",
            ]
        )

        assert captured["config"].language == "en"
        assert captured["config"].desired_case == "snakeCase"


class TestLoadOverrideCategoryMapWarning:
    """Test _load_override_category_map OSError warning (lines 185-186)."""

    def test_load_override_category_map_os_error(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """OSError when reading CSV -> warning logged, empty dict returned."""
        from ai_pdf_renamer.cli import _load_override_category_map

        p = tmp_path / "overrides.csv"
        p.write_text("filename,category\ninvoice.pdf,finance\n", encoding="utf-8")

        # Make open() raise OSError
        def fake_open(*args: Any, **kwargs: Any) -> Any:
            raise OSError("Permission denied")

        monkeypatch.setattr("builtins.open", fake_open)

        with caplog.at_level(logging.WARNING):
            result = _load_override_category_map(p)

        assert result == {}
        assert any("Could not read override-category file" in r.message for r in caplog.records)


class TestMainDoctorPath:
    """Test main() --doctor path (line 433-434)."""

    def test_main_doctor_exits(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """--doctor calls run_doctor_checks and exits."""
        import ai_pdf_renamer.cli as cli

        monkeypatch.setattr(cli, "setup_logging", lambda **k: None)
        monkeypatch.setattr(cli, "run_doctor_checks", lambda args: 0)

        with pytest.raises(SystemExit) as exc_info:
            cli.main(["--doctor", "--dir", "/tmp"])

        assert exc_info.value.code == 0


class TestMainRequestsError:
    """Test main() requests/OSError error handling (line 410-411).

    Note: requests.RequestException inherits from OSError, so it's caught
    by the ``except (FileNotFoundError, NotADirectoryError, OSError)`` handler.
    """

    def test_main_requests_error(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """requests.RequestException (subclass of OSError) -> SystemExit with its message."""
        import requests

        import ai_pdf_renamer.cli as cli

        monkeypatch.setattr(cli, "setup_logging", lambda **k: None)
        monkeypatch.setattr(cli, "_is_interactive", lambda: False)

        def fake_rename(*args: Any, **kwargs: Any) -> None:
            raise requests.RequestException("Connection refused")

        monkeypatch.setattr(cli, "rename_pdfs_in_directory", fake_rename)

        with pytest.raises(SystemExit) as exc_info:
            cli.main(
                [
                    "--dir",
                    str(tmp_path),
                    "--language",
                    "de",
                    "--case",
                    "kebabCase",
                    "--project",
                    "",
                    "--version",
                    "",
                ]
            )
        assert exc_info.value.code == 1


class TestMainGenericError:
    """Test main() generic Exception handling (lines 421-423)."""

    def test_main_generic_error(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Generic Exception during rename -> SystemExit with exit code 1."""
        import ai_pdf_renamer.cli as cli

        monkeypatch.setattr(cli, "setup_logging", lambda **k: None)
        monkeypatch.setattr(cli, "_is_interactive", lambda: False)

        def fake_rename(*args: Any, **kwargs: Any) -> None:
            raise Exception("Unexpected failure")

        monkeypatch.setattr(cli, "rename_pdfs_in_directory", fake_rename)

        with pytest.raises(SystemExit) as exc_info:
            cli.main(
                [
                    "--dir",
                    str(tmp_path),
                    "--language",
                    "de",
                    "--case",
                    "kebabCase",
                    "--project",
                    "",
                    "--version",
                    "",
                ]
            )
        assert exc_info.value.code == 1
