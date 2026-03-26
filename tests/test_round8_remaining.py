"""Round 8: Targeted coverage for pdf_extract.py, cli.py, llm_prompts.py, llm_schema.py, data_paths.py.

Focuses on uncovered lines identified by coverage analysis.
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

# ===========================================================================
# pdf_extract.py — shrink density, max_pages, vision render, extract_pages
# ===========================================================================


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


class TestExtractPagesBlocksFallbackWithData:
    """Test _extract_pages blocks fallback with real block data (lines 360-363)."""

    def test_extract_pages_blocks_fallback_with_data(self) -> None:
        """Text mode empty, blocks mode returns block tuples with content."""
        from ai_pdf_renamer import pdf_extract

        mock_page = MagicMock()

        def fake_get_text(mode: str = "text") -> Any:
            if mode == "text":
                return ""  # Empty -> triggers blocks fallback
            if mode == "blocks":
                return [
                    (0, 0, 100, 50, "First block content.", 0, 0),
                    (0, 50, 100, 100, "Second block content.", 0, 0),
                ]
            return ""

        mock_page.get_text = fake_get_text

        mock_doc = MagicMock()
        mock_doc.page_count = 1
        mock_doc.__getitem__ = MagicMock(return_value=mock_page)
        mock_doc.load_page = MagicMock(return_value=mock_page)

        pieces, errors = pdf_extract._extract_pages(mock_doc, Path("/tmp/test.pdf"))
        assert len(pieces) == 1
        assert "First block content." in pieces[0]
        assert "Second block content." in pieces[0]
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


class TestExtractPagesRawdictFallback:
    """Test _extract_pages rawdict fallback when text and blocks return empty (lines 371-381)."""

    def test_extract_pages_rawdict_fallback(self) -> None:
        """Text and blocks empty -> rawdict extraction produces text."""
        from ai_pdf_renamer import pdf_extract

        mock_page = MagicMock()

        def fake_get_text(mode: str = "text") -> Any:
            if mode == "text":
                return ""
            if mode == "blocks":
                return []  # Empty blocks
            if mode == "rawdict":
                return {
                    "blocks": [
                        {
                            "lines": [
                                {
                                    "spans": [
                                        {"text": "Rawdict span text."},
                                        {"text": "More span text."},
                                    ]
                                }
                            ]
                        }
                    ]
                }
            return ""

        mock_page.get_text = fake_get_text

        mock_doc = MagicMock()
        mock_doc.page_count = 1
        mock_doc.__getitem__ = MagicMock(return_value=mock_page)
        mock_doc.load_page = MagicMock(return_value=mock_page)

        pieces, errors = pdf_extract._extract_pages(mock_doc, Path("/tmp/test.pdf"))
        assert len(pieces) == 1
        assert "Rawdict span text." in pieces[0]
        assert "More span text." in pieces[0]
        assert errors == []


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


# ===========================================================================
# cli.py — config loading, log config, interactive prompts, main paths
# ===========================================================================


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
        """No args set -> defaults to error.log and INFO."""
        from ai_pdf_renamer.cli import _resolve_log_config

        args = argparse.Namespace()
        log_file, log_level = _resolve_log_config(args)
        assert log_file == "error.log"
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


# ===========================================================================
# llm_prompts.py — German language prompt variants
# ===========================================================================


class TestSummaryPromptChunkGerman:
    """Test _summary_prompt_chunk with language='de' (lines 68-73)."""

    def test_summary_prompt_chunk_german(self) -> None:
        """German chunk prompt contains expected German text."""
        from ai_pdf_renamer.llm_prompts import _summary_prompt_chunk

        result = _summary_prompt_chunk("de", "", "Testinhalt des Dokuments.")
        assert "Fasse den folgenden Text" in result
        assert "kurzen Sätzen" in result
        assert '{"summary":"..."}' in result
        assert "Testinhalt des Dokuments." in result


class TestSummaryPromptCombineGerman:
    """Test _summary_prompt_combine with language='de' (lines 83-91)."""

    def test_summary_prompt_combine_german(self) -> None:
        """German combine prompt contains expected German text."""
        from ai_pdf_renamer.llm_prompts import _summary_prompt_combine

        result = _summary_prompt_combine("de", "", "Teil 1. Teil 2.")
        assert "Teilzusammenfassungen" in result
        assert "prägnanten Sätzen" in result
        assert "Dokumenttyp" in result
        assert '{"summary":"..."}' in result
        assert "Teil 1. Teil 2." in result


class TestCategoryPromptGermanWithAllowed:
    """Test _build_allowed_categories_instruction German + allowed_categories (line 176)."""

    def test_category_prompt_german_with_allowed(self) -> None:
        """German with allowed_categories returns constrained instruction."""
        from ai_pdf_renamer.llm_prompts import _build_allowed_categories_instruction

        result = _build_allowed_categories_instruction(
            allowed_categories=["Rechnung", "Vertrag", "Brief"],
            language="de",
        )
        assert "genau eine dieser Kategorien" in result
        assert "unknown" in result
        assert "Brief" in result
        assert "Rechnung" in result
        assert "Vertrag" in result


class TestSummaryPromptsShortGerman:
    """Test _summary_prompts_short with language='de' (lines 34-48)."""

    def test_summary_prompts_short_german(self) -> None:
        """German short prompts contain expected German text and return 2 prompts."""
        from ai_pdf_renamer.llm_prompts import _summary_prompts_short

        result = _summary_prompts_short("de", "", "Kurzer Testtext.")
        assert len(result) == 2
        assert "präzisen Sätzen" in result[0]
        assert '{"summary":"..."}' in result[0]
        assert "wichtigsten Informationen" in result[1]
        assert "Kurzer Testtext." in result[0]
        assert "Kurzer Testtext." in result[1]


class TestSummaryPromptsShortGermanWithDocType:
    """Test _summary_prompts_short with German doc type hint."""

    def test_summary_prompts_short_german_with_doc_type(self) -> None:
        """German short prompts include doc type hint."""
        from ai_pdf_renamer.llm_prompts import _summary_doc_type_hint, _summary_prompts_short

        hint = _summary_doc_type_hint("de", "Rechnung")
        result = _summary_prompts_short("de", hint, "Inhalt.")
        assert "Rechnung" in result[0]
        assert "heuristisch" in result[0]


class TestBuildAnalysisPromptGerman:
    """Test build_analysis_prompt with language='de' (lines 141-152)."""

    def test_build_analysis_prompt_german(self) -> None:
        """German analysis prompt contains expected structure."""
        from ai_pdf_renamer.llm_prompts import build_analysis_prompt

        result = build_analysis_prompt("de", "Testdokument Inhalt.", suggested_doc_type="Rechnung")
        assert "Analysiere das folgende Dokument" in result
        assert "JSON" in result
        assert "summary" in result
        assert "keywords" in result
        assert "category" in result
        assert "Testdokument Inhalt." in result
        assert "Rechnung" in result


class TestBuildAllowedCategoriesGermanSuggested:
    """Test _build_allowed_categories_instruction German with suggested_categories (lines 181-182)."""

    def test_category_german_suggested(self) -> None:
        """German with suggested_categories returns suggestion instruction."""
        from ai_pdf_renamer.llm_prompts import _build_allowed_categories_instruction

        result = _build_allowed_categories_instruction(
            suggested_categories=["Rechnung", "Vertrag"],
            language="de",
        )
        assert "Vorschläge" in result or "Vorschl" in result
        assert "Rechnung" in result

    def test_category_german_no_categories(self) -> None:
        """German with no categories returns generic instruction."""
        from ai_pdf_renamer.llm_prompts import _build_allowed_categories_instruction

        result = _build_allowed_categories_instruction(language="de")
        assert "passende Kategorie" in result


# ===========================================================================
# llm_schema.py — jsonschema validation
# ===========================================================================


class TestValidateResultJsonschemaAvailable:
    """Test validate_llm_document_result when jsonschema is available (lines 76-83)."""

    def test_validate_result_jsonschema_available(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When jsonschema is available, validation runs and logs on error."""
        from ai_pdf_renamer import llm_schema

        mock_jsonschema = MagicMock()

        class FakeValidationError(Exception):
            pass

        mock_jsonschema.ValidationError = FakeValidationError
        mock_jsonschema.validate.side_effect = FakeValidationError("Bad field")

        monkeypatch.setitem(sys.modules, "jsonschema", mock_jsonschema)

        # Clear lru_cache to ensure schema is freshly loaded
        llm_schema._load_llm_response_schema.cache_clear()

        parsed = {"summary": "Test summary", "keywords": ["a", "b"], "category": "finance"}
        result = llm_schema.validate_llm_document_result(parsed)

        # Should still return result despite validation error (validation is advisory)
        assert result.summary == "Test summary"
        assert result.category == "finance"
        mock_jsonschema.validate.assert_called_once()

    def test_validate_result_jsonschema_passes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When jsonschema validates successfully, no error is logged."""
        from ai_pdf_renamer import llm_schema

        mock_jsonschema = MagicMock()
        mock_jsonschema.ValidationError = Exception
        mock_jsonschema.validate.return_value = None  # No error

        monkeypatch.setitem(sys.modules, "jsonschema", mock_jsonschema)

        llm_schema._load_llm_response_schema.cache_clear()

        parsed = {"summary": "Test", "keywords": ["x"], "category": "report"}
        result = llm_schema.validate_llm_document_result(parsed)

        assert result.summary == "Test"
        assert result.category == "report"
        mock_jsonschema.validate.assert_called_once()


# ===========================================================================
# data_paths.py — edge cases
# ===========================================================================


class TestDataDirNoPyproject:
    """Test data_dir when no pyproject.toml is found (project_root CWD fallback)."""

    def test_data_dir_no_pyproject(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When _discover_repo_root returns None, data_dir uses package data path."""
        from ai_pdf_renamer import data_paths

        monkeypatch.setattr(data_paths, "_discover_repo_root", lambda start=None: None)
        monkeypatch.delenv("AI_PDF_RENAMER_DATA_DIR", raising=False)

        result = data_paths.data_dir()
        # Should be the package data directory
        expected = (Path(data_paths.__file__).resolve().parent / "data").resolve()
        assert result == expected


class TestDataPathPackageFallback:
    """Test data_path when env not set, repo data missing -> package_data_path tried (lines 76-78)."""

    def test_data_path_package_fallback(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """data_dir points to empty dir -> falls back to package_data_path."""
        from ai_pdf_renamer import data_paths

        monkeypatch.setattr(data_paths, "data_dir", lambda: tmp_path)

        # package_data_path should have the actual files
        result = data_paths.data_path("meta_stopwords.json")
        assert result.exists()
        assert result.name == "meta_stopwords.json"

    def test_data_path_raises_when_both_missing(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Neither data_dir nor package_data_path has the file -> FileNotFoundError."""
        from ai_pdf_renamer import data_paths

        monkeypatch.setattr(data_paths, "data_dir", lambda: tmp_path)
        monkeypatch.setattr(data_paths, "package_data_path", lambda f: tmp_path / "nonexistent" / f)

        with pytest.raises(FileNotFoundError, match="Data file"):
            data_paths.data_path("meta_stopwords.json")


class TestProjectRootNoPyproject:
    """Test project_root falls back to CWD when no pyproject.toml found."""

    def test_project_root_cwd_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """No pyproject.toml found -> project_root returns CWD."""
        from ai_pdf_renamer import data_paths

        monkeypatch.setattr(data_paths, "_discover_repo_root", lambda start=None: None)
        result = data_paths.project_root()
        assert result == Path.cwd()


class TestResolveDirsInteractiveDefault:
    """Test _resolve_dirs interactive prompt with default value."""

    def test_resolve_dirs_interactive_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Interactive mode, user presses Enter (empty) -> uses ./input_files default."""
        import ai_pdf_renamer.cli as cli

        monkeypatch.setattr(cli, "_is_interactive", lambda: True)
        monkeypatch.setattr("builtins.input", lambda _prompt: "")

        args = argparse.Namespace(dirs=None, single_file=None, manual_file=None, dirs_from_file=None)
        dirs, _single_file = cli._resolve_dirs(args)
        assert dirs == [str(Path("./input_files").resolve())]
