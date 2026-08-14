"""Cover rendering, OCR, metadata, and filesystem failure paths."""

from __future__ import annotations

import base64
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from folionym import pdf_extract
from tests.conftest import install_fitz_mock, make_fitz_doc, make_pdf_to_text_sequence
from tests.helpers import install_fitz_document, make_pdf


class TestPdfToTextRaisesOnExtractionError:
    """Test that RuntimeError is raised when all pages fail (line 128)."""

    def test_pdf_to_text_raises_on_extraction_error(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """All pages fail extraction -> RuntimeError with error details."""

        mock_page = MagicMock()
        # All get_text calls raise, triggering error recording on all methods.
        mock_page.get_text.side_effect = RuntimeError("extraction failed")

        install_fitz_mock(monkeypatch, make_fitz_doc(mock_page, item_access=True))

        pdf_path = tmp_path / "broken.pdf"
        pdf_path.write_bytes(b"%PDF-1.4")

        with pytest.raises(RuntimeError, match="Extraction failed"):
            pdf_extract.pdf_to_text(pdf_path)


class TestVisionRenderJpegSuccess:
    """Test vision render JPEG success path (lines 183-184)."""

    def test_vision_render_jpeg_success(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """tobytes(output='jpeg') succeeds -> base64-encoded result."""

        fake_jpeg = b"\xff\xd8\xff\xe0JFIF-test-data"

        mock_pix = MagicMock(spec=["tobytes"])
        mock_pix.tobytes.return_value = fake_jpeg

        mock_page = MagicMock()
        mock_page.get_pixmap.return_value = mock_pix

        install_fitz_mock(monkeypatch, make_fitz_doc(mock_page))

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

        fake_png = b"\x89PNG-test-data"

        mock_pix = MagicMock(spec=["tobytes"])

        def tobytes_side_effect(output: str = "png", **kwargs: Any) -> bytes:
            if output == "jpeg":
                raise TypeError("JPEG not supported")
            return fake_png

        mock_pix.tobytes.side_effect = tobytes_side_effect

        mock_page = MagicMock()
        mock_page.get_pixmap.return_value = mock_pix

        install_fitz_mock(monkeypatch, make_fitz_doc(mock_page))

        pdf_path = tmp_path / "render_png_fallback.pdf"
        pdf_path.write_bytes(b"%PDF-1.4")

        result = pdf_extract.pdf_first_page_to_image_base64(pdf_path)
        assert result is not None
        assert base64.b64decode(result) == fake_png


class TestVisionRenderGetPNGDataFallback:
    """Test getPNGData fallback when tobytes and getImageData are absent (lines 189-190)."""

    def test_vision_render_getpngdata_fallback(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """No tobytes, no getImageData -> getPNGData is used."""

        fake_png = b"\x89PNG-via-getPNGData"

        # Create a pix object with only getPNGData
        mock_pix = MagicMock(spec=["getPNGData"])
        mock_pix.getPNGData.return_value = fake_png

        mock_page = MagicMock()
        mock_page.get_pixmap.return_value = mock_pix

        install_fitz_document(monkeypatch, mock_page)
        pdf_path = make_pdf(tmp_path, "render_getpngdata.pdf")

        result = pdf_extract.pdf_first_page_to_image_base64(pdf_path)
        assert result is not None
        assert base64.b64decode(result) == fake_png
        mock_pix.getPNGData.assert_called_once()


class TestVisionRenderNoMethods:
    """Test vision render returns None when no rendering methods are available (lines 191-192)."""

    def test_vision_render_no_methods(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Pix object has no tobytes/getImageData/getPNGData -> returns None."""

        # spec=[] means no attributes at all -> hasattr checks all return False
        mock_pix = MagicMock(spec=[])

        mock_page = MagicMock()
        mock_page.get_pixmap.return_value = mock_pix

        install_fitz_document(monkeypatch, mock_page)
        pdf_path = make_pdf(tmp_path, "render_no_methods.pdf")

        result = pdf_extract.pdf_first_page_to_image_base64(pdf_path)
        assert result is None


class TestVisionRenderEmptyBytes:
    """Test vision render returns None when tobytes returns b'' (line 193-194)."""

    def test_vision_render_empty_bytes(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """tobytes returns b'' -> returns None."""

        mock_pix = MagicMock(spec=["tobytes"])
        mock_pix.tobytes.return_value = b""

        mock_page = MagicMock()
        mock_page.get_pixmap.return_value = mock_pix

        install_fitz_document(monkeypatch, mock_page)
        pdf_path = make_pdf(tmp_path, "render_empty.pdf")

        result = pdf_extract.pdf_first_page_to_image_base64(pdf_path)
        assert result is None


class TestExtractPagesEmptyTextResult:
    """Test _extract_pages when text mode returns empty (no fallback since S3 simplification)."""

    def test_extract_pages_empty_text_yields_nothing(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Text mode empty -> no pieces extracted, no errors."""

        mock_page = MagicMock()
        mock_page.get_text.return_value = ""

        install_fitz_document(monkeypatch, mock_page)
        pdf_path = make_pdf(tmp_path, "test.pdf")

        assert pdf_extract.pdf_to_text(pdf_path) == ""


class TestOcrTempFileCleanup:
    """Test OCR temp file is cleaned up after OCR (lines 237, 249, 253-274)."""

    def test_ocr_temp_file_cleanup(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """After OCR, temp file is deleted regardless of outcome."""

        monkeypatch.setattr(
            pdf_extract,
            "pdf_to_text",
            make_pdf_to_text_sequence("Hi", "OCR result text with enough characters to pass."),
        )

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

        result = pdf_extract.pdf_first_page_to_image_base64(None)
        assert result is None


class TestVisionRenderZeroPages:
    """Test vision render with 0 pages (lines 176-177)."""

    def test_vision_render_zero_pages(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Doc with 0 page_count -> returns None."""

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

        mock_page = MagicMock()
        mock_page.get_pixmap.side_effect = RuntimeError("render error")

        install_fitz_document(monkeypatch, mock_page)
        pdf_path = make_pdf(tmp_path, "render_error.pdf")

        result = pdf_extract.pdf_first_page_to_image_base64(pdf_path)
        assert result is None


class TestExtractPagesAccessError:
    """Test _extract_pages with page access error (lines 343-347)."""

    def test_extract_pages_access_error(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Page access raises IndexError -> error recorded, continues."""

        mock_doc = MagicMock()
        mock_doc.is_encrypted = False
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

        mock_fitz = MagicMock()
        mock_fitz.open.return_value = mock_doc
        monkeypatch.setitem(sys.modules, "fitz", mock_fitz)

        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"%PDF-1.4")

        assert pdf_extract.pdf_to_text(pdf_path) == "Page 1 text."


class TestExtractPagesTextExtractionError:
    """Test _extract_pages records error when text extraction fails."""

    def test_extract_pages_text_error_recorded(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Text extraction raises -> error recorded with OCR suggestion."""

        mock_page = MagicMock()
        mock_page.get_text.side_effect = RuntimeError("corrupt page")

        install_fitz_document(monkeypatch, mock_page)
        pdf_path = make_pdf(tmp_path, "test.pdf")

        with pytest.raises(RuntimeError, match="corrupt page"):
            pdf_extract.pdf_to_text(pdf_path)


class TestPdfToTextEmptyContentLargeFile:
    """Test pdf_to_text with no text but large file size triggers ValueError (lines 133-143)."""

    def test_pdf_to_text_empty_content_large_file(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """No text, page_count > 0, file > 1024 bytes -> ValueError with OCR suggestion."""

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
