from __future__ import annotations

import base64
import logging
import os
import re
import tempfile
from datetime import date
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# PDF metadata date format: D:YYYYMMDDHHmmss... or D:YYYYMMDD
_PDF_DATE_PREFIX = re.compile(r"^D:(\d{4})(\d{2})(\d{2})")

# Minimum extracted characters below which we try OCR (image-only PDFs).
MIN_CHARS_BEFORE_OCR = 50

# Default DPI for first-page render when using vision fallback.
VISION_FALLBACK_DPI = 300

# Default token limit for local LLMs: 32K is more compatible than 128K.
# reserve ~4K tokens for prompt + response.
DEFAULT_MAX_CONTENT_TOKENS = 28_000

# Cached tiktoken encoding to avoid repeated get_encoding() in _shrink_to_token_limit.
_tiktoken_encoding: Any = None


def _token_count(text: str) -> int:
    global _tiktoken_encoding
    try:
        import tiktoken

        if _tiktoken_encoding is None:
            _tiktoken_encoding = tiktoken.get_encoding("cl100k_base")
        return len(_tiktoken_encoding.encode(text))
    except Exception:
        # Fallback heuristic: ~4 chars per token for typical text.
        return max(1, len(text) // 4)


def _shrink_to_token_limit(text: str, *, max_tokens: int) -> str:
    """
    Shrink text to a token limit. Uses tiktoken if available, else heuristic.
    Optimized to jump close to the target length to avoid multiple expensive encodings.
    """
    count = _token_count(text)
    if count <= max_tokens:
        return text

    # Approximate target length based on density
    density = len(text) / max(1, count)
    target_char_count = int(max_tokens * density * 1.1)  # 10% buffer
    if target_char_count < len(text):
        text = text[:target_char_count]

    # Final fine-tuning
    while _token_count(text) > max_tokens and len(text) > 200:
        new_len = int(len(text) * 0.95)
        # Prefer cut at last space to avoid mid-word truncation
        chunk = text[:new_len]
        last_space = chunk.rfind(" ")
        if last_space > len(text) // 2:
            new_len = last_space
        text = text[:new_len]
    return text


def pdf_to_text(
    filepath: str | Path | None,
    *,
    max_tokens: int = DEFAULT_MAX_CONTENT_TOKENS,
    max_pages: int = 0,
) -> str:
    """
    Extracts text from a PDF via PyMuPDF (fitz). Import is done lazily so that
    core functionality can be tested without optional deps installed.
    """
    if filepath is None:
        return ""
    try:
        import fitz  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("PyMuPDF is required for PDF extraction. Install with: pip install -e '.[pdf]'") from exc

    path = Path(filepath)
    try:
        doc = fitz.open(path)
    except Exception as exc:
        raise OSError(f"Could not open PDF file {path.name}: {exc}") from exc

    page_count = getattr(doc, "page_count", 0) or 0
    if max_pages > 0:
        page_count = min(page_count, max_pages)
    try:
        if getattr(doc, "is_encrypted", False):
            logger.warning("PDF %s is encrypted/password-protected. Skipping text extraction.", path.name)
            return ""
        pieces, errors = _extract_pages(doc, path, max_pages=max_pages)
    finally:
        closer = getattr(doc, "close", None)
        if callable(closer):
            closer()

    content = "\n".join(pieces).strip()
    if not content:
        if errors and page_count > 0:
            raise RuntimeError(
                f"Extraction failed for {path.name}: {len(errors)} error(s) occurred during page processing. "
                f"First error: {errors[0]}"
            )
        if page_count > 0:
            msg = (
                f"No text extracted from {path.name} ({page_count} page(s)). "
                "File may be encrypted, image-only, or extraction failed for all pages."
            )
            # Differentiate: If file size is significant but no text, it's likely image/encrypted.
            try:
                if path.stat().st_size > 1024:
                    raise ValueError(f"{msg} Consider using --ocr.")
            except OSError:
                pass
            logger.warning(msg)
        return ""

    return _shrink_to_token_limit(content, max_tokens=max_tokens)


def pdf_first_page_to_image_base64(
    filepath: str | Path | None,
    *,
    dpi: int = VISION_FALLBACK_DPI,
) -> str | None:
    """
    Render the first page of the PDF to an image and return base64-encoded JPEG.
    Returns None if rendering fails (no fitz, encrypted, or error).
    Used by the optional vision fallback when text extraction is empty or very short.
    """
    if filepath is None:
        return None
    try:
        import fitz  # type: ignore[import-not-found]
    except Exception:
        return None
    path = Path(filepath)
    try:
        doc = fitz.open(path)
    except Exception as exc:
        logger.debug("Could not open PDF for vision render %s: %s", path.name, exc)
        return None
    try:
        if getattr(doc, "is_encrypted", False):
            logger.debug("PDF %s is encrypted; skipping vision render.", path.name)
            return None
        page_count = getattr(doc, "page_count", 0) or 0
        if page_count == 0:
            return None
        page = doc.load_page(0)
        pix = page.get_pixmap(dpi=dpi, alpha=False)
        # Prefer JPEG for smaller payload; Pixmap.tobytes(output=...) in PyMuPDF 1.23+.
        image_bytes: bytes
        if hasattr(pix, "tobytes"):
            try:
                image_bytes = pix.tobytes(output="jpeg", jpg_quality=85)  # type: ignore[attr-defined]
            except (TypeError, ValueError):
                image_bytes = pix.tobytes(output="png")  # type: ignore[attr-defined]
        elif hasattr(pix, "getImageData"):
            image_bytes = pix.getImageData("jpeg")  # type: ignore[attr-defined]
        elif hasattr(pix, "getPNGData"):
            image_bytes = pix.getPNGData()  # type: ignore[attr-defined]
        else:
            return None
        if not image_bytes:
            return None
        return base64.b64encode(image_bytes).decode("ascii")
    except Exception as exc:
        logger.debug("Vision render failed for %s: %s", path.name, exc)
        return None
    finally:
        closer = getattr(doc, "close", None)
        if callable(closer):
            closer()


def _ocr_language_code(lang: str) -> str:
    """Map config language (de/en) to Tesseract/OCRmyPDF language code.
    Can be overridden by AI_PDF_RENAMER_OCR_LANG.
    """
    override = (os.environ.get("AI_PDF_RENAMER_OCR_LANG") or "").strip()
    if override:
        return override
    if (lang or "").strip().lower() == "en":
        return "eng"
    return "deu"


def pdf_to_text_with_ocr(
    filepath: str | Path | None,
    *,
    max_tokens: int = DEFAULT_MAX_CONTENT_TOKENS,
    max_pages: int = 0,
    min_chars_for_ocr: int = MIN_CHARS_BEFORE_OCR,
    language: str = "de",
) -> str:
    """
    Extract text from a PDF; if too little text is found and OCRmyPDF is
    available, run OCR first then extract. Requires optional dependency
    ocrmypdf and system Tesseract. Falls back to non-OCR extraction on
    missing dependency or OCR failure.
    """
    text = pdf_to_text(
        filepath,
        max_tokens=max_tokens,
        max_pages=max_pages,
    )
    if not filepath or len(text.strip()) >= min_chars_for_ocr:
        return text

    try:
        import ocrmypdf
    except ImportError:
        logger.warning(
            "OCR requested but ocrmypdf not installed. Install with: pip install -e '.[ocr]' (and install Tesseract)."
        )
        return text

    path = Path(filepath)
    if not path.exists() or not path.is_file():
        return text

    tmp = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False, prefix="ai_pdf_renamer_ocr_") as f:
            tmp = Path(f.name)
        ocrmypdf.ocr(
            str(path),
            str(tmp),
            language=_ocr_language_code(language),
        )
        text_ocr = pdf_to_text(tmp, max_tokens=max_tokens, max_pages=max_pages)
        if text_ocr.strip():
            logger.info("OCR produced %s chars for %s", len(text_ocr.strip()), path.name)
            return text_ocr
    except Exception as exc:
        logger.warning("OCR failed for %s: %s. Using original extraction.", path, exc)
    finally:
        if tmp is not None and tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
    return text


def _parse_pdf_date(value: str | None) -> date | None:
    """Parse PDF metadata date string (D:YYYYMMDD...) to date. Returns None if invalid or missing."""
    if not value or not isinstance(value, str):
        return None
    m = _PDF_DATE_PREFIX.match(value.strip())
    if not m:
        logger.debug("PDF metadata date field exists but does not match expected D:YYYYMMDD format: %r", value)
        return None
    try:
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        return date(y, mo, d)
    except (ValueError, TypeError) as exc:
        logger.debug("Failed to create date object from PDF metadata %r: %s", value, exc)
        return None


def get_pdf_metadata(filepath: str | Path | None) -> dict[str, Any]:
    """
    Read PDF metadata (Title, Author, CreationDate, ModDate) without extracting text.
    Returns dict with keys: title (str), author (str), creation_date (YYYY-MM-DD or None),
    mod_date (YYYY-MM-DD or None). Empty dict on error or missing PyMuPDF.
    """
    result: dict[str, Any] = {
        "title": "",
        "author": "",
        "creation_date": None,
        "mod_date": None,
    }
    if not filepath:
        return result
    try:
        import fitz  # type: ignore[import-not-found]
    except Exception:
        return result
    path = Path(filepath)
    try:
        doc = fitz.open(path)
    except Exception as exc:
        logger.debug("Could not open PDF for metadata %s: %s", path, exc)
        return result
    try:
        meta = doc.metadata or {}
        result["title"] = (meta.get("title") or "").strip()
        result["author"] = (meta.get("author") or "").strip()
        for key, out_key in (
            ("creationDate", "creation_date"),
            ("modDate", "mod_date"),
        ):
            d = _parse_pdf_date(meta.get(key))
            result[out_key] = d.strftime("%Y-%m-%d") if d else None
    finally:
        closer = getattr(doc, "close", None)
        if callable(closer):
            closer()
    return result


def _extract_pages(doc: Any, path: Path, *, max_pages: int = 0) -> tuple[list[str], list[str]]:
    """Extract text from pages. Returns (pieces, errors)."""
    pieces: list[str] = []
    errors: list[str] = []
    limit = min(doc.page_count, max_pages) if max_pages > 0 else doc.page_count
    for page_number in range(limit):
        try:
            page = doc[page_number]
        except Exception as exc:
            msg = f"Error accessing page {page_number} in {path.name}: {exc}"
            logger.error(msg)
            errors.append(msg)
            continue

        # Prefer single strategy to avoid triple text (text + blocks + rawdict overlap)
        page_text = ""
        try:
            page_text = (page.get_text("text") or "").strip()
        except Exception as exc:
            logger.warning(
                "Page %s get_text('text') failed in %s: %s",
                page_number,
                path,
                exc,
            )
        if not page_text:
            try:
                blocks = page.get_text("blocks") or []
                page_text = " ".join(b[4] for b in blocks if len(b) > 4 and str(b[4]).strip()).strip()
            except Exception as exc:
                logger.warning(
                    "Page %s get_text('blocks') failed in %s: %s",
                    page_number,
                    path,
                    exc,
                )
        if not page_text:
            try:
                rawdict = page.get_text("rawdict") or {}
                parts = []
                for block in rawdict.get("blocks", []):
                    for line in block.get("lines", []):
                        for span in line.get("spans", []):
                            t = span.get("text", "")
                            if t and t.strip():
                                parts.append(t.strip())
                page_text = " ".join(parts)
            except Exception as exc:
                msg = (
                    f"Page {page_number} get_text('rawdict') failed in {path.name}: {exc}. "
                    f"Use --ocr to try OCR extraction. Skipping {path.name}."
                )
                logger.info(msg)
                errors.append(msg)

        combined = page_text
        if combined:
            pieces.append(combined)
            logger.debug(
                "Combined extracted %s characters from page %s of %s",
                len(combined),
                page_number,
                path,
            )
        else:
            logger.info("Page %s in %s yields no text.", page_number, path)

    return pieces, errors
