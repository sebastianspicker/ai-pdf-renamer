"""Low-level PDF extraction adapters.

Higher-level modules decide when OCR or vision should run; this module handles
PyMuPDF text/metadata access, optional OCR fallback, and first-page rendering.
"""

from __future__ import annotations

import base64
import contextlib
import logging
import os
import sys
import tempfile
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .metadata import empty_pdf_metadata, normalize_pdf_metadata
from .models import OcrExtractionRequest

if TYPE_CHECKING:
    import fitz as _fitz_mod

logger = logging.getLogger(__name__)

# Minimum extracted characters below which we try OCR (image-only PDFs).
MIN_CHARS_BEFORE_OCR = 50

# Default DPI for first-page render when using vision fallback.
VISION_FALLBACK_DPI = 300

# Default token limit for local LLMs: 32K is more compatible than 128K.
# reserve ~4K tokens for prompt + response.
DEFAULT_MAX_CONTENT_TOKENS = 28_000

# Token-shrink loop constants
_DENSITY_BUFFER_FACTOR = 1.1  # 10% over-estimate when jumping to approximate target length
_MIN_SHRINK_TEXT_LEN = 200  # Stop shrinking below this many characters
_SHRINK_FACTOR = 0.95  # Remove ~5% of text per fine-tuning iteration

# Cached tiktoken encoding (Any: tiktoken lacks type stubs, ignore_missing_imports applies).
_TIKTOKEN_MISSING = object()  # sentinel: import failed, don't retry
_tiktoken_encoding: Any = None
_tiktoken_lock = threading.Lock()


def _initialize_tiktoken_encoding() -> Any:
    """Load the tokenizer once, using a sentinel when it is unavailable."""
    # fmt: off
    try:
        import tiktoken

        return tiktoken.get_encoding("cl100k_base")
    except (ImportError, LookupError):
        return _TIKTOKEN_MISSING
    # fmt: on


def _get_tiktoken_encoding() -> Any:
    """Return the cached tokenizer, initializing it under the module lock."""
    module = sys.modules[__name__]
    encoding = module.__dict__["_tiktoken_encoding"]
    if encoding is None:
        with _tiktoken_lock:
            encoding = module.__dict__["_tiktoken_encoding"]
            if encoding is None:  # double-checked locking
                encoding = _initialize_tiktoken_encoding()
                module.__dict__["_tiktoken_encoding"] = encoding
    return encoding


def _token_count(text: str) -> int:
    """Count tokens with cached tiktoken support, falling back to a four-character heuristic."""
    encoding = _get_tiktoken_encoding()
    if encoding is not None and encoding is not _TIKTOKEN_MISSING:
        # fmt: off
        try:
            return len(encoding.encode(text))
        except (AttributeError, RuntimeError, ValueError):
            pass
        # fmt: on
    # Fallback heuristic: ~4 chars per token for typical text.
    return max(1, len(text) // 4)


def estimate_token_count(text: str) -> int:
    """Estimate token count for extracted PDF text."""
    return _token_count(text)


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
    target_char_count = int(max_tokens * density * _DENSITY_BUFFER_FACTOR)
    if target_char_count < len(text):
        text = text[:target_char_count]

    # Keep shrinking bounded; pathological tokenizers should not turn this into
    # an expensive loop.
    _MAX_SHRINK_ITERATIONS = 50
    for _ in range(_MAX_SHRINK_ITERATIONS):
        if _token_count(text) <= max_tokens or len(text) <= _MIN_SHRINK_TEXT_LEN:
            break
        new_len = int(len(text) * _SHRINK_FACTOR)
        # Prefer cut at last space to avoid mid-word truncation
        chunk = text[:new_len]
        last_space = chunk.rfind(" ")
        if last_space > len(text) // 2:
            new_len = last_space
        text = text[:new_len]
    return text


def shrink_to_token_limit(text: str, *, max_tokens: int) -> str:
    """Return text shortened to the configured token budget."""
    return _shrink_to_token_limit(text, max_tokens=max_tokens)


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
    fitz = _required_fitz_module()
    path = Path(filepath)
    pieces, errors, page_count = _extract_pdf_text_from_document(fitz, path, max_pages=max_pages)
    return _finalize_pdf_text(path, pieces, errors, page_count, max_tokens=max_tokens)


def _required_fitz_module() -> Any:
    """Import PyMuPDF or raise an installation error."""
    try:
        import fitz
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("PyMuPDF is required for PDF extraction. Install with: pip install -e '.[pdf]'") from exc
    return fitz


def _optional_fitz_module() -> Any | None:
    """Import PyMuPDF when available; otherwise return None."""
    try:
        import fitz
    except ImportError:
        return None
    return fitz


def _close_document(doc: Any) -> None:
    """Close a document when it exposes a callable close method."""
    closer = getattr(doc, "close", None)
    if callable(closer):
        closer()


def _open_pdf_for_text(fitz: Any, path: Path) -> Any:
    """Open a PDF for text extraction and normalize expected failures to OSError."""
    try:
        return fitz.open(path)
    except (RuntimeError, OSError, ValueError) as exc:
        raise OSError(f"Could not open PDF file {path.name}: {exc}") from exc


def _extract_pdf_text_from_document(fitz: Any, path: Path, *, max_pages: int) -> tuple[list[str], list[str], int]:
    """Extract page text and errors, returning no text for encrypted PDFs and always closing the document."""
    doc = _open_pdf_for_text(fitz, path)
    page_count = getattr(doc, "page_count", 0) or 0
    if max_pages > 0:
        page_count = min(page_count, max_pages)
    try:
        if getattr(doc, "is_encrypted", False):
            logger.warning("PDF %s is encrypted/password-protected. Skipping text extraction.", path.name)
            return [], [], page_count
        pieces, errors = _extract_pages(doc, path, max_pages=max_pages)
        return pieces, errors, page_count
    finally:
        _close_document(doc)


def _finalize_pdf_text(
    path: Path,
    pieces: list[str],
    errors: list[str],
    page_count: int,
    *,
    max_tokens: int,
) -> str:
    """Join extracted pages, enforce the token limit, or handle an empty extraction."""
    content = "\n".join(pieces).strip()
    if content:
        return _shrink_to_token_limit(content, max_tokens=max_tokens)
    _handle_empty_pdf_text(path, errors, page_count)
    return ""


def _handle_empty_pdf_text(path: Path, errors: list[str], page_count: int) -> None:
    """Raise or log the appropriate outcome for an empty PDF extraction."""
    if errors and page_count > 0:
        raise RuntimeError(
            f"Extraction failed for {path.name}: {len(errors)} error(s) occurred during page processing. "
            f"First error: {errors[0]}"
        )
    if page_count <= 0:
        return
    msg = (
        f"No text extracted from {path.name} ({page_count} page(s)). "
        "File may be encrypted, image-only, or extraction failed for all pages."
    )
    if _looks_like_image_only_pdf(path):
        raise ValueError(f"{msg} Consider using --ocr.")
    logger.warning(msg)


def _looks_like_image_only_pdf(path: Path) -> bool:
    """Use file size as a coarse image-only PDF heuristic."""
    try:
        return path.stat().st_size > 1024
    except OSError as exc:
        logger.debug("Could not stat %s for size check: %s", path.name, exc)
        return False


def _vision_render_payload(rendered: tuple[bytes, str] | None) -> dict[str, str] | None:
    """Base64-encode rendered image bytes with their MIME type."""
    if rendered is None:
        return None
    image_bytes, mime_type = rendered
    if not image_bytes:
        return None
    return {
        "image_b64": base64.b64encode(image_bytes).decode("ascii"),
        "mime_type": mime_type,
    }


def _open_pdf_for_vision(fitz: Any, path: Path) -> Any | None:
    """Open a PDF for vision rendering, returning None on expected failures."""
    try:
        return fitz.open(path)
    except (RuntimeError, OSError, ValueError) as exc:
        logger.debug("Could not open PDF for vision render %s: %s", path.name, exc)
        return None


def _render_vision_payload(doc: Any, path: Path, *, dpi: int) -> dict[str, str] | None:
    """Render a first-page vision payload, returning None and logging expected failures."""
    try:
        return _vision_render_payload(_render_first_page_for_vision(doc, path, dpi=dpi))
    except (RuntimeError, OSError, ValueError) as exc:
        logger.debug("Vision render failed for %s: %s", path.name, exc)
        return None


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
    image_payload = pdf_first_page_to_image_payload(filepath, dpi=dpi)
    if image_payload is None:
        return None
    return image_payload["image_b64"]


def _encode_pixmap_for_vision(pix: Any) -> tuple[bytes, str] | None:
    """Encode a pixmap as JPEG when supported, falling back to PNG."""
    if hasattr(pix, "tobytes"):
        # fmt: off
        try:
            return (pix.tobytes(output="jpeg", jpg_quality=85), "image/jpeg")
        except (TypeError, ValueError):
            return (pix.tobytes(output="png"), "image/png")
        # fmt: on
    if hasattr(pix, "getImageData"):
        return (pix.getImageData("jpeg"), "image/jpeg")
    if hasattr(pix, "getPNGData"):
        return (pix.getPNGData(), "image/png")
    return None


def _render_first_page_for_vision(doc: Any, path: Path, *, dpi: int) -> tuple[bytes, str] | None:
    """Render and encode the first page of a non-encrypted PDF for vision."""
    if getattr(doc, "is_encrypted", False):
        logger.debug("PDF %s is encrypted; skipping vision render.", path.name)
        return None
    page_count = getattr(doc, "page_count", 0) or 0
    if page_count == 0:
        return None
    page = doc.load_page(0)
    pix = page.get_pixmap(dpi=dpi, alpha=False)
    return _encode_pixmap_for_vision(pix)


def pdf_first_page_to_image_payload(
    filepath: str | Path | None,
    *,
    dpi: int = VISION_FALLBACK_DPI,
) -> dict[str, str] | None:
    """Render the first page and preserve the actual MIME type for vision requests."""
    if filepath is None:
        return None
    fitz = _optional_fitz_module()
    if fitz is None:
        return None
    path = Path(filepath)
    doc = _open_pdf_for_vision(fitz, path)
    if doc is None:
        return None
    try:
        return _render_vision_payload(doc, path, dpi=dpi)
    finally:
        _close_document(doc)


def _ocr_language_code(lang: str) -> str:
    """Map config language (de/en) to Tesseract/OCRmyPDF language code.
    Can be overridden by FOLIONYM_OCR_LANG.
    """
    override = (os.environ.get("FOLIONYM_OCR_LANG") or "").strip()
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
    text = _initial_text_for_ocr(filepath, max_tokens=max_tokens, max_pages=max_pages)
    if not _should_attempt_ocr(filepath, text, min_chars_for_ocr=min_chars_for_ocr):
        return text
    ocrmypdf = _ocrmypdf_module_or_none()
    if ocrmypdf is None:
        return text

    if filepath is None:
        return text
    path = Path(filepath)
    if not path.exists() or not path.is_file():
        return text
    return _ocr_text_or_original(
        ocrmypdf,
        OcrExtractionRequest(
            path=path,
            original_text=text,
            max_tokens=max_tokens,
            max_pages=max_pages,
            language=language,
        ),
    )


def _initial_text_for_ocr(filepath: str | Path | None, *, max_tokens: int, max_pages: int) -> str:
    """Try ordinary extraction first; return empty text after expected failures so OCR can run."""
    try:
        return pdf_to_text(
            filepath,
            max_tokens=max_tokens,
            max_pages=max_pages,
        )
    except (RuntimeError, ValueError) as exc:
        # Extraction failed entirely, so proceed to OCR if available.
        logger.info("Text extraction failed for %s, will try OCR: %s", filepath, exc)
        return ""


def _should_attempt_ocr(
    filepath: str | Path | None,
    text: str,
    *,
    min_chars_for_ocr: int,
) -> bool:
    """Return whether a supplied path produced less than the OCR text threshold."""
    return bool(filepath) and len(text.strip()) < min_chars_for_ocr


def _ocrmypdf_module_or_none() -> Any | None:
    """Import OCRmyPDF or warn and return None."""
    try:
        import ocrmypdf
    except ImportError:
        logger.warning(
            "OCR requested but ocrmypdf not installed. Install with: pip install -e '.[ocr]' (and install Tesseract)."
        )
        return None
    return ocrmypdf


def _ocr_text_or_original(
    ocrmypdf: Any,
    request: OcrExtractionRequest,
) -> str:
    """Run OCR and return its text, falling back to the original extraction on expected failures."""
    tmp = None
    try:
        tmp = _create_ocr_temp_path()
        _run_ocr_to_temp(ocrmypdf, request.path, tmp, language=request.language)
        text_ocr = _extract_ocr_temp_text(
            tmp,
            request.path,
            max_tokens=request.max_tokens,
            max_pages=request.max_pages,
        )
        if text_ocr is not None:
            return text_ocr
    except (RuntimeError, OSError, ValueError) as exc:
        logger.warning("OCR failed for %s: %s. Using original extraction.", request.path, exc)
    finally:
        _remove_ocr_temp_path(tmp)
    return request.original_text


def _create_ocr_temp_path() -> Path:
    """Create a temporary PDF path for OCRmyPDF output."""
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False, prefix="folionym_ocr_") as f:
        return Path(f.name)


def _run_ocr_to_temp(ocrmypdf: Any, path: Path, tmp: Path, *, language: str) -> None:
    """Run OCRmyPDF and best-effort restrict its replacement output to mode 0600."""
    ocrmypdf.ocr(
        str(path),
        str(tmp),
        language=_ocr_language_code(language),
    )
    # ocrmypdf typically renames its own temp output into `tmp`, creating a new inode
    # with umask-based permissions. Restore 0600 so the OCR output isn't world-readable.
    with contextlib.suppress(OSError):
        tmp.chmod(0o600)


def _extract_ocr_temp_text(tmp: Path, source_path: Path, *, max_tokens: int, max_pages: int) -> str | None:
    """Extract text from OCR output, returning None when it remains empty."""
    text_ocr = pdf_to_text(tmp, max_tokens=max_tokens, max_pages=max_pages)
    if not text_ocr.strip():
        return None
    logger.info("OCR produced %s chars for %s", len(text_ocr.strip()), source_path.name)
    return text_ocr


def _remove_ocr_temp_path(tmp: Path | None) -> None:
    """Best-effort remove the OCR temporary file."""
    if tmp is not None and tmp.exists():
        with contextlib.suppress(OSError):
            tmp.unlink()


def get_pdf_metadata(filepath: str | Path | None) -> dict[str, object]:
    """
    Read PDF metadata (Title, Author, CreationDate, ModDate) without extracting text.
    Returns dict with keys: title (str), author (str), creation_date (YYYY-MM-DD or None),
    mod_date (YYYY-MM-DD or None). Empty dict on error or missing PyMuPDF.
    """
    result = empty_pdf_metadata()
    if not filepath:
        return result
    fitz = _optional_fitz_module()
    if fitz is None:
        return result
    path = Path(filepath)
    doc = _open_pdf_for_metadata(fitz, path)
    if doc is None:
        return result
    try:
        return normalize_pdf_metadata(doc)
    finally:
        _close_document(doc)


def _open_pdf_for_metadata(fitz: Any, path: Path) -> Any | None:
    """Open a PDF for metadata, returning None and debug-logging expected failures."""
    try:
        return fitz.open(path)
    except (RuntimeError, OSError, ValueError) as exc:
        logger.debug("Could not open PDF for metadata %s: %s", path, exc)
        return None


def _extract_pages(doc: _fitz_mod.Document, path: Path, *, max_pages: int = 0) -> tuple[list[str], list[str]]:
    """Extract text from pages. Returns (pieces, errors)."""
    pieces: list[str] = []
    errors: list[str] = []
    limit = min(doc.page_count, max_pages) if max_pages > 0 else doc.page_count
    for page_number in range(limit):
        try:
            # load_page() is stable across supported PyMuPDF versions.
            page = doc.load_page(page_number)
        except (IndexError, RuntimeError, OSError, ValueError) as exc:
            msg = f"Error accessing page {page_number} in {path.name}: {exc}"
            logger.error(msg)
            errors.append(msg)
            continue

        page_text = ""
        try:
            page_text = (page.get_text("text") or "").strip()
        except (RuntimeError, OSError, ValueError) as exc:
            msg = f"Page {page_number} text extraction failed in {path.name}: {exc}. Use --ocr to try OCR extraction."
            logger.warning(msg)
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


def extract_pages(doc: Any, path: Path, *, max_pages: int = 0) -> tuple[list[str], list[str]]:
    """Extract text fragments and page-level errors from an opened PDF document."""
    return _extract_pages(doc, path, max_pages=max_pages)
