"""Pure PDF metadata normalization helpers.

The extraction facade owns optional PyMuPDF access and document lifetime. This
module keeps date parsing and result-shape normalization independent from that
runtime dependency.
"""

from __future__ import annotations

import logging
import re
from datetime import date
from typing import Any

logger = logging.getLogger(__name__)

_PDF_DATE_PREFIX = re.compile(r"^D:(\d{4})(\d{2})(\d{2})")


def _parse_pdf_date(value: str | None) -> date | None:
    """Parse a PDF ``D:YYYYMMDD`` prefix into a date when valid."""
    if not value or not isinstance(value, str):
        return None
    match = _PDF_DATE_PREFIX.match(value.strip())
    if not match:
        logger.debug("PDF metadata date field exists but does not match expected D:YYYYMMDD format: %r", value)
        return None
    try:
        year, month, day = (int(part) for part in match.groups())
        return date(year, month, day)
    except (TypeError, ValueError) as exc:
        logger.debug("Failed to create date object from PDF metadata %r: %s", value, exc)
        return None


def _empty_pdf_metadata() -> dict[str, object]:
    """Return the stable metadata shape for unavailable or unreadable PDFs."""
    return {
        "title": "",
        "author": "",
        "creation_date": None,
        "mod_date": None,
    }


def _metadata_result_from_doc(doc: Any) -> dict[str, object]:
    """Normalize title, author, and metadata dates from an open PDF."""
    metadata = doc.metadata or {}
    result = _empty_pdf_metadata()
    result["title"] = (metadata.get("title") or "").strip()
    result["author"] = (metadata.get("author") or "").strip()
    result["creation_date"] = _metadata_date_string(metadata.get("creationDate"))
    result["mod_date"] = _metadata_date_string(metadata.get("modDate"))
    return result


def _metadata_date_string(value: str | None) -> str | None:
    """Parse a PDF metadata date and return its ISO representation."""
    parsed = _parse_pdf_date(value)
    return parsed.isoformat() if parsed else None
