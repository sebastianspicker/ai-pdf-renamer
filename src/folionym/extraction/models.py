"""Typed values exchanged between extraction adapters and strategies."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from ..llm.protocol import LLMClient
from ..settings.models import RenamerConfig


@dataclass(frozen=True)
class OcrExtractionRequest:
    """Parameters for one OCR fallback attempt."""

    path: Path
    original_text: str
    max_tokens: int
    max_pages: int
    language: str


@dataclass(frozen=True)
class ExtractionFunctions:
    """Injectable adapters used by configured extraction strategies and tests."""

    image_fn: Callable[..., str | dict[str, str] | None]
    pdf_to_text_fn: Callable[..., str]
    pdf_to_text_with_ocr_fn: Callable[..., str]
    prompt_fn: Callable[..., str] | None = None
    sanitize_fn: Callable[..., str] | None = None


@dataclass(frozen=True)
class VisionExtractionRequest:
    """Inputs required for one first-page vision extraction request."""

    path: Path
    config: RenamerConfig
    client: LLMClient
    extraction_fns: ExtractionFunctions


@dataclass(frozen=True)
class VisionAttempt:
    """Outcome of a vision-first attempt and its reusable caller-owned client."""

    content: str | None
    client: LLMClient | None
    attempted: bool


@dataclass(frozen=True)
class ExtractionResult:
    """Content selected by the extraction pipeline and its vision provenance."""

    content: str
    used_vision: bool

    def as_tuple(self) -> tuple[str, bool]:
        """Return the established renamer-facing extraction contract."""
        return self.content, self.used_vision
