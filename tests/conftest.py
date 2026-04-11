from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from ai_pdf_renamer.config import RenamerConfig


def make_config(**overrides: Any) -> RenamerConfig:
    """Build a RenamerConfig with sensible test defaults and optional overrides."""
    defaults: dict[str, Any] = {
        "use_llm": False,
        "use_single_llm_call": False,
        "dry_run": False,
    }
    defaults.update(overrides)
    return RenamerConfig(**defaults)


@pytest.fixture
def default_config() -> RenamerConfig:
    """A default RenamerConfig with LLM disabled for fast tests."""
    return make_config()


@pytest.fixture
def tmp_pdf(tmp_path: Path) -> Path:
    """Create a minimal syntactically valid PDF file in a temp directory."""
    pdf = tmp_path / "test.pdf"
    # Minimal valid PDF with proper xref table and trailer
    pdf.write_bytes(
        b"%PDF-1.4\n"
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
        b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>\nendobj\n"
        b"xref\n0 4\n"
        b"0000000000 65535 f \n"
        b"0000000009 00000 n \n"
        b"0000000058 00000 n \n"
        b"0000000115 00000 n \n"
        b"trailer\n<< /Size 4 /Root 1 0 R >>\n"
        b"startxref\n190\n%%EOF\n"
    )
    return pdf
