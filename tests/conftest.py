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
    """Create a minimal valid PDF file in a temp directory."""
    pdf = tmp_path / "test.pdf"
    pdf.write_bytes(b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog >>\nendobj\n%%EOF")
    return pdf
