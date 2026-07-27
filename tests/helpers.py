"""Small, reusable test seams that keep scenario tests focused on their contracts."""

from __future__ import annotations

import builtins
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import folionym.renamer as renamer
from folionym.config import RenamerConfig, build_config_from_flat_dict
from folionym.heuristics import (
    CategoryCombineOptions,
    ConflictResolutionOptions,
    _combine_resolve_conflict,
)
from folionym.heuristics import (
    combine_categories as _combine_categories,
)


def patch_renamer_process_result(
    monkeypatch: Any,
    _renamer_module: object,
    new_base: object,
    *,
    meta: dict[str, object] | None = None,
    error: BaseException | None = None,
) -> None:
    """Stub pipeline content processing while preserving the requested file path."""

    def fake_process(
        request: renamer.ContentProcessingRequest,
    ) -> tuple[Path, str | None, dict[str, object] | None, BaseException | None]:
        base = new_base(request.file_path) if callable(new_base) else new_base
        return (request.file_path, base, meta, error)  # type: ignore[return-value]

    monkeypatch.setattr(renamer, "_process_content_to_result", fake_process)


def make_filename_config(**values: object) -> RenamerConfig:
    """Build the flat test configuration accepted by filename-focused scenarios."""
    return build_config_from_flat_dict(values)


def resolve_category_conflict(cat_llm: str, cat_heuristic: str, **values: object) -> str:
    """Exercise conflict resolution with concise named overrides."""
    return _combine_resolve_conflict(cat_llm, cat_heuristic, ConflictResolutionOptions(**values))


def combine_test_categories(cat_llm: str, cat_heuristic: str, **values: object) -> str:
    """Exercise category combination with the optional scoring inputs used in tests."""
    return _combine_categories(
        cat_llm,
        cat_heuristic,
        CategoryCombineOptions(
            heuristic_score=values.get("heuristic_score"),
            heuristic_gap=values.get("heuristic_gap"),
            params=values.get("params"),
            context_for_overlap=values.get("context_for_overlap"),
        ),
    )


def install_fitz_document(monkeypatch: Any, page: object) -> MagicMock:
    """Install a one-page unencrypted fitz mock and return the document mock."""
    document = MagicMock()
    document.is_encrypted = False
    document.page_count = 1
    document.load_page.return_value = page
    fitz = MagicMock()
    fitz.open.return_value = document
    monkeypatch.setitem(sys.modules, "fitz", fitz)
    return document


def make_pdf(path: Path, name: str, payload: bytes = b"%PDF-1.4") -> Path:
    """Create a named minimal PDF payload for a scenario."""
    pdf_path = path / name
    pdf_path.write_bytes(payload)
    return pdf_path


def block_fitz_import(monkeypatch: Any) -> None:
    """Ensure a lazy ``import fitz`` fails without changing other imports."""
    monkeypatch.delitem(sys.modules, "fitz", raising=False)
    real_import = builtins.__import__

    def import_without_fitz(name: str, *args: object, **kwargs: object) -> object:
        if name == "fitz":
            raise ImportError("no fitz")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_fitz)
