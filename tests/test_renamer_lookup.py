"""Tests for renamer_lookup._lookup_override_category."""

from __future__ import annotations

from pathlib import Path

from ai_pdf_renamer.renamer_lookup import _lookup_override_category


def test_returns_none_for_none_map(tmp_path: Path) -> None:
    assert _lookup_override_category(tmp_path / "file.pdf", None) is None


def test_returns_none_for_empty_map(tmp_path: Path) -> None:
    assert _lookup_override_category(tmp_path / "file.pdf", {}) is None


def test_matches_by_filename(tmp_path: Path) -> None:
    path = tmp_path / "invoice.pdf"
    result = _lookup_override_category(path, {"invoice.pdf": "finance"})
    assert result == "finance"


def test_matches_by_str_path(tmp_path: Path) -> None:
    path = tmp_path / "invoice.pdf"
    result = _lookup_override_category(path, {str(path): "finance"})
    assert result == "finance"


def test_matches_by_posix_path(tmp_path: Path) -> None:
    path = tmp_path / "invoice.pdf"
    result = _lookup_override_category(path, {path.as_posix(): "tax"})
    assert result == "tax"


def test_matches_resolved_path(tmp_path: Path) -> None:
    path = tmp_path / "doc.pdf"
    path.touch()
    resolved = path.resolve()
    result = _lookup_override_category(path, {resolved.as_posix(): "legal"})
    assert result == "legal"


def test_returns_none_when_no_match(tmp_path: Path) -> None:
    path = tmp_path / "other.pdf"
    result = _lookup_override_category(path, {"invoice.pdf": "finance"})
    assert result is None


def test_matches_normalized_suffix(tmp_path: Path) -> None:
    path = tmp_path / "docs" / "report.pdf"
    result = _lookup_override_category(path, {"docs/report.pdf": "archive"})
    assert result == "archive"


def test_skips_normalized_key_without_slash(tmp_path: Path) -> None:
    # Keys without '/' are skipped in the normalized-suffix path
    path = tmp_path / "report.pdf"
    result = _lookup_override_category(path, {"report.pdf": "archive"})
    # Should still match via direct filename lookup, not normalized path
    assert result == "archive"


def test_backslash_normalized_in_map_key(tmp_path: Path) -> None:
    path = tmp_path / "docs" / "report.pdf"
    # Windows-style path in map key — should normalize to forward slash
    win_key = "docs\\report.pdf"
    result = _lookup_override_category(path, {win_key: "archive"})
    assert result == "archive"
