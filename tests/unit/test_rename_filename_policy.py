"""Filename sanitization and path-boundary tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from folionym.rename_ops import is_path_within, sanitize_filename_base, sanitize_filename_from_llm
from folionym.rename_ops.naming import _validate_path_within_parent


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("", "document"),
        ("   ", "document"),
        ('a/b\\c:d*e?f"g<h>i|j', "a_b_c_d_e_f_g_h_i_j"),
        ("INVOICE_AMAZON.PDF", "INVOICE_AMAZON"),
        ("a  b\nc\rd", "a_b_c_d"),
        ("._only_._", "only"),
    ],
)
def test_sanitize_filename_from_llm(raw: str, expected: str) -> None:
    assert sanitize_filename_from_llm(raw) == expected


def test_sanitize_filename_from_llm_length() -> None:
    assert len(sanitize_filename_from_llm("a" * 150)) == 120


@pytest.mark.parametrize(("name", "expected"), [("INVOICE", "INVOICE"), ("", "unnamed"), ("   ", "unnamed")])
def test_sanitize_filename_base(name: str, expected: str) -> None:
    assert sanitize_filename_base(name) == expected


@pytest.mark.parametrize("reserved", ["CON", "NUL", "AUX", "PRN", "COM1", "LPT9", "con", "Nul"])
def test_sanitize_filename_base_windows_reserved(reserved: str) -> None:
    assert sanitize_filename_base(reserved) == f"{reserved}_"


def test_validate_path_within_parent_valid(tmp_path: Path) -> None:
    child = tmp_path / "subdir" / "file.pdf"
    assert _validate_path_within_parent(child, tmp_path) == child.resolve()


def test_validate_path_within_parent_traversal(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Path traversal detected"):
        _validate_path_within_parent(tmp_path / ".." / "escape.pdf", tmp_path)


def test_is_path_within_oserror_returns_false(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    path = tmp_path / "file.pdf"
    original = Path.resolve

    def fail_resolve(self: Path, *args: object, **kwargs: object) -> Path:
        if self == path:
            raise OSError("resolve failed")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", fail_resolve)
    assert not is_path_within(path, tmp_path)
