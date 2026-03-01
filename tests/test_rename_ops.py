"""Tests for rename and filename-sanitization helpers."""

from __future__ import annotations

from pathlib import Path

from ai_pdf_renamer.rename_ops import (
    apply_single_rename,
    sanitize_filename_base,
    sanitize_filename_from_llm,
)


def test_sanitize_filename_from_llm_empty() -> None:
    assert sanitize_filename_from_llm("") == "document"
    assert sanitize_filename_from_llm("   ") == "document"


def test_sanitize_filename_from_llm_unsafe_chars() -> None:
    assert sanitize_filename_from_llm('a/b\\c:d*e?f"g<h>i|j') == "a_b_c_d_e_f_g_h_i_j"


def test_sanitize_filename_from_llm_strip_pdf() -> None:
    assert sanitize_filename_from_llm("INVOICE_AMAZON.PDF") == "INVOICE_AMAZON"
    assert sanitize_filename_from_llm("doc.pdf") == "doc"


def test_sanitize_filename_from_llm_newlines_and_spaces() -> None:
    assert sanitize_filename_from_llm("a  b\nc\rd") == "a_b_c_d"


def test_sanitize_filename_from_llm_length() -> None:
    long_str = "a" * 150
    result = sanitize_filename_from_llm(long_str)
    assert len(result) == 120


def test_sanitize_filename_from_llm_strip_leading_trailing_dots_underscores() -> None:
    assert sanitize_filename_from_llm("._only_._") == "only"
    assert sanitize_filename_from_llm("._.") == "document"


def test_sanitize_filename_base_unchanged() -> None:
    assert sanitize_filename_base("INVOICE_AMAZON_2023") == "INVOICE_AMAZON_2023"


def test_sanitize_filename_base_empty() -> None:
    assert sanitize_filename_base("") == "unnamed"
    assert sanitize_filename_base("   ") == "unnamed"


def test_apply_single_rename_backup_collision_does_not_overwrite(tmp_path: Path) -> None:
    src = tmp_path / "doc.pdf"
    src.write_text("new-content", encoding="utf-8")

    backup_dir = tmp_path / "backups"
    backup_dir.mkdir()
    existing_backup = backup_dir / "doc.pdf"
    existing_backup.write_text("old-content", encoding="utf-8")

    success, target = apply_single_rename(
        src,
        "renamed",
        plan_file_path=None,
        plan_entries=[],
        dry_run=False,
        backup_dir=backup_dir,
        on_success=None,
        max_filename_chars=None,
    )

    assert success is True
    assert target.exists()
    assert existing_backup.read_text(encoding="utf-8") == "old-content"
    assert (backup_dir / "doc_1.pdf").read_text(encoding="utf-8") == "new-content"
