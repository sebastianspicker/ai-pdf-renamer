"""Rename planning, collision, and completed-operation behavior."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from folionym.rename_ops import RenameApplyOptions, apply_single_rename
from tests.conftest import rename_pdf


def test_concurrent_renames_same_target_no_overwrite(tmp_path: Path) -> None:
    sources = []
    for number in range(5):
        source = tmp_path / f"source_{number}.pdf"
        source.write_text(str(number), encoding="utf-8")
        sources.append(source)
    occupied = tmp_path / "report.pdf"
    occupied.write_text("occupied", encoding="utf-8")
    with ThreadPoolExecutor(max_workers=5) as pool:
        results = list(pool.map(lambda source: rename_pdf(source, "report"), sources))
    assert all(success for success, _ in results)
    assert len({target for _, target in results}) == len(results)
    assert occupied.read_text(encoding="utf-8") == "occupied"


def test_rename_collision_suffix_increments(tmp_path: Path) -> None:
    (tmp_path / "invoice.pdf").write_text("occupied", encoding="utf-8")
    first, second = tmp_path / "a.pdf", tmp_path / "b.pdf"
    first.write_text("first", encoding="utf-8")
    second.write_text("second", encoding="utf-8")
    assert rename_pdf(first, "invoice")[1].name == "invoice_1.pdf"
    assert rename_pdf(second, "invoice")[1].name == "invoice_2.pdf"


def test_rename_dry_run_no_filesystem_change(tmp_path: Path) -> None:
    source = tmp_path / "original.pdf"
    source.write_text("keep", encoding="utf-8")
    before = sorted(tmp_path.iterdir())
    assert rename_pdf(source, "new", dry_run=True)[0]
    assert sorted(tmp_path.iterdir()) == before


def test_rename_dry_run_uses_same_collision_suffix_as_apply(tmp_path: Path) -> None:
    source = tmp_path / "original.pdf"
    source.write_text("keep", encoding="utf-8")
    (tmp_path / "new.pdf").write_text("occupied", encoding="utf-8")
    preview = rename_pdf(source, "new", dry_run=True)
    applied = rename_pdf(source, "new")
    assert preview[1].name == "new_1.pdf"
    assert applied[1] == preview[1]


def test_rename_plan_uses_same_collision_suffix_as_apply(tmp_path: Path) -> None:
    source = tmp_path / "original.pdf"
    source.write_text("keep", encoding="utf-8")
    (tmp_path / "new.pdf").write_text("occupied", encoding="utf-8")
    entries: list[dict[str, str]] = []
    preview = rename_pdf(source, "new", dry_run=True, plan_file_path=tmp_path / "plan.json", plan_entries=entries)
    assert entries == [{"old": str(source), "new": str(preview[1])}]
    assert rename_pdf(source, "new")[1] == preview[1]


def test_rename_dry_run_treats_dangling_symlink_as_collision(tmp_path: Path) -> None:
    source = tmp_path / "original.pdf"
    source.write_text("keep", encoding="utf-8")
    target = tmp_path / "new.pdf"
    try:
        target.symlink_to(tmp_path / "missing.pdf")
    except NotImplementedError, OSError:
        pytest.skip("symlinks are unavailable on this platform")
    assert rename_pdf(source, "new", dry_run=True)[1].name == "new_1.pdf"
    assert rename_pdf(source, "new")[1].name == "new_1.pdf"


def test_rename_callback_failure_does_not_mark_completed_rename_as_failed(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    source = tmp_path / "original.pdf"
    source.write_text("keep", encoding="utf-8")
    with caplog.at_level("ERROR", logger="folionym.rename_ops"):
        success, target = rename_pdf(source, "new", on_success=lambda *_: (_ for _ in ()).throw(RuntimeError("failed")))
    assert success and target.exists() and not source.exists()
    assert "Rename completed but its post-rename callback failed" in caplog.text


def test_path_traversal_blocked(tmp_path: Path) -> None:
    source = tmp_path / "legit.pdf"
    source.write_text("payload", encoding="utf-8")
    with pytest.raises(ValueError):
        rename_pdf(source, "../escape")
    assert source.read_text(encoding="utf-8") == "payload"


def test_apply_single_rename_plan_entries_none(tmp_path: Path) -> None:
    source = tmp_path / "doc.pdf"
    source.write_text("content", encoding="utf-8")
    success, target = rename_pdf(source, "planned", plan_file_path=tmp_path / "plan.json", plan_entries=None)
    assert success and target.name == "planned.pdf" and source.exists()


def test_exact_target_collision_does_not_select_a_suffix(tmp_path: Path) -> None:
    source = tmp_path / "doc.pdf"
    source.write_text("content", encoding="utf-8")
    occupied = tmp_path / "renamed.pdf"
    occupied.write_text("occupied", encoding="utf-8")
    success, target = apply_single_rename(source, "renamed", RenameApplyOptions(exact_target=True))
    assert not success and target == occupied and source.exists()
