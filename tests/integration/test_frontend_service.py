"""Structured browser workflow tests."""

from __future__ import annotations

from pathlib import Path

import folionym.frontend_service as frontend_service
from folionym.config_resolver import build_config
from folionym.frontend_service import ApplyStatus, PreviewStatus, apply_reviewed_plan, create_preview_plan


def test_preview_plan_classifies_ready_review_and_failed_items(tmp_path: Path, monkeypatch) -> None:
    ready = tmp_path / "ready.pdf"
    review = tmp_path / "review.pdf"
    failed = tmp_path / "failed.pdf"
    for path in (ready, review, failed):
        path.write_bytes(b"%PDF")
    config = build_config({"use_llm": False, "dry_run": True})

    monkeypatch.setattr(
        frontend_service,
        "produce_rename_results",
        lambda files, _config, **_kwargs: [
            (files[0], "20260723-invoice-acme", {"category": "invoice"}, None),
            (files[1], "20260723-document", {"category": "document"}, None),
            (files[2], None, {}, ValueError("Unreadable PDF")),
        ],
    )

    plan = create_preview_plan(tmp_path, config)

    assert [item.status for item in plan.items] == [
        PreviewStatus.READY,
        PreviewStatus.REVIEW,
        PreviewStatus.FAILED,
    ]
    assert [item.included for item in plan.items] == [True, False, False]


def test_apply_reviewed_plan_uses_exact_names_and_preserves_unselected_items(tmp_path: Path, monkeypatch) -> None:
    first = tmp_path / "first.pdf"
    second = tmp_path / "second.pdf"
    first.write_bytes(b"%PDF first")
    second.write_bytes(b"%PDF second")
    config = build_config({"use_llm": False, "dry_run": True})

    monkeypatch.setattr(
        frontend_service,
        "produce_rename_results",
        lambda files, _config, **_kwargs: [
            (files[0], "approved-first", {"category": "invoice"}, None),
            (files[1], "approved-second", {"category": "invoice"}, None),
        ],
    )
    plan = create_preview_plan(tmp_path, config)
    unselected_source = plan.items[1].source

    report = apply_reviewed_plan(plan, [plan.items[0].id])

    assert report.items[0].status is ApplyStatus.RENAMED
    assert report.items[0].target_name == "approved-first.pdf"
    assert report.items[1].status is ApplyStatus.UNCHANGED
    assert (tmp_path / "approved-first.pdf").exists()
    assert unselected_source.exists()


def test_apply_reviewed_plan_rejects_target_created_after_preview(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "source.pdf"
    source.write_bytes(b"%PDF source")
    config = build_config({"use_llm": False, "dry_run": True})
    monkeypatch.setattr(
        frontend_service,
        "produce_rename_results",
        lambda files, _config, **_kwargs: [
            (files[0], "approved", {"category": "invoice"}, None),
        ],
    )
    plan = create_preview_plan(tmp_path, config)
    (tmp_path / "approved.pdf").write_bytes(b"%PDF occupied")

    report = apply_reviewed_plan(plan, [plan.items[0].id])

    assert report.items[0].status is ApplyStatus.FAILED
    assert report.items[0].reason == "The reviewed target already exists."
    assert source.exists()
