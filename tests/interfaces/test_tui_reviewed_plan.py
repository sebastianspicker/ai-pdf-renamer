"""Contract tests for the Textual directory reviewed-plan controller."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from folionym.application.models import ApplyReport, PreviewItem, PreviewPlan, PreviewStatus
from folionym.config import build_config
from folionym.interfaces.tui.reviewed_plan import DirectoryReviewedPlanController


def _plan(tmp_path: Path) -> PreviewPlan:
    config = build_config({"use_llm": False, "dry_run": True})
    return PreviewPlan(
        id="preview-1",
        source=tmp_path,
        source_kind="directory",
        config=config,
        created_at=datetime.now(UTC),
        items=(
            PreviewItem(
                id="ready-included",
                source=tmp_path / "ready.pdf",
                proposed_base="ready-reviewed",
                metadata={},
                status=PreviewStatus.READY,
                included=True,
                fingerprint=None,
            ),
            PreviewItem(
                id="ready-excluded",
                source=tmp_path / "excluded.pdf",
                proposed_base="excluded-reviewed",
                metadata={},
                status=PreviewStatus.READY,
                included=False,
                fingerprint=None,
            ),
            PreviewItem(
                id="review",
                source=tmp_path / "review.pdf",
                proposed_base="review-reviewed",
                metadata={},
                status=PreviewStatus.REVIEW,
                included=True,
                fingerprint=None,
                reason="Needs an operator decision.",
            ),
        ),
    )


def test_retained_plan_rows_map_directly_from_immutable_items_and_apply_exact_ready_ids(tmp_path: Path) -> None:
    controller = DirectoryReviewedPlanController()
    plan = _plan(tmp_path)
    controller.retain(plan, complete=True)

    rows = controller.rows()
    assert [(row.item_id, row.status, row.source_name, row.proposed_name) for row in rows] == [
        ("ready-included", "READY", "ready.pdf", "ready-reviewed.pdf"),
        ("ready-excluded", "READY", "excluded.pdf", "excluded-reviewed.pdf"),
        ("review", "REVIEW", "review.pdf", "review-reviewed.pdf"),
    ]
    seen: list[tuple[PreviewPlan, tuple[str, ...]]] = []

    def apply_exact(retained_plan: PreviewPlan, item_ids: tuple[str, ...]) -> ApplyReport:
        seen.append((retained_plan, item_ids))
        return ApplyReport(
            id="report-1",
            plan_id=retained_plan.id,
            source=retained_plan.source,
            started_at=datetime.now(UTC),
            completed_at=datetime.now(UTC),
            items=(),
        )

    assert controller.apply(apply_exact) is not None
    assert seen == [(plan, ("ready-included",))]


def test_material_control_invalidation_discards_the_retained_plan(tmp_path: Path) -> None:
    controller = DirectoryReviewedPlanController()
    controller.retain(_plan(tmp_path), complete=True)

    controller.invalidate("Source or settings changed after Preview.")

    assert controller.plan is None
    assert controller.rows() == ()
    assert controller.invalidation_reason == "Source or settings changed after Preview."
    with pytest.raises(ValueError, match="Preview the folder"):
        controller.require_apply()


def test_no_plan_and_partial_plan_refuse_apply_without_calling_a_mutator(tmp_path: Path) -> None:
    controller = DirectoryReviewedPlanController()
    calls = 0

    def mutator(_plan: PreviewPlan, _item_ids: tuple[str, ...]) -> ApplyReport:
        nonlocal calls
        calls += 1
        return ApplyReport(
            id="report-2",
            plan_id=_plan.id,
            source=_plan.source,
            started_at=datetime.now(UTC),
            completed_at=datetime.now(UTC),
            items=(),
        )

    with pytest.raises(ValueError, match="Preview the folder"):
        controller.apply(mutator)

    controller.retain(_plan(tmp_path), complete=False)
    with pytest.raises(ValueError, match="incomplete or cancelled"):
        controller.apply(mutator)

    assert calls == 0
