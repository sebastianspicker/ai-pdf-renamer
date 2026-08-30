"""Framework-free retained-plan state for the directory Textual workflow."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from ...application.models import ApplyReport, PreviewPlan, PreviewStatus


@dataclass(frozen=True)
class PreviewRow:
    """One direct projection of an immutable preview item into the TUI table."""

    item_id: str
    status: str
    source_name: str
    proposed_name: str
    detail: str


class DirectoryReviewedPlanController:
    """Retain only a complete directory plan and apply its exact ready targets."""

    def __init__(self) -> None:
        self._plan: PreviewPlan | None = None
        self._complete = False
        self._invalidation_reason: str | None = None

    @property
    def plan(self) -> PreviewPlan | None:
        """Return the immutable plan retained by the active Textual process."""
        return self._plan

    @property
    def complete(self) -> bool:
        """Return whether Preview completed without cancellation or invalidation."""
        return self._complete

    @property
    def invalidation_reason(self) -> str | None:
        """Explain why Apply needs a fresh Preview when one was invalidated."""
        return self._invalidation_reason

    def retain(self, plan: PreviewPlan, *, complete: bool) -> None:
        """Keep the typed plan, even when cancellation made it visibly incomplete."""
        self._plan = plan
        self._complete = complete
        self._invalidation_reason = None if complete else "Preview was cancelled before it completed."

    def invalidate(self, reason: str) -> None:
        """Discard an old plan when any material source or setting changes."""
        self._plan = None
        self._complete = False
        self._invalidation_reason = reason

    def clear(self) -> None:
        """Discard retained state without presenting an additional stale-plan reason."""
        self._plan = None
        self._complete = False
        self._invalidation_reason = None

    def rows(self) -> tuple[PreviewRow, ...]:
        """Render from PreviewPlan.items rather than from batch log strings."""
        if self._plan is None:
            return ()
        return tuple(
            PreviewRow(
                item_id=item.id,
                status=item.status.value.upper(),
                source_name=item.source.name,
                proposed_name=item.proposed_name or "No reviewed filename",
                detail=item.reason or self._row_detail(item.status),
            )
            for item in self._plan.items
        )

    @staticmethod
    def _row_detail(status: PreviewStatus) -> str:
        """Describe an item state without implying a confidence score."""
        return {
            PreviewStatus.READY: "Included for exact reviewed apply.",
            PreviewStatus.REVIEW: "Needs review and remains excluded in this TUI.",
            PreviewStatus.SKIPPED: "No rename will be applied.",
            PreviewStatus.FAILED: "Preview could not produce a reviewed name.",
        }[status]

    def ready_included_ids(self) -> tuple[str, ...]:
        """Return only ready items the canonical plan explicitly includes."""
        if self._plan is None:
            return ()
        return tuple(item.id for item in self._plan.items if item.included and item.status is PreviewStatus.READY)

    def require_apply(self) -> tuple[PreviewPlan, tuple[str, ...]]:
        """Validate that Apply can use the current immutable reviewed plan."""
        if self._plan is None:
            raise ValueError("Preview the folder before applying reviewed names.")
        if not self._complete:
            raise ValueError("Preview is incomplete or cancelled. Preview the folder again before applying.")
        item_ids = self.ready_included_ids()
        if not item_ids:
            raise ValueError("Preview has no included ready names to apply.")
        return self._plan, item_ids

    def apply(self, apply_plan: Callable[[PreviewPlan, tuple[str, ...]], ApplyReport]) -> ApplyReport:
        """Invoke the canonical exact-plan apply operation with retained ready IDs."""
        plan, item_ids = self.require_apply()
        return apply_plan(plan, item_ids)
