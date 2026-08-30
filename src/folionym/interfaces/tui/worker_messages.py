"""Typed worker-to-UI messages for the Textual interface."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from textual.message import Message as _MessageBase

if TYPE_CHECKING:
    from ...application.models import ApplyReport, PreviewPlan


class _RunLog(_MessageBase):
    """A single-file worker line rendered on Textual's UI thread."""

    def __init__(self, line: str) -> None:
        super().__init__()
        self.line = line


class _RunFinished(_MessageBase):
    """Terminal result for the immediate single-file operation."""

    def __init__(self, ok: bool, message: str) -> None:
        super().__init__()
        self.ok = ok
        self.message = message


class _RunProgress(_MessageBase):
    """Structured plan progress independent of batch log formatting."""

    def __init__(self, current: int, total: int, source: Path) -> None:
        super().__init__()
        self.current = current
        self.total = total
        self.source = source


class _PreviewFinished(_MessageBase):
    """Retain the preview plan, including a visibly incomplete cancelled plan."""

    def __init__(self, plan: PreviewPlan, *, complete: bool) -> None:
        super().__init__()
        self.plan = plan
        self.complete = complete


class _PlanFailed(_MessageBase):
    """Report a preview or reviewed-apply failure without parsing log text."""

    def __init__(self, action: str, message: str) -> None:
        super().__init__()
        self.action = action
        self.message = message


class _ApplyFinished(_MessageBase):
    """Deliver the structured exact-plan apply report to the UI thread."""

    def __init__(self, report: ApplyReport) -> None:
        super().__init__()
        self.report = report
