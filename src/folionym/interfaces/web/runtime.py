"""Thread-safe background run registry for the local browser frontend."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from threading import Event, Lock, Thread
from typing import Literal
from uuid import uuid4

from ...application.models import ApplyReport, PreviewPlan
from ...application.reviewed_plan import apply_reviewed_plan, create_preview_plan
from ..ui_settings import build_config_from_ui_settings, save_ui_settings

logger = logging.getLogger(__name__)

RunKind = Literal["preview", "apply"]
RunState = Literal["queued", "running", "completed", "cancelled", "failed"]
TERMINAL_STATES = frozenset({"completed", "cancelled", "failed"})


@dataclass(frozen=True)
class RunEvent:
    """One ordered event retained for SSE reconnects."""

    sequence: int
    event: str
    payload: dict[str, object]


@dataclass
class RunRecord:
    """Mutable state for one local background operation."""

    id: str
    kind: RunKind
    state: RunState = "queued"
    completed: int = 0
    total: int = 0
    current_file: str = ""
    message: str = "Queued"
    plan_id: str | None = None
    report_id: str | None = None
    error: str | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    stop_event: Event = field(default_factory=Event)
    events: list[RunEvent] = field(default_factory=list)


class RunConflictError(RuntimeError):
    """Raised when a second filesystem operation is requested concurrently."""


class RunRegistry:
    """Own preview plans, apply reports, run state, and cooperative cancellation."""

    def __init__(self) -> None:
        """Initialize isolated run, plan, and report stores."""
        self._lock = Lock()
        self._runs: dict[str, RunRecord] = {}
        self._plans: dict[str, PreviewPlan] = {}
        self._reports: dict[str, ApplyReport] = {}

    def _active_run(self) -> RunRecord | None:
        """Return the sole nonterminal operation, when present."""
        return next((run for run in self._runs.values() if run.state not in TERMINAL_STATES), None)

    def _new_run(self, kind: RunKind) -> RunRecord:
        """Reserve the one active-operation slot."""
        with self._lock:
            active = self._active_run()
            if active is not None:
                raise RunConflictError(f"A {active.kind} run is already active.")
            run = RunRecord(uuid4().hex, kind)
            self._runs[run.id] = run
            self._emit_locked(run, "run.queued", {"kind": kind})
            return run

    @staticmethod
    def _snapshot_locked(run: RunRecord) -> dict[str, object]:
        """Serialize one run while its owner holds the registry lock."""
        return {
            "id": run.id,
            "kind": run.kind,
            "state": run.state,
            "completed": run.completed,
            "total": run.total,
            "current_file": run.current_file,
            "message": run.message,
            "plan_id": run.plan_id,
            "report_id": run.report_id,
            "error": run.error,
            "started_at": run.started_at.isoformat() if run.started_at else None,
            "completed_at": run.completed_at.isoformat() if run.completed_at else None,
        }

    def _emit_locked(self, run: RunRecord, event: str, payload: dict[str, object]) -> None:
        """Append one ordered event while holding the registry lock."""
        sequence = run.events[-1].sequence + 1 if run.events else 1
        run.events.append(RunEvent(sequence, event, payload))

    def _set_running(self, run: RunRecord, message: str) -> None:
        """Move a queued operation into its running state."""
        with self._lock:
            run.state = "running"
            run.started_at = datetime.now(UTC)
            run.message = message
            self._emit_locked(run, "run.started", self._snapshot_locked(run))

    def _progress(self, run: RunRecord, completed: int, total: int, path: Path) -> None:
        """Record progress for the latest completed file."""
        with self._lock:
            run.completed = completed
            run.total = total
            run.current_file = path.name
            run.message = f"Processing {completed} of {total}"
            self._emit_locked(run, "run.progress", self._snapshot_locked(run))

    def _finish(self, run: RunRecord, *, message: str) -> None:
        """Finalize a successful or cooperatively cancelled operation."""
        with self._lock:
            run.state = "cancelled" if run.stop_event.is_set() else "completed"
            run.message = message
            run.completed_at = datetime.now(UTC)
            event = "run.cancelled" if run.state == "cancelled" else "run.completed"
            self._emit_locked(run, event, self._snapshot_locked(run))

    def _fail(self, run: RunRecord, error: BaseException) -> None:
        """Finalize a failed operation and preserve a safe user-facing error."""
        logger.exception("Web %s run failed", run.kind, exc_info=error)
        with self._lock:
            run.state = "failed"
            run.error = str(error)
            run.message = "The operation could not be completed."
            run.completed_at = datetime.now(UTC)
            self._emit_locked(run, "run.failed", self._snapshot_locked(run))

    def start_preview(self, source: Path, settings: dict[str, object]) -> str:
        """Validate, persist, and start one structured Preview."""
        if not source.exists():
            raise FileNotFoundError(f"Source does not exist: {source}")
        if source.is_file() and source.suffix.lower() != ".pdf":
            raise ValueError("Single-file sources must be PDFs.")
        if not source.is_dir() and not source.is_file():
            raise ValueError("Source must be a folder or PDF.")

        run = self._new_run("preview")
        config = build_config_from_ui_settings(settings, run.stop_event, dry_run=True)
        save_ui_settings(settings)

        def worker() -> None:
            """Build and retain the Preview plan on a background thread."""
            self._set_running(run, "Building filename proposals")
            try:
                plan = create_preview_plan(
                    source,
                    config,
                    progress_callback=lambda current, total, path: self._progress(run, current, total, path),
                )
                with self._lock:
                    self._plans[plan.id] = plan
                    run.plan_id = plan.id
                    run.total = len(plan.items)
                    run.completed = len(plan.items)
                self._finish(run, message="Preview completed")
            except Exception as exc:
                self._fail(run, exc)

        Thread(target=worker, name=f"folionym-preview-{run.id[:8]}", daemon=True).start()
        return run.id

    def start_apply(self, plan_id: str, revision: int, selected_ids: list[str]) -> str:
        """Start exact reviewed-plan application."""
        plan = self.get_plan(plan_id)
        if plan.revision != revision:
            raise ValueError("The preview plan revision is stale.")
        run = self._new_run("apply")

        def worker() -> None:
            """Apply and retain the reviewed-plan report on a background thread."""
            self._set_running(run, "Applying reviewed filenames")
            try:
                report = apply_reviewed_plan(
                    plan,
                    selected_ids,
                    stop_event=run.stop_event,
                    progress_callback=lambda current, total, path: self._progress(run, current, total, path),
                )
                with self._lock:
                    self._reports[report.id] = report
                    run.report_id = report.id
                    run.total = len(selected_ids)
                    run.completed = min(run.completed, run.total) if run.stop_event.is_set() else run.total
                self._finish(run, message="Reviewed changes complete")
            except Exception as exc:
                self._fail(run, exc)

        Thread(target=worker, name=f"folionym-apply-{run.id[:8]}", daemon=True).start()
        return run.id

    def cancel(self, run_id: str) -> dict[str, object]:
        """Request cooperative cancellation and return the updated snapshot."""
        with self._lock:
            run = self._runs.get(run_id)
            if run is None:
                raise KeyError(run_id)
            if run.state in TERMINAL_STATES:
                return self._snapshot_locked(run)
            run.stop_event.set()
            run.message = "Cancelling after the active file"
            self._emit_locked(run, "run.cancelling", self._snapshot_locked(run))
            return self._snapshot_locked(run)

    def snapshot(self, run_id: str) -> dict[str, object]:
        """Return one immutable HTTP-friendly run snapshot."""
        with self._lock:
            run = self._runs.get(run_id)
            if run is None:
                raise KeyError(run_id)
            return self._snapshot_locked(run)

    def events_after(self, run_id: str, sequence: int) -> tuple[list[RunEvent], bool]:
        """Return retained events after ``sequence`` and whether the run is terminal."""
        with self._lock:
            run = self._runs.get(run_id)
            if run is None:
                raise KeyError(run_id)
            return ([event for event in run.events if event.sequence > sequence], run.state in TERMINAL_STATES)

    def get_plan(self, plan_id: str) -> PreviewPlan:
        """Return a retained preview plan."""
        with self._lock:
            plan = self._plans.get(plan_id)
        if plan is None:
            raise KeyError(plan_id)
        return plan

    def get_report(self, report_id: str) -> ApplyReport:
        """Return a retained apply report."""
        with self._lock:
            report = self._reports.get(report_id)
        if report is None:
            raise KeyError(report_id)
        return report
