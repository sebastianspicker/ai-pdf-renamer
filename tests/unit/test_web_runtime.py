"""Behavioral tests for the browser run registry."""

from __future__ import annotations

import os
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from threading import Event
from typing import cast

import pytest

import folionym.web_runtime as web_runtime
from folionym.config import RenamerConfig
from folionym.frontend_service import ApplyReport, PreviewItem, PreviewPlan, PreviewStatus


class ImmediateThread:
    """Thread stand-in that exposes asynchronous outcomes synchronously."""

    def __init__(self, *, target: Callable[[], None], name: str, daemon: bool) -> None:
        self.target = target
        self.name = name
        self.daemon = daemon

    def start(self) -> None:
        self.target()


def _plan(source: Path, *, revision: int = 1) -> PreviewPlan:
    item = PreviewItem(
        id="item-1",
        source=source / "invoice.pdf",
        proposed_base="20260806-invoice-acme",
        metadata={"category": "invoice"},
        status=PreviewStatus.READY,
        included=True,
        fingerprint=None,
    )
    return PreviewPlan(
        id="plan-1",
        source=source,
        source_kind="directory",
        config=cast(RenamerConfig, object()),
        items=(item,),
        created_at=datetime.now(UTC),
        revision=revision,
    )


def _use_immediate_threads(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(web_runtime, "Thread", ImmediateThread)


def test_preview_retains_plan_and_emits_ordered_progress_events(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A completed preview exposes its plan and a replayable state-event sequence."""
    _use_immediate_threads(monkeypatch)
    source = tmp_path / "source"
    source.mkdir()
    settings: dict[str, object] = {"language": "en"}
    plan = _plan(source)
    captured: dict[str, object] = {}
    config = cast(RenamerConfig, object())

    def fake_config(snapshot: dict[str, object], stop_event: object, *, dry_run: bool) -> RenamerConfig:
        captured.update(snapshot=snapshot, stop_event=stop_event, dry_run=dry_run)
        return config

    def fake_preview(
        actual_source: Path,
        actual_config: RenamerConfig,
        *,
        progress_callback: Callable[[int, int, Path], None],
    ) -> PreviewPlan:
        assert actual_source == source
        assert actual_config is config
        progress_callback(1, 1, source / "nested" / "invoice.pdf")
        return plan

    monkeypatch.setattr(web_runtime, "build_config_from_snapshot", fake_config)
    monkeypatch.setattr(web_runtime, "save_ui_settings", lambda saved: captured.setdefault("saved", saved))
    monkeypatch.setattr(web_runtime, "create_preview_plan", fake_preview)

    registry = web_runtime.RunRegistry()
    run_id = registry.start_preview(source, settings)

    snapshot = registry.snapshot(run_id)
    events, terminal = registry.events_after(run_id, 0)
    assert captured["snapshot"] == settings
    assert captured["dry_run"] is True
    assert captured["saved"] == settings
    assert snapshot["state"] == "completed"
    assert snapshot["plan_id"] == plan.id
    assert snapshot["completed"] == snapshot["total"] == 1
    assert snapshot["current_file"] == "invoice.pdf"
    assert registry.get_plan(plan.id) is plan
    assert [event.sequence for event in events] == [1, 2, 3, 4]
    assert [event.event for event in events] == ["run.queued", "run.started", "run.progress", "run.completed"]
    assert events[2].payload["message"] == "Processing 1 of 1"
    assert terminal is True
    assert registry.events_after(run_id, events[-1].sequence) == ([], True)


def test_preview_cancellation_and_failure_are_visible_to_event_consumers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cancellation stays cooperative while exceptions yield a safe failed state."""
    _use_immediate_threads(monkeypatch)
    source = tmp_path / "source"
    source.mkdir()
    registry = web_runtime.RunRegistry()
    plan = _plan(source)

    monkeypatch.setattr(
        web_runtime,
        "build_config_from_snapshot",
        lambda _settings, _stop_event, *, dry_run: cast(RenamerConfig, object()),
    )
    monkeypatch.setattr(web_runtime, "save_ui_settings", lambda _settings: None)

    def cancel_during_preview(
        _source: Path,
        _config: RenamerConfig,
        *,
        progress_callback: Callable[[int, int, Path], None],
    ) -> PreviewPlan:
        progress_callback(1, 1, source / "invoice.pdf")
        active = next(run for run in registry._runs.values() if run.state == "running")
        cancelling = registry.cancel(active.id)
        assert cancelling["message"] == "Cancelling after the active file"
        return plan

    monkeypatch.setattr(web_runtime, "create_preview_plan", cancel_during_preview)

    cancelled_id = registry.start_preview(source, {})
    cancelled_events, cancelled_terminal = registry.events_after(cancelled_id, 0)
    assert registry.snapshot(cancelled_id)["state"] == "cancelled"
    assert [event.event for event in cancelled_events] == [
        "run.queued",
        "run.started",
        "run.progress",
        "run.cancelling",
        "run.cancelled",
    ]
    assert cancelled_terminal is True
    assert registry.cancel(cancelled_id)["state"] == "cancelled"

    monkeypatch.setattr(
        web_runtime,
        "create_preview_plan",
        lambda _source, _config, **_kwargs: (_ for _ in ()).throw(RuntimeError("unreadable file")),
    )

    failed_id = registry.start_preview(source, {})
    failed_snapshot = registry.snapshot(failed_id)
    failed_events, failed_terminal = registry.events_after(failed_id, 0)
    assert failed_snapshot["state"] == "failed"
    assert failed_snapshot["error"] == "unreadable file"
    assert failed_snapshot["message"] == "The operation could not be completed."
    assert failed_events[-1].event == "run.failed"
    assert failed_terminal is True


def test_apply_retains_report_and_rejects_stale_plan_revisions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Applying a reviewed plan stores its report and guards against stale revisions."""
    _use_immediate_threads(monkeypatch)
    source = tmp_path / "source"
    source.mkdir()
    registry = web_runtime.RunRegistry()
    plan = _plan(source, revision=3)
    registry._plans[plan.id] = plan
    report = ApplyReport(
        id="report-1",
        plan_id=plan.id,
        source=source,
        started_at=datetime.now(UTC),
        completed_at=datetime.now(UTC),
        items=(),
    )
    captured: dict[str, object] = {}

    def fake_apply(
        actual_plan: PreviewPlan,
        selected_ids: list[str],
        *,
        stop_event: object,
        progress_callback: Callable[[int, int, Path], None],
    ) -> ApplyReport:
        captured.update(plan=actual_plan, selected_ids=selected_ids, stop_event=stop_event)
        progress_callback(1, 1, source / "invoice.pdf")
        return report

    monkeypatch.setattr(web_runtime, "apply_reviewed_plan", fake_apply)

    with pytest.raises(ValueError, match="revision is stale"):
        registry.start_apply(plan.id, revision=2, selected_ids=["item-1"])

    run_id = registry.start_apply(plan.id, revision=3, selected_ids=["item-1"])
    snapshot = registry.snapshot(run_id)
    events, terminal = registry.events_after(run_id, 0)
    assert captured["plan"] is plan
    assert captured["selected_ids"] == ["item-1"]
    assert snapshot["state"] == "completed"
    assert snapshot["report_id"] == report.id
    assert snapshot["completed"] == snapshot["total"] == 1
    assert registry.get_report(report.id) is report
    assert [event.event for event in events] == ["run.queued", "run.started", "run.progress", "run.completed"]
    assert terminal is True

    def cancelled_apply(
        _plan: PreviewPlan,
        _selected_ids: list[str],
        *,
        stop_event: Event,
        progress_callback: Callable[[int, int, Path], None],
    ) -> ApplyReport:
        progress_callback(1, 2, source / "invoice.pdf")
        stop_event.set()
        return report

    monkeypatch.setattr(web_runtime, "apply_reviewed_plan", cancelled_apply)
    cancelled_id = registry.start_apply(plan.id, revision=3, selected_ids=["item-1", "item-2"])
    cancelled_snapshot = registry.snapshot(cancelled_id)
    cancelled_events, cancelled_terminal = registry.events_after(cancelled_id, 0)
    assert cancelled_snapshot["state"] == "cancelled"
    assert cancelled_snapshot["completed"] == 1
    assert cancelled_snapshot["total"] == 2
    assert cancelled_events[-1].event == "run.cancelled"
    assert cancelled_terminal is True

    def failing_apply(
        _plan: PreviewPlan,
        _selected_ids: list[str],
        *,
        stop_event: Event,
        progress_callback: Callable[[int, int, Path], None],
    ) -> ApplyReport:
        raise RuntimeError("target collision")

    monkeypatch.setattr(web_runtime, "apply_reviewed_plan", failing_apply)
    failed_id = registry.start_apply(plan.id, revision=3, selected_ids=["item-1"])
    assert registry.snapshot(failed_id)["state"] == "failed"
    assert registry.events_after(failed_id, 0)[0][-1].event == "run.failed"


def test_registry_rejects_invalid_sources_unknown_ids_and_active_conflicts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Validation failures and unknown resources never create misleading run state."""
    registry = web_runtime.RunRegistry()
    text_file = tmp_path / "notes.txt"
    text_file.write_text("not a PDF", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="Source does not exist"):
        registry.start_preview(tmp_path / "missing.pdf", {})
    with pytest.raises(ValueError, match="must be PDFs"):
        registry.start_preview(text_file, {})
    pipe = tmp_path / "source.pipe"
    os.mkfifo(pipe)
    with pytest.raises(ValueError, match="folder or PDF"):
        registry.start_preview(pipe, {})
    for action in (registry.cancel, registry.snapshot, registry.get_plan, registry.get_report):
        with pytest.raises(KeyError):
            action("missing")
    with pytest.raises(KeyError):
        registry.events_after("missing", 0)

    class PendingThread:
        def __init__(self, *, target: Callable[[], None], name: str, daemon: bool) -> None:
            self.target = target
            self.name = name
            self.daemon = daemon

        def start(self) -> None:
            return None

    source = tmp_path / "source"
    source.mkdir()
    monkeypatch.setattr(web_runtime, "Thread", PendingThread)
    monkeypatch.setattr(
        web_runtime,
        "build_config_from_snapshot",
        lambda _settings, _stop_event, *, dry_run: cast(RenamerConfig, object()),
    )
    monkeypatch.setattr(web_runtime, "save_ui_settings", lambda _settings: None)
    active_id = registry.start_preview(source, {})

    with pytest.raises(web_runtime.RunConflictError, match="preview run is already active"):
        registry.start_preview(source, {})
    assert registry.cancel(active_id)["state"] == "queued"
