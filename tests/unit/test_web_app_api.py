"""API-boundary tests for the local browser application.

The installed Starlette test client intentionally depends on the optional
``httpx2`` extra.  These tests use the ASGI contract directly so they exercise
the same middleware and routes without making that optional development
dependency a production requirement.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from urllib.parse import urlencode

import pytest
from fastapi import HTTPException

import folionym.web_app as web_app
from folionym.config import RenamerConfig
from folionym.frontend_service import ApplyItemResult, ApplyReport, ApplyStatus, PreviewItem, PreviewPlan, PreviewStatus
from folionym.web_runtime import RunConflictError, RunEvent


@dataclass(frozen=True)
class ASGIResponse:
    """Minimal completed HTTP response collected from the application boundary."""

    status_code: int
    headers: dict[str, str]
    body: bytes

    def json(self) -> Any:
        """Decode the JSON body returned by an API endpoint."""
        return json.loads(self.body)


async def _asgi_request(
    app: Any,
    method: str,
    target: str,
    *,
    headers: dict[str, str] | None = None,
    json_body: object | None = None,
) -> ASGIResponse:
    """Issue one complete in-process HTTP request through FastAPI's ASGI edge."""
    path, separator, query = target.partition("?")
    request_headers = {"host": "127.0.0.1", **(headers or {})}
    body = b"" if json_body is None else json.dumps(json_body).encode()
    if json_body is not None:
        request_headers.setdefault("content-type", "application/json")
    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": method,
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "query_string": query.encode() if separator else b"",
        "root_path": "",
        "headers": [(name.lower().encode(), value.encode()) for name, value in request_headers.items()],
        "client": ("127.0.0.1", 50000),
        "server": ("127.0.0.1", 8000),
    }
    messages: list[dict[str, Any]] = []

    async def receive() -> dict[str, Any]:
        return {"type": "http.request", "body": body, "more_body": False}

    async def send(message: dict[str, Any]) -> None:
        messages.append(message)

    await app(scope, receive, send)
    start = next(message for message in messages if message["type"] == "http.response.start")
    response_headers = {name.decode().lower(): value.decode() for name, value in start["headers"]}
    response_body = b"".join(
        message.get("body", b"") for message in messages if message["type"] == "http.response.body"
    )
    return ASGIResponse(start["status"], response_headers, response_body)


def _request(app: Any, method: str, target: str, **kwargs: Any) -> ASGIResponse:
    """Synchronously run a fully collected ASGI request for ordinary route tests."""
    return asyncio.run(_asgi_request(app, method, target, **kwargs))


def _api_headers() -> dict[str, str]:
    """Return the fixed test session required by protected browser API calls."""
    return {"cookie": "folionym_session=test-session"}


class StubRegistry:
    """A deterministic registry substitute exposing only API-facing state."""

    def __init__(self, plan: PreviewPlan, report: ApplyReport) -> None:
        self.plan = plan
        self.report = report
        self.preview_error: BaseException | None = None
        self.apply_error: BaseException | None = None
        self.preview_calls: list[tuple[Path, dict[str, object]]] = []
        self.apply_calls: list[tuple[str, int, list[str]]] = []

    def start_preview(self, source: Path, settings: dict[str, object]) -> str:
        if self.preview_error is not None:
            raise self.preview_error
        self.preview_calls.append((source, settings))
        return "preview-run"

    def start_apply(self, plan_id: str, revision: int, selected_ids: list[str]) -> str:
        if self.apply_error is not None:
            raise self.apply_error
        self.apply_calls.append((plan_id, revision, selected_ids))
        return "apply-run"

    def snapshot(self, run_id: str) -> dict[str, object]:
        if run_id != "preview-run":
            raise KeyError(run_id)
        return {"id": run_id, "state": "completed", "total": 2}

    def events_after(self, run_id: str, sequence: int) -> tuple[list[RunEvent], bool]:
        if run_id != "preview-run":
            raise KeyError(run_id)
        if sequence:
            return ([], True)
        return ([RunEvent(1, "run.completed", {"id": run_id})], True)

    def cancel(self, run_id: str) -> dict[str, object]:
        if run_id != "preview-run":
            raise KeyError(run_id)
        return {"id": run_id, "state": "cancelled", "message": "Cancellation requested"}

    def get_plan(self, plan_id: str) -> PreviewPlan:
        if plan_id != self.plan.id:
            raise KeyError(plan_id)
        return self.plan

    def get_report(self, report_id: str) -> ApplyReport:
        if report_id != self.report.id:
            raise KeyError(report_id)
        return self.report


@pytest.fixture
def web_state(tmp_path: Path) -> tuple[Any, StubRegistry, PreviewPlan, ApplyReport]:
    """Create an isolated app with fixture-backed plans and downloadable artifacts."""
    source = tmp_path / "source"
    source.mkdir()
    invoice = source / "invoice.pdf"
    invoice.write_bytes(b"%PDF test document")
    artifact = tmp_path / "rename-log.json"
    artifact.write_text('{"renamed": 1}', encoding="utf-8")
    config = cast(
        RenamerConfig,
        SimpleNamespace(
            output=SimpleNamespace(
                paths=SimpleNamespace(
                    rename_log_path=artifact,
                    export_metadata_path=None,
                    summary_json_path=None,
                )
            )
        ),
    )
    plan = PreviewPlan(
        id="plan-1",
        source=source,
        source_kind="directory",
        config=config,
        items=(
            PreviewItem("ready", invoice, "20260807-invoice", {"category": "invoice"}, PreviewStatus.READY, True, None),
            PreviewItem(
                "review",
                source / "missing.pdf",
                None,
                {"when": datetime(2026, 8, 7, tzinfo=UTC)},
                PreviewStatus.REVIEW,
                False,
                None,
                "Needs review",
            ),
        ),
        created_at=datetime(2026, 8, 7, tzinfo=UTC),
        revision=4,
    )
    report = ApplyReport(
        id="report-1",
        plan_id=plan.id,
        source=source,
        started_at=datetime(2026, 8, 7, tzinfo=UTC),
        completed_at=datetime(2026, 8, 7, 1, tzinfo=UTC),
        items=(
            ApplyItemResult("ready", "invoice.pdf", "20260807-invoice.pdf", ApplyStatus.RENAMED),
            ApplyItemResult("review", "missing.pdf", None, ApplyStatus.SKIPPED, "Not selected"),
        ),
    )
    static_dir = tmp_path / "web_dist"
    static_dir.mkdir()
    (static_dir / "index.html").write_text("<main>Folionym</main>", encoding="utf-8")
    registry = StubRegistry(plan, report)
    app = web_app.create_app(registry=registry, static_dir=static_dir, session_token="test-session")
    return app, registry, plan, report


def test_session_static_shell_and_api_security_boundary(
    web_state: tuple[Any, StubRegistry, PreviewPlan, ApplyReport],
) -> None:
    """Only loopback, cookie-backed JSON API calls reach application routes."""
    app, _registry, _plan, _report = web_state

    session = _request(app, "GET", "/api/v1/session")
    assert session.status_code == 200
    assert session.json() == {"ready": True}
    assert "folionym_session=test-session" in session.headers["set-cookie"]
    assert session.headers["content-security-policy"].startswith("default-src 'self'")
    assert session.headers["cache-control"] == "no-store"

    shell = _request(app, "GET", "/source")
    assert shell.status_code == 200
    assert shell.body == b"<main>Folionym</main>"
    assert shell.headers["cache-control"] == "no-cache"

    assert _request(app, "GET", "/api/v1/session", headers={"host": "example.test"}).status_code == 400
    assert _request(app, "GET", "/api/v1/bootstrap").status_code == 403
    assert _request(app, "GET", "/api/v1/bootstrap", headers={"cookie": "folionym_session=wrong"}).status_code == 403
    assert _request(
        app,
        "GET",
        "/api/v1/bootstrap",
        headers={**_api_headers(), "origin": "http://localhost:8000"},
    ).json() == {"detail": "Invalid request origin."}
    assert _request(
        app,
        "POST",
        "/api/v1/runs/preview-run/cancel",
        headers=_api_headers(),
    ).json() == {"detail": "JSON request required."}


def test_bootstrap_and_filesystem_routes_expose_only_navigable_local_state(
    web_state: tuple[Any, StubRegistry, PreviewPlan, ApplyReport], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Bootstrap and directory routes expose settings and visible directories, not file contents."""
    app, _registry, plan, _report = web_state
    child = plan.source / "Alpha"
    child.mkdir()
    (child / "visible.PDF").write_bytes(b"%PDF")
    (plan.source / ".private").mkdir()
    monkeypatch.setattr(web_app, "load_ui_settings", lambda: {"language": "en"})
    monkeypatch.setattr(web_app, "merged_ui_settings", lambda settings: settings)
    monkeypatch.setattr(web_app, "_roots", lambda: [])

    bootstrap = _request(app, "GET", "/api/v1/bootstrap", headers=_api_headers())
    assert bootstrap.status_code == 200
    assert bootstrap.json()["settings"]["language"] == "en"
    assert bootstrap.json()["capabilities"]["external_model_confirmation"] is True

    listing = _request(
        app,
        "GET",
        f"/api/v1/filesystem?{urlencode({'path': str(plan.source)})}",
        headers=_api_headers(),
    )
    assert listing.status_code == 200
    assert listing.json()["pdf_count"] == 1
    assert listing.json()["entries"] == [{"name": "Alpha", "path": str(child), "pdf_count": 1}]
    assert _request(app, "GET", "/api/v1/filesystem?path=relative", headers=_api_headers()).status_code == 400
    missing = _request(
        app,
        "GET",
        f"/api/v1/filesystem?{urlencode({'path': str(plan.source / 'gone')})}",
        headers=_api_headers(),
    )
    assert missing.status_code == 404


def test_preview_run_plan_apply_and_report_routes_preserve_review_state(
    web_state: tuple[Any, StubRegistry, PreviewPlan, ApplyReport],
) -> None:
    """The HTTP layer retains preview selection, state snapshots, and exact apply results."""
    app, registry, plan, report = web_state
    preview_payload = {"source_kind": "directory", "path": str(plan.source), "settings": {"use_llm": False}}

    preview = _request(app, "POST", "/api/v1/previews", headers=_api_headers(), json_body=preview_payload)
    assert preview.status_code == 202
    assert preview.json() == {"run_id": "preview-run"}
    preview_source, preview_settings = registry.preview_calls[0]
    assert preview_source == plan.source
    assert preview_settings["directory"] == str(plan.source)
    assert preview_settings["single_file"] == ""
    assert preview_settings["use_llm"] is False

    snapshot = _request(app, "GET", "/api/v1/runs/preview-run", headers=_api_headers())
    cancellation = _request(app, "POST", "/api/v1/runs/preview-run/cancel", headers=_api_headers(), json_body={})
    assert snapshot.json()["state"] == "completed"
    assert cancellation.json()["state"] == "cancelled"

    returned_plan = _request(app, "GET", f"/api/v1/plans/{plan.id}", headers=_api_headers())
    assert returned_plan.json()["counts"] == {"all": 2, "ready": 1, "review": 1, "skipped": 0, "failed": 0}
    assert returned_plan.json()["items"][1]["size"] == 0
    assert returned_plan.json()["items"][1]["metadata"]["when"].startswith("2026-08-07")

    apply = _request(
        app,
        "POST",
        f"/api/v1/plans/{plan.id}/apply",
        headers=_api_headers(),
        json_body={"plan_revision": plan.revision, "selected_ids": ["ready"]},
    )
    returned_report = _request(app, "GET", f"/api/v1/reports/{report.id}", headers=_api_headers())
    assert apply.status_code == 202
    assert registry.apply_calls == [(plan.id, plan.revision, ["ready"])]
    assert returned_report.json()["counts"] == {
        "renamed": 1,
        "skipped": 1,
        "unchanged": 0,
        "failed": 0,
        "cancelled": 0,
    }


def test_route_errors_preserve_conflict_validation_and_missing_resource_meaning(
    web_state: tuple[Any, StubRegistry, PreviewPlan, ApplyReport],
) -> None:
    """Route-specific errors remain distinguishable for browser recovery behavior."""
    app, registry, plan, _report = web_state
    registry.preview_error = RunConflictError("A preview run is already active.")
    conflict = _request(
        app,
        "POST",
        "/api/v1/previews",
        headers=_api_headers(),
        json_body={"source_kind": "directory", "path": str(plan.source), "settings": {"use_llm": False}},
    )
    registry.apply_error = ValueError("The preview plan revision is stale.")
    stale = _request(
        app,
        "POST",
        f"/api/v1/plans/{plan.id}/apply",
        headers=_api_headers(),
        json_body={"plan_revision": plan.revision, "selected_ids": []},
    )
    assert conflict.status_code == 409
    assert stale.status_code == 422
    assert _request(app, "GET", "/api/v1/runs/missing", headers=_api_headers()).status_code == 404
    assert _request(app, "POST", "/api/v1/runs/missing/cancel", headers=_api_headers(), json_body={}).status_code == 404
    assert _request(app, "GET", "/api/v1/plans/missing", headers=_api_headers()).status_code == 404
    assert _request(app, "GET", "/api/v1/reports/missing", headers=_api_headers()).status_code == 404


def test_artifacts_and_serializers_fail_closed_for_unavailable_data(
    web_state: tuple[Any, StubRegistry, PreviewPlan, ApplyReport],
) -> None:
    """Downloads require retained plans and configured existing artifact paths."""
    app, _registry, plan, _report = web_state
    artifact = _request(app, "GET", f"/api/v1/plans/{plan.id}/artifacts/rename-log", headers=_api_headers())
    assert artifact.status_code == 200
    assert artifact.body == b'{"renamed": 1}'
    assert artifact.headers["content-disposition"].endswith('filename="rename-log.json"')
    assert _request(app, "GET", f"/api/v1/plans/{plan.id}/artifacts/summary", headers=_api_headers()).status_code == 404
    assert _request(app, "GET", "/api/v1/plans/missing/artifacts/rename-log", headers=_api_headers()).status_code == 404
    with pytest.raises(HTTPException, match="Artifact is not configured"):
        web_app._artifact_path(plan, "unknown")


def test_thumbnail_renderer_bounds_preview_output_and_handles_missing_pdf_content(tmp_path: Path) -> None:
    """Thumbnails use only the first owned page and present unavailable content as a 404."""
    import fitz

    source = tmp_path / "sample.pdf"
    document = fitz.open()
    page = document.new_page(width=1200, height=800)
    page.insert_text((72, 72), "Invoice")
    document.save(source)
    document.close()
    plan = PreviewPlan(
        id="thumbnail-plan",
        source=tmp_path,
        source_kind="file",
        config=cast(RenamerConfig, object()),
        items=(PreviewItem("item", source, "invoice", {}, PreviewStatus.READY, True, None),),
        created_at=datetime.now(UTC),
    )

    assert web_app._render_thumbnail(plan, "item").startswith(b"\x89PNG")
    with pytest.raises(HTTPException, match="Preview item not found"):
        web_app._render_thumbnail(plan, "other")
    missing = PreviewPlan(
        id="missing-thumbnail",
        source=tmp_path,
        source_kind="file",
        config=cast(RenamerConfig, object()),
        items=(PreviewItem("gone", tmp_path / "gone.pdf", "gone", {}, PreviewStatus.READY, True, None),),
        created_at=datetime.now(UTC),
    )
    with pytest.raises(HTTPException, match="No page preview is available"):
        web_app._render_thumbnail(missing, "gone")


def test_static_route_returns_actionable_error_when_packaged_shell_is_missing(tmp_path: Path) -> None:
    """A broken package fails visibly while still establishing the local session cookie."""
    app = web_app.create_app(static_dir=tmp_path, session_token="test-session")

    response = _request(app, "GET", "/source")

    assert response.status_code == 503
    assert b"Frontend assets are not built" in response.body
    assert "folionym_session=test-session" in response.headers["set-cookie"]


async def _collect(stream: AsyncIterator[str]) -> list[str]:
    """Collect the finite portion of an event stream for direct SSE assertions."""
    return [chunk async for chunk in stream]


def test_event_stream_replays_events_reports_missing_runs_and_sends_keepalives(monkeypatch: pytest.MonkeyPatch) -> None:
    """SSE reconnection receives ordered events, a missing-run error, and idle heartbeats."""
    event = RunEvent(3, "run.progress", {"completed": 2})

    class CompletedRegistry:
        def events_after(self, _run_id: str, sequence: int) -> tuple[list[RunEvent], bool]:
            return ([event], True) if sequence == 0 else ([], True)

    assert asyncio.run(_collect(web_app._event_stream(CompletedRegistry(), "run", 0))) == [
        'id: 3\nevent: run.progress\ndata: {"completed": 2}\n\n'
    ]

    class MissingRegistry:
        def events_after(self, _run_id: str, _sequence: int) -> tuple[list[RunEvent], bool]:
            raise KeyError("missing")

    assert asyncio.run(_collect(web_app._event_stream(MissingRegistry(), "missing", 0))) == [
        'event: run.failed\ndata: {"error":"Run not found."}\n\n'
    ]

    class IdleRegistry:
        calls = 0

        def events_after(self, _run_id: str, _sequence: int) -> tuple[list[RunEvent], bool]:
            self.calls += 1
            return ([], self.calls > 40)

    async def no_wait(_seconds: float) -> None:
        return None

    monkeypatch.setattr(web_app.asyncio, "sleep", no_wait)
    assert asyncio.run(_collect(web_app._event_stream(IdleRegistry(), "idle", 0))) == [": keep-alive\n\n"]
