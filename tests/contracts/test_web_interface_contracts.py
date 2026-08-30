"""ASGI contracts for the loopback browser interface."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from folionym.application.models import FileFingerprint, PreviewItem, PreviewPlan, PreviewStatus
from folionym.config import RenamerConfig
from folionym.interfaces.web.app import create_app
from folionym.interfaces.web.runtime import RunRegistry


def _asgi_request(
    app: Any,
    method: str,
    path: str,
    *,
    headers: dict[str, str] | None = None,
    payload: dict[str, object] | None = None,
) -> tuple[int, dict[str, str], object]:
    """Exercise the local ASGI app without a network listener or HTTP client dependency."""
    body = json.dumps(payload).encode() if payload is not None else b""
    request_headers = {"host": "127.0.0.1", **(headers or {})}
    if payload is not None:
        request_headers.setdefault("content-type", "application/json")
    messages: list[dict[str, object]] = []
    received = False

    async def receive() -> dict[str, object]:
        nonlocal received
        if received:
            return {"type": "http.disconnect"}
        received = True
        return {"type": "http.request", "body": body, "more_body": False}

    async def send(message: dict[str, object]) -> None:
        messages.append(message)

    scope: dict[str, object] = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": method,
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "headers": [(name.encode(), value.encode()) for name, value in request_headers.items()],
        "client": ("127.0.0.1", 12345),
        "server": ("127.0.0.1", 8765),
        "root_path": "",
    }
    asyncio.run(app(scope, receive, send))
    start = next(message for message in messages if message["type"] == "http.response.start")
    response_headers = {name.decode(): value.decode() for name, value in start["headers"]}  # type: ignore[index]
    response_body = b"".join(
        message.get("body", b"") for message in messages if message["type"] == "http.response.body"
    )
    return int(start["status"]), response_headers, json.loads(response_body)  # type: ignore[index]


def _app_with_plan(tmp_path: Path) -> tuple[Any, RunRegistry, PreviewPlan]:
    source = tmp_path / "source.pdf"
    source.write_bytes(b"pdf")
    static_dir = tmp_path / "web_dist"
    static_dir.mkdir()
    (static_dir / "index.html").write_text("<title>Folionym</title>", encoding="utf-8")
    item = PreviewItem(
        id="item-1",
        source=source,
        proposed_base="reviewed-invoice",
        metadata={"category": "invoice"},
        status=PreviewStatus.READY,
        included=True,
        fingerprint=FileFingerprint.capture(source),
    )
    plan = PreviewPlan(
        id="plan-1",
        source=tmp_path,
        source_kind="directory",
        config=RenamerConfig(),
        items=(item,),
        created_at=datetime(2026, 8, 27, tzinfo=UTC),
        revision=3,
    )
    registry = RunRegistry()
    registry._plans[plan.id] = plan
    return create_app(registry=registry, static_dir=static_dir, session_token="test-session"), registry, plan


def test_plan_serialization_and_stale_revision_rejection_are_stable(tmp_path: Path) -> None:
    app, _registry, plan = _app_with_plan(tmp_path)

    missing_status, _missing_headers, missing_body = _asgi_request(app, "GET", f"/api/v1/plans/{plan.id}")
    assert (missing_status, missing_body) == (403, {"detail": "Local session required."})

    session_status, session_headers, session_body = _asgi_request(app, "GET", "/api/v1/session")
    assert (session_status, session_body) == (200, {"ready": True})
    assert session_headers["content-security-policy"].startswith("default-src 'self'")
    assert session_headers["x-frame-options"] == "DENY"
    assert session_headers["cache-control"] == "no-store"
    cookie = session_headers["set-cookie"].partition(";")[0]

    status, _headers, response = _asgi_request(app, "GET", f"/api/v1/plans/{plan.id}", headers={"cookie": cookie})
    assert status == 200
    assert response == {
        "id": "plan-1",
        "revision": 3,
        "source": str(tmp_path),
        "source_kind": "directory",
        "created_at": "2026-08-27T00:00:00+00:00",
        "items": [
            {
                "id": "item-1",
                "current_name": "source.pdf",
                "source_path": str(tmp_path / "source.pdf"),
                "proposed_name": "reviewed-invoice.pdf",
                "status": "ready",
                "included": True,
                "reason": None,
                "size": 3,
                "modified_at": response["items"][0]["modified_at"],
                "metadata": {"category": "invoice"},
            }
        ],
        "counts": {"all": 1, "ready": 1, "review": 0, "skipped": 0, "failed": 0},
    }

    stale_status, _stale_headers, stale_body = _asgi_request(
        app,
        "POST",
        f"/api/v1/plans/{plan.id}/apply",
        headers={"cookie": cookie},
        payload={"plan_revision": 2, "selected_ids": ["item-1"]},
    )
    assert (stale_status, stale_body) == (422, {"detail": "The preview plan revision is stale."})


def test_cancel_route_preserves_cooperative_cancellation_state(tmp_path: Path) -> None:
    app, registry, _plan = _app_with_plan(tmp_path)
    _session_status, session_headers, _session_body = _asgi_request(app, "GET", "/api/v1/session")
    run = registry._new_run("preview")

    status, _headers, response = _asgi_request(
        app,
        "POST",
        f"/api/v1/runs/{run.id}/cancel",
        headers={"cookie": session_headers["set-cookie"].partition(";")[0]},
        payload={},
    )

    assert status == 200
    assert response["state"] == "queued"
    assert response["message"] == "Cancelling after the active file"
    assert run.stop_event.is_set()
    events, terminal = registry.events_after(run.id, 0)
    assert [event.event for event in events] == ["run.queued", "run.cancelling"]
    assert terminal is False
