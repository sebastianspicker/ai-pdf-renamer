"""Loopback-only FastAPI application for the Folionym browser frontend."""

from __future__ import annotations

import asyncio
import json
import os
import secrets
import string
from collections.abc import AsyncIterator, Awaitable, Callable
from datetime import datetime
from pathlib import Path
from urllib.parse import urlsplit

from fastapi import FastAPI, Header, HTTPException, Request, Response
from fastapi.encoders import jsonable_encoder
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .frontend_service import ApplyReport, PreviewItem, PreviewPlan
from .http_url import validate_http_endpoint
from .ui_settings import load_ui_settings, merged_ui_settings
from .web_runtime import RunConflictError, RunEvent, RunRegistry
from .web_schema import (
    ApplyRequest,
    DirectoryEntry,
    DirectoryListing,
    PreviewRequest,
    RunStartedResponse,
    UISettingsPayload,
)

_SESSION_COOKIE = "folionym_session"
_ALLOWED_HOSTS = frozenset({"127.0.0.1", "localhost"})
_SECURITY_HEADERS = {
    "Content-Security-Policy": (
        "default-src 'self'; img-src 'self' blob:; connect-src 'self'; "
        "font-src 'self'; style-src 'self'; script-src 'self'; object-src 'none'; "
        "base-uri 'none'; frame-ancestors 'none'"
    ),
    "Referrer-Policy": "no-referrer",
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
}


def _host_name(host_header: str) -> str:
    """Return a normalized hostname from the HTTP Host header."""
    if host_header.startswith("["):
        return host_header.partition("]")[0].lstrip("[")
    return host_header.rsplit(":", 1)[0].lower()


def _directory_pdf_count(path: Path) -> int:
    """Count visible direct PDF files, tolerating entries that disappear."""
    try:
        return sum(
            entry.is_file() and entry.suffix.lower() == ".pdf" and not entry.name.startswith(".")
            for entry in path.iterdir()
        )
    except OSError:
        return 0


def _resolve_directory(path_value: str) -> Path:
    """Resolve one absolute directory path to a readable filesystem location."""
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        raise HTTPException(400, "Directory paths must be absolute.")
    try:
        resolved = path.resolve(strict=True)
    except FileNotFoundError as exc:
        raise HTTPException(404, "Directory does not exist.") from exc
    if not resolved.is_dir():
        raise HTTPException(400, "Path is not a directory.")
    return resolved


def _visible_directories(directory: Path) -> list[Path]:
    """List non-hidden, non-symlink child directories in a stable order."""
    try:
        return sorted(
            (
                child
                for child in directory.iterdir()
                if child.is_dir() and not child.is_symlink() and not child.name.startswith(".")
            ),
            key=lambda child: child.name.casefold(),
        )
    except PermissionError as exc:
        raise HTTPException(403, "Directory is not readable.") from exc


def _directory_listing(path_value: str) -> DirectoryListing:
    """Build one permission-aware filesystem navigator response."""
    resolved = _resolve_directory(path_value)
    directories = _visible_directories(resolved)
    entries = [
        DirectoryEntry(name=child.name, path=str(child), pdf_count=_directory_pdf_count(child)) for child in directories
    ]
    parent = str(resolved.parent) if resolved.parent != resolved else None
    return DirectoryListing(
        path=str(resolved), parent=parent, entries=entries, pdf_count=_directory_pdf_count(resolved)
    )


def _is_same_local_origin(origin_header: str, host_header: str) -> bool:
    """Accept only a parsed HTTP origin exactly matching the validated request host."""
    try:
        origin = urlsplit(origin_header)
    except ValueError:
        return False
    return (
        origin.scheme == "http"
        and origin.netloc.casefold() == host_header.casefold()
        and origin.username is None
        and origin.password is None
        and origin.path == ""
        and origin.query == ""
        and origin.fragment == ""
    )


def _requires_api_session(request: Request) -> bool:
    """Return whether the request is an API call other than session bootstrap."""
    is_session_bootstrap = request.url.path == "/api/v1/session" and request.method == "GET"
    return request.url.path.startswith("/api/") and not is_session_bootstrap


def _requires_json_body(request: Request) -> bool:
    """Return whether this API method requires a JSON content type."""
    is_mutating = request.method in {"POST", "PUT", "PATCH", "DELETE"}
    return is_mutating and not request.headers.get("content-type", "").startswith("application/json")


def _api_request_security_error(request: Request, session_token: str) -> JSONResponse | None:
    """Return the first failed API-boundary response, or ``None`` when validation passes."""
    if not _requires_api_session(request):
        return None
    if request.cookies.get(_SESSION_COOKIE) != session_token:
        return JSONResponse({"detail": "Local session required."}, status_code=403)
    origin = request.headers.get("origin")
    host = request.headers.get("host", "")
    if origin is not None and not _is_same_local_origin(origin, host):
        return JSONResponse({"detail": "Invalid request origin."}, status_code=403)
    if _requires_json_body(request):
        return JSONResponse({"detail": "JSON request required."}, status_code=415)
    return None


def _roots() -> list[DirectoryEntry]:
    """Return useful local filesystem roots without exposing file contents."""
    roots: list[Path] = [Path.home()]
    if os.name == "nt":
        roots.extend(Path(f"{letter}:\\") for letter in string.ascii_uppercase if Path(f"{letter}:\\").exists())
    else:
        roots.append(Path("/"))
    unique: dict[str, DirectoryEntry] = {}
    for root in roots:
        try:
            resolved = root.resolve(strict=True)
        except OSError:
            continue
        unique[str(resolved)] = DirectoryEntry(
            name="Home" if resolved == Path.home().resolve() else str(resolved),
            path=str(resolved),
            pdf_count=_directory_pdf_count(resolved),
        )
    return list(unique.values())


def _json_metadata(metadata: dict[str, object]) -> dict[str, object]:
    """Return JSON-compatible metadata without serializing arbitrary objects."""
    encoded = jsonable_encoder(metadata)
    return encoded if isinstance(encoded, dict) else {}


def _item_payload(item: PreviewItem) -> dict[str, object]:
    """Serialize one retained preview item for the browser."""
    try:
        current = item.source.stat(follow_symlinks=False)
        size = current.st_size
        modified = datetime.fromtimestamp(current.st_mtime).astimezone().isoformat()
    except OSError:
        size = 0
        modified = None
    return {
        "id": item.id,
        "current_name": item.source.name,
        "source_path": str(item.source),
        "proposed_name": item.proposed_name,
        "status": item.status.value,
        "included": item.included,
        "reason": item.reason,
        "size": size,
        "modified_at": modified,
        "metadata": _json_metadata(item.metadata),
    }


def _plan_payload(plan: PreviewPlan) -> dict[str, object]:
    """Serialize a preview plan and factual status counts."""
    items = [_item_payload(item) for item in plan.items]
    counts = {
        status: sum(item["status"] == status for item in items) for status in ("ready", "review", "skipped", "failed")
    }
    return {
        "id": plan.id,
        "revision": plan.revision,
        "source": str(plan.source),
        "source_kind": plan.source_kind,
        "created_at": plan.created_at.isoformat(),
        "items": items,
        "counts": {"all": len(items), **counts},
    }


def _report_payload(report: ApplyReport) -> dict[str, object]:
    """Serialize an exact reviewed-plan apply report."""
    items = [
        {
            "item_id": item.item_id,
            "source_name": item.source_name,
            "target_name": item.target_name,
            "status": item.status.value,
            "reason": item.reason,
        }
        for item in report.items
    ]
    counts = {
        status: sum(item["status"] == status for item in items)
        for status in ("renamed", "skipped", "unchanged", "failed", "cancelled")
    }
    return {
        "id": report.id,
        "plan_id": report.plan_id,
        "source": str(report.source),
        "started_at": report.started_at.isoformat(),
        "completed_at": report.completed_at.isoformat(),
        "items": items,
        "counts": counts,
    }


def _sse_event(event: RunEvent) -> str:
    """Encode one retained run event in the Server-Sent Events wire format."""
    return f"id: {event.sequence}\nevent: {event.event}\ndata: {json.dumps(event.payload)}\n\n"


async def _event_stream(registry: RunRegistry, run_id: str, last_sequence: int) -> AsyncIterator[str]:
    """Stream retained and live run events, including keep-alive comments."""
    sequence = last_sequence
    idle_ticks = 0
    while True:
        try:
            events, terminal = registry.events_after(run_id, sequence)
        except KeyError:
            yield 'event: run.failed\ndata: {"error":"Run not found."}\n\n'
            return
        for event in events:
            sequence = event.sequence
            idle_ticks = 0
            yield _sse_event(event)
        if terminal and not events:
            return
        idle_ticks += 1
        if idle_ticks >= 40:
            idle_ticks = 0
            yield ": keep-alive\n\n"
        await asyncio.sleep(0.25)


def _external_endpoint_requirement(request: PreviewRequest) -> str | None:
    """Return the normalized external model URL when acknowledgement is required."""
    settings = request.settings
    if not settings.use_llm or not settings.llm_url.strip():
        return None
    try:
        endpoint = validate_http_endpoint(settings.llm_url)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    if endpoint.is_loopback or settings.acknowledged_external_endpoint == endpoint.url:
        return None
    return endpoint.url


def _prepare_preview_settings(request: PreviewRequest) -> dict[str, object]:
    """Validate external-boundary consent and normalize the selected source into settings."""
    endpoint = _external_endpoint_requirement(request)
    if endpoint is not None and not request.acknowledge_external_endpoint:
        raise HTTPException(
            409,
            {
                "code": "external_endpoint_ack_required",
                "message": "Document-derived content may leave this machine.",
                "endpoint": endpoint,
            },
        )
    settings = request.settings.model_dump()
    if endpoint is not None:
        settings["acknowledged_external_endpoint"] = endpoint
    source = str(Path(request.path).expanduser())
    if request.source_kind == "file":
        settings["single_file"] = source
        settings["directory"] = str(Path(source).parent)
    else:
        settings["directory"] = source
        settings["single_file"] = ""
    return settings


def _artifact_path(plan: PreviewPlan, kind: str) -> Path:
    """Resolve a configured machine-readable artifact for download."""
    paths = plan.config.output.paths
    configured: dict[str, str | Path | None] = {
        "rename-log": paths.rename_log_path,
        "metadata": paths.export_metadata_path,
        "summary": paths.summary_json_path,
    }
    value = configured.get(kind)
    if not value:
        raise HTTPException(404, "Artifact is not configured.")
    path = Path(value).expanduser()
    if not path.is_file():
        raise HTTPException(404, "Artifact is not available.")
    return path


def _install_security_middleware(app: FastAPI, session_token: str) -> None:
    """Install the loopback host, same-origin, session, and response-header boundary."""

    @app.middleware("http")
    async def local_security(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        """Enforce the loopback session and attach browser security headers."""
        host = request.headers.get("host", "")
        if _host_name(host) not in _ALLOWED_HOSTS:
            return JSONResponse({"detail": "Invalid local host."}, status_code=400)
        security_error = _api_request_security_error(request, session_token)
        if security_error is not None:
            return security_error
        response = await call_next(request)
        for name, value in _SECURITY_HEADERS.items():
            response.headers[name] = value
        response.headers["Cache-Control"] = (
            "no-store" if request.url.path.startswith("/api/") else response.headers.get("Cache-Control", "no-cache")
        )
        return response


def _register_source_routes(app: FastAPI, registry: RunRegistry) -> None:
    """Register bootstrap, filesystem navigation, and Preview creation."""

    @app.get("/api/v1/session")
    def session(request: Request) -> Response:
        """Create the same-origin browser session cookie."""
        response = JSONResponse({"ready": True})
        response.set_cookie(
            _SESSION_COOKIE,
            request.app.state.session_token,
            httponly=True,
            samesite="strict",
            secure=False,
            path="/",
        )
        return response

    @app.get("/api/v1/bootstrap")
    def bootstrap() -> dict[str, object]:
        """Return persisted settings, local roots, and available capabilities."""
        settings = UISettingsPayload.model_validate(merged_ui_settings(load_ui_settings()))
        return {
            "settings": settings.model_dump(),
            "roots": [root.model_dump() for root in _roots()],
            "capabilities": {"thumbnails": True, "ocr": True, "external_model_confirmation": True},
        }

    @app.get("/api/v1/filesystem")
    def filesystem(path: str) -> DirectoryListing:
        """List one absolute local directory for the folder picker."""
        return _directory_listing(path)

    @app.post("/api/v1/previews", response_model=RunStartedResponse, status_code=202)
    def start_preview(payload: PreviewRequest) -> RunStartedResponse:
        """Validate and enqueue one structured Preview run."""
        settings = _prepare_preview_settings(payload)
        try:
            run_id = registry.start_preview(Path(payload.path).expanduser(), settings)
        except RunConflictError as exc:
            raise HTTPException(409, str(exc)) from exc
        except (FileNotFoundError, NotADirectoryError, ValueError) as exc:
            raise HTTPException(422, str(exc)) from exc
        return RunStartedResponse(run_id=run_id)


def _register_run_routes(app: FastAPI, registry: RunRegistry) -> None:
    """Register run snapshots, event streaming, and cancellation."""

    @app.get("/api/v1/runs/{run_id}")
    def run_snapshot(run_id: str) -> dict[str, object]:
        """Return the latest state for one background operation."""
        try:
            return registry.snapshot(run_id)
        except KeyError as exc:
            raise HTTPException(404, "Run not found.") from exc

    @app.get("/api/v1/runs/{run_id}/events")
    def run_events(run_id: str, last_event_id: str | None = Header(default=None)) -> StreamingResponse:
        """Stream retained and live operation events."""
        try:
            sequence = int(last_event_id or "0")
        except ValueError:
            sequence = 0
        return StreamingResponse(
            _event_stream(registry, run_id, sequence),
            media_type="text/event-stream",
            headers={"X-Accel-Buffering": "no", "Cache-Control": "no-store"},
        )

    @app.post("/api/v1/runs/{run_id}/cancel")
    def cancel_run(run_id: str, _payload: dict[str, object]) -> dict[str, object]:
        """Request cooperative cancellation for one active operation."""
        try:
            return registry.cancel(run_id)
        except KeyError as exc:
            raise HTTPException(404, "Run not found.") from exc


def _register_plan_routes(app: FastAPI, registry: RunRegistry) -> None:
    """Register structured plan retrieval, application, and reports."""

    @app.get("/api/v1/plans/{plan_id}")
    def get_plan(plan_id: str) -> dict[str, object]:
        """Return one retained structured Preview plan."""
        try:
            return _plan_payload(registry.get_plan(plan_id))
        except KeyError as exc:
            raise HTTPException(404, "Preview plan not found.") from exc

    @app.post("/api/v1/plans/{plan_id}/apply", response_model=RunStartedResponse, status_code=202)
    def apply_plan(plan_id: str, payload: ApplyRequest) -> RunStartedResponse:
        """Enqueue exact application of selected reviewed targets."""
        try:
            run_id = registry.start_apply(plan_id, payload.plan_revision, payload.selected_ids)
        except KeyError as exc:
            raise HTTPException(404, "Preview plan not found.") from exc
        except RunConflictError as exc:
            raise HTTPException(409, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        return RunStartedResponse(run_id=run_id)

    @app.get("/api/v1/reports/{report_id}")
    def get_report(report_id: str) -> dict[str, object]:
        """Return one retained exact-apply report."""
        try:
            return _report_payload(registry.get_report(report_id))
        except KeyError as exc:
            raise HTTPException(404, "Apply report not found.") from exc


def _render_thumbnail(plan: PreviewPlan, item_id: str) -> bytes:
    """Render a bounded first-page PNG for one plan-owned PDF."""
    item = next((candidate for candidate in plan.items if candidate.id == item_id), None)
    if item is None:
        raise HTTPException(404, "Preview item not found.")
    try:
        import fitz

        with fitz.open(item.source) as document:
            if document.page_count < 1:
                raise HTTPException(404, "No page preview is available.")
            page = document.load_page(0)
            scale = min(1.5, 900 / max(float(page.rect.width), 1.0))
            pixmap = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
            return bytes(pixmap.tobytes("png"))
    except ImportError as exc:
        raise HTTPException(503, "PDF thumbnails require the web extra.") from exc
    except (OSError, RuntimeError, ValueError) as exc:
        raise HTTPException(404, "No page preview is available.") from exc


def _register_media_routes(app: FastAPI, registry: RunRegistry) -> None:
    """Register plan-owned thumbnails and configured artifacts."""

    @app.get("/api/v1/plans/{plan_id}/items/{item_id}/thumbnail")
    def thumbnail(plan_id: str, item_id: str) -> Response:
        """Render a bounded first-page thumbnail for a plan-owned PDF."""
        try:
            plan = registry.get_plan(plan_id)
        except KeyError as exc:
            raise HTTPException(404, "Preview plan not found.") from exc
        return Response(
            _render_thumbnail(plan, item_id),
            media_type="image/png",
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/api/v1/plans/{plan_id}/artifacts/{kind}")
    def artifact(plan_id: str, kind: str) -> FileResponse:
        """Download one configured output artifact for the retained plan."""
        try:
            plan = registry.get_plan(plan_id)
        except KeyError as exc:
            raise HTTPException(404, "Preview plan not found.") from exc
        path = _artifact_path(plan, kind)
        return FileResponse(path, filename=path.name)


def _register_static_routes(app: FastAPI, static_dir: Path, session_token: str) -> None:
    """Serve packaged Vite assets and the history-fallback application shell."""
    assets = static_dir / "assets"
    if assets.is_dir():
        app.mount("/assets", StaticFiles(directory=assets), name="assets")

    @app.get("/{route:path}", response_class=HTMLResponse)
    def spa(route: str) -> Response:
        """Serve the packaged application shell for every browser route."""
        index = static_dir / "index.html"
        if index.is_file():
            response: Response = FileResponse(index)
        else:
            response = HTMLResponse(
                "<!doctype html><title>Folionym</title><p>Frontend assets are not built.</p>",
                status_code=503,
            )
        response.set_cookie(
            _SESSION_COOKIE,
            session_token,
            httponly=True,
            samesite="strict",
            secure=False,
            path="/",
        )
        return response


def create_app(
    *,
    registry: RunRegistry | None = None,
    static_dir: Path | None = None,
    session_token: str | None = None,
) -> FastAPI:
    """Create the loopback browser application with isolated mutable state."""
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    active_registry = registry or RunRegistry()
    active_token = session_token or secrets.token_urlsafe(32)
    active_static_dir = static_dir or Path(__file__).with_name("web_dist")
    app.state.registry = active_registry
    app.state.session_token = active_token
    app.state.static_dir = active_static_dir
    _install_security_middleware(app, active_token)
    _register_source_routes(app, active_registry)
    _register_run_routes(app, active_registry)
    _register_plan_routes(app, active_registry)
    _register_media_routes(app, active_registry)
    _register_static_routes(app, active_static_dir, active_token)
    return app


app = create_app()
