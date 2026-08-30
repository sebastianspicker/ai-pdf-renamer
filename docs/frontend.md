# Browser interface

Install the browser and PDF extras, then start the loopback service:

```bash
python -m pip install -e '.[web,pdf]'
folionym-web
```

It serves the packaged React application at `http://127.0.0.1:8765/source`.
Use `folionym-web --port 9000 --no-open` to select another loopback port.
There is no supported remote, multi-user, container, or hosted deployment.

## Reviewed workflow

1. **Source** selects one PDF or a directory and supplies naming, extraction,
   endpoint, and output settings.
2. **Preview** creates a process-local immutable plan of source-to-target
   proposals. It does not rename files.
3. **Apply** confirms selected reviewed targets and never generates names again.

Before a rename, the application checks the source fingerprint, duplicate
selected target, and exact target availability. A changed source or occupied
target fails; the system does not select a suffixed replacement. Only one
browser operation runs at a time. Cancellation is cooperative, and plans and
reports disappear when the server process stops.

## Runtime map

| Path | Responsibility |
| --- | --- |
| `frontend/` | React, TypeScript, and Vite source |
| `frontend/src/App.tsx` and `frontend/src/pages/` | browser workflow and routes |
| `frontend/src/components/` | shared browser components |
| `src/folionym/interfaces/web/cli.py` | loopback Uvicorn launcher |
| `src/folionym/interfaces/web/app.py` | HTTP boundary, filesystem navigation, API, static files, and headers |
| `src/folionym/interfaces/web/runtime.py` | one active run, events, cancellation, plans, and reports |
| `src/folionym/application/reviewed_plan.py` | shared preview creation and exact reviewed Apply |
| `src/folionym/interfaces/ui_settings.py` | browser and TUI settings persistence |
| `src/folionym/web_dist/` | Vite output included in the wheel |

Vite development uses `127.0.0.1:5173` and proxies `/api` to the loopback
service. The production build writes to `src/folionym/web_dist/`.

## Local security boundary

The server accepts loopback Host values only. API requests require a
process-local HTTP-only same-site cookie; mutating requests must be same-origin
JSON. Documentation and OpenAPI routes are disabled. Responses use a restrictive
content security policy, no-store cache headers, and framing protection.

These controls do not make the service a remote authentication system. A
non-loopback LLM endpoint or post-rename hook can receive document-derived
content. The browser requires acknowledgement before the first Preview for each
external endpoint. See [SECURITY.md](../SECURITY.md).

The browser and TUI share `~/.folionym_ui.json`; on POSIX, the application uses
owner-only writes where supported. An existing `~/.folionym_tui.json` is
migrated without deleting the old file.

## Development

```bash
make frontend-check
make release-check
```

The release gate verifies the packaged `web_dist` asset through an isolated
installed-wheel check.
