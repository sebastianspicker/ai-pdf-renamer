# Browser interface

Install the `web` and `pdf` extras, then launch the local service:

```bash
python -m pip install -e '.[web,pdf]'
folionym-web
```

The command binds Uvicorn to `127.0.0.1:8765`, serves the packaged React
application and API from one origin, opens `/source`, and remains attached to
the terminal. Use `folionym-web --port 9000 --no-open` to select another
loopback port without opening a browser.

There is no supported remote, multi-user, container, or hosted deployment.

## Workflow

The browser has three stages:

1. Source selects one PDF or a directory and sets naming, extraction, endpoint,
   and output options.
2. Preview displays source-to-target proposals, evidence, and per-file status.
   It does not rename files.
3. Apply confirms selected exact targets and reports each result.

Preview creates an in-memory plan. Apply does not calculate names again. Before
each rename, it checks the source fingerprint, exact target, duplicate targets,
and new filesystem collisions. Changed sources and unsafe targets fail without
selecting a replacement name.

Only one browser operation can run at a time. Cancellation is cooperative and
takes effect at the next supported boundary. Plans and reports are process
local. Restarting the service invalidates open Preview and Apply routes.

The browser does not expose CLI watch mode, diagnostics, or undo.

## Runtime architecture

| Path | Responsibility |
| --- | --- |
| `frontend/` | React 19 and TypeScript source and Vite configuration |
| `frontend/src/App.tsx` | routes and top-level interface state |
| `frontend/src/pages/` | Source, Preview, and Apply pages |
| `frontend/src/components/` | shared browser components |
| `frontend/src/styles.css` and `frontend/src/styles/` | global CSS |
| `src/folionym/web_cli.py` | loopback Uvicorn launcher |
| `src/folionym/web_app.py` | HTTP boundary, filesystem navigation, endpoints, static files, and headers |
| `src/folionym/web_runtime.py` | active run, progress events, cancellation, plans, and reports |
| `src/folionym/frontend_service.py` | preview construction and exact apply handling |
| `src/folionym/ui_settings.py` | shared browser and TUI settings persistence |
| `src/folionym/web_dist/` | Vite output included in the wheel |

Vite development uses `127.0.0.1:5173` and proxies `/api` to
`127.0.0.1:8765`. The production build writes directly to
`src/folionym/web_dist/`.

## Local security boundary

The server accepts loopback Host values only. API requests require a
process-local, HTTP-only, same-site cookie. Mutating requests must be
same-origin JSON requests. FastAPI documentation and OpenAPI routes are
disabled. Responses use a restrictive Content Security Policy, no-store cache
headers, and framing protection.

These controls protect the local browser boundary. They are not a remote
authentication system. A configured non-loopback LLM endpoint can still
receive document-derived content. The browser requires acknowledgement before
the first Preview for each exact external endpoint. See
[SECURITY.md](../SECURITY.md).

## Settings

The browser and TUI share `~/.folionym_ui.json`. The file uses owner-only
permissions on POSIX systems where that is supported. An existing
`~/.folionym_tui.json` is migrated once without deleting the old file.

## Development and testing

Install the contributor environment:

```bash
make install-dev
```

Run the frontend gate:

```bash
make frontend-check
```

This runs TypeScript checking and the production Vite build. The full
repository gate also verifies that the wheel contains
`src/folionym/web_dist/index.html`:

```bash
make release-check
```

For an interactive development server, run the Python service in one terminal
and Vite in another:

```bash
folionym-web --no-open
```

```bash
cd frontend
npm run dev
```
