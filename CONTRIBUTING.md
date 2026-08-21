# Contributing to Folionym

## Development setup

Folionym requires CPython 3.14. The repository and CI use Python 3.14.6.
Install Python dependencies, all optional extras, and frontend dependencies:

```bash
make install-dev
```

This target runs `uv sync --all-extras` and `npm ci` in `frontend/`.

## Repository orientation

Start with these files:

1. `pyproject.toml` and `Makefile` define dependencies, packaging, and checks.
2. `src/folionym/cli.py` and `src/folionym/config_resolver.py` normalize CLI,
   environment, and file configuration.
3. `src/folionym/renamer.py` orchestrates discovery, proposal calculation, and
   sequential rename handling.
4. `src/folionym/filename.py` constructs names.
5. `src/folionym/rename_ops/` contains the filesystem mutation boundary; its
   `__init__.py` is the stable import facade described by
   [ADR 0001](docs/decisions/0001-rename-filesystem-boundary.md).
6. `src/folionym/web_app.py`, `src/folionym/web_runtime.py`, and
   `src/folionym/frontend_service.py` implement the browser service.
7. `src/folionym/tui.py` and the other `tui_*` modules implement the terminal
   interface.
8. `tests/test_core.py` covers the retained direct filesystem, undo, parsing,
   and hook contracts.

The package uses a `src` layout. Import public objects from their owning
modules. Do not introduce compatibility re-exports without a documented API
requirement. The `folionym.rename_ops` facade is the documented exception.

The deliberately compact suite lives in one module:

| Path | Scope |
| --- | --- |
| `tests/test_core.py` | rename safety, undo boundaries, and LLM response parsing |

Construct small inputs directly in each test and use temporary directories for
filesystem behavior. Test output is ignored; test source is not.

## Change workflow

1. Reproduce the current behavior or failing case.
2. Make the smallest change that fixes the owning code path.
3. Add or update tests that exercise observable behavior.
4. Run the narrowest relevant test or check.
5. Run the broad local gate before opening a pull request.

Useful focused commands:

```bash
make lint
make typecheck
make test
make frontend-check
```

`make format` applies Ruff formatting. Review its diff before including it in a
change.

The broad gate is:

```bash
make release-check
```

It includes frontend type checking, the Vite build, repository hygiene,
Ruff format and lint checks, strict mypy, focused Python tests, package
builds, and installed-wheel verification. CI runs the Linux Python 3.14.6 release gate
and macOS and Windows targeted smoke jobs.

## Frontend changes

The browser source is in `frontend/`. Use:

```bash
cd frontend
npm run typecheck
npm run build
```

The Vite build writes to `src/folionym/web_dist/`, which is packaged with the
wheel.

## Code and documentation standards

- Target Python 3.14 and keep mypy strict checks passing.
- Use Ruff for Python formatting and linting.
- Keep user-facing behavior, defaults, paths, and commands consistent across
  code, tests, and documentation.
- Use direct technical language. Do not add claims that are not supported by
  implementation or verification.
- Add new runtime dependencies only after maintainer agreement.
- Keep large behavior or public API changes in a focused issue or proposal
  before implementation.

## Sensitive data

Do not commit:

- PDFs or extracted document content
- filenames, paths, metadata, or summaries from private documents
- model request or response bodies
- local logs, caches, exports, rename logs, or backup files
- credentials, tokens, private environment files, or machine-specific state

Use small programmatic data in tests. Follow [SECURITY.md](SECURITY.md)
for transport and reporting requirements.

## Pull requests

- Use the pull request template.
- Describe the user-visible behavior and the checks you ran.
- Keep unrelated formatting and refactoring out of the change.
- Link relevant issues.
- Do not tag or publish a release from an ordinary pull request.
- Follow [RELEASING.md](RELEASING.md) only when preparing an authorized
  release.

Report security vulnerabilities through the private channel in
[SECURITY.md](SECURITY.md), not through a public issue.
