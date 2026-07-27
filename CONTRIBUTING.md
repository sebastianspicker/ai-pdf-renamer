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
5. `src/folionym/rename_ops.py` contains the filesystem mutation boundary.
6. `src/folionym/web_app.py`, `src/folionym/web_runtime.py`, and
   `src/folionym/frontend_service.py` implement the browser service.
7. `src/folionym/tui.py` and the other `tui_*` modules implement the terminal
   interface.
8. `tests/contracts/test_repo_contracts.py` pins documentation, package, and public
   interface contracts.

The package uses a `src` layout. Import public objects from their owning
modules. Do not introduce compatibility re-exports without a documented API
requirement.

Tests are grouped by execution boundary:

| Path | Scope |
| --- | --- |
| `tests/unit/` | focused module behavior and edge cases |
| `tests/integration/` | workflows that cross module or interface boundaries |
| `tests/contracts/` | repository, CI, distribution, documentation, and asset contracts |
| `tests/e2e/` | installed-command and subprocess workflows |
| `frontend/src/*.test.tsx` | co-located Vitest component tests |

Keep reusable pytest fixtures in `tests/conftest.py` and narrow helper seams in
`tests/helpers.py`. Test outputs and browser failure artifacts are ignored;
test source is not.

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
make e2e
```

`make format` applies Ruff formatting. Review its diff before including it in a
change.

The broad gate is:

```bash
make release-check
```

It includes frontend type checking, Vitest, the Vite build, repository hygiene,
Ruff format and lint checks, strict mypy, coverage-gated Python tests, package
builds, and installed-wheel verification. CI runs the Linux Python 3.14.6 release gate
and macOS and Windows targeted smoke jobs.

The end-to-end CLI tests are separate from `release-check`:

```bash
make e2e
```

## Frontend and screenshot changes

The browser source is in `frontend/`. Use:

```bash
cd frontend
npm run typecheck
npm run test
npm run build
```

The Vite build writes to `src/folionym/web_dist/`, which is packaged with the
wheel.

When a visible TUI change makes the tracked captures stale, refresh them from
the repository root:

```bash
make docs-screenshots
```

The capture uses fixture paths and an isolated settings file. Review all three
SVG changes.

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

Use fixture data in tests and screenshots. Follow [SECURITY.md](SECURITY.md)
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
