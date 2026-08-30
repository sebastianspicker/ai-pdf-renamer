# Contributing to Folionym

## Development setup

Folionym requires CPython 3.14; the repository and CI use 3.14.6. Install all
Python extras and frontend dependencies with:

```bash
make install-dev
```

## Architecture

Read [DESIGN.md](DESIGN.md) before changing package boundaries. The principal
packages are `settings`, `naming`, `extraction`, `llm`, `application`,
`rename_ops`, `infrastructure`, and `interfaces` under `src/folionym/`.
`rename_ops` is the filesystem mutation boundary described by
[ADR 0001](docs/decisions/0001-rename-filesystem-boundary.md); reviewed-plan
boundaries are described by
[ADR 0002](docs/decisions/0002-modular-monolith-boundaries.md).

Preserve the public facades `folionym.config`, `folionym.filename`,
`folionym.heuristics`, `folionym.renamer`, and `folionym.rename_ops`. Keep
interface adaptation in `interfaces/`, low-level primitives in
`infrastructure/`, and application orchestration out of both. Do not introduce
new compatibility re-exports without a documented public requirement.

The test layout reflects those boundaries:

| Path | Scope |
| --- | --- |
| `tests/contracts/` | public, infrastructure, and web contracts |
| `tests/domain/` | settings, extraction, LLM, and application behavior |
| `tests/integration/` | artifacts and reviewed-plan integration |
| `tests/interfaces/` | Textual retained-plan behavior |
| `tests/test_core.py` | retained end-to-end filesystem, undo, parser, and hook coverage |

## Change workflow

1. Reproduce the behavior or failure at the owning boundary.
2. Make the smallest change that preserves public contracts and safety policy.
3. Add or adjust behavioral tests in the relevant test area.
4. Run the narrowest relevant check, then the release gate for release-ready work.

```bash
make lint
make typecheck
make test
make frontend-check
make architecture-check
make release-check
```

`make release-check` includes dependency-lock validation, the frontend build,
repository hygiene, the architecture check, Ruff, strict mypy, tests, package
builds, and installed-wheel verification. The frontend build writes packaged
assets to `src/folionym/web_dist/`.

## Rename behavior

Conventional CLI runs and immediate TUI single-file rename use the
unique-available policy. Browser and directory-TUI Apply use exact retained
reviewed targets, revalidating source identity and collisions without
regenerating names. A material TUI source or setting edit invalidates the
directory plan; an incomplete or cancelled Preview cannot apply. CLI dry-run
and apply are separate recomputing invocations, and `--plan-file` is export
only.

## Standards and sensitive data

- Target Python 3.14, keep Ruff and strict mypy passing, and use direct
  technical language in user-facing documentation.
- Keep unrelated formatting and refactoring out of a change. Do not commit,
  push, tag, or publish unless explicitly authorized.
- Do not commit PDFs, extracted content, filenames, paths, metadata, model
  bodies, logs, caches, exports, backups, credentials, tokens, or local state.
- Add runtime dependencies only with maintainer agreement. Follow
  [SECURITY.md](SECURITY.md) for transport and vulnerability reporting.
