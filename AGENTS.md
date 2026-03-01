# Agent map – AI-PDF-Renamer

This file is the repository map for agents. It points to source-of-truth docs and avoids duplicating full manuals.

## Quick orientation

- **Repo purpose:** local-first PDF renaming by content (`date + category + keywords + summary`).
- **Entrypoints:** `python ren.py`, `ai-pdf-renamer`, `ai-pdf-renamer-gui`, `ai-pdf-renamer-undo`.

## Source-of-truth documentation

- [README.md](README.md)
- [CONTRIBUTING.md](CONTRIBUTING.md)
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- [docs/RUNBOOK.md](docs/RUNBOOK.md)
- [docs/product-specs/PRD.md](docs/product-specs/PRD.md)
- [BUGS_AND_FIXES.md](BUGS_AND_FIXES.md)
- [SECURITY.md](SECURITY.md)
- [docs/RELEASE.md](docs/RELEASE.md)
- [CHANGELOG.md](CHANGELOG.md)

## Where to look for what

| Need | Document |
|---|---|
| Product goals and scope | [docs/product-specs/PRD.md](docs/product-specs/PRD.md) |
| Runtime flow and module boundaries | [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) |
| Setup, commands, operations, troubleshooting | [docs/RUNBOOK.md](docs/RUNBOOK.md) |
| Release process and gates | [docs/RELEASE.md](docs/RELEASE.md) |
| Bug backlog and remediation items | [BUGS_AND_FIXES.md](BUGS_AND_FIXES.md) |
| Security posture and reporting | [SECURITY.md](SECURITY.md) |
| User-facing change history | [CHANGELOG.md](CHANGELOG.md) |

## Conventions for agents

- **Language/runtime:** Python 3.13+
- **Lint/format:** `ruff check .`, `ruff format .`
- **Tests:** `pytest -q`
- **Data files:** resolve through `data_path()` and `AI_PDF_RENAMER_DATA_DIR`; no path traversal.
- **Compatibility:** preserve CLI/public API behavior unless explicitly changed.

## Harness alignment

- Keep `AGENTS.md` as a compact map.
- Prefer in-repo docs over external references.
- Update source docs when behavior changes; avoid parallel stale docs.
