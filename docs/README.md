# Documentation

Public documentation for AI-PDF-Renamer lives in the repository root and this
directory.

## Current status

- Project: local-first Python 3.11 CLI/TUI for renaming PDFs from document
  content.
- Active user surfaces: `ai-pdf-renamer`, `ai-pdf-renamer-tui`, and
  `ai-pdf-renamer-undo`. The `ai-pdf-renamer-gui` command remains a compatibility
  alias for the terminal UI.
- Current verification gate: `make release-check`.

## Public docs

- [README.md](../README.md) - user guide, quick start, configuration, public
  API compatibility, and operational behavior.
- [CONTRIBUTING.md](../CONTRIBUTING.md) - contributor setup, checks, architecture
  overview, and pull request expectations.
- [SECURITY.md](../SECURITY.md) - local LLM traffic, logs, caches, hooks, and
  vulnerability reporting.
- [CHANGELOG.md](../CHANGELOG.md) - release history and notable changes.

## Maintainer surfaces

- [GitHub Actions CI](../.github/workflows/ci.yml) runs the complete local
  release gate on Python 3.11.
- [GitHub Actions Security](../.github/workflows/security.yml) runs CodeQL,
  dependency review on pull requests, pip-audit, and TruffleHog.
- [Pull request template](../.github/pull_request_template.md) asks for scope,
  verification, and sensitive-data impact.
- [Issue templates](../.github/ISSUE_TEMPLATE/) route bug reports, feature
  requests, and private vulnerability reporting.

## Local-only archives

Internal audit, remediation, archive, deprecated, superseded planning, status,
and ledger files are intentionally excluded from the public docs set. Keep
current local agent status in `docs/agent/` and move retired packets into
`docs/archive/` or `docs/agent/archive/`; those paths are ignored and excluded
from GitHub/Codacy analysis.

Active public docs should not link to archived audit packets, local agent
ledgers, or private status files as current project state. Re-read live source,
tests, and configuration before copying anything from those lanes back into a
public page.
