# Documentation

Folionym `0.4.0a1` is a Python 3.14 application for naming local PDF files.
The installed user interfaces are `folionym`, `folionym-tui`,
`folionym-web`, and `folionym-undo`.

## User and operator guides

- [README](../README.md): purpose, requirements, installation, configuration,
  usage, limitations, operation, and troubleshooting
- [Browser interface](frontend.md): Source, Preview, and Apply behavior,
  process lifetime, local API boundaries, screenshots, and frontend checks
- [Terminal interface](tui.md): controls, settings persistence, confirmation
  behavior, terminal requirements, and screenshot maintenance
- [Security policy](../SECURITY.md): endpoint transport, local data, parser
  boundaries, hooks, and private vulnerability reporting

## Maintainer guides

- [Contributing](../CONTRIBUTING.md): development setup, source orientation,
  checks, data handling, and pull requests
- [Product scope](../PRODUCT.md): supported workflows and interface
  requirements
- [Interface reference](../DESIGN.md): implemented visual and interaction
  conventions
- [Releasing](../RELEASING.md): clean-checkout verification, tagging, and
  GitHub prerelease publication
- [0.4.0a1 release notes](releases/0.4.0a1.md): install command, included
  surfaces, and alpha restrictions
- [Changelog](../CHANGELOG.md): release history

## Configuration and automation

- [`pyproject.toml`](../pyproject.toml): package metadata, optional extras,
  tool settings, and installed commands
- [`Makefile`](../Makefile): development and release checks
- [CI workflow](../.github/workflows/ci.yml): Linux release gate and targeted
  macOS and Windows smoke tests
- [Security workflow](../.github/workflows/security.yml): CodeQL, dependency
  review, pip-audit, and verified-secret scanning
- [Issue templates](../.github/ISSUE_TEMPLATE/): bug and feature reports
- [Pull request template](../.github/pull_request_template.md): change summary,
  verification, and sensitive-data impact

The full local release gate runs on exact CPython 3.14.6:

```bash
make release-check
```

The CLI end-to-end suite is separate:

```bash
make e2e
```
