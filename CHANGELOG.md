# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `make release-check` target combining repository hygiene, lint, and tests.
- CI repository hygiene step to fail on committed generated/temp artifacts.
- `docs/RELEASE.md` with manual GitHub release procedure and checklist.
- README runtime flowchart and lifecycle state diagram (Mermaid).

### Changed

- README reorganized for release-readiness onboarding (`what it does`, quick start, flow, lifecycle, config precedence, troubleshooting).
- RUNBOOK updated with release-gate usage and release-readiness checklist.
- CONTRIBUTING now recommends `make release-check` as pre-PR validation.
- SECURITY documentation updated to match hook execution behavior (`shell=False` process creation).

## [0.1.0] - 2026-03-01

### Added

- Initial public release baseline for local-first PDF renaming with CLI and GUI.

