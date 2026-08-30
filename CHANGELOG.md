# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Target prerelease: `0.4.0a1`. These entries remain unreleased until the exact
candidate commit is tagged and published as a GitHub prerelease.

### Added

- Linux full-gate CI plus targeted macOS and Windows smoke coverage on exact
  CPython 3.14.6.
- Alpha release notes, structured GitHub release-note categories, and a
  maintainer prerelease checklist.
- Alpha package metadata and repository checks for public documentation,
  workflows, and built distributions.
- Architecture guidance and a decision record for the modular-monolith package
  boundaries and shared reviewed-plan contract.

### Changed

- Rebranded the product, distribution, Python package, command-line tools,
  environment variables, local state paths, documentation, and release assets
  as Folionym.
- Raised the supported interpreter floor to Python 3.14 and release verification
  to exact CPython 3.14.6 using the standard GIL build.
- Made PyYAML a core runtime dependency for documented YAML configuration.
- Decomposed the CLI, filename, heuristic, LLM, renamer, text, and TUI runtime
  modules around canonical request/configuration objects and module-owned APIs.
- Grouped implementation into settings, naming, extraction, LLM, application,
  rename, infrastructure, and interface packages while retaining documented
  public facades.
- Made directory TUI Preview retain an immutable reviewed plan. Its Apply now
  uses exact included ready targets without recomputation; material edits
  invalidate the plan and incomplete Preview cannot apply. The confirmed
  single-file TUI path remains immediate and unique-available.
- Added the architecture check to the local release gate.
- Restricted post-rename hooks to HTTPS and literal-loopback HTTP endpoints and
  redacted schema-validation logs so document-derived values are not exposed.
- Restricted LLM endpoints to structurally valid HTTP(S) URLs while retaining
  the documented warning for remote plain HTTP and optional HTTPS enforcement.
- Made CI hygiene and secret scanning unconditional for applicable pushes and
  pull requests.
- Added Ruff complexity guards (`C901`, `PLR0911`, `PLR0912`, `PLR0913`, and
  `PLR0915`) and aligned mypy with Python 3.14.

### Fixed

- Removed duplicate wheel package-data inclusion and added installed-artifact
  verification for the wheel, source distribution, and all four entry points.
- Centralized tracked-path hygiene policy and added regression checks for
  private, credential, document, cache, and local automation paths.
- Switched response-cache identity to full-file hashing with change detection.
- Moved TUI work to managed Textual workers with cancellation-aware shutdown.
- Disabled redirects for LLM and diagnostic POST requests and redacted endpoint
  and request details from LLM failure logs.
- Corrected the Textual worker-message types used by the strict mypy gate.

### Removed

- The obsolete GUI command alias and the former Python compatibility façade.
- Local model-loading, automatic backend selection, embedding-assisted
  conflict resolution, and their retired configuration fields.

## [0.2.0] - 2026-04-19

Development milestone; it was not published as a GitHub release.

### Added

- `--require-https` flag and `FOLIONYM_REQUIRE_HTTPS` env var for HTTPS enforcement.
- `CategoryCombineParams` frozen dataclass for category merge configuration.
- Path traversal validation in rename operations.
- Thread-safe tiktoken initialization with double-checked locking.
- The test suite grew from 161 to 786 cases, and its coverage threshold rose
  from 50% to 85%.
- LLM backend abstraction: HTTP (llama.cpp / Ollama) and in-process (llama-cpp-python).
- Single-call LLM mode for combined summary/keywords/category extraction.
- Chat API mode with JSON response format support.
- LLM hardware presets: `apple-silicon` (default) and `gpu`.
- Terminal UI (`folionym-tui`) replacing Tkinter GUI.
- Vision fallback and vision-first modes for scanned PDFs.
- `--preset` flag (`high-confidence-heuristic`, `scanned`).
- `make release-check` target combining hygiene, lint, and tests.
- CI repository hygiene check and security workflow (CodeQL, pip-audit, TruffleHog).
- README flowchart and lifecycle state diagram (Mermaid).
- Environment variables table in README.

### Changed

- `renamer.py` decomposed: category override lookup → `renamer_lookup.py`, CSV/JSON output → `renamer_output.py`, progress reporting → `renamer_progress.py`.
- `tui.py` decomposed: TUI constants, CSS, and log formatters → `tui_assets.py`.
- Exception handlers narrowed from bare `except Exception` to specific types.
- `build_config()` decomposed into 4 focused helper functions.
- `rename_pdfs_in_directory()` decomposed with extracted `_write_rename_outputs()`.
- `combine_categories()` parameter count reduced from 13 to 5 via params object.
- CSV formula injection prevention strengthened per OWASP guidelines.
- Coverage threshold raised from 50% to 85%.
- tui.py brought under mypy strict type checking.
- Default LLM endpoint uses Ollama (`http://127.0.0.1:11434`) via presets.
- README reorganized for clarity and quick start.
- CONTRIBUTING updated with architecture table.

### Fixed

- TOCTOU race condition in `apply_single_rename()` eliminated.
- Mutable default parameter pattern in rename closure documented.
- Silent failure paths now log at debug level.
- `combine_categories()` returned LLM category even when invalid (both branches of conditional returned the same value).
- `_write_pdf_title_metadata()` tests updated to match atomic tempfile-based save implementation.
- Ruff lint violations fixed: import ordering, `contextlib.suppress` usage, line length.

### Removed

- Tkinter GUI (`gui.py`), replaced by the TUI.
- Ollama-specific code and global thread-local session management.
- `requirements.txt`; use `pyproject.toml` optional dependency groups.

## [0.1.0] - 2026-03-01

Development milestone; it was not published as a GitHub release.

### Added

- Initial public release baseline for local-first PDF renaming with CLI and GUI.
