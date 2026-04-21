# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-04-19

### Added

- `--require-https` flag and `AI_PDF_RENAMER_REQUIRE_HTTPS` env var for HTTPS enforcement.
- `CategoryCombineParams` frozen dataclass for category merge configuration.
- Path traversal validation in rename operations.
- Thread-safe tiktoken initialization with double-checked locking.
- 786 tests (up from 161); current coverage gate is 85% (up from 50%).
- LLM backend abstraction: HTTP (llama.cpp / Ollama) and in-process (llama-cpp-python).
- Single-call LLM mode for combined summary/keywords/category extraction.
- Chat API mode with JSON response format support.
- LLM hardware presets: `apple-silicon` (default) and `gpu`.
- Terminal UI (`ai-pdf-renamer-tui`) replacing Tkinter GUI.
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

- Tkinter GUI (`gui.py`) — replaced by TUI.
- Ollama-specific code and global thread-local session management.
- Internal documentation (AGENTS.md, BUGS_AND_FIXES.md, docs/, scripts/).
- `requirements.txt` — use `pyproject.toml` optional dependency groups.

## [0.1.0] - 2026-03-01

### Added

- Initial public release baseline for local-first PDF renaming with CLI and GUI.
