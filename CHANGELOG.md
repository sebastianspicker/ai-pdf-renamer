# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

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

- Default LLM endpoint uses Ollama (`http://127.0.0.1:11434`) via presets.
- README reorganized for clarity and quick start.
- CONTRIBUTING updated with architecture table.

### Removed

- Tkinter GUI (`gui.py`) — replaced by TUI.
- Ollama-specific code and global thread-local session management.
- Internal documentation (AGENTS.md, BUGS_AND_FIXES.md, docs/, scripts/).
- `requirements.txt` — use `pyproject.toml` optional dependency groups.

## [0.1.0] - 2026-03-01

### Added

- Initial public release baseline for local-first PDF renaming with CLI and GUI.
