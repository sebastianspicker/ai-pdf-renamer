# Security Review Notes

Date: 2026-05-01
Branch: `security-review-fixes`
Scope: local repository and test environment only.

## Repository Index

- Entry points: `ai_pdf_renamer.cli:main`, `ai_pdf_renamer.tui:main`, `ai_pdf_renamer.undo_cli:main`.
- Runtime configuration: `src/ai_pdf_renamer/cli_parser.py`, `src/ai_pdf_renamer/cli.py`, `src/ai_pdf_renamer/config.py`, `src/ai_pdf_renamer/config_resolver.py`.
- Main side-effect path: `src/ai_pdf_renamer/renamer.py`, `src/ai_pdf_renamer/renamer_files.py`, `src/ai_pdf_renamer/rename_ops.py`, `src/ai_pdf_renamer/undo_cli.py`.
- Document extraction and naming: `src/ai_pdf_renamer/pdf_extract.py`, `src/ai_pdf_renamer/renamer_extract.py`, `src/ai_pdf_renamer/filename.py`, `src/ai_pdf_renamer/heuristics.py`, `src/ai_pdf_renamer/rules.py`.
- LLM boundary: `src/ai_pdf_renamer/llm_backend.py`, `src/ai_pdf_renamer/llm.py`, `src/ai_pdf_renamer/llm_prompts.py`, `src/ai_pdf_renamer/llm_parsing.py`, `src/ai_pdf_renamer/cache.py`.
- Output boundary: `src/ai_pdf_renamer/renamer_output.py`, `src/ai_pdf_renamer/logging_utils.py`.
- Security automation: `.github/workflows/security.yml`, `.github/dependabot.yml`.
- Regression surface: `tests/test_llm_backend.py`, `tests/test_rename_ops.py`, `tests/test_undo_cli.py`, `tests/test_data_paths.py`, `tests/e2e/test_cli_e2e.py`.

## Threat Model

Primary assets are PDF text, rendered page images, extracted metadata, generated summaries/keywords/categories, local file paths, rename logs, and LLM responses.

Trust boundaries:

- CLI/TUI/env/config boundary: operator input controls directories, files, LLM endpoints, hooks, exports, cache paths, and logging.
- File boundary: candidate PDFs must stay within the selected directory unless an explicit single-file flow narrows the directory to that file's parent.
- Network boundary: HTTP LLM and HTTP post-rename hooks can receive document-derived content or metadata.
- Process boundary: post-rename hook commands run with the current user's privileges and must be treated as operator-controlled code.
- Persistence boundary: logs, persistent cache files, metadata exports, plan files, summary JSON, and rename logs can outlive the run.

## Findings

### MEDIUM: LLM HTTP error logging could persist PDF-derived content

Evidence: prompts include document text in `src/ai_pdf_renamer/llm_backend.py` request payloads. The prior HTTP error path logged the first 500 response-body characters, while normal logs are written to a persistent file sink.

Impact: if a local or remote OpenAI-compatible endpoint echoes request text in an error body, PDF content can be written to `error.log`.

Status: fixed in `security-review-fixes`.

Patch: keep HTTP error/status logging, but redact response bodies. Regression test: `tests/test_llm_backend.py::test_http_backend_http_error_does_not_log_response_body`.

### MEDIUM: persistent LLM cache stores document-derived responses as plaintext

Evidence: `src/ai_pdf_renamer/cache.py` writes raw response strings to JSON files when a persistent cache directory is configured. `src/ai_pdf_renamer/config_resolver.py` enables a persistent cache directory for batch preset runs.

Impact: summaries, keywords, and other model responses can persist outside the source PDF directory and may be captured by backups or sync tools.

Status: fixed in `security-review-fixes`.

Patch: create persistent cache directories as owner-only where supported, write cache files with owner-only permissions, force existing cache files back to owner-only after write, and keep `SECURITY.md` guidance that persistent cache contains plaintext document-derived text.

### LOW: `files_override` bypasses selected-directory containment for internal callers

Evidence: `rename_pdfs_in_directory(..., files_override=...)` passes override entries into `collect_pdf_files()`, which currently filters override entries by file existence and `.pdf` suffix but not by containment under the selected directory.

Impact: normal CLI `--file` is constrained to the file's parent, but an internal caller that accepts untrusted override input can process and rename a PDF outside the selected root.

Status: fixed in `security-review-fixes`.

Patch: require override entries to resolve inside `directory` with the existing `is_path_within()` helper. Regression test: `tests/test_data_paths.py::test_collect_files_override_rejects_paths_outside_root`.

### LOW: `--explain` can log document-derived classification details

Evidence: `--explain` logs detailed heuristic and LLM reasoning. These values can include summaries and keywords derived from PDFs.

Impact: this is operator opt-in, but sensitive document-derived details can be written to the configured log sink.

Status: documented in `security-review-fixes`.

Patch: update CLI help and security docs to make the persistence risk explicit. `--explain` remains an operator opt-in diagnostic mode and continues logging detailed reasoning.

## Non-Findings Checked

- Recursive and non-recursive PDF discovery reject symlinked PDFs resolving outside the selected root.
- Rename target collision handling avoids overwriting existing files in the tested Unix paths.
- Undo validates old and new paths under the rename-log parent and denies cross-directory undo.
- Config loading uses JSON or `yaml.safe_load`.
- LLM and hook HTTP clients disable proxy environment inheritance with `trust_env=False`.
- Security workflow includes CodeQL, dependency review, pip-audit, and TruffleHog with read-only default permissions.

## PR Summary

Title: `fix: harden document-derived security surfaces`

Summary:

- Stop logging LLM HTTP response bodies when requests fail.
- Add a regression test proving PDF-derived sentinel text from an echoed error response is not written to logs.
- Create persistent LLM cache directories and JSON files with owner-only permissions where supported.
- Reject `files_override` PDFs whose resolved path is outside the selected root directory.
- Document that `--explain` logs may include sensitive document-derived summaries or keywords.

Verification:

- Red regression before patch: `uv run pytest tests/test_llm_backend.py::test_http_backend_http_error_does_not_log_response_body -q` failed because the sentinel appeared in the logged response body.
- Red regression before patch: `uv run pytest tests/test_cache.py tests/test_data_paths.py tests/test_new_features.py tests/test_llm_backend.py -q` failed on cache permissions, `files_override` containment, and `--explain` help text.
- `uv run pytest tests/test_llm_backend.py::test_http_backend_http_error_does_not_log_response_body -q`
- `uv run pytest tests/test_cache.py tests/test_data_paths.py tests/test_new_features.py tests/test_llm_backend.py -q`
- `make release-check`
- `make e2e`
- `uv run ai-pdf-renamer --validate-config --dir . --no-llm --dry-run`
- `git diff --check`

Note: `AGENTS.md` was added for local repo-agent guidance, but the current `.gitignore` ignores `AGENTS.md`, so it will remain local unless explicitly force-added or the ignore rule changes.
