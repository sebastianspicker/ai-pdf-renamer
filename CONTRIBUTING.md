# Contributing to AI-PDF-Renamer

Thank you for your interest in contributing. This document covers local setup, checks, and conventions.

## Local setup

```bash
uv sync --extra dev --extra pdf --extra tui
```

Optional extras beyond the default contributor setup: `--extra tokens` for token counting, `--extra ocr` for OCR support, `--extra llama-cpp` for the in-process LLM backend.

## Run checks

Before submitting a pull request, run:

```bash
make release-check
```

Clean ignored local artifacts (optional but recommended before reviews):

```bash
make clean
```

`make release-check` runs the same hygiene, lint, type-check, and coverage-gated test commands as CI.
CI still adds `uv sync --frozen --extra dev --extra pdf --extra tui` and a single Python 3.11 job, so local runs are close parity rather than byte-for-byte identical.
Run `make typecheck` alone to run `mypy` in isolation. Fix any reported issues locally first.

## Architecture overview

Key source modules under `src/ai_pdf_renamer/`:

| Module | Purpose |
|---|---|
| `cli.py` / `cli_parser.py` | CLI entry point and argument parsing |
| `config.py` / `config_resolver.py` | Config dataclass and normalization |
| `renamer.py` | Main orchestration pipeline |
| `renamer_files.py` | PDF file collection |
| `renamer_extract.py` | Extraction helpers |
| `renamer_lookup.py` | Category override lookup helpers |
| `renamer_output.py` | CSV / JSON output and CSV injection sanitization |
| `renamer_progress.py` | Rich / null progress reporter abstraction |
| `llm_backend.py` | LLM backend abstraction (HTTP / in-process) |
| `llm.py` | LLM helper functions (summary, category, keywords) |
| `llm_prompts.py` / `llm_parsing.py` | Prompt templates and JSON parsing |
| `filename.py` | Filename generation pipeline |
| `heuristics.py` | Heuristic scoring engine |
| `pdf_extract.py` | PDF text / image extraction |
| `rules.py` | Processing rules engine |
| `tui.py` | Terminal UI (textual) |
| `tui_assets.py` | TUI constants, CSS, and log-line formatters |
| `data/` | Bundled JSON data files |

Data flow: `cli.py` builds a `RenamerConfig` → `renamer.py` iterates PDFs (collecting via `renamer_files.py`) → `renamer_extract.py` extracts text → `filename.py` generates a filename (using heuristics + optional LLM) → `rename_ops.py` performs the rename. Progress is reported via `renamer_progress.py`, output written via `renamer_output.py`, and category overrides resolved via `renamer_lookup.py`.

## Scope and alignment

- **Features and behavior changes:** Open an issue for discussion before implementing large changes.
- **Bugs:** Open a GitHub issue and link it in your PR.
- **Data files:** Only allowlisted filenames are resolved (no path traversal). See `src/ai_pdf_renamer/data_paths.py`.

## Code style

- Python 3.11.
- Format with Ruff: `ruff format .`
- Lint with Ruff: `ruff check .`
- Type-checked with mypy strict: `mypy src/ai_pdf_renamer/` (required, enforced in CI).

## Security

- Do not commit PDFs or sensitive content. Use `input_files/` locally (it is gitignored).
- Security vulnerabilities: see [SECURITY.md](SECURITY.md) for reporting. Do not disclose in public issues.

## Pull requests

- Use the pull request template; describe what changed and why.
- Keep PRs focused. For large changes, consider splitting into smaller steps.
- Ensure all checks pass and the branch is up to date with the target branch.

## Questions

- Open a GitHub issue for questions or discussion.
- See [README.md](README.md) for primary project documentation.
