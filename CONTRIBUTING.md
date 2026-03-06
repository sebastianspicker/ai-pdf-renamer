# Contributing to AI-PDF-Renamer

Thank you for your interest in contributing. This document covers local setup, checks, and conventions.

## Local setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
python -m pip install -U pip
python -m pip install -e '.[dev,pdf]'
```

Optional extras: `.[tokens]` for token counting, `.[ocr]` for OCR support, `.[llama-cpp]` for in-process LLM backend, `.[tui]` for the terminal UI.

## Run checks

Before submitting a pull request, run:

```bash
make release-check
```

Clean ignored local artifacts (optional but recommended before reviews):

```bash
make clean
```

`make release-check` runs hygiene guard + lint + tests and matches release gate expectations.
CI runs the same lint/test/hygiene checks. Fix any reported issues locally first.

## Architecture overview

Key source modules under `src/ai_pdf_renamer/`:

| Module | Purpose |
|---|---|
| `cli.py` / `cli_parser.py` | CLI entry point and argument parsing |
| `config.py` / `config_resolver.py` | Config dataclass and normalization |
| `renamer.py` | Main orchestration pipeline |
| `llm_backend.py` | LLM backend abstraction (HTTP / in-process) |
| `llm.py` | LLM helper functions (summary, category, keywords) |
| `llm_prompts.py` / `llm_parsing.py` | Prompt templates and JSON parsing |
| `filename.py` | Filename generation pipeline |
| `heuristics.py` | Heuristic scoring engine |
| `pdf_extract.py` | PDF text / image extraction |
| `renamer_extract.py` | Extraction helpers |
| `rules.py` | Processing rules engine |
| `tui.py` | Terminal UI (textual) |
| `data/` | Bundled JSON data files |

Data flow: `cli.py` builds a `RenamerConfig` → `renamer.py` iterates PDFs → `renamer_extract.py` extracts text → `filename.py` generates a filename (using heuristics + optional LLM) → `rename_ops.py` performs the rename.

## Scope and alignment

- **Features and behavior changes:** Open an issue for discussion before implementing large changes.
- **Bugs:** Open a GitHub issue and link it in your PR.
- **Data files:** Only allowlisted filenames are resolved (no path traversal). See `src/ai_pdf_renamer/data_paths.py`.

## Code style

- Python 3.13+.
- Format with Ruff: `ruff format .`
- Lint with Ruff: `ruff check .`
- Type hints and docstrings are encouraged for public APIs.

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
