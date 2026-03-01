# Contributing to AI-PDF-Renamer

Thank you for your interest in contributing. This document covers local setup, checks, and conventions.

## Local setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
python -m pip install -U pip
python -m pip install -e '.[dev,pdf]'
```

Optional extras: `.[tokens]` for token counting, `.[ocr]` for OCR support (see [docs/RUNBOOK.md](docs/RUNBOOK.md)).

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

## Scope and alignment

- **Features and behavior changes:** Align with [docs/product-specs/PRD.md](docs/product-specs/PRD.md) and [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).
- **Bugs:** Consider documenting or linking in [BUGS_AND_FIXES.md](BUGS_AND_FIXES.md) and opening an issue.
- **Data files:** Only allowlisted filenames are resolved (no path traversal). See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) and `src/ai_pdf_renamer/data_paths.py`.

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
- See [README.md](README.md) and [docs/RUNBOOK.md](docs/RUNBOOK.md) for primary project documentation.
