# Runbook

Operational setup, execution, validation, and troubleshooting.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e '.[dev,pdf]'
```

Optional extras:

```bash
python -m pip install -e '.[tokens,ocr,embeddings]'
```

## Core Commands

```bash
ruff format .
ruff check .
pytest -q
```

Release gate bundle:

```bash
make release-check
```

CLI:

```bash
ai-pdf-renamer --dir ./input_files
```

GUI:

```bash
ai-pdf-renamer-gui
```

Undo:

```bash
ai-pdf-renamer-undo --rename-log rename.log --dry-run
```

## New Ops Commands

Preflight diagnostics:

```bash
ai-pdf-renamer --doctor
```

Run summary output:

```bash
ai-pdf-renamer --dir ./input_files --summary-json run-summary.json
```

Post-rename hook:

```bash
ai-pdf-renamer --dir ./input_files --post-rename-hook "/path/to/hook.sh"
```

## Defaults and Environment

- Extraction token default: `28000` (`DEFAULT_MAX_CONTENT_TOKENS`).
- LLM URL default: `http://127.0.0.1:11434/v1/completions`.
- LLM model default: `qwen3:8b`.
- Log file default: `error.log`.

Supported env vars:

- `AI_PDF_RENAMER_DATA_DIR`
- `AI_PDF_RENAMER_LLM_URL`
- `AI_PDF_RENAMER_LLM_MODEL`
- `AI_PDF_RENAMER_LLM_TIMEOUT`
- `AI_PDF_RENAMER_MAX_TOKENS`
- `AI_PDF_RENAMER_MAX_CONTENT_CHARS`
- `AI_PDF_RENAMER_MAX_CONTENT_TOKENS`
- `AI_PDF_RENAMER_LOG_FILE`
- `AI_PDF_RENAMER_LOG_LEVEL`
- `AI_PDF_RENAMER_STRUCTURED_LOGS`
- `AI_PDF_RENAMER_POST_RENAME_HOOK`
- `AI_PDF_RENAMER_USE_VISION_FALLBACK`
- `AI_PDF_RENAMER_VISION_FIRST`

## Repository Hygiene

Clean local generated artifacts:

```bash
make clean
```

`make release-check` includes a tracked-artifact guard for:

- `.DS_Store`
- `__pycache__`, `.pytest_cache`, `.ruff_cache`, `.mypy_cache`
- `*.egg-info`
- `build/`, `dist/`

## Important Flags

Selection and execution:

- `--dir`, `--file`, `--manual`
- `--recursive`, `--max-depth`
- `--include`, `--exclude`
- `--watch`, `--watch-interval`

Behavior:

- `--no-llm`
- `--lenient-llm-json`
- `--date-format {dmy,mdy}`
- `--date-prefer-leading-chars`
- `--no-pdf-metadata-date`
- `--ocr`
- `--vision-fallback`, `--vision-first`, `--vision-model`
- `--simple-naming`

Outputs:

- `--dry-run`
- `--plan-file`
- `--rename-log`
- `--export-metadata`
- `--summary-json`
- `--write-pdf-metadata`

Rules and tuning:

- `--rules-file`
- `--skip-llm-if-heuristic-score-ge`
- `--skip-llm-if-heuristic-gap-ge`
- `--heuristic-override-min-score`
- `--heuristic-override-min-gap`
- `--heuristic-score-weight`
- `--title-weight-region`, `--title-weight-factor`
- `--max-pages-for-extraction`

## Reliability Notes

- Single-process assumption for best safety in a target directory.
- Cross-filesystem rename uses copy+unlink fallback.
- Post-rename hook is best-effort; failures are logged.
- GUI cancel is cooperative via stop-event and may not interrupt already-running external operations.

## Performance Notes

- Extraction defaults to 28k tokens for broad local-model compatibility.
- Increase `--max-tokens` only if your local model runtime supports larger contexts.
- For long/mixed documents, use `--max-pages-for-extraction` to focus on leading pages.
- Keep `--workers` conservative if LLM backend is single-GPU/single-request.

## Processing Rules File

`--rules-file` JSON supports:

- `skip_llm_if_heuristic_category`
- `force_category_by_pattern`
- `skip_files_by_pattern`
- `allowed_categories`

`allowed_categories` now constrains LLM category candidates when provided.

## Troubleshooting

| Symptom | Action |
|---|---|
| `ModuleNotFoundError: ai_pdf_renamer` in tests | Run from repo root; test bootstrap in `tests/conftest.py` should resolve `src/`. |
| LLM unreachable | Use `--doctor`; verify local endpoint; try `--no-llm` as fallback. |
| Weak scanned PDF extraction | Install OCR deps and use `--ocr`; optionally use vision fallback. |
| Data file errors | Validate JSON files under package data or `AI_PDF_RENAMER_DATA_DIR`. |
| Unexpected naming quality | Use `--rules-file`, tune heuristic thresholds, inspect logs and summary JSON. |

## GitHub Release Readiness

Before tagging a release:

1. Run `make release-check`.
2. Confirm docs are canonical and links resolve (`README`, `ARCHITECTURE`, `RUNBOOK`, `PRD`, `BUGS_AND_FIXES`, `SECURITY`, `RELEASE`, `CHANGELOG`).
3. Confirm license and security policy are present.
4. Confirm issue templates and PR template are current.
5. Confirm package version is consistent in `pyproject.toml` and `src/ai_pdf_renamer/__init__.py`.

Manual release procedure is documented in [RELEASE.md](RELEASE.md).

## Exit Codes

- `0`: success
- `1`: runtime/config failure
- `130`: interrupted (Ctrl+C)
