# Runbook

Reproducible commands for setup, lint, test, build, and security checks. Use this for local development and CI alignment.

## Prerequisites

- **Python:** 3.13+ (see pyproject.toml; CI runs 3.13).
- **Local LLM (optional):** An endpoint that accepts JSON completions at `http://127.0.0.1:11434/v1/completions` is used by default for summary, keywords, and category. Use `--no-llm` to disable all LLM calls (heuristic-only: category from heuristics, empty summary/keywords). Default model: **qwen3:8b**.
- **Scanned PDFs:** Use `--ocr` so the tool runs OCR (OCRmyPDF) when a PDF has little or no text. Install with `pip install -e '.[ocr]'` and install [Tesseract](https://github.com/tesseract-ocr/tesseract) on your system. Language follows `--language` (de/en).

### Ollama 128K context (Qwen3 8B)

The tool is optimized for Qwen3 8B with a 128K context window. To allow Ollama to accept long prompts:

- Set context size when starting Ollama: `OLLAMA_NUM_CTX=131072 ollama serve` (or `export OLLAMA_NUM_CTX=131072`), or
- In the model’s Modelfile: `PARAMETER num_ctx 131072`

Then run: `ollama pull qwen3:8b`

**Using 16GB VRAM fully:** Keep a single model in VRAM (e.g. `OLLAMA_MAX_LOADED_MODELS=1`). With one GPU, `OLLAMA_NUM_GPU=1` reserves the full 16GB for the active model. See [PERFORMANCE.md](PERFORMANCE.md).

**Typical Ollama/Qwen-128K issues:** If you get empty or truncated LLM responses, ensure `OLLAMA_NUM_CTX=131072` and the model supports 128K (e.g. `qwen3:8b`). Timeouts: increase `AI_PDF_RENAMER_LLM_TIMEOUT` or `--llm-timeout` for large PDFs. Use `--no-llm` to run heuristic-only when the endpoint is unavailable.

**Vision fallback (optional):** When using `--vision-fallback`, run a vision-capable model (e.g. `ollama pull llava`). The tool calls Ollama’s `/api/chat` with the first page as an image; the same Ollama host as the completions URL is used.

**Scanned or image-only PDFs:** For folders where PDFs are scans or image-only (little or no extractable text), use `--vision-fallback --simple-naming`, or the `--preset scanned` shortcut. A vision-capable model (e.g. `llava`) must be available; the tool will use the first page image to generate a short filename when text extraction is short.

## Environment

- **Optional:** `AI_PDF_RENAMER_DATA_DIR` – directory containing:
  - `heuristic_scores.json`
  - `meta_stopwords.json`
  - `heuristic_patterns.json` (legacy; unused by code)
  - `category_aliases.json` (optional; if present overrides package aliases for LLM→category mapping)
- **Optional LLM:** `AI_PDF_RENAMER_LLM_URL`, `AI_PDF_RENAMER_LLM_MODEL`, `AI_PDF_RENAMER_LLM_TIMEOUT` – override default endpoint, model, and timeout (seconds). See also `--llm-url`, `--llm-model`, `--llm-timeout`.
- **Optional extraction:** `AI_PDF_RENAMER_MAX_TOKENS` – max tokens for PDF text (default 120000). See also `--max-tokens`.
- **Optional LLM input cap:** `AI_PDF_RENAMER_MAX_CONTENT_CHARS` – cap on characters of text sent to the LLM (summary, simple naming, etc.). When set, all such text is truncated to this length. See also `--max-content-chars`.
- **Optional token cap:** When `[tokens]` (tiktoken) is installed, config key `max_content_tokens` (env `AI_PDF_RENAMER_MAX_CONTENT_TOKENS` or `--max-content-tokens`) can limit LLM input by token count; when set and tiktoken is available, token truncation overrides the character limit for that input.
- **Optional vision fallback:** `AI_PDF_RENAMER_USE_VISION_FALLBACK` – set to `1`, `true`, or `yes` to enable vision fallback when extracted text is short (same as `--vision-fallback`). Requires a vision-capable model (e.g. llava).
- **Optional vision first:** `AI_PDF_RENAMER_VISION_FIRST` – set to `1`, `true`, or `yes` to try vision on the first page first and skip text extraction when vision succeeds (scan-only workflow; same as `--vision-first`). Requires a vision-capable model; on failure the tool falls back to text extraction.
- **Multi-language heuristics:** Rules in `heuristic_scores.json` can include `"language": "de"` or `"en"`; only rules matching `--language` (or language-agnostic) are applied. Optional per-locale files `heuristic_scores_de.json` and `heuristic_scores_en.json` in the same directory are loaded in addition when `--language` is de or en.
- **Logging:** `AI_PDF_RENAMER_LOG_FILE` – log file path (default: error.log). `AI_PDF_RENAMER_LOG_LEVEL` – DEBUG, INFO, WARNING, ERROR (default: INFO). CLI: `--quiet` (WARNING), `--verbose` (DEBUG), `--log-file`, `--log-level`.
- **Structured logs:** Set `AI_PDF_RENAMER_STRUCTURED_LOGS=1` (or true/yes) for one JSON object per line (e.g. for CI/monitoring).
- **Batch config:** `--config FILE` loads JSON or YAML as defaults (e.g. `language`, `desired_case`, `project`, `version`); CLI options override. YAML requires PyYAML.
- **Override category per file:** `--override-category-file FILE` – CSV with columns `filename`,`category` to force category per PDF.
- **Processing rules:** `--rules-file FILE` – JSON file with optional keys: `skip_llm_if_heuristic_category` (list of category strings; when heuristic category is in this list, LLM is skipped), `force_category_by_pattern` (list of `{"pattern": "fnmatch", "category": "invoice"}`; first match on basename forces category), `skip_files_by_pattern` (list of fnmatch patterns; matching files are skipped), `allowed_categories` (list of category strings; when non-empty, LLM category is restricted to this list). Precedence: override CSV > force_category_by_pattern > normal. If the file is missing or invalid, the tool runs with no rules.
- **Post-rename hook:** `AI_PDF_RENAMER_POST_RENAME_HOOK` – command or URL run after each successful rename (e.g. `curl -X POST https://...` or `/path/to/script.sh`). Receives `AI_PDF_RENAMER_OLD_PATH`, `AI_PDF_RENAMER_NEW_PATH`, and `AI_PDF_RENAMER_META` (JSON) as env vars. Hook runs with the user’s privileges; on failure the tool logs and continues (does not fail the run). Do not put untrusted PDF content into the hook string.
- **Structured export:** `--export-metadata-path FILE` writes CSV or JSON with columns: path, new_name, category, summary, keywords, category_source, llm_failed, used_vision_fallback, invoice_id, amount, company. Use for downstream tooling.
- **Embeddings for conflict resolution:** Install `.[embeddings]` and use `--embeddings-conflict` so that when heuristic and LLM disagree, embedding similarity (sentence-transformers) chooses the category.
- **Category aliases (typo/OCR):** Extend `category_aliases.json` with typo or OCR variants (e.g. rechung, vertag, invoce) so LLM output maps to the same category. Use `scripts/derive_category_aliases.py` for suggestions; see RECOGNITION-RATE §4.2.

## Setup (venv)

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
python -m pip install -U pip
python -m pip install -e '.[dev,pdf]'
```

Optional OCR for scanned PDFs (requires [Tesseract](https://github.com/tesseract-ocr/tesseract) installed):

```bash
python -m pip install -e '.[ocr]'
```

Optional token-counting support:

```bash
python -m pip install -e '.[tokens]'
```

## GUI

A simple desktop GUI (folder picker, language/case options, dry-run preview, apply) is included. It uses tkinter (stdlib) and calls the same renamer as the CLI. Use **Single file (optional)** and **Process one file** to run the pipeline on one PDF, see the suggestion (editable), and apply the rename without using the CLI. Run after install:

```bash
ai-pdf-renamer-gui
```

or:

```bash
python -m ai_pdf_renamer.gui
```

Requires the same dependencies as the CLI (install with `.[pdf]` for PDF extraction).

## Manual mode

Use `--manual path/to/file.pdf` to process a single PDF: the tool runs the full pipeline, prints the suggested filename and metadata (category, summary, keywords, category_source), then prompts to confirm (y/n) or edit (e). When you choose edit, the suggested name is shown as the default so you can press Enter to accept or type a new name. Implies interactive mode.

## New options (recursive, single file, undo, template, plan, watch, metadata)

- **Recursive:** `--recursive` / `-r` collects PDFs from subdirectories; `--max-depth N` limits depth (0 = unlimited).
- **Single file:** `--file path/to/document.pdf` processes only that PDF.
- **Undo renames:** `ai-pdf-renamer-undo --rename-log FILE` (or `python scripts/undo_renames.py -l FILE`). Use `--dry-run` to preview.
- **Template:** `--template "{date}_{category}.pdf"` with placeholders `{date}`, `{project}`, `{category}`, `{keywords}`, `{summary}`, `{version}`, `{invoice_id}`, `{amount}`, `{company}`. Structured fields (invoice_id, amount, company) are extracted heuristically from the first 5000 characters. Use `--no-structured-fields` to disable. Config key: `filename_template`.
- **Include/exclude:** `--include '*.pdf'`, `--exclude 'draft-*'` (fnmatch; repeatable).
- **Multiple dirs:** `--dir ./inbox ./archive` or `--dirs-from-file dirs.txt` (one path per line).
- **Interactive:** `--interactive` / `-i`: prompt for each file (y/n/e=edit).
- **Watch:** `--watch` scans the directory periodically and processes new/changed PDFs; `--watch-interval SEC` (default 60).
- **Plan only:** `--plan-file plan.json` writes old→new to JSON or CSV without renaming.
- **PDF metadata:** `--write-pdf-metadata` writes the new filename into the PDF `/Title` field after rename.
- **Timestamp fallback:** When category, keywords, and summary are all empty (heuristic + LLM both fail), the filename is `YYYYMMDD-document-HHMMSS.pdf` by default. Use `--no-timestamp-fallback` to disable; `--timestamp-fallback-segment NAME` to change the segment (default: document).
- **Simple naming:** `--simple-naming` uses a single LLM call to get a short filename (3–6 words, underscores) instead of the full category/summary/keywords pipeline. Useful for faster runs or when the full pipeline is too verbose.
- **Vision fallback:** When text extraction is very short (e.g. image-only PDFs), `--vision-fallback` uses Ollama’s vision API on the first page to get a short description used as content. Requires a vision-capable model (e.g. `llava`). Use `--vision-fallback-min-len N` (default 50) and optionally `--vision-model MODEL` if different from the main LLM model.
- **Vision first:** `--vision-first` (or env `AI_PDF_RENAMER_VISION_FIRST`) skips text extraction and uses only the first-page image + vision for content (scan-only workflow). If vision fails, the tool falls back to normal text extraction. Requires a vision-capable model (e.g. llava).

## Date and locale

- **Date order:** Ambiguous numeric dates (e.g. `01/02/2025`) are interpreted by `--date-format`: `dmy` (day-month-year, default) or `mdy` (month-day-year). Use `--date-format mdy` for US-style documents so the filename date matches the document date.
- **PDF metadata fallback:** When no date is found in the text, the tool uses PDF CreationDate/ModDate if available. Disable with `--no-pdf-metadata-date`.

## Recommended presets and scenarios

- **Many clear document types (e.g. invoices, payslips):** Use the high-confidence heuristic preset so the heuristic leads and the LLM category call is skipped when the heuristic is confident. Saves latency and reduces wrong overrides:

  ```bash
  ai-pdf-renamer --dir ./pdfs --preset high-confidence-heuristic
  ```

  Or set skip/override explicitly: `--skip-llm-if-heuristic-score-ge 0.5 --skip-llm-if-heuristic-gap-ge 0.3`.

- **Mixed or unclear documents:** Rely on the default (LLM leading, heuristic supports). Use `--verbose` and inspect `CategorySource` / `CategoryConflict` logs to tune aliases or thresholds.

- **Heuristic only (no LLM):** For speed or when the LLM is unavailable:

  ```bash
  ai-pdf-renamer --dir ./pdfs --no-llm
  ```

- **US date format:** For documents with month-day-year dates:

  ```bash
  ai-pdf-renamer --dir ./pdfs --date-format mdy
  ```

- **Scanned / image-only:** Use the scanned preset (vision fallback + simple naming; requires a vision model such as llava):

  ```bash
  ai-pdf-renamer --dir ./pdfs --preset scanned
  ```

  Or explicitly: `--vision-fallback --simple-naming`.

## Tuning (recognition and conflicts)

- **Heuristic vs LLM on conflict:** Increase `--heuristic-score-weight` (e.g. `0.2`) to favour the heuristic when both overlap and score are considered; default is 0.15. See [RECOGNITION-RATE.md](RECOGNITION-RATE.md).
- **Long PDFs (heuristic):** For very long documents, only the first N characters are used for heuristic scoring. Tune with `--heuristic-long-doc-chars-threshold` and `--heuristic-long-doc-leading` (e.g. lower leading to 8000 to focus on the start). See RECOGNITION-RATE §4.3.
- **Long or mixed PDFs (extraction):** Use `--max-pages-for-extraction N` (e.g. 30 or 50) to limit extraction to the first N pages and improve recognition on catalogues or mixed packs. See RECOGNITION-RATE §4.6.
- **Log-based tuning:** Run with INFO logging; grep for `CategoryConflict` and `CategorySource` to adjust aliases and thresholds. See RECOGNITION-RATE §5–6.

## Scripts

- **Derive category aliases from heuristics:** `python scripts/derive_category_aliases.py [path_to_heuristic_scores.json]` (or run from repo root with `PYTHONPATH=src`). Outputs a JSON fragment of suggested alias → category for manual review and merge into `category_aliases.json`.

## Fast loop (recommended)

```bash
ruff check .
pytest -q
```

## Format

```bash
ruff format .
```

## Lint

```bash
ruff check .
```

## Tests

```bash
pytest -q
```

CI installs `.[dev]` only; tests that need PyMuPDF (e.g. pdf_extract) are skipped if `.[pdf]` is not installed. For full coverage locally use `pip install -e '.[dev,pdf]'` or run with `PYTHONPATH=src` and optional deps installed.

## Typecheck / static analysis

Not configured in this project. Optional: run `mypy` if installed.

## Build (optional)

To produce a wheel or sdist:

```bash
python -m pip install -U build
python -m build
```

## Security checks

- **Secret scanning (CI):** TruffleHog in `.github/workflows/security.yml`.
  - Optional local (TruffleHog v2 CLI): `trufflehog --regex --entropy 1 --repo_path . .`
  - Note: CI uses the TruffleHog v3 action (different CLI).
- **SAST (CI):** CodeQL in `.github/workflows/security.yml`.
- **SCA / dependency scanning (CI):** `pip-audit -r requirements.txt`.
  - Optional local: `python -m pip install -U pip-audit` then `pip-audit -r requirements.txt`.
- **Dependency Review (CI, PR only):** GitHub Dependency Review action.
- **Optional SAST:** For deeper Python security checks: `pip install bandit semgrep` then e.g. `bandit -r src/ -ll` or Semgrep with rule set `p/python`.
- **Known issues and tech debt:** Tracked in [../BUGS_AND_FIXES.md](../BUGS_AND_FIXES.md) and [exec-plans/tech-debt-tracker.md](exec-plans/tech-debt-tracker.md).

## Heuristic options (CLI)

Optional tuning for category heuristics (see [RECOGNITION-RATE.md](RECOGNITION-RATE.md) and [ARCHITECTURE.md](ARCHITECTURE.md)):

- `--dry-run` – Do not rename; log what would be done.
- `--no-llm` – Do not call the LLM at all; category from heuristics only, summary/keywords empty (no HTTP requests).
- `--lenient-llm-json` – Try to extract JSON from LLM responses that don't start with `{` (regex fallback; use if your model often wraps JSON in prose).
- `--prefer-heuristic` – On category conflict, use heuristic instead of LLM (default: use LLM; heuristics support LLM).
- `--min-heuristic-gap DELTA` – Require best category to lead by DELTA; else use `unknown`.
- `--min-heuristic-score T` – If heuristic score &lt; T, prefer LLM category.
- `--title-weight-region N` – Weight matches in the first N characters (default 2000; use 0 to disable).
- `--title-weight-factor F` – Multiplier for title region (default 1.5).
- `--max-score-per-category M` – Cap score per category at M.
- Keyword overlap is **on by default** when heuristic and LLM disagree (pick by context). Use `--no-keyword-overlap` to disable.
- `--category-display {specific,with_parent,parent_only}` – How category appears in filename (default: `specific`).
- `--skip-llm-if-heuristic-score-ge S` and `--skip-llm-if-heuristic-gap-ge G` – Skip LLM category call when heuristic is confident (saves latency).
- `--heuristic-suggestions-top-n N` – Top-N heuristic categories to LLM (default 5).
- `--heuristic-score-weight W` – Weight heuristic score in overlap comparison (default 0.15).
- `--heuristic-override-min-score S` and `--heuristic-override-min-gap G` – Use heuristic when score and gap above threshold (default: 0.55, 0.3). Use `--no-heuristic-override` to disable override entirely.
- `--no-constrained-llm` – Do not restrict LLM to heuristic category list (default: constrained).
- `--date-prefer-leading-chars N` – Prefer date found in the first N characters (default 8000). Use 0 to search full text.
- `--heuristic-leading-chars N` – Use only the first N characters for category heuristic (0 = full text).
- `--heuristic-long-doc-threshold N` – When text length ≥ N, use only the first `--heuristic-long-doc-leading` chars for heuristic (default 40000; set 0 to disable).
- `--heuristic-long-doc-leading N` – For long docs, number of leading characters used for heuristic (default 12000). Lower (e.g. 8000) to focus on the very beginning when document type is declared early.
- `--preset high-confidence-heuristic` – Skip LLM category when heuristic is confident (score ≥ 0.5, gap ≥ 0.3). Recommended for high-volume clear document types (invoices, payslips, contracts) to reduce wrong overrides by the LLM.
- `--preset scanned` – Enable vision fallback and simple naming (for scanned or image-only PDFs). Requires a vision-capable model (e.g. llava).
- `--max-pages-for-extraction N` – Extract text only from the first N pages (0 = all). For mixed or very long PDFs (e.g. catalogues, multi-document packs), setting N (e.g. 30 or 50) can improve recognition by focusing on the main body and avoiding appendix/boilerplate.

## Exit codes

| Code | Meaning |
|------|---------|
| 0 | Success (all requested work completed or dry-run logged). |
| 1 | Configuration error, missing directory/data, LLM or rename failure, or other runtime error. |
| 130 | Interrupted by user (Ctrl+C). |

Use these in scripts (e.g. `if [ $? -eq 0 ]; then ...`).

## Troubleshooting

| Issue | Action |
|-------|--------|
| `RuntimeError: PyMuPDF is required for PDF extraction` | Install PDF extra: `python -m pip install -e '.[pdf]'`. |
| Empty or low-quality LLM output | Ensure the local LLM endpoint is running and returns JSON with the expected keys. Use `--no-llm` for heuristic-only when the endpoint is unavailable. |
| Data files not found | The package ships JSON under `src/ai_pdf_renamer/data/`. Run from the project root or set `AI_PDF_RENAMER_DATA_DIR`. |
| Heuristic-only mode | If you see "Heuristic-only mode (LLM disabled)" in the log, category comes from regex rules only; summary/keywords are empty. Use `--no-llm` explicitly or fix the LLM endpoint. |
