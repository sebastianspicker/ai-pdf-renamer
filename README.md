# AI-PDF-Renamer

[![CI](https://github.com/sebastianspicker/AI-PDF-Renamer/actions/workflows/ci.yml/badge.svg)](https://github.com/sebastianspicker/AI-PDF-Renamer/actions/workflows/ci.yml)
![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

Local-first tool to rename PDF files by content.

It extracts text, applies heuristic category scoring with optional local-LLM enrichment, and produces deterministic names like:

```text
YYYYMMDD-category-keywords-summary.pdf
```

The release gate includes a coverage threshold, static checks, repository
hygiene, and built-distribution verification.

## Project status

- Active public surface: CLI, TUI, undo CLI, local-first defaults, and documented Python API compatibility.
- Current verification gate: `make release-check`, including Ruff format/lint, mypy, and coverage-gated tests.
- Public docs index: [docs/README.md](docs/README.md).

Internal audit, remediation, archive, status, ledger, and superseded planning
packets are local-only working artifacts and are not part of the public
documentation set.

## What it does

- Renames PDFs from extracted document content.
- Uses heuristics first and optional LLM for enrichment.
- Single-call LLM mode: summary, keywords, and category in one request.
- OCR and vision fallback/vision-first modes for scanned or low-text PDFs.
- LLM hardware presets for Apple Silicon and dedicated GPU setups.
- Dry-run, plan export, metadata export, and undo.
- Runs as CLI and TUI (Textual).

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e '.[dev,pdf]'
```

Optional extras:

```bash
# TUI (terminal UI)
python -m pip install -e '.[tui]'

# Tokenization and OCR
python -m pip install -e '.[tokens,ocr]'
```

The in-process LLM backend still supports `llama-cpp-python` when you install
that package yourself. It is not bundled as a project extra, so review that
package's native build and dependency chain before adding it to your
environment.

Embedding-assisted category conflict resolution likewise remains available
when `sentence-transformers` is installed manually; it is not bundled as a
project extra because its transitive native dependency set requires separate
security review.

## Quick start

Run once against a directory. The default `apple-silicon` preset expects a
running Ollama-compatible endpoint with `qwen2.5:3b`; use `--no-llm` for a
heuristics-only run when that service or model is unavailable.

```bash
ai-pdf-renamer --dir ./input_files --dry-run
```

If you have a dedicated GPU (e.g. RTX 4080 Super 16 GB), use the `gpu` preset for the larger model:

```bash
ai-pdf-renamer --dir ./input_files --llm-preset gpu --dry-run
```

For scanned PDFs with vision-first extraction:

```bash
ai-pdf-renamer --dir ./input_files --preset scanned --dry-run
```

Apply changes:

```bash
ai-pdf-renamer --dir ./input_files
```

Preflight diagnostics:

```bash
ai-pdf-renamer --doctor
```

TUI (Textual-based terminal UI with three tabs -- Settings, Advanced, and Run):

```bash
ai-pdf-renamer-tui
```

`ai-pdf-renamer-gui` is a compatibility alias for this same terminal UI; it
does not launch a separate desktop application.

The TUI provides a complete graphical interface in the terminal:

- **Settings tab** -- configure input folder, language, case style, date format, preset, and processing flags (dry run, use LLM, OCR, vision)
- **Advanced tab** -- LLM backend selection (HTTP/in-process/auto), model URL, content limits, post-rename hook, rules file, worker count
- **Run tab** -- live progress bar, scrollable log output with color-coded rename results, and Preview / Apply / Cancel buttons

Keyboard shortcuts: `Ctrl+P` preview, `Ctrl+A` apply, `Ctrl+C` cancel, `Ctrl+Q` quit. Settings persist across sessions in `~/.ai_pdf_renamer_gui.json`.

Undo preview from rename log:

```bash
ai-pdf-renamer-undo --rename-log rename.log --dry-run
```

### LLM hardware presets

| Preset | Model | Size (Q4) | Context | Target hardware |
| --- | --- | --- | --- | --- |
| `apple-silicon` (default) | `qwen2.5:3b` | ~2 GB | 32K | Apple Silicon M4 16 GB |
| `gpu` | `qwen2.5:7b-instruct` | ~4.5 GB | 128K | RTX 4080 Super 16 GB |

Both presets use Ollama (`http://127.0.0.1:11434`). Explicit `--llm-model`, `--llm-url`, or `--max-content-chars` override preset values.

## How it works

1. Validate the input path and runtime configuration, then load rules and data.
2. Collect candidate PDFs and extract text, using OCR when configured.
3. Derive metadata with rules and heuristics. When enabled, the LLM enriches a
   structured response that is validated before use.
4. Build and sanitize the filename, resolve collisions, and either preview or
   apply the rename.
5. After an applied rename, call the optional HTTP(S) hook and aggregate the
   run summary.

LLM is optional at every stage; the heuristic path alone produces a valid filename. Both `--dry-run` and apply mode produce summary output.

## File lifecycle

Each file moves through scanning, extraction, classification, naming, and then
either preview or rename. Extraction and metadata failures are recorded before
processing continues with the next file. A configured post-rename hook runs
only after a successful rename; hook failures are recorded but remain
non-fatal. Processing ends when no candidate files remain.

Per-file failures are recorded and processing continues to the next file. Hook failures are non-fatal.

## Configuration model

Precedence is:

1. CLI flags
2. Environment defaults (for supported settings)
3. Config file values (`--config` JSON/YAML)
4. Built-in defaults

CLI options with parser defaults are treated as unset when the user omitted the flag, so a config file can still set
values such as `dry_run`, `workers`, or `use_llm`. Explicit CLI values always win.

Important defaults:

- LLM URL: `http://127.0.0.1:11434/v1/completions` (Ollama, via preset; code default without preset is `http://127.0.0.1:8080/v1/completions`)
- LLM model: `qwen2.5:3b` (apple-silicon preset) / `qwen2.5:7b-instruct` (gpu preset)
- Extraction token default: `28000` (`DEFAULT_MAX_CONTENT_TOKENS`)
- Log file: `~/.local/share/ai-pdf-renamer/error.log` (cwd fallback: `./error.log`)

High-impact operational flags:

- `--post-rename-hook URL`
- `--summary-json FILE`
- `--doctor`
- `--rules-file FILE`
- `--max-tokens N`, `--max-content-chars N`, `--max-content-tokens N`
- `--workers N`
- `--preset` (`high-confidence-heuristic`, `scanned`, `fast`, `accurate`, `batch`)
- `--llm-preset` (`apple-silicon`, `gpu`)
- `--no-single-llm-call`, `--no-chat-api`, `--no-json-mode`
- `--llm-backend` (`http`, `in-process`, `auto`)
- `--require-https` (enforce HTTPS for LLM endpoint URLs)
- `--vision-fallback`, `--vision-first`

## Environment variables

| Variable | Description |
| --- | --- |
| `AI_PDF_RENAMER_LLM_BACKEND` | LLM backend: `http`, `in-process`, or `auto` |
| `AI_PDF_RENAMER_LLM_URL` | HTTP endpoint URL |
| `AI_PDF_RENAMER_LLM_MODEL` | Model name for HTTP backend |
| `AI_PDF_RENAMER_LLM_MODEL_PATH` | Path to GGUF model file (in-process backend) |
| `AI_PDF_RENAMER_LLM_TIMEOUT` | LLM request timeout in seconds |
| `AI_PDF_RENAMER_MAX_TOKENS` | PDF extraction token cap |
| `AI_PDF_RENAMER_MAX_CONTENT_CHARS` | Cap chars of text sent to LLM |
| `AI_PDF_RENAMER_MAX_CONTENT_TOKENS` | Cap tokens for LLM (requires tiktoken) |
| `AI_PDF_RENAMER_CACHE_DIR` | Override the persistent cache directory for LLM responses |
| `AI_PDF_RENAMER_DATA_DIR` | Override path for bundled JSON data files |
| `AI_PDF_RENAMER_OCR_LANG` | OCR language override |
| `AI_PDF_RENAMER_POST_RENAME_HOOK` | HTTP(S) endpoint called after each successful rename |
| `AI_PDF_RENAMER_LOG_FILE` | Log file path (default: `~/.local/share/ai-pdf-renamer/error.log`) |
| `AI_PDF_RENAMER_LOG_LEVEL` | Log level (default: `INFO`) |
| `AI_PDF_RENAMER_STRUCTURED_LOGS` | Enable structured JSON logging (`1` or `true`) |
| `AI_PDF_RENAMER_REQUIRE_HTTPS` | Enforce HTTPS for LLM endpoint URLs (`1` or `true`) |
| `AI_PDF_RENAMER_USE_VISION_FALLBACK` | Enable vision fallback for low-text PDFs |
| `AI_PDF_RENAMER_VISION_FIRST` | Use vision extraction before text extraction |

## Public API compatibility

Stable interfaces from `ai_pdf_renamer.renamer`:

- `rename_pdfs_in_directory(directory, config, files_override=None)`
- `generate_filename(pdf_content, FilenameGenerationRequest(config=...))`; the
  current major version still accepts the legacy keyword form
  `generate_filename(pdf_content, config=..., ...)`
- `RenamerConfig`
- `CategoryCombineParams` — frozen dataclass for `combine_categories()` configuration

No signature-breaking changes within the current major version.

## Safety and limitations

- Runs locally and talks only to local LLM endpoints by default.
- Built-in LLM HTTP calls use `trust_env=False` to avoid proxy leakage.
- Post-rename hooks support HTTP(S) endpoints only. Local command hooks are not
  executed.
- Run only one instance at a time per target directory.

See [SECURITY.md](SECURITY.md) for security policy and reporting.

## Development

Install the contributor environment:

```bash
uv sync --extra dev --extra pdf --extra tui
```

Run the same local gate used before release handoff:

```bash
make release-check
```

Run the process-level CLI end-to-end tests:

```bash
make e2e
```

The E2E suite creates temporary PDFs, runs the real CLI entry points with `--no-llm`, and does not require external
services or secrets.

Smoke-test the CLI without processing files:

```bash
uv run ai-pdf-renamer --validate-config --dir . --no-llm --dry-run
```

## Troubleshooting

- Run `ai-pdf-renamer --doctor` for dependency/data/LLM diagnostics.
- If LLM endpoint is unavailable, retry with `--no-llm`.
- For scanned PDFs, install OCR deps and use `--ocr` or `--preset scanned`.
- For strict local validation, run `make release-check`.

## Documentation

- [docs/README.md](docs/README.md)
- [CONTRIBUTING.md](CONTRIBUTING.md)
- [SECURITY.md](SECURITY.md)
- [CHANGELOG.md](CHANGELOG.md)

## License

MIT — see [LICENSE](LICENSE).
