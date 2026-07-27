# Folionym

[![CI](https://github.com/sebastianspicker/AI-PDF-Renamer/actions/workflows/ci.yml/badge.svg)](https://github.com/sebastianspicker/AI-PDF-Renamer/actions/workflows/ci.yml)
[![Security](https://github.com/sebastianspicker/AI-PDF-Renamer/actions/workflows/security.yml/badge.svg)](https://github.com/sebastianspicker/AI-PDF-Renamer/actions/workflows/security.yml)

Folionym extracts content and metadata from PDF files and constructs structured
filenames. It combines category heuristics with optional OCR, vision, and an
HTTP LLM endpoint. The package provides a command-line interface, a Textual
terminal interface, a local browser interface, and an undo command.

The current source identifies itself as `0.4.0a1`. This is an alpha candidate,
not evidence of a published release. Treat rename operations as supervised file
operations: preview proposed names, keep backups of important documents, and
review the [limitations](#current-capabilities-and-limitations).

## Project purpose and scope

Folionym processes one PDF, one or more directories, or a directory watch loop.
The default filename shape is:

```text
YYYYMMDD-category-keywords-summary.pdf
```

The filename can also use a template with date, project, category, keyword,
summary, version, invoice ID, amount, and company fields. Processing runs under
the current operating-system account. Folionym does not provide a hosted
service, remote authentication, or a container deployment.

## Current capabilities and limitations

Available capabilities:

- PDF text and metadata extraction with PyMuPDF
- heuristic category scoring with bundled JSON data
- optional HTTP(S) endpoint client for summaries, keywords, categories, and
  vision
- optional OCR through OCRmyPDF
- dry runs, per-file confirmation, plan files, metadata exports, and summaries
- recursive discovery, include and exclude patterns, category overrides, and
  rules files
- backups, rename logs, and constrained undo
- browser, terminal, batch CLI, manual single-file, and watch workflows
- sequential or worker-based proposal calculation with ordered rename handling

Current limitations:

- Python 3.14 or later is required.
- Run one Folionym process per target directory. There is no interprocess lock
  for concurrent writers.
- Input size has no hard byte limit. Page extraction is unlimited unless
  `--max-pages-for-extraction` is set, worker count has no hard maximum, and
  OCR has no application-level timeout.
- OCR, vision, heuristic classification, and LLM output can be incomplete or
  incorrect. Review proposed names before applying them.
- The browser retains a reviewed Preview plan and applies its selected exact
  targets without another model call. The TUI behaves differently: Apply
  starts a separate run and can produce different proposals.
- Browser plans and reports exist only in the running process. Restarting
  `folionym-web` invalidates them.
- The browser server binds only to loopback. There is no supported remote or
  multi-user deployment.
- Configured LLM endpoints and post-rename hooks can receive content or metadata
  derived from documents. See [SECURITY.md](SECURITY.md).

## Requirements

Runtime requirements:

- CPython 3.14 or later
- PyMuPDF through the `pdf` extra for normal PDF processing
- an OpenAI-compatible HTTP endpoint when LLM or vision features are enabled
- OCRmyPDF and its platform prerequisites, including Tesseract, when OCR is
  enabled

Contributor requirements:

- CPython 3.14.6, matching `.python-version` and CI
- [uv](https://docs.astral.sh/uv/)
- Node.js 22 and npm for the React frontend
- Make

## Installation

Create an isolated environment and install from a source checkout:

```bash
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[pdf]'
```

Install only the interfaces and optional processing features you use:

```bash
python -m pip install -e '.[pdf,tui,web]'
python -m pip install -e '.[pdf,tokens,ocr,tui,web]'
```

The extras are:

| Extra | Adds |
| --- | --- |
| `pdf` | PyMuPDF text, metadata, image, and thumbnail support |
| `tokens` | token-aware content limits through tiktoken |
| `ocr` | OCRmyPDF integration |
| `tui` | Textual terminal interface |
| `web` | FastAPI, Uvicorn, PyMuPDF, and the local browser interface |
| `dev` | Python test, lint, format, type-check, and coverage tools |

For contributor setup, install every extra and the frontend dependencies:

```bash
make install-dev
```

No package-index publication is documented for `0.4.0a1`. If a GitHub
prerelease exists, install its wheel by local path after verifying its
checksum.

## Configuration

Configuration precedence is:

1. CLI flags
2. Environment defaults (for supported settings)
3. Config file values (`--config` JSON/YAML)
4. Named preset and built-in defaults

This precedence applies to the `folionym` CLI. The browser and TUI load form
values from `~/.folionym_ui.json` and pass the current form as explicit runtime
configuration. That settings file is UI state, not a `--config` file or another
CLI precedence layer.

An explicit CLI value always wins. JSON and YAML configuration files must
contain a mapping and use the same option names as the internal configuration
keys. For example:

```yaml
language: en
desired_case: kebabCase
dry_run: true
use_llm: false
workers: 2
max_pages_for_extraction: 20
```

Validate a configuration without processing files:

```bash
folionym --config ./folionym.yaml --dir ./input_files --validate-config
```

The default LLM preset is `apple-silicon`, which selects
`qwen2.5:3b` at `http://127.0.0.1:11434/v1/completions`. The `gpu` preset
selects `qwen2.5:7b-instruct` at the same endpoint. Select it with
`--llm-preset gpu`:

```bash
folionym --dir ./input_files --llm-preset gpu --dry-run
```

Override both endpoint values when your service differs:

```bash
FOLIONYM_LLM_URL=http://127.0.0.1:11434/v1/completions \
FOLIONYM_LLM_MODEL=qwen2.5:3b \
folionym --dir ./input_files --dry-run
```

The configured log path defaults to
`~/.local/share/folionym/error.log`, with `./error.log` as a fallback when the
data directory cannot be used.

Supported runtime environment variables:

| Variable | Purpose |
| --- | --- |
| `FOLIONYM_LLM_URL` | LLM HTTP endpoint |
| `FOLIONYM_LLM_MODEL` | model identifier sent to the endpoint |
| `FOLIONYM_LLM_TIMEOUT` | request timeout in seconds |
| `FOLIONYM_REQUIRE_HTTPS` | reject non-loopback HTTP LLM endpoints when true |
| `FOLIONYM_MAX_TOKENS` | PDF extraction token cap |
| `FOLIONYM_MAX_CONTENT_CHARS` | character cap for LLM input |
| `FOLIONYM_MAX_CONTENT_TOKENS` | token cap for LLM input |
| `FOLIONYM_CACHE_DIR` | persistent LLM response cache directory |
| `FOLIONYM_DATA_DIR` | directory containing the required JSON data files |
| `FOLIONYM_OCR_LANG` | OCR language |
| `FOLIONYM_POST_RENAME_HOOK` | HTTP(S) endpoint called after a rename |
| `FOLIONYM_LOG_FILE` | log file path |
| `FOLIONYM_LOG_LEVEL` | `DEBUG`, `INFO`, `WARNING`, or `ERROR` |
| `FOLIONYM_STRUCTURED_LOGS` | write JSON log records when true |
| `FOLIONYM_USE_VISION_FALLBACK` | enable vision for low-text PDFs when true |
| `FOLIONYM_VISION_FIRST` | try vision before text extraction when true |

Boolean environment values accept the forms implemented by the resolver,
including `1` and `true`.

The CLI presets are `--preset` (`high-confidence-heuristic`, `scanned`, `fast`, `accurate`, `batch`).
The related extraction switches are `--vision-fallback`, `--vision-first`,
`--vision-model`, and `--ocr`.
Use `folionym --help` for the complete option list.

## Usage

Preview a directory without an LLM call:

```bash
folionym --dir ./input_files --no-llm --dry-run
```

Preview a recursive selection:

```bash
folionym --dir ./input_files --recursive --include '*.pdf' \
  --exclude 'draft-*' --dry-run
```

Use vision fallback for a scanned document set:

```bash
folionym --dir ./input_files --preset scanned --vision-model llava --dry-run
```

Try vision before text extraction:

```bash
folionym --dir ./input_files --vision-first --vision-model llava --dry-run
```

Use OCR:

```bash
folionym --dir ./input_files --ocr --dry-run
```

Apply a reviewed CLI configuration:

```bash
folionym --dir ./input_files
```

CLI dry run and apply are separate invocations. An apply invocation calculates
proposals again from the current files and configuration; it does not retain
targets from an earlier dry run. Use `--interactive` to confirm each proposal
in the apply invocation, or use `--plan-file` when you need an exported plan
without renaming files.

Use `--interactive` to confirm each proposed rename. Use `--backup-dir` and
`--rename-log` when you need a copy and an undo record:

```bash
folionym --dir ./input_files --interactive \
  --backup-dir ./backup --rename-log ./rename.log
folionym-undo --rename-log ./rename.log --dry-run
```

Launch the terminal interface:

```bash
folionym-tui
```

Apply and Rename one PDF require explicit confirmation in the TUI. Settings
from a valid directory Preview or confirmed Apply run are stored in
`~/.folionym_ui.json`; edits are not saved merely by closing the interface.
See [docs/tui.md](docs/tui.md).

Launch the local browser interface:

```bash
folionym-web
```

It serves `http://127.0.0.1:8765/source` and opens the default browser. To
choose another loopback port without opening a browser:

```bash
folionym-web --port 9000 --no-open
```

See [docs/frontend.md](docs/frontend.md) for the browser workflow and runtime
boundaries.

Other useful commands:

```bash
folionym --doctor
folionym --dir ./input_files --watch --watch-interval 60
folionym --manual ./input_files/document.pdf
folionym-undo --rename-log ./rename.log
```

## Repository structure

| Path | Purpose |
| --- | --- |
| `src/folionym/` | Python package, bundled data, and packaged browser files |
| `frontend/` | React 19, TypeScript, Vite, Vitest, and screenshot source |
| `tests/unit/` | focused Python module and helper tests |
| `tests/integration/` | multi-module CLI, rename, browser, and security workflows |
| `tests/contracts/` | repository, CI, packaging, documentation, and screenshot contracts |
| `tests/e2e/` | subprocess CLI and undo tests |
| `tests/conftest.py`, `tests/helpers.py` | shared fixtures and test seams |
| `scripts/` | distribution checks, repository hygiene, and TUI capture |
| `docs/` | interface guides, screenshots, and release notes |
| `.github/workflows/` | CI and security workflows |
| `pyproject.toml` | package metadata, dependencies, and Python tool settings |
| `Makefile` | contributor and release verification commands |
| `uv.lock` | locked Python dependency graph |

The installed commands are:

| Command | Entry point |
| --- | --- |
| `folionym` | `folionym.cli:main` |
| `folionym-tui` | `folionym.tui:main` |
| `folionym-undo` | `folionym.undo_cli:main` |
| `folionym-web` | `folionym.web_cli:main` |

The public Python interfaces are owned by their modules:

- `folionym.config.RenamerConfig`
- `folionym.filename.FilenameGenerationRequest`
- `folionym.filename.generate_filename`
- `folionym.renamer.rename_pdfs_in_directory`
- `folionym.renamer.suggest_rename_for_file`
- `folionym.heuristics.CategoryCombineParams`

The CLI uses standard-library argparse. The supported model integration is the
HTTP-only LLM client. Here, HTTP-only distinguishes the network client from
loading a model inside the Folionym process; endpoint URLs may use HTTP or
HTTPS.

## Development workflow

Install dependencies:

```bash
make install-dev
```

Run focused checks while editing:

```bash
make format
make lint
make typecheck
make test
make frontend-check
```

Run `make docs-screenshots` only when the visible TUI changes. It replaces the
three tracked TUI SVG files from the Textual application.

Follow [CONTRIBUTING.md](CONTRIBUTING.md) for code organization, pull request
scope, and sensitive-data rules.

## Testing

The broad local gate is:

```bash
make release-check
```

It runs frontend type checking, Vitest, and the Vite build; repository hygiene;
Ruff formatting and linting; strict mypy; the coverage-gated Python suite; a
wheel and source distribution build; and an isolated installed-wheel check.

The process-level CLI tests are separate:

```bash
make e2e
```

CI runs the full release gate on Linux with Python 3.14.6. macOS and Windows
run targeted smoke jobs. The security workflow separately runs CodeQL,
dependency review, pip-audit, and verified-secret scanning.

## Deployment and operation

Folionym is installed and run as a local Python application. The browser
interface is a loopback Uvicorn process, not a remote server deployment. Keep
the terminal that started `folionym-web` open for the duration of the browser
session. Only one browser operation can be active at a time.

Release publication is a manual GitHub prerelease procedure. Maintainers should
follow [RELEASING.md](RELEASING.md) from a clean checkout of the exact commit
being tagged.

## Troubleshooting

- Run `folionym --doctor` to check optional dependencies, data files, and LLM
  connectivity.
- Run `folionym --validate-config --dir . --no-llm --dry-run` to isolate
  configuration errors without processing PDFs.
- Add `--no-llm` when the configured endpoint is unavailable.
- Install `.[pdf]` if PyMuPDF is missing.
- Install `.[ocr]` and OCRmyPDF's system prerequisites if `--ocr` is
  unavailable.
- Use `--require-https` or `FOLIONYM_REQUIRE_HTTPS=1` for non-loopback LLM
  endpoints.
- If the browser package is missing, install `.[web]`. If the packaged frontend
  is absent in a source checkout, run `make frontend-check`.
- A rename log can undo only safe same-directory moves whose target still
  exists and whose original path is free.
- Inspect the configured log file for operational failures, but treat it as
  document-adjacent private data.

## Security considerations

Process only documents you trust. PDF and OCR libraries run without an
application sandbox. Keep caches, logs, exports, rename logs, backups, and
browser artifacts on private storage.

LLM requests do not follow redirects and ignore proxy environment variables.
Remote HTTP emits a warning unless HTTPS is required. Post-rename hooks accept
HTTP only for literal loopback addresses and require HTTPS elsewhere. Review
[SECURITY.md](SECURITY.md) before sending document-derived data to either
integration.

## Contribution guidance

Keep changes focused, add tests for behavior changes, and run the narrowest
relevant check before `make release-check`. Do not add PDFs, extracted text,
model request or response bodies, local logs, caches, or rename records to a
pull request. Report vulnerabilities through the private channel described in
[SECURITY.md](SECURITY.md).

See [CONTRIBUTING.md](CONTRIBUTING.md) for the contributor workflow and
[docs/README.md](docs/README.md) for the documentation index.

## License

Folionym is licensed under the [MIT License](LICENSE).
