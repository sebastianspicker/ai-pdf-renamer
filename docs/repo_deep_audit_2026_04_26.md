# Deep Repository Audit (2026-04-26)

## Objective

Comprehensively review architecture, module interactions, feature surfaces, and operational quality, then remediate P0/P1/P2 issues discovered during the pass.

## End-to-end architecture

1. **CLI/TUI entrypoints**
   - `ai-pdf-renamer` (`cli.py` + `cli_parser.py`)
   - `ai-pdf-renamer-tui` (`tui.py`)
   - `ai-pdf-renamer-undo` (`undo_cli.py`)
2. **Config normalization**
   - `config.py` dataclass model
   - `config_resolver.py` precedence + presets + validation
3. **Core pipeline orchestration**
   - `renamer.py`: scan -> extract -> classify -> generate -> rename -> export/log/hook
   - `renamer_files.py`, `renamer_extract.py`, `renamer_lookup.py`, `renamer_output.py`, `renamer_progress.py`
4. **Content intelligence**
   - `pdf_extract.py` (text/OCR/vision image payload)
   - `heuristics.py`, `rules.py`
   - `filename.py`, `text_utils.py`, `llm_parsing.py`, `llm_prompts.py`, `llm_schema.py`
5. **LLM/runtime integrations**
   - `llm_backend.py`, `llm.py`, `cache.py`, `loaders.py`, `data_paths.py`

## How features work together

- File discovery from directory settings is handled in `renamer_files.py`, then each candidate file enters extraction/classification flow in `renamer.py`.
- Extraction strategy delegates to `renamer_extract.py`, combining native PDF parsing (`pdf_extract.py`) with optional OCR and optional vision fallback.
- Generated metadata (date/category/keywords/summary) is synthesized in `filename.py` using heuristics first and optional LLM augmentation (`llm_backend.py`).
- Final names are sanitized and collision-resolved in `rename_ops.py`; post-rename side effects (logs/export/hook/PDF metadata) are coordinated in `renamer.py` + `renamer_output.py`.
- Caching and data loading (`cache.py`, `loaders.py`) reduce repeated LLM work and repetitive disk parsing.

## Module-by-module audit notes

### Entrypoints / UX

- CLI parser has extensive operational flags and sensible defaults.
- Doctor mode and validation mode provide good preflight diagnostics.
- TUI remains optional and isolated behind optional dependency gates.

### Config / preset model

- Config precedence logic is centralized and testable.
- Presets (`fast`, `accurate`, `batch`, `scanned`) compose cleanly with explicit overrides.
- Enum-style validation is explicit and user-facing.

### Extraction and naming

- PDF extraction and OCR fallback paths are defensive (best-effort with warnings).
- Token-limit shrinking is bounded and includes a max-iteration guard.
- Rename logic contains collision and EXDEV handling for cross-device moves.

### Integration/safety surfaces

- LLM HTTP requests disable environment proxies (`trust_env=False`) to reduce accidental leakage.
- Post-rename hook behavior supports local commands + HTTP callback integrations.
- Structured logging path supports machine-readable CI ingestion.

## Findings and remediation tracking

### P0

- **None identified in this pass** (no deterministic data-loss or guaranteed crash paths found in reviewed codepaths).

### P1 (fixed)

1. **Cache persistence robustness**
   - **Issue:** `ResponseCache.set()` previously wrote directly to target file and propagated write interruptions more easily through partial/failed writes.
   - **Fix:** switched to atomic temp-write + replace and added OSError handling with warning logs while preserving in-memory cache hit behavior.
   - **Impact:** lower risk of cache corruption and fewer run failures in constrained filesystems.

2. **HTTP hook payload hygiene**
   - **Issue:** path values were sanitized only for env vars, not HTTP JSON payload fields; also non-JSON-native metadata objects could cause brittle payload behavior.
   - **Fix:** sanitize hook payload path strings consistently and normalize metadata through JSON round-trip (`default=str`) before POST.
   - **Impact:** better transport safety and more reliable hook interoperability.

### P2 (improved)

- Added tests validating new cache and hook hardening behaviors.
- Documented full audit method, module map, and remediation results for maintainability.

## 20-pass iterative audit log

1. CLI/parser walkthrough: validated flag surface and mode gates; no P0/P1.
2. Config resolver pass: validated precedence/preset interactions; no P0/P1.
3. Core renamer orchestration pass: identified hook payload sanitization gap (fixed).
4. Rename operations pass: collision/EXDEV handling reviewed; no new P0/P1.
5. PDF extraction/OCR pass: fallback behavior reviewed; no new P0/P1.
6. LLM backend HTTP flow pass: endpoint and failure handling reviewed; no new P0/P1.
7. Caching pass #1: identified direct-write persistence fragility (fixed via atomic replace).
8. Caching pass #2: identified corrupt-entry repeated parse overhead (fixed via cleanup on read failure).
9. Output/export pass: CSV/formula sanitization reviewed; no new P0/P1.
10. Logging pass: structured log behavior reviewed; no new P0/P1.
11. Rules/heuristics pass: loading/scoring paths reviewed; no new P0/P1.
12. TUI boundary pass: optional dependency boundaries reviewed; no new P0/P1.
13. Undo flow pass: log format assumptions reviewed; no new P0/P1.
14. Path safety pass: traversal and rename target guards reviewed; no new P0/P1.
15. Hook security pass #2: added optional HTTPS enforcement for remote hooks via env policy.
16. Hook serialization pass: normalized HTTP payload metadata to JSON-native values (fixed).
17. Test suite pass #1: added cache failure regression tests (new + passing).
18. Test suite pass #2: added hook payload/HTTPS policy regression tests (new + passing).
19. Integration pass: targeted changed-module suites executed (passing).
20. Final review pass: no additional P0/P1/P2 found in inspected surfaces.

## Refactor / dedup / optimization opportunities (next steps)

1. **Hook policy unification**
   - Introduce a dedicated hook security policy (`require_https_hooks`) matching LLM `require_https` semantics.
2. **Exception taxonomy tightening**
   - Gradually replace broad `except Exception` blocks in non-UI paths with narrower categories + stable error codes.
3. **Extraction parallelism tuning**
   - Consider bounded queues / adaptive worker strategy for very large directory scans to smooth memory spikes.
4. **Config parsing simplification**
   - Consolidate small conversion helpers (`_str`, `_bool`, optional parsers) into one typed coercion utility.

## Validation commands run

```bash
PYTHONPATH=src:.venv/lib/python3.11/site-packages pytest -q tests/test_cache.py tests/test_new_features.py tests/test_renamer_deep.py
PYTHONPATH=src:.venv/lib/python3.11/site-packages pytest -q tests/test_cache.py tests/test_rename_ops.py tests/test_new_features.py
PYTHONPATH=src:.venv/lib/python3.11/site-packages pytest -q
```

## Validation summary

- Targeted suites covering modified codepaths pass.
- Full-suite execution in this environment still fails at collection for TUI-related tests because `textual` is not installed.
