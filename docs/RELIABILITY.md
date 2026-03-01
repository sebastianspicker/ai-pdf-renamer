# Reliability – AI-PDF-Renamer

Failure modes, mitigations, and operational assumptions. Complements [RUNBOOK.md](RUNBOOK.md) and [BUGS_AND_FIXES.md](../BUGS_AND_FIXES.md).

## 1. Assumptions

- **Single process, single directory:** One run operates on one directory; no design guarantee for concurrent runs or external processes creating files in the same directory.
- **Local LLM optional:** Tool can run with heuristics only; use `--no-llm` to disable all LLM calls (heuristic-only: category from heuristics, empty summary/keywords).
- **Data files present:** heuristic_scores.json and meta_stopwords.json must exist and be valid JSON; missing or malformed data causes exit or traceback (see BUGS §6).

## 2. Failure modes and mitigations

| Failure mode | Mitigation | Limitation |
|--------------|------------|------------|
| Empty or invalid `--dir` | Reject empty; document; CLI validates (BUGS §1). | - |
| Non-interactive block (no TTY) | EOF on input returns defaults; pass all flags in scripts. | - |
| LLM unreachable or non-JSON | Retries and fallback prompts; can still yield `na` or empty. Use `--no-llm` for heuristic-only. Metadata includes `category_source` and `llm_failed`; a warning is logged when LLM was used but category is unknown. Timestamp fallback yields `YYYYMMDD-document-HHMMSS.pdf` when both heuristic and LLM fail. | User may see degraded filenames; warning and metadata aid diagnosis. |
| LLM `choices` empty or wrong shape | Guard with isinstance and str(); avoid IndexError/AttributeError. | - |
| Data files missing / bad JSON | FileNotFoundError for missing; catch JSONDecodeError/ValueError at boundary and exit with message (BUGS §6, §8). | - |
| PDF extraction fails or empty | Log “PDF appears to be empty”; skip file. Don’t distinguish empty vs extraction error (BUGS §9, §22). | - |
| One PDF crashes batch | Per-file try/except and summary (BUGS §10); partial. | Single exception can abort whole run. |
| Filename too long (ENAMETOOLONG) | Catch and raise with clear message. Optional proactive truncation via `max_filename_chars` (BUGS §12). | - |
| Rename collision / TOCTOU | Suffix _1,_2…; rename used as existence check to reduce TOCTOU; after 20 attempts raise with clear message (BUGS §14, §17). | Concurrency can still cause overwrite or inconsistent suffixes; single-process recommended. |
| EXDEV (cross-filesystem) | copy2 + unlink; on unlink failure remove target and re-raise. Copy failure can leave partial (BUGS §19). | - |
| Proxy sends local LLM traffic off-device | Disable proxy for LLM client or set NO_PROXY (BUGS §16). | Document in SECURITY/README. |
| Post-rename hook fails or times out | Hook runs in subprocess; on failure or timeout (120s) the tool logs and continues; renames are not rolled back. | Hook is best-effort; do not rely on it for critical path. |

## 3. Recovery

- **After partial run:** No built-in rollback; user may restore from backup or re-run on remaining PDFs. Logging and final summary help identify which files were renamed or skipped.
- **After crash:** Re-run is idempotent for already-renamed files (different basename); collision logic avoids overwriting same target.

## 4. Observability

- **Logs:** Structured logging; destination configurable (default error.log). See RUNBOOK. When the LLM returns no useful category, a warning is logged: "LLM was used but category is unknown; using heuristic or timestamp fallback."
- **Metadata (per-file):** Export and rename metadata include `category_source` (heuristic / llm / combined / override) and `llm_failed` (true when LLM was used but category ended up unknown). Useful for auditing and tuning.
- **Exit codes:** 0 = success, 1 = error, 130 = interrupt (Ctrl+C). See RUNBOOK.
- **Metrics/traces:** None; add if needed for production-style ops.

## 5. References

- [BUGS_AND_FIXES.md](../BUGS_AND_FIXES.md) – Quick reference table and full issue list.
- [RUNBOOK.md](RUNBOOK.md) – Setup, lint, test, security checks.
- [docs/DESIGN.md](DESIGN.md) – Design principles.
