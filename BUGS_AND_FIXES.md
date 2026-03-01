# Bugs & Required Fixes

List derived from documentation, known limitations, and code review. Each item can be turned into a separate issue.

---

## Known Limitations / Bugs

### 1. [Bug] Empty `--dir` is treated as current directory

**Status:** ✅ FIXED (2026-02-23)

**Description:** When `--dir ""` is passed (e.g. from `--dir "$DIR"` with unset variable), the CLI uses the current working directory. There is no validation that the directory argument is non-empty before calling `rename_pdfs_in_directory()`.

**Impact:** Accidental bulk renames in whatever directory the command is run from (data integrity risk).

**Fix:** Reject empty `--dir` (e.g. treat as invalid and exit with a clear message or prompt); normalize/strip and require a non-empty path.

**Sources:** `src/ai_pdf_renamer/cli.py:65-70`

---

### 2. [Bug/Operational] CLI is implicitly interactive; blocks when flags are omitted

**Status:** ✅ FIXED (2026-02-23)

**Description:** If `--dir`, `--language`, `--case`, or optional `--project`/`--version` are omitted, the tool prompts for input. There is no non-interactive default for "optional" parameters.

**Impact:** In CI, cron, or any run without a TTY/stdin, the process can block indefinitely. "Optional" in docs does not match behavior (no silent default).

**Fix:** Provide true defaults for optional flags (e.g. empty project/version when not passed) and document; for required params either require flags in non-interactive mode or document that stdin must be available.

**Sources:** `src/ai_pdf_renamer/cli.py:65-100`

---

### 3. [Bug] LLM variability and non-JSON output

**Status:** ✅ FIXED (2026-02-23)

**Description:** Some local LLMs return non-JSON or partial answers. The code retries with fallback prompts but may still return `na` or empty; response parsing rejects anything not starting with `{` (no extraction from code fences or leading prose).

**Impact:** Filenames degrade to heuristic-only or `na` without clear user feedback; wrapped or preambled JSON is dropped.

**Fix:** Improve robustness: try to extract first JSON object from response (strip code fences, leading text); surface a clear message or exit when LLM is unreachable or consistently invalid instead of silent degradation.

**Sources:** README Known Issues, `src/ai_pdf_renamer/llm.py`

---

### 4. [Bug] Date extraction can pick misleading or locale-wrong dates

**Description:** Date parsing is heuristic; `\d{1,2}/\d{1,2}/\d{4}` is interpreted as DMY, so US-style MM/DD/YYYY documents can be misdated. Invalid calendar dates (e.g. 2024-02-30) are now validated and fall back to today; prior behavior is documented.

**Impact:** Wrong date prefix in generated filenames for some documents.

**Fix:** Document locale assumption (DMY); optionally add config or detection for date format; keep invalid-date fallback to today.

**Sources:** README Known Issues, `src/ai_pdf_renamer/text_utils.py`

---

### 5. [Bug] Heuristic false positives

**Description:** PDFs can match category regex patterns unintentionally; category comes from heuristic when it wins over LLM.

**Impact:** Misleading category in filenames; user may need to tune `heuristic_scores.json`.

**Fix:** Document; consider confidence or manual override; keep tuning as operator responsibility.

**Sources:** README Known Issues

---

### 6. [Bug/Config] Data files not found – unclear or traceback

**Status:** ✅ FIXED (2026-02-23)

**Description:** When data files (e.g. `heuristic_scores.json`, `meta_stopwords.json`) are missing, `data_path()` now raises a clear `FileNotFoundError`. Malformed JSON in those files still raises `json.JSONDecodeError` and is not caught at CLI level. Unsupported `data_path(filename)` raises `ValueError`.

**Impact:** Missing files: clear message. Malformed JSON or wrong filename: traceback instead of user-facing error.

**Fix:** Catch `JSONDecodeError` and `ValueError` in CLI (or in loaders), exit with a short message pointing to data dir and file names.

**Sources:** README Troubleshooting, `src/ai_pdf_renamer/data_paths.py`, `src/ai_pdf_renamer/renamer.py`

---

## Required Fixes / Improvements

### 7. [Enhancement] Guard interactive prompts against EOF/KeyboardInterrupt

**Status:** ✅ FIXED (2026-02-23)

**Description:** `input()` and `_prompt_choice()` have no try/except; EOF (Ctrl-D) or Ctrl-C during prompts can raise `EOFError`/`KeyboardInterrupt` and produce a traceback.

**Fix:** Wrap interactive input in try/except; on EOF/KeyboardInterrupt, print a short message and exit with a non-zero code (e.g. 130 for interrupt).

**Sources:** `src/ai_pdf_renamer/cli.py:50,68,95,99`

---

### 8. [Enhancement] Broader top-level exception handling in CLI

**Status:** ✅ FIXED (2026-02-23)

**Description:** `main()` only catches `(FileNotFoundError, NotADirectoryError, OSError)` around the rename flow. JSON decode errors, `ValueError` from data_paths, PDF/LLM library errors, etc. result in an unhandled traceback.

**Fix:** Catch a wider set of exceptions (e.g. `ValueError`, `json.JSONDecodeError`, `requests.RequestException`) and exit with a single-line user message; optionally log full traceback at DEBUG.

**Sources:** `src/ai_pdf_renamer/cli.py:107-110`

---

### 9. [Enhancement] Distinguish "empty PDF" from "extraction failed"

**Status:** ✅ FIXED (2026-02-23)

**Description:** When `pdf_to_text()` returns `""`, the renamer logs "PDF appears to be empty. Skipping." and continues. The same behavior occurs when the PDF could not be opened, is encrypted, or extraction raised and was swallowed inside `pdf_to_text()`.

**Fix:** Differentiate: e.g. let `pdf_to_text()` signal failure (return a sentinel or raise), or document that "empty" includes extraction failure; consider logging at WARNING when content is empty and the file is non-zero size.

**Sources:** `src/ai_pdf_renamer/renamer.py:199-205`, `src/ai_pdf_renamer/pdf_extract.py`

---

### 10. [Enhancement] Per-file error containment in batch rename

**Status:** ✅ FIXED (2026-02-23)

**Description:** A single exception during PDF extraction or filename generation (e.g. malformed JSON in data, LLM crash) aborts the entire directory run with no summary of which files succeeded or failed.

**Fix:** Wrap per-file logic (extract → generate_filename → rename) in try/except; on failure log the file and error, then continue with the next file; optionally print a short summary at the end (e.g. "Renamed N, skipped M, failed K").

**Sources:** `src/ai_pdf_renamer/renamer.py:199-236`

---

### 11. [Enhancement] `--dir` help text and default behavior aligned

**Status:** ✅ FIXED (2026-02-23)

**Description:** Help says "default: ./input_files" but the parser default is `None` and the actual default is applied only after prompting when `--dir` is omitted.

**Fix:** Document that when `--dir` is omitted the tool prompts, or implement a real default (e.g. `./input_files`) when not in interactive mode.

**Sources:** `src/ai_pdf_renamer/cli.py:12-16,65-70`

---

### 12. [Enhancement] Filename length and cross-platform safety

**Status:** ✅ FIXED (2026-02-23)

**Description:** No total filename-length cap; long LLM keywords/summary or long project/version can exceed path-component limits (e.g. 255 bytes) and cause `os.rename()` to fail (e.g. ENAMETOOLONG), aborting the run. `clean_token()` does not handle all reserved names (e.g. trailing dots, device names on Windows).

**Fix:** Enforce a max filename length (e.g. truncate or cap segment lengths); harden `clean_token()` for reserved names and trailing/control chars; document platform limits.

**Sources:** `src/ai_pdf_renamer/renamer.py:129-171`, `src/ai_pdf_renamer/text_utils.py:101-118`

---

### 13. [Operational] Stuck or ambiguous run summary

**Status:** ✅ FIXED (2026-02-23)

**Description:** When no PDFs are found or all are skipped, there is no explicit user-facing summary; success vs. no-op can be ambiguous.

**Fix:** Log or print a one-line summary at end (e.g. "No PDFs found in …" or "Processed N files, renamed M, skipped K").

**Sources:** `src/ai_pdf_renamer/renamer.py:185-190,199-205`

---

## Critical

### 14. [Bug] Rename loop TOCTOU: non-atomic exists check and rename

**Status:** ⚠️ DOCUMENTED (2026-02-23) - Added warning in docstring

**Description:** The code checks `target.exists()` then later calls `os.rename(file_path, target)`. Another process can create `target` in between, leading to overwrite (data loss) or platform-dependent rename failure.

**Impact:** Concurrent runs or external tools creating files in the same directory can cause overwrites or unpredictable failures.

**Fix:** Use atomic primitives where possible (e.g. open with O_EXCL for destination, then move); or document as single-process/single-user and accept best-effort collision handling.

**Sources:** `src/ai_pdf_renamer/renamer.py:210-217`

---

### 15. [Bug] LLM response parsing can crash on empty/invalid `choices`

**Status:** ✅ FIXED (2026-02-23)

**Description:** `LocalLLMClient.complete()` uses `data.get("choices", [{}])[0].get("text", "").strip()`. If the server returns `"choices": []`, indexing `[0]` raises `IndexError`. If `text` is not a string, `.strip()` raises `AttributeError`. These are not caught by the current `except (requests.RequestException, json.JSONDecodeError)`.

**Impact:** A single bad or empty LLM response can abort the entire rename run with a traceback.

**Fix:** Safely resolve completion text (e.g. check `choices` length and type of `text`); coerce to string before `.strip()`; optionally catch `IndexError`/`AttributeError`/`TypeError` and return `""` with a log line.

**Sources:** `src/ai_pdf_renamer/llm.py:107-120`

---

### 16. [Bug] Security: proxy can route "local" LLM traffic off-device

**Status:** ✅ FIXED (2026-02-23)

**Description:** `requests.post(self.base_url, ...)` honors proxy env vars (e.g. `HTTP_PROXY`). If the environment has a proxy and loopback is not in `NO_PROXY`, requests to `http://127.0.0.1:11434` can be sent via the proxy, exposing PDF-derived prompt content off-device.

**Impact:** Document text (and thus potentially sensitive content) may leave the machine despite "local" LLM configuration.

**Fix:** Disable proxy for the LLM client (e.g. pass `trust_env=False` or set session proxies to empty for this host); document in README/SECURITY and recommend `NO_PROXY` for 127.0.0.1 when using a local endpoint.

**Sources:** `src/ai_pdf_renamer/llm.py`, README, SECURITY.md

---

## High

### 17. [Bug] Collision suffix logic can skip suffixes (e.g. `_1`) under races

**Status:** ✅ FIXED (2026-02-23) - MAX_RENAME_RETRIES=20 added

**Description:** On `FileExistsError`, the code increments `counter` and builds a new target. Depending on ordering, the first retry can be `base_2` instead of `base_1`; combined with the `exists()` loop, suffix numbering can be inconsistent.

**Fix:** Ensure suffix progression is deterministic (e.g. re-check existing files and pick the next free suffix in one place); document best-effort behavior under concurrency.

**Sources:** `src/ai_pdf_renamer/renamer.py:208-220`

---

### 18. [Bug] Unbounded retry loop on rename under contention

**Status:** ✅ FIXED (2026-02-23) - MAX_RENAME_RETRIES=20 added

**Description:** The rename loop has no upper bound on retries. If another process repeatedly creates the next candidate filename, the loop can run indefinitely.

**Fix:** Add a max retry count (e.g. 100); after that, log and skip the file or exit with a clear error.

**Sources:** `src/ai_pdf_renamer/renamer.py:210-220`

---

### 19. [Bug] EXDEV fallback (copy2 + unlink) is non-atomic

**Status:** ✅ FIXED (2026-02-23)

**Description:** Cross-filesystem rename uses `shutil.copy2` then `file_path.unlink()`. On unlink failure the code removes the target and re-raises, but copy failure is not wrapped; crashes or I/O errors can leave duplicate or partial files.

**Fix:** On copy failure, remove the target if it was created; document that EXDEV path is best-effort and that duplicates are possible on failure.

**Sources:** `src/ai_pdf_renamer/renamer.py:221-235`

---

## New Issues Found (2026-02-23 Code Review)

### 20. [P0-Critical] Missing `import os` in pdf_extract.py

**Status:** ✅ FIXED (2026-02-23)

**Description:** The function `_ocr_language_code()` in `pdf_extract.py` uses `os.environ.get("AI_PDF_RENAMER_OCR_LANG")` but the `os` module was never imported in this file. This causes a `NameError` at runtime when OCR is used.

**Impact:** OCR feature completely broken - any use of `--ocr` flag would crash with `NameError: name 'os' is not defined`.

**Fix:** Added `import os` to the imports at the top of `pdf_extract.py`.

**Sources:** `src/ai_pdf_renamer/pdf_extract.py:130`

---

### 21. [P2-Enhancement] ReDoS risk via heuristic regex patterns

**Status:** ✅ DOCUMENTED (2026-02-23) - Added security note in docstring

**Description:** The `_score_text()` function applies regex patterns loaded from `heuristic_scores.json`. If a malicious pattern with catastrophic backtracking is inserted, it could cause denial of service.

**Mitigating factors:** Text length is capped at 100,000 characters, and patterns are loaded from a local file (not user input).

**Fix:** Added security documentation in `_score_text()` docstring noting that pattern files should only be modified by trusted users.

**Sources:** `src/ai_pdf_renamer/heuristics.py:105-133`

---

### 22. [P2-Enhancement] Global mutable state for sessions/cache

**Status:** ✅ DOCUMENTED (2026-02-23) - Added thread-safety notes

**Description:** Global variables `_llm_sessions`, `_CATEGORY_ALIASES`, and `_embedding_model` are mutable and could cause issues in multi-threaded environments.

**Mitigating factors:** `_llm_sessions` has a lock (`_llm_sessions_lock`), and the application is primarily single-threaded.

**Fix:** Added documentation comments noting thread-safety considerations and suggesting dependency injection for multi-process deployments.

**Sources:** `src/ai_pdf_renamer/llm.py:16-17`, `src/ai_pdf_renamer/heuristics.py:286,342`

---

### 20. [Bug] LLM JSON salvage can corrupt list/multi-key responses

**Description:** The sanitizer's "single-key string" salvage path is used after any JSON decode error. For list values or multi-key objects, the salvage logic (first/last quote scan, escape inner quotes) can produce wrong but parseable JSON and thus wrong summary/keywords/category in filenames.

**Fix:** Restrict salvage to responses that look like single-key string objects; for list or multi-key decode failures, do not run salvage and return None (retry or use fallback).

**Sources:** `src/ai_pdf_renamer/llm.py:32-69`

---

### 21. [Bug] LLM always invoked; no CLI "off" switch

**Description:** The main rename flow always uses the LLM (summary, keywords, category, final_summary_tokens) when processing a non-empty PDF. Docs describe the LLM as "optional," but there is no flag or config to disable sending document text to HTTP.

**Impact:** Users may assume "optional" means off by default; in practice every run with sufficient text triggers HTTP requests with document content.

**Fix:** Add a CLI/config option to disable LLM (e.g. heuristic-only mode); document default as "LLM on when endpoint is available" and how to run without it.

**Sources:** README, `src/ai_pdf_renamer/renamer.py:90-116`

---

### 22. [Bug] PDF extraction swallows all exceptions (silent partial/empty)

**Description:** In `_extract_pages`, each extraction path (`get_text("text")`, blocks, rawdict) is wrapped in `except Exception: pass`. Any failure yields partial or empty text with no error log, and the caller cannot tell "empty PDF" from "extraction failed."

**Fix:** At minimum log at WARNING on exception; optionally propagate a sentinel or raise so the renamer can distinguish empty from error.

**Sources:** `src/ai_pdf_renamer/pdf_extract.py:71-93`

---

### 23. [Bug] Response must start with `{`; no extraction from fences/preambles

**Description:** `parse_json_field()` returns None if the response does not start with `{`. Many LLMs return JSON inside code fences (```json ...```) or with a short preamble; that content is dropped.

**Fix:** Attempt to extract the first JSON object from the response (e.g. find first `{`, then matching `}`); optionally strip markdown code fences before parsing.

**Sources:** `src/ai_pdf_renamer/llm.py:72-75`

---

### 24. [Bug] snakeCase filename still uses `-` between segments

**Description:** For `desired_case="snakeCase"`, segment tokens are joined with underscores inside segments, but the overall filename is still built with `-` between date/project/category/keywords/summary/version, and a final `split("-")`/rejoin is applied. Result is mixed separators (e.g. `20260211-project_part-category-key_word-summary`).

**Fix:** When building the final filename for snakeCase, join segments with underscores (or a single consistent delimiter) and avoid treating `-` as the only structural delimiter.

**Sources:** `src/ai_pdf_renamer/renamer.py:158-171`

---

## Medium / Low (summary)

- Case handling and magic strings duplicated between argparse and prompts (low).
- Legacy `ren.py` duplicates entry point (low).
- Uppercase umlauts not transliterated in `clean_token()` (low).
- kebabCase can emit underscores when keywords contain spaces (medium).
- Date format DMY vs MM/DD/YYYY (medium).
- Magic numbers for token limits (low).
- PyMuPDF `blocks` structure assumed (`b[4]`) (medium).
- Token truncation can stop at 200 chars while still over token limit (medium).
- `_token_count()` fallback masks errors (medium).
- Multiple extraction strategies concatenated, possible duplication (medium).
- Silent degradation and retry returning non-JSON (medium).
- No connection reuse; retry log on last iteration (low).
- Quote escaping backslash parity in JSON sanitizer (medium).
- Default logging to `error.log` includes renamed filenames (medium).
- Hardcoded endpoint, logged exceptions, security docs minimal (medium/low).

---

## Quick reference: common failure causes

### 25. [Bug] [P0] Rename loop TOCTOU: silent overwrite on Unix

**Description:** On Unix, `os.rename()` silently overwrites the target file if it exists. The collision check `target.exists()` is separated from the `os.rename()` call. If another process creates the destination between the check and the rename, data is lost.

**Impact:** Permanent loss of existing files in the destination directory.

**Fix:** Use atomic primitives; or on Unix, check for existence *after* rename if possible, though `os.rename` with `EXCL` equivalent is preferred.

**Sources:** `src/ai_pdf_renamer/rename_ops.py:65-80`

---

### 26. [Security] [P0] Potential ReDoS in heuristic scoring

**Description:** Heuristic regexes are run against large PDF content using `re.search`. Poorly constructed regexes in `heuristic_scores.json` can cause exponential backtracking.

**Impact:** Application hang or DoS via malicious PDF or misconfigured rules.

**Fix:** Audit heuristic regexes; use `regex` module with timeout or cap text length per regex scan.

**Sources:** `src/ai_pdf_renamer/heuristics.py:118`

---

### 27. [Security] [P1] LLM Prompt Injection via PDF content

**Description:** PDF text is directly concatenated into LLM prompts. A malicious PDF can "hijack" the prompt instructions.

**Impact:** Filename forgery or unintended LLM behavior.

**Fix:** Use XML-like tags (e.g. `<content>...</content>`) and instruct the LLM to ignore tags inside the content; or escape triple backticks.

**Sources:** `src/ai_pdf_renamer/llm.py:222-283`

---

### 28. [Bug] [P1] requests.Session is not thread-safe in parallel mode

**Description:** A shared `requests.Session` is used across multiple threads in the `LocalLLMClient`. `requests.Session` is not guaranteed to be thread-safe for concurrent POST requests modifying internal state.

**Impact:** Intermittent network crashes or corrupted data when running with `--workers > 1`.

**Fix:** Use thread-local storage for sessions or create a new session/adapter for each thread.

**Sources:** `src/ai_pdf_renamer/llm.py:15, 311-316`

---

| Symptom | Typical cause | Fix / see |
|--------|----------------|-----------|
| `RuntimeError: PyMuPDF is required` | PDF extra not installed | `pip install -e '.[pdf]'` |
| Empty or `na` in filenames | LLM not running / non-JSON / parsing strict | Start LLM, check logs; §3, §23 |
| Data files not found | Wrong cwd or missing `AI_PDF_RENAMER_DATA_DIR` | Set env or run from repo root; README Troubleshooting |
| Traceback on run | JSON decode, ValueError, or LLM response shape | Fix data JSON; §6, §8, §15 |
| "PDF appears to be empty" | Actually empty or extraction failed | Check file; §9, §22 |
| Rename fails (e.g. ENAMETOOLONG) | Filename too long | §12; cap length or shorten project/version |
| Block/hang | Missing flags in non-interactive run | Pass all args or ensure stdin; §2 |
| Accidental rename in cwd | `--dir ""` | Reject empty dir; §1 |
| Proxy sending data off-device | Proxy set, loopback not in NO_PROXY | §16; disable proxy for client or set NO_PROXY |

---

## Deep code inspection (2026-02-28)

Findings from a thorough pass over `src/ai_pdf_renamer` (excluding `_local`) for potential errors and security risks. Prioritized P0 (critical) through P3 (nice-to-have).

### Suspicious areas and priority

| ID | Area | Why suspicious | Priority | Why it could occur |
|----|------|----------------|----------|---------------------|
| D1 | Post-rename hook uses `shell=True` | Hook string is executed by the shell; if it ever included user-controlled content (e.g. PDF or filename), risk of injection. Paths are passed via env vars only. | P2 | Design: user configures the hook; doc must state not to embed untrusted data. |
| D2 | `_load_override_category_map`: `except OSError: pass` | Silently returns empty dict on permission error or lock; user may believe overrides are applied when they are not. | P2 | Intent was to allow missing file; OSError also covers unreadable file. |
| D3 | `--dirs-from-file`: `read_text()` loads entire file | Very large file could cause MemoryError or long delay. | P3 | No line limit; edge case for misconfigured or malicious input. |
| D4 | `rename_log_path` / `export_metadata_path` | User can set any path; we create parent and write. Could write into sensitive location if user config is wrong. | P3 | By design (user-controlled paths); document as operator responsibility. |
| D5 | PDF paths / symlinks | Opening PDFs under a directory could follow symlinks to sensitive files (e.g. /etc/shadow). | P3 | Standard for file tools; document or accept. |
| D6 | `data_path(filename)` | Only allows fixed allowlist of filenames; no path traversal. `data_dir()` is env-controlled. | OK | Allowlist prevents `../` in filename. |
| D7 | Override category CSV: keys/values | Keys are basenames (lookup); values go through filename pipeline (sanitized). No path injection. | OK | Sanitization in place. |
| D8 | Config validation | `RenamerConfig.__post_init__` validates desired_case, date_locale, category_display. | OK | Invoked on build. |
| D9 | ReDoS in heuristics | Heuristic regex runs on capped text (`max_text_length`). | OK | Documented and mitigated in code. |

### Resolution plan

- **P0/P1:** None identified; existing BUGS_AND_FIXES items cover critical/breaking items.
- **P2:** D1 → document in SECURITY.md (done); D2 → log OSError in _load_override_category_map (done).
- **P3:** D3 → cap lines read from dirs-from-file at 10_000 (done); D4/D5 → document only (paths and PDF/symlinks are operator responsibility; see SECURITY.md for hook).

---

## Using this list for issues

- **Labels:** `bug`, `enhancement`, `documentation`, `operational`, `security` as appropriate.
- **Title:** Use the **[Bug]** / **[Enhancement]** prefix or label.
- **Body:** Copy the relevant section (description, impact, fix, sources) into the issue.
- The **quick reference** table can be linked from README or a meta-issue for troubleshooting.
