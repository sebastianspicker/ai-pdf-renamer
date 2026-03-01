# Design overview – AI-PDF-Renamer

Short design summary. For structure see [ARCHITECTURE.md](ARCHITECTURE.md).

## Design goals

1. **Content-based naming:** Filenames reflect date, category, keywords, and summary derived from PDF text.
2. **Local-first:** Optional local LLM; no cloud dependency; proxy disabled for LLM client so data stays on-device.
3. **Deterministic and safe:** Same inputs → same filename shape; sanitized basenames; collision suffixes; no path injection.
4. **Scriptable and clear:** CLI flags for non-interactive use; clear errors for config and data; EOF-safe prompts.

## Key decisions

- **Heuristics + LLM:** Heuristic category from regex (heuristic_scores.json) is combined with LLM category; heuristic wins unless `--prefer-llm`. This keeps predictable categories for known document types while allowing LLM to suggest for ambiguous content.
- **Chunking:** Large documents are chunked before summarization so the LLM sees bounded context; partial summaries are merged.
- **Data files in repo:** heuristic_scores.json and meta_stopwords.json are versioned; path override via `AI_PDF_RENAMER_DATA_DIR`. Only allowed filenames are resolved (no path traversal).
- **Single-directory batch:** One process, one directory per run; concurrency and TOCTOU are documented limitations.
- **LLM response schema:** A declarative JSON schema describes the document-analysis response (summary, keywords, category, final_summary_tokens). It lives in package data as `data/llm_response_schema.json` and can be overridden via `AI_PDF_RENAMER_DATA_DIR/llm_response_schema.json`. `validate_llm_document_result` uses this schema (with optional `jsonschema` when installed) and fills defaults so downstream code always sees a consistent shape (`DocumentAnalysisResult`: e.g. `"na"`/`"unknown"`/`[]` when parsing fails).
- **Timestamp fallback:** When heuristic and LLM both yield nothing useful, the filename is `YYYYMMDD-<segment>-HHMMSS.pdf` to avoid sparse or empty filenames.
- **Optional vision fallback:** When text extraction is very short, the tool can call Ollama’s vision API on the first page (optional, gated by `--vision-fallback`). No new system dependency beyond PyMuPDF for rendering.
- **Processing rules (optional):** A JSON rules file (`--rules-file`) can define: `skip_llm_if_heuristic_category` (list of categories for which the LLM is skipped and heuristic is used), `force_category_by_pattern` (fnmatch pattern + category), `skip_files_by_pattern` (fnmatch patterns to skip files), `allowed_categories` (list of category strings; when non-empty, the LLM category is restricted to this list, overriding the heuristic-based list from `use_constrained_llm_category`). Precedence: override category (CSV) > force_category_by_pattern > normal heuristic+LLM. On load error, the tool proceeds with no rules.
- **Vision prompt:** When using `--vision-fallback` or `--vision-first`, the tool uses a strict Montscan-style prompt (3–6 words, underscores only, optional date, uppercase preferred, “Respond with ONLY the filename, nothing else”). Vision output is sanitized via `sanitize_filename_from_llm` before being used as content.

## Out of scope

- Cloud LLM; GUI; PDF content editing; automatic backup of original names; multi-process safety guarantees.
- **Full RAG:** Persistent embedding index, multi-turn chat, and semantic search over a stored document corpus are out of scope. A minimal one-shot “query this batch” (no persistent index) is a possible future option only; not part of the current product.

## References

- [ARCHITECTURE.md](ARCHITECTURE.md) – Components and data flow.
- [BUGS_AND_FIXES.md](../BUGS_AND_FIXES.md) – Known issues and required fixes.
