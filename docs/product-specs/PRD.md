# Product Requirements Document – AI-PDF-Renamer

Product goals, requirements, and personas. For architecture and design see [ARCHITECTURE.md](../ARCHITECTURE.md) and [DESIGN.md](../DESIGN.md).

---

## 1. Product summary

**AI-PDF-Renamer** is a CLI (and optional GUI) tool that renames PDF files based on their content. It extracts text from PDFs, derives a date, category, keywords, and short summary using heuristics and an optional local LLM, and produces deterministic filenames in the form:

```text
YYYYMMDD-category-keywords-summary.pdf
```

The tool is **local-first**: no cloud dependency; optional LLM runs on a local endpoint (e.g. Ollama). Data stays on the user’s machine.

---

## 2. Goals

1. **Content-based naming** – Filenames reflect what the document is about (date, category, keywords, summary) so users can find and sort PDFs without opening them.
2. **Predictable and safe** – Same inputs yield the same filename shape; sanitized names; collision handling; no path injection.
3. **Scriptable** – Non-interactive use via CLI flags; clear errors for config and data; suitable for cron or automation when all options are passed.
4. **Transparent and maintainable** – Versioned data files (heuristic rules, stopwords); documented failure modes and limitations.

---

## 3. Personas

| Persona | Need |
|--------|------|
| **Individual user** | Batch-rename scanned or downloaded PDFs (e.g. `Scan0001.pdf`) to readable names without manual editing. |
| **Power user / script author** | Run the tool from scripts or cron with fixed options; get consistent output and clear exit codes. |
| **Contributor** | Understand architecture, data flow, and quality/reliability trade-offs to extend or fix the codebase. |

---

## 4. Requirements

### 4.1 Core

- Extract text from PDFs (PyMuPDF; optional OCR for scanned PDFs).
- Derive date from document content (heuristic parsing; fallback to today for invalid/missing).
- Assign a category via weighted regex rules (`heuristic_scores.json`) and optionally via LLM; combine with configurable preference (heuristic vs LLM).
- Generate keywords and short summary (optional local LLM); filter prompt artifacts via `meta_stopwords.json`.
- Build a single filename per PDF: date, optional project/version, category, keywords, summary; apply case (kebab/camel/snake) and sanitization.
- Rename files in a target directory with collision suffixes (`_1`, `_2`, …) and support cross-filesystem move (copy + unlink).

### 4.2 Non-functional

- **Local-first:** No cloud API; LLM traffic to user-configured local endpoint only; proxy disabled for built-in client so content stays on-device.
- **Single process, single directory:** One run operates on one directory; no guarantee for concurrent runs or external processes modifying the same directory.
- **Data files:** Heuristic and stopword data are versioned in the repo; path override via `AI_PDF_RENAMER_DATA_DIR`; only allowlisted filenames (no path traversal).

### 4.3 Out of scope

- Cloud LLM; PDF content editing; automatic backup of original names; multi-process safety guarantees; official support for non-English product docs (docs are in English).
- **Full RAG:** Persistent embedding index, multi-turn chat over documents, and semantic search over a stored corpus are out of scope. A minimal one-shot “query this batch” (no persistent index) may be considered later; it would not be part of the core rename workflow.

---

## 5. Success criteria

- Users can batch-rename PDFs in a directory and get readable, content-based filenames.
- Heuristic-only mode (`--no-llm`) works without any LLM; with LLM, category/keywords/summary improve when the endpoint returns valid JSON.
- Scripts can run the CLI with all required flags and get deterministic behavior and clear error messages for invalid config or missing data.

---

## 6. References

- [ARCHITECTURE.md](../ARCHITECTURE.md) – Workflow and component map.
- [DESIGN.md](../DESIGN.md) – Design goals and key decisions.
- [BUGS_AND_FIXES.md](../../BUGS_AND_FIXES.md) – Known bugs and required fixes.
