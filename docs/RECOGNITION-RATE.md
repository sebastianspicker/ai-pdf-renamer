# Recognition rate: leading logic and improvements

Overview: **Who is leading (heuristic vs LLM)?** When does the heuristic apply? Use of **Qwen-8B-128K** and further ways to improve recognition rate.

---

## 1. Who is leading? Category decision tree

**Order of checks** (in `combine_categories` and the renamer):

| # | Condition | Leading | Note |
|---|-----------|--------|------|
| 0 | `override_category` set (CLI) | **Override** | Manually chosen category; no LLM call for category (summary/keywords still run). |
| 1 | Heuristic = `unknown` | **LLM** | No heuristic suggestion; normalized LLM category is used (if not document/na/unknown). |
| 2 | Heuristic score < `min_heuristic_score` | **LLM** | Heuristic treated as too weak. CLI: `--min-heuristic-score T`. |
| 3 | LLM = `unknown`/`na`/`document`/empty | **Heuristic** | LLM yields nothing useful; heuristic wins. |
| 4 | **High-confidence override** active | **Heuristic** | Score ≥ S and gap ≥ G → force heuristic. CLI: `--heuristic-override-min-score S`, `--heuristic-override-min-gap G`. |
| 5 | **Agreement** (exact or parent/sibling) | **Heuristic** | Same category or parent–child/sibling → heuristic form is used (consistent vocabulary). |
| 6 | **Conflict** (LLM ≠ heuristic) | **Overlap or prefer_llm** | See below. |

**On conflict (LLM ≠ heuristic):**

- **Keyword overlap on** (default):  
  Overlap of category tokens with (summary + keywords).  
  Weighted: `overlap_heur + heuristic_score_weight * heuristic_score` vs `overlap_llm`.  
  → Higher value wins; **tie → heuristic**.
- **Overlap off** or no context:  
  `prefer_llm=True` (default) → **LLM**; `--prefer-heuristic` → **Heuristic**.

**Summary:** Default is “LLM leading, heuristic supports” (fills gaps, suggests, overlap resolves conflicts). Heuristic leads when: override, LLM unknown/empty, high-confidence override, agreement, or overlap tie / overlap favours heuristic.

---

## 2. When does the heuristic actually apply?

- **Always:** The heuristic is **always** computed first (on PDF text, or only the first N characters for long documents).
- **Skip LLM:** When **both** are set: `--skip-llm-if-heuristic-score-ge S` and `--skip-llm-if-heuristic-gap-ge G`, **and** heuristic ≠ unknown, score ≥ S, gap ≥ G → **no** LLM call for category; `cat_llm` is set to `cat_heur`. Saves latency/cost; heuristic is then de facto the only source for category.
- **Text used for heuristic:**
  - `--heuristic-leading-chars N` (N>0): only first N characters.
  - Else: `len(content) >= heuristic_long_doc_chars_threshold` (default 40 000) → only first `heuristic_long_doc_leading_chars` (default 12 000).
  - Else: full text.

So the heuristic “applies” as: (1) suggestion to the LLM (top-N / allowed list), (2) result when LLM is unknown/empty, override, agreement, or skip-LLM, (3) in conflict via overlap or prefer_llm.

---

## 3. Qwen-8B-128K: current usage

- **PDF text:** `pdf_extract` caps at ~120 000 tokens (`CONTEXT_128K_MAX_CONTENT_TOKENS`); beyond that, text is truncated.
- **Summary:** Up to ~480k characters (~120k tokens) a single LLM call; above that, chunking (100k chars, 5k overlap), then a final summary from partial summaries.
- **Category/keywords:** The LLM sees only **summary + keywords**, not the full PDF text. The 128k context helps mainly for **summary** on long PDFs (fewer chunks or larger single-shot texts).

**Implication:** The LLM’s category decision is based on compressed information (summary + keywords). The heuristic uses (possibly truncated) raw text—so for clear patterns (e.g. “Rechnung”, “Lohnabrechnung”) the heuristic can often be more stable.

**Single-shot vs chunking:** For Qwen-8B-128K, PDFs whose extracted text stays below ~480k characters (~120k tokens) use a single summary call; only longer documents are chunked (100k chars per chunk, 5k overlap) and then summarized again. Most typical business PDFs therefore never trigger chunking.

**Timestamp fallback:** When both heuristic and LLM yield no useful category/summary/keywords, the filename becomes `YYYYMMDD-document-HHMMSS.pdf` (or a custom segment via `--timestamp-fallback-segment`). Disable with `--no-timestamp-fallback`.

**Simple naming mode:** `--simple-naming` skips the full pipeline and asks the LLM for a single short filename (3–6 words). Reduces malformed or verbose output at the cost of less structured filenames. Prompts are kept strict (bullet rules, concrete example, “ONLY the filename”) to minimise junk output.

---

## 4. Concrete improvements derived from behaviour

The following improvements are derived directly from how the pipeline works and where it is weak or underspecified.

### 4.1 Category: make heuristic lead when it is clearly right

- **Observation:** The LLM never sees full PDF text for category; it only sees summary + keywords. For document types with strong lexical cues (invoice, payslip, contract), the heuristic sees those cues in raw text and can be more reliable.
- **Improvement A – Skip-LLM defaults:** Document or add optional defaults that skip the LLM category call when the heuristic is confident, e.g. `--skip-llm-if-heuristic-score-ge 0.5 --skip-llm-if-heuristic-gap-ge 0.3`. This reduces wrong overrides by the LLM when the heuristic is already unambiguous. *Concrete:* add a RUNBOOK “recommended for high-volume clear document types” snippet and/or a preset (e.g. `--preset high-confidence-heuristic` that sets these two plus optional override thresholds).
- **Improvement B – High-confidence override defaults:** When heuristic score and gap are above thresholds, the code already forces heuristic over LLM. Today both options are `None` by default, so this never triggers unless the user sets them. *Concrete:* either document recommended values (e.g. 0.6 / 0.35) in RUNBOOK and RECOGNITION-RATE, or introduce non-None defaults (e.g. 0.55 and 0.3) so that very clear heuristic wins even if the LLM disagrees.

### 4.2 Category: reduce false conflicts via overlap and vocabulary

- **Observation:** Conflicts are resolved by token overlap of category names with (summary + keywords). If the LLM returns a synonym or typo that is not in the alias list, we get a false conflict and then rely on overlap or prefer_llm.
- **Improvement C – More category aliases:** Every LLM/OCR variant that maps to the same canonical category reduces false conflicts. *Concrete:* extend `category_aliases.json` with systematic typo/OCR variants (e.g. “rechung”, “vertag”, “abrechnug”, “invoce”, “reciept”) and optionally derive a small set from heuristic pattern tokens (with manual review) so vocabulary stays aligned.
- **Improvement D – Heuristic score weight in overlap:** The overlap comparison uses `overlap_heur + heuristic_score_weight * heuristic_score` (default weight 0.1). For strong heuristic scores, a higher weight (e.g. 0.2) makes the heuristic more likely to win on conflict without changing the rest of the logic. *Concrete:* document in RUNBOOK that increasing `--heuristic-score-weight` (e.g. to 0.2) favours heuristic when both overlap and score are considered; optionally raise default from 0.1 to 0.15.

### 4.3 Category: focus heuristic on the informative part of the document

- **Observation:** Long documents mix many themes (e.g. cover letter + contract + appendix). Heuristic scores accumulate over the full text (or first N chars); footer/boilerplate can dilute the true document type.
- **Improvement E – Title region by default:** When the document type is usually in the first pages, weighting the first N characters (e.g. 2000–4000) reduces noise. *Concrete:* document `--title-weight-region 2000 --title-weight-factor 1.5` (or 2.0) as a recommended option for “document type in header/title”; consider non-zero default for `title_weight_region` (e.g. 2000) so short headers are favoured unless the user disables it.
- **Improvement F – Long-doc leading (documented):** For long docs (≥40k chars), only the first 12k chars are used for heuristic. For very long PDFs, lowering `--heuristic-long-doc-leading` (e.g. to 8000) can further focus on the beginning if the type is declared early. See RUNBOOK.

### 4.4 Date: prefer document date over footer/page dates

- **Observation:** Dates are extracted from the full text (or first N chars if `date_prefer_leading_chars` is set). Footers and page numbers often contain print/version dates that can override the real document date.
- **Improvement G – Date leading chars by default:** For typical business documents, the document date is in the header or first page. *Concrete:* document `--date-prefer-leading-chars 8000` (or 10000) as the recommended option; consider a non-zero default (e.g. 8000) so that the first 8k characters are searched first and the first valid date there wins, reducing footer-date wins.
- **Improvement H – More date patterns:** Add patterns in `text_utils` for common labels, e.g. “Erstellt: DD.MM.YYYY”, “Datum Rechnung: …”, “Invoice date:”, “Document date:”, so that explicitly labelled dates are found more reliably.

### 4.5 LLM (Qwen-8B-128K): summary quality and doc-type hint

- **Observation:** Category and keywords are derived from summary + keywords. If the summary does not state the document type clearly, the LLM’s category can drift; for chunked long docs, the final summary is built from partial summaries and may lose the “document type” emphasis.
- **Improvement I – Doc-type in chunked summary aggregation:** When `suggested_doc_type` is set and the document is chunked, the final aggregation prompt could include: “The document was heuristically classified as [type]. Ensure the combined summary reflects this document type.” *Concrete:* in `llm.get_document_summary`, when building `final_prompt` for the chunked path, append the same doc-type hint used in chunk prompts so the final 1–2 sentence summary consistently emphasises type.
- **Improvement J – Keep single-shot below 128K:** The current single-shot limit (~480k chars) and chunk sizes are already aligned with 128k tokens. No code change; document in RECOGNITION-RATE or PERFORMANCE that for Qwen-128K, only PDFs whose extracted text exceeds that size use chunking, so most documents get a single summary call.

### 4.6 PDF extraction and length

- **Observation:** Very long PDFs (e.g. catalogues, multi-document packs) can push the heuristic and the summary toward “noisiest” or generic content; the true type is often in the first few pages.
- **Improvement K – Max pages as tuning knob:** `--max-pages-for-extraction N` (e.g. 30 or 50) limits extraction to the first N pages. *Concrete:* document in RUNBOOK that for mixed or very long PDFs, setting `--max-pages-for-extraction 50` (or lower) can improve recognition by focusing on the main body and avoiding appendix/boilerplate.

### 4.7 Observability and tuning

- **Observation:** `CategorySource source=heuristic|llm|combined|override` and `CategoryConflict chosen=...` logs show who decided and when heuristic and LLM disagreed. Without analysing these, it is hard to tune thresholds and aliases.
- **Improvement L – Log analysis guidance:** In RECOGNITION-RATE or RUNBOOK, add a short “Tuning” subsection: (1) run with INFO logging; (2) grep for `CategoryConflict` to see frequent (llm, heuristic) pairs; (3) add aliases for LLM outputs that should map to heuristic category, or adjust override/skip-LLM thresholds; (4) use `CategorySource` to see ratio heuristic vs llm vs combined and decide whether to strengthen heuristic (e.g. skip-LLM, override, score weight) or leave LLM leading.

### 4.8 Summary of concrete actions

| ID | Area | Concrete action |
|----|------|-----------------|
| A | Skip-LLM | Document or add preset for `--skip-llm-if-heuristic-score-ge 0.5 --skip-llm-if-heuristic-gap-ge 0.3`. |
| B | Override | Document or set default `heuristic_override_min_score` / `heuristic_override_min_gap` (e.g. 0.55, 0.3). |
| C | Aliases | Add typo/OCR variants to `category_aliases.json`; optionally derive some from heuristic patterns. |
| D | Overlap weight | Document or raise default `heuristic_score_weight` (e.g. 0.15 or 0.2). |
| E | Title region | Document or set default `title_weight_region` (e.g. 2000) and `title_weight_factor` (e.g. 1.5). |
| F | Long-doc | Document tuning of `--heuristic-long-doc-leading` for long PDFs. |
| G | Date leading | Document or set default `date_prefer_leading_chars` (e.g. 8000). |
| H | Date patterns | Add date patterns in `text_utils` (Erstellt:, Invoice date:, etc.). |
| I | Chunked summary | Add doc-type hint to final aggregation prompt when `suggested_doc_type` is set. |
| J | 128K | Document single-shot vs chunking for Qwen-128K (no code change). |
| K | Max pages | Document `--max-pages-for-extraction` for long/mixed PDFs. |
| L | Tuning | Add “Tuning” subsection: use CategorySource and CategoryConflict logs to adjust aliases and thresholds. |

---

## 5. Tuning (quick reference)

1. Run with INFO logging.  
2. Grep `CategoryConflict` → add aliases for frequent LLM outputs.  
3. Tune skip-LLM / override thresholds or use `--preset high-confidence-heuristic`.  
4. Use `CategorySource` ratio (heuristic vs llm vs combined) to decide whether to strengthen heuristic (score weight, override) or leave LLM leading.  
Details: §6.

---

## 6. Tuning using logs

Use runtime logs to tune aliases and thresholds:

1. **Run with INFO logging** so that `CategorySource` and `CategoryConflict` lines are emitted.
2. **Inspect conflicts:** Grep for `CategoryConflict` to see frequent (llm, heuristic) pairs. If the LLM often returns a synonym or typo that should map to the heuristic category, add that string to `category_aliases.json` (key = normalized LLM output, value = heuristic category).
3. **Adjust thresholds:** If the heuristic is often right but the LLM overrides it, consider `--skip-llm-if-heuristic-score-ge` / `--skip-llm-if-heuristic-gap-ge` or `--preset high-confidence-heuristic`; if the heuristic is too weak, lower `--heuristic-override-min-score` / `--heuristic-override-min-gap` or use `--no-heuristic-override` to rely more on the LLM.
4. **CategorySource ratio:** Use `CategorySource source=heuristic|llm|combined|override` to see how often the category comes from heuristic only, LLM only, or both. A high share of `combined` with many conflicts may warrant more aliases or a higher `--heuristic-score-weight`.

---

## 7. Summary

- **Leading:** Default is LLM on conflict (with overlap resolution). Heuristic leads when: override, LLM unknown/empty, high-confidence override, agreement, or skip-LLM (no LLM category call then).
- **When heuristic applies:** Always as first estimate; as the result when LLM is missing/unknown, on override/agreement/skip-LLM/overlap tie, or when overlap/config favours heuristic.
- **128K:** Used for long PDFs in summary (single-shot up to ~480k chars, else chunking); category/keywords are based on summary + keywords, not full text.
- **Improvements:** The table in §4.8 lists concrete, derived improvements (defaults, aliases, date patterns, doc-type in chunked summary, and log-based tuning).

See also: [RUNBOOK.md](RUNBOOK.md), [ARCHITECTURE.md](ARCHITECTURE.md).
