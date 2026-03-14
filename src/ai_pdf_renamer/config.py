"""
Renamer configuration: dataclass and build from flat dict.

Used by CLI and GUI; re-exported from renamer for backward compatibility.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

_VALID_DESIRED_CASES = frozenset({"camelCase", "kebabCase", "snakeCase"})
_VALID_DATE_LOCALES = frozenset({"dmy", "mdy"})
_VALID_CATEGORY_DISPLAY = frozenset({"specific", "with_parent", "parent_only"})


@dataclass(frozen=True)
class RenamerConfig:
    language: str = "de"
    desired_case: str = "kebabCase"
    project: str = ""
    version: str = ""
    prefer_llm_category: bool = True  # Prefer LLM; --prefer-heuristic uses heuristic
    date_locale: str = "dmy"
    date_prefer_leading_chars: int = 8000  # Prefer date from first N chars (0 = full text)
    use_pdf_metadata_for_date: bool = True  # Use PDF CreationDate/ModDate when content has no date
    dry_run: bool = False
    min_heuristic_score_gap: float = 0.0
    min_heuristic_score: float = 0.0
    title_weight_region: int = 2000  # Weight first N chars (0 = off)
    title_weight_factor: float = 1.5
    max_score_per_category: float | None = None
    use_keyword_overlap_for_category: bool = True  # On conflict pick by context overlap
    use_embeddings_for_conflict: bool = False  # If True and [embeddings], use embedding similarity instead of overlap
    category_display: str = "specific"
    # Skip LLM category call when heuristic is confident (saves latency/cost)
    skip_llm_category_if_heuristic_score_ge: float | None = None
    skip_llm_category_if_heuristic_gap_ge: float | None = None
    heuristic_suggestions_top_n: int = 5  # Top-N heuristic categories to LLM
    heuristic_score_weight: float = 0.15  # Bonus for heuristic in overlap comparison
    heuristic_override_min_score: float | None = None  # Override LLM when score >= this
    heuristic_override_min_gap: float | None = None  # and gap >= this
    use_constrained_llm_category: bool = True  # Pass full category list to LLM
    heuristic_leading_chars: int = 0  # If > 0, score category only on first N chars
    # When len(content) >= threshold, use first leading_chars for heuristic
    heuristic_long_doc_chars_threshold: int = 40_000
    heuristic_long_doc_leading_chars: int = 12_000
    max_pages_for_extraction: int = 0  # If > 0, only extract text from first N pages
    # LLM backend (env: AI_PDF_RENAMER_LLM_BACKEND; choices: "http", "in-process", "auto")
    llm_backend: str = "http"
    # LLM HTTP endpoint (env: AI_PDF_RENAMER_LLM_URL; default: llama.cpp port 8080)
    llm_base_url: str | None = None
    # LLM model name for HTTP backend (env: AI_PDF_RENAMER_LLM_MODEL)
    llm_model: str | None = None
    # LLM request timeout in seconds (env: AI_PDF_RENAMER_LLM_TIMEOUT)
    llm_timeout_s: float | None = None
    # Path to GGUF model file for in-process backend (env: AI_PDF_RENAMER_LLM_MODEL_PATH)
    llm_model_path: str | None = None
    # PDF extraction token cap (env: AI_PDF_RENAMER_MAX_TOKENS)
    max_tokens_for_extraction: int | None = None
    # UX: skip PDFs whose name already matches YYYYMMDD-*.pdf
    skip_if_already_named: bool = False
    # Optional backup dir (copy before rename) and/or rename log (old_path -> new_path)
    backup_dir: str | Path | None = None
    rename_log_path: str | Path | None = None
    # Optional export of proposed renames + metadata to CSV/JSON before applying
    export_metadata_path: str | Path | None = None
    # Optional run summary JSON output (processed/renamed/skipped/failed and details)
    summary_json_path: str | Path | None = None
    # Cap filename length (truncate from right at separator)
    max_filename_chars: int | None = None
    # Per-file category override: filename -> category (from --override-category-file)
    override_category_map: dict[str, str] | None = None
    # If True, run OCR (OCRmyPDF) when extracted text is too short (scanned PDFs). Optional [ocr].
    use_ocr: bool = False
    # Number of parallel workers for extract+generate_filename (default 1). Renames are applied sequentially.
    workers: int = 1
    # Recursive: collect PDFs from subdirectories (default False).
    recursive: bool = False
    # Max depth when recursive (0 = unlimited). Only used when recursive=True.
    max_depth: int = 0
    # Include/exclude fnmatch patterns for basename (e.g. ["*.pdf"], ["draft-*"]). None = no filter.
    include_patterns: list[str] | None = None
    exclude_patterns: list[str] | None = None
    # Custom filename template. Placeholders: date, project, category, keywords, summary, version,
    # invoice_id, amount, company. None = default schema.
    filename_template: str | None = None
    # If True, extract structured fields (invoice_id, amount, company) from content for template placeholders.
    use_structured_fields: bool = True
    # Write rename plan to this file without applying (old_path, new_path). Implies no renames.
    plan_file_path: str | Path | None = None
    # Interactive: prompt for each file (y/n/e=edit) before renaming.
    interactive: bool = False
    # Write new filename as PDF /Title metadata after rename (requires PyMuPDF).
    write_pdf_metadata: bool = False
    # If False, do not call LLM (heuristic-only: category from heuristics, summary/keywords empty).
    use_llm: bool = True
    # If True, try to extract JSON fields from LLM responses that don't start with "{" (regex fallback).
    lenient_llm_json: bool = False
    # When heuristic+LLM both yield no useful category/summary/keywords, use date + segment + HHMMSS.
    use_timestamp_fallback: bool = True
    timestamp_fallback_segment: str = "document"
    # When True, use a single LLM call for a short "filename only" instead of summary+keywords+category.
    simple_naming_mode: bool = False
    # When True and extracted text length < vision_fallback_min_text_len, try Ollama vision on first page.
    use_vision_fallback: bool = False
    vision_fallback_min_text_len: int = 50
    # Model for vision (default: same as llm_model; e.g. llava for vision-capable models).
    vision_model: str | None = None
    # When True, try vision on first page first; only if that fails, extract text (scan-only workflow).
    vision_first: bool = False
    # Cap on chars of text sent to LLM (summary, simple naming, etc.). None = use internal limits.
    max_content_chars: int | None = None
    # When set and [tokens] (tiktoken) is installed, truncate LLM input by token count.
    max_content_tokens: int | None = None
    # Optional rules file (JSON): skip_llm_if_heuristic_category, force_category_by_pattern, skip_files_by_pattern.
    rules_file: str | Path | None = None
    # Optional command or URL after each rename (env: AI_PDF_RENAMER_POST_RENAME_HOOK). Receives old/new path via env.
    post_rename_hook: str | None = None
    # Manual mode: single file, print suggestion and metadata, then prompt with suggestion as default for edit.
    manual_mode: bool = False
    # When True, use a single LLM call for summary+keywords+category instead of 4 separate calls.
    use_single_llm_call: bool = True
    # When True, use /v1/chat/completions instead of /v1/completions for text LLM calls.
    llm_use_chat_api: bool = True
    # When True, request JSON mode from the LLM server (response_format: json_object).
    llm_json_mode: bool = True
    # Hardware preset: "apple-silicon" (default, Qwen 2.5 3B) or "gpu" (Qwen 2.5 7B).
    llm_preset: str | None = None
    # Max chars of document text to send to LLM (set by llm_preset or explicit override).
    max_context_chars: int | None = None
    # Optional cooperative stop signal (used by GUI cancel button).
    stop_event: object | None = None

    def __post_init__(self) -> None:
        if self.desired_case not in _VALID_DESIRED_CASES:
            raise ValueError(f"desired_case must be one of {sorted(_VALID_DESIRED_CASES)}, got {self.desired_case!r}")
        loc = (self.date_locale or "dmy").strip().lower()
        if loc not in _VALID_DATE_LOCALES:
            raise ValueError(f"date_locale must be one of {sorted(_VALID_DATE_LOCALES)}, got {self.date_locale!r}")
        disp = (self.category_display or "specific").strip().lower()
        if disp not in _VALID_CATEGORY_DISPLAY:
            raise ValueError(
                f"category_display must be one of {sorted(_VALID_CATEGORY_DISPLAY)}, got {self.category_display!r}"
            )


def build_config_from_flat_dict(data: dict) -> RenamerConfig:
    """Build RenamerConfig from a flat dict of option names -> values. Used by CLI and GUI to avoid duplication."""
    allowed = set(RenamerConfig.__dataclass_fields__)
    kwargs = {k: v for k, v in data.items() if k in allowed}
    return RenamerConfig(**kwargs)
