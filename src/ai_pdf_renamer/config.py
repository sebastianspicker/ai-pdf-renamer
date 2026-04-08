"""
Renamer configuration: dataclass and build from flat dict.

Used by CLI and GUI; re-exported from renamer for backward compatibility.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

_VALID_DESIRED_CASES = frozenset({"camelCase", "kebabCase", "snakeCase"})
_VALID_DATE_LOCALES = frozenset({"dmy", "mdy"})
_VALID_CATEGORY_DISPLAY = frozenset({"specific", "with_parent", "parent_only"})


@dataclass(frozen=True)
class LLMConfig:
    """LLM backend and model configuration."""

    llm_backend: str = "http"
    llm_base_url: str | None = None
    llm_model: str | None = None
    llm_timeout_s: float | None = None
    llm_model_path: str | None = None
    require_https: bool = False
    use_llm: bool = True
    lenient_llm_json: bool = False
    simple_naming_mode: bool = False
    use_single_llm_call: bool = True
    llm_use_chat_api: bool = True
    llm_json_mode: bool = True
    llm_preset: str | None = None
    max_context_chars: int | None = None
    use_vision_fallback: bool = False
    vision_fallback_min_text_len: int = 50
    vision_model: str | None = None
    vision_first: bool = False
    max_content_chars: int | None = None
    max_content_tokens: int | None = None
    use_cache: bool = True
    cache_dir: str | Path | None = None


@dataclass(frozen=True)
class HeuristicConfig:
    """Heuristic scoring and category resolution configuration."""

    min_heuristic_score_gap: float = 0.0
    min_heuristic_score: float = 0.0
    title_weight_region: int = 2000
    title_weight_factor: float = 1.5
    max_score_per_category: float | None = None
    use_keyword_overlap_for_category: bool = True
    use_embeddings_for_conflict: bool = False
    category_display: str = "specific"
    skip_llm_category_if_heuristic_score_ge: float | None = None
    skip_llm_category_if_heuristic_gap_ge: float | None = None
    heuristic_suggestions_top_n: int = 5
    heuristic_score_weight: float = 0.15
    heuristic_override_min_score: float | None = None
    heuristic_override_min_gap: float | None = None
    use_constrained_llm_category: bool = True
    heuristic_leading_chars: int = 0
    heuristic_long_doc_chars_threshold: int = 40_000
    heuristic_long_doc_leading_chars: int = 12_000
    prefer_llm_category: bool = True


@dataclass(frozen=True)
class ExtractionConfig:
    """PDF text extraction configuration."""

    max_pages_for_extraction: int = 0
    max_tokens_for_extraction: int | None = None
    use_ocr: bool = False
    use_structured_fields: bool = True
    use_pdf_metadata_for_date: bool = True


@dataclass(frozen=True)
class OutputConfig:
    """Output, naming, and file-handling configuration."""

    language: str = "de"
    desired_case: str = "kebabCase"
    project: str = ""
    version: str = ""
    date_locale: str = "dmy"
    date_prefer_leading_chars: int = 8000
    dry_run: bool = False
    skip_if_already_named: bool = False
    backup_dir: str | Path | None = None
    rename_log_path: str | Path | None = None
    export_metadata_path: str | Path | None = None
    summary_json_path: str | Path | None = None
    max_filename_chars: int | None = None
    override_category_map: dict[str, str] | None = None
    workers: int = 1
    recursive: bool = False
    max_depth: int = 0
    include_patterns: list[str] | None = None
    exclude_patterns: list[str] | None = None
    filename_template: str | None = None
    plan_file_path: str | Path | None = None
    interactive: bool = False
    manual_mode: bool = False
    write_pdf_metadata: bool = False
    use_timestamp_fallback: bool = True
    timestamp_fallback_segment: str = "document"
    rules_file: str | Path | None = None
    post_rename_hook: str | None = None
    stop_event: object | None = None
    progress: bool = False
    quiet_progress: bool = False
    explain: bool = False


# Build field -> sub-config mapping at import time
_LLM_FIELDS = frozenset(f.name for f in fields(LLMConfig))
_HEURISTIC_FIELDS = frozenset(f.name for f in fields(HeuristicConfig))
_EXTRACTION_FIELDS = frozenset(f.name for f in fields(ExtractionConfig))
_OUTPUT_FIELDS = frozenset(f.name for f in fields(OutputConfig))

_SUB_CONFIG_FIELD_MAP: dict[str, str] = {}
for _fn in _LLM_FIELDS:
    _SUB_CONFIG_FIELD_MAP[_fn] = "llm"
for _fn in _HEURISTIC_FIELDS:
    _SUB_CONFIG_FIELD_MAP[_fn] = "heuristic"
for _fn in _EXTRACTION_FIELDS:
    _SUB_CONFIG_FIELD_MAP[_fn] = "extraction"
for _fn in _OUTPUT_FIELDS:
    _SUB_CONFIG_FIELD_MAP[_fn] = "output"

# All known flat field names (for backward compat __init__)
_ALL_FLAT_FIELDS = _LLM_FIELDS | _HEURISTIC_FIELDS | _EXTRACTION_FIELDS | _OUTPUT_FIELDS


class RenamerConfig:
    """Main configuration. Sub-configs group related fields; flat access for full backward compat.

    Accepts either sub-config objects (llm=LLMConfig(...), ...) or flat kwargs
    (use_llm=True, desired_case="snakeCase", ...) — both styles work.
    """

    __slots__ = ("extraction", "heuristic", "llm", "output")

    def __init__(
        self,
        *,
        llm: LLMConfig | None = None,
        heuristic: HeuristicConfig | None = None,
        extraction: ExtractionConfig | None = None,
        output: OutputConfig | None = None,
        **flat_kwargs: Any,
    ) -> None:
        # Detect unknown flat kwargs before dispatching to sub-configs so typos
        # (e.g. ``date_format`` instead of ``date_locale``) are not silently dropped.
        unknown = flat_kwargs.keys() - _ALL_FLAT_FIELDS
        if unknown:
            raise TypeError(f"RenamerConfig() got unexpected keyword argument(s): {', '.join(sorted(unknown))}")

        # Split flat kwargs into sub-config buckets
        llm_kw = {k: v for k, v in flat_kwargs.items() if k in _LLM_FIELDS}
        heur_kw = {k: v for k, v in flat_kwargs.items() if k in _HEURISTIC_FIELDS}
        ext_kw = {k: v for k, v in flat_kwargs.items() if k in _EXTRACTION_FIELDS}
        out_kw = {k: v for k, v in flat_kwargs.items() if k in _OUTPUT_FIELDS}

        # Merge flat overrides into provided sub-configs (preserves existing values)
        def _merge(cls: type, existing: Any, overrides: dict[str, Any]) -> Any:
            if existing is not None and overrides:
                base = {f.name: getattr(existing, f.name) for f in fields(cls)}
                base.update(overrides)
                return cls(**base)
            if existing is not None:
                return existing
            return cls(**overrides)

        object.__setattr__(self, "llm", _merge(LLMConfig, llm, llm_kw))
        object.__setattr__(self, "heuristic", _merge(HeuristicConfig, heuristic, heur_kw))
        object.__setattr__(self, "extraction", _merge(ExtractionConfig, extraction, ext_kw))
        object.__setattr__(self, "output", _merge(OutputConfig, output, out_kw))

        # Validation
        if self.desired_case not in _VALID_DESIRED_CASES:
            raise ValueError(
                f"Invalid --case value: {self.desired_case!r}. Choose from: {', '.join(sorted(_VALID_DESIRED_CASES))}"
            )
        loc = (self.date_locale or "dmy").strip().lower()
        if loc not in _VALID_DATE_LOCALES:
            raise ValueError(
                f"Invalid --date-format value: {self.date_locale!r}. "
                f"Choose from: {', '.join(sorted(_VALID_DATE_LOCALES))}"
            )
        disp = (self.category_display or "specific").strip().lower()
        if disp not in _VALID_CATEGORY_DISPLAY:
            raise ValueError(
                f"Invalid --category-display value: {self.category_display!r}. "
                f"Choose from: {', '.join(sorted(_VALID_CATEGORY_DISPLAY))}"
            )

    def __getattr__(self, name: str) -> Any:
        sub = _SUB_CONFIG_FIELD_MAP.get(name)
        if sub is not None:
            return getattr(object.__getattribute__(self, sub), name)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute {name!r}")

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("RenamerConfig is immutable")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("RenamerConfig is immutable")

    def __repr__(self) -> str:
        parts = [
            f"llm={self.llm!r}",
            f"heuristic={self.heuristic!r}",
            f"extraction={self.extraction!r}",
            f"output={self.output!r}",
        ]
        return f"RenamerConfig({', '.join(parts)})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, RenamerConfig):
            return False
        return bool(
            object.__getattribute__(self, "llm") == object.__getattribute__(other, "llm")
            and object.__getattribute__(self, "heuristic") == object.__getattribute__(other, "heuristic")
            and object.__getattribute__(self, "extraction") == object.__getattribute__(other, "extraction")
            and object.__getattribute__(self, "output") == object.__getattribute__(other, "output")
        )

    def __hash__(self) -> int:
        return hash((self.llm, self.heuristic, self.extraction, self.output))


# Keep __dataclass_fields__ for code that introspects it (e.g. build_config_from_flat_dict)
RenamerConfig.__dataclass_fields__ = {}  # type: ignore[attr-defined]
for _sc_cls in (LLMConfig, HeuristicConfig, ExtractionConfig, OutputConfig):
    for _f in fields(_sc_cls):
        RenamerConfig.__dataclass_fields__[_f.name] = _f  # type: ignore[attr-defined]


def build_config_from_flat_dict(data: dict[str, Any]) -> RenamerConfig:
    """Build RenamerConfig from a flat dict of option names -> values. Used by CLI and GUI to avoid duplication."""
    allowed = set(_ALL_FLAT_FIELDS)
    kwargs = {k: v for k, v in data.items() if k in allowed}
    return RenamerConfig(**kwargs)
