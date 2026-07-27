"""Canonical grouped renamer configuration and flat-input adapter."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

_VALID_DESIRED_CASES = frozenset({"camelCase", "kebabCase", "snakeCase"})
_VALID_DATE_LOCALES = frozenset({"dmy", "mdy"})
_VALID_CATEGORY_DISPLAY = frozenset({"specific", "with_parent", "parent_only"})


@dataclass(frozen=True)
class LLMBackendConfig:
    """Hold the LLM endpoint, model, timeout, and hardware-preset settings."""

    llm_base_url: str | None = None
    llm_model: str | None = None
    llm_timeout_s: float | None = None
    llm_preset: str | None = None


@dataclass(frozen=True)
class LLMRuntimeConfig:
    """Hold LLM enablement, transport, call-mode, and response-parsing flags."""

    require_https: bool = False
    use_llm: bool = True
    lenient_llm_json: bool = False
    simple_naming_mode: bool = False
    use_single_llm_call: bool = True
    llm_use_chat_api: bool = True
    llm_json_mode: bool = True


@dataclass(frozen=True)
class LLMVisionConfig:
    """Hold the opt-in vision fallback configuration."""

    use_vision_fallback: bool = False
    vision_fallback_min_text_len: int = 50
    vision_model: str | None = None
    vision_first: bool = False


@dataclass(frozen=True)
class LLMContentConfig:
    """Hold LLM content bounds and response-cache settings."""

    max_context_chars: int | None = None
    max_content_chars: int | None = None
    max_content_tokens: int | None = None
    use_cache: bool = True
    cache_dir: str | Path | None = None


@dataclass(frozen=True)
class LLMConfig:
    """Group LLM connection, runtime, vision, and content/cache settings."""

    backend: LLMBackendConfig = LLMBackendConfig()
    runtime: LLMRuntimeConfig = LLMRuntimeConfig()
    vision: LLMVisionConfig = LLMVisionConfig()
    content: LLMContentConfig = LLMContentConfig()


@dataclass(frozen=True)
class HeuristicScoreConfig:
    """Hold scoring thresholds used to decide heuristic confidence."""

    min_heuristic_score_gap: float = 0.0
    min_heuristic_score: float = 0.0
    title_weight_region: int = 2000
    title_weight_factor: float = 1.5
    max_score_per_category: float | None = None
    heuristic_score_weight: float = 0.15


@dataclass(frozen=True)
class HeuristicCategoryConfig:
    """Hold policy for reconciling heuristic and LLM categories."""

    use_keyword_overlap_for_category: bool = True
    category_display: str = "specific"
    use_constrained_llm_category: bool = True
    prefer_llm_category: bool = True


@dataclass(frozen=True)
class HeuristicSkipConfig:
    """Hold thresholds that permit skipping an LLM category request."""

    skip_llm_category_if_heuristic_score_ge: float | None = None
    skip_llm_category_if_heuristic_gap_ge: float | None = None
    heuristic_suggestions_top_n: int = 5
    heuristic_override_min_score: float | None = None
    heuristic_override_min_gap: float | None = None


@dataclass(frozen=True)
class HeuristicWindowConfig:
    """Hold bounded text windows for heuristic processing."""

    heuristic_leading_chars: int = 0
    heuristic_long_doc_chars_threshold: int = 40_000
    heuristic_long_doc_leading_chars: int = 12_000


@dataclass(frozen=True)
class HeuristicConfig:
    """Heuristic scoring and category resolution configuration."""

    scoring: HeuristicScoreConfig = HeuristicScoreConfig()
    category: HeuristicCategoryConfig = HeuristicCategoryConfig()
    skip: HeuristicSkipConfig = HeuristicSkipConfig()
    window: HeuristicWindowConfig = HeuristicWindowConfig()


@dataclass(frozen=True)
class ExtractionConfig:
    """PDF text extraction configuration."""

    max_pages_for_extraction: int = 0
    max_tokens_for_extraction: int | None = None
    use_ocr: bool = False
    use_structured_fields: bool = True
    use_pdf_metadata_for_date: bool = True


@dataclass(frozen=True)
class OutputNamingConfig:
    """Hold filename-shaping options shared by all output modes."""

    language: str = "de"
    desired_case: str = "kebabCase"
    project: str = ""
    version: str = ""
    date_locale: str = "dmy"
    date_prefer_leading_chars: int = 8000
    max_filename_chars: int | None = None


@dataclass(frozen=True)
class OutputTemplateConfig:
    """Hold filename templates, category overrides, and timestamp-fallback policy."""

    override_category_map: dict[str, str] | None = None
    filename_template: str | None = None
    use_timestamp_fallback: bool = True
    timestamp_fallback_segment: str = "document"


@dataclass(frozen=True)
class OutputPathConfig:
    """Hold paths for output artifacts created during a run."""

    backup_dir: str | Path | None = None
    rename_log_path: str | Path | None = None
    export_metadata_path: str | Path | None = None
    summary_json_path: str | Path | None = None
    plan_file_path: str | Path | None = None
    rules_file: str | Path | None = None


@dataclass(frozen=True)
class OutputTraversalConfig:
    """Hold discovery and traversal choices that control file scope."""

    workers: int = 1
    recursive: bool = False
    max_depth: int = 0
    include_patterns: list[str] | None = None
    exclude_patterns: list[str] | None = None


@dataclass(frozen=True)
class OutputModeConfig:
    """Hold execution and interaction flags, including dry-run and PDF-metadata writes."""

    dry_run: bool = False
    skip_if_already_named: bool = False
    interactive: bool = False
    manual_mode: bool = False
    write_pdf_metadata: bool = False


@dataclass(frozen=True)
class OutputHookConfig:
    """Hold the post-rename hook and cooperative cancellation state."""

    post_rename_hook: str | None = None
    stop_event: object | None = None


@dataclass(frozen=True)
class OutputProgressConfig:
    """Hold progress-reporting settings for interactive and batch runs."""

    progress: bool = False
    quiet_progress: bool = False
    explain: bool = False


@dataclass(frozen=True)
class OutputConfig:
    """Output, naming, and file-handling configuration."""

    naming: OutputNamingConfig = OutputNamingConfig()
    template: OutputTemplateConfig = OutputTemplateConfig()
    paths: OutputPathConfig = OutputPathConfig()
    traversal: OutputTraversalConfig = OutputTraversalConfig()
    mode: OutputModeConfig = OutputModeConfig()
    hooks: OutputHookConfig = OutputHookConfig()
    progress_options: OutputProgressConfig = OutputProgressConfig()


def _validate_choice(value: str, valid_values: frozenset[str], option_name: str) -> None:
    """Reject unsupported enum values before grouped configuration is constructed."""
    if value not in valid_values:
        raise ValueError(f"Invalid {option_name} value: {value!r}. Choose from: {', '.join(sorted(valid_values))}")


def _validate_renamer_config(config: RenamerConfig) -> None:
    """Validate cross-section configuration invariants before rename work starts."""
    _validate_choice(config.output.naming.desired_case, _VALID_DESIRED_CASES, "--case")
    _validate_choice((config.output.naming.date_locale or "dmy").strip().lower(), _VALID_DATE_LOCALES, "--date-format")
    _validate_choice(
        (config.heuristic.category.category_display or "specific").strip().lower(),
        _VALID_CATEGORY_DISPLAY,
        "--category-display",
    )


@dataclass(frozen=True)
class RenamerConfig:
    """Canonical configuration assembled from explicit grouped dataclasses."""

    llm: LLMConfig = LLMConfig()
    heuristic: HeuristicConfig = HeuristicConfig()
    extraction: ExtractionConfig = ExtractionConfig()
    output: OutputConfig = OutputConfig()

    def __post_init__(self) -> None:
        """Validate cross-field invariants after dataclass initialization."""
        _validate_renamer_config(self)


def build_config_from_flat_dict(data: dict[str, Any]) -> RenamerConfig:
    """Translate supported flat CLI, environment, JSON, or YAML values into grouped configuration."""

    def group_values(cls: type[Any], values: dict[str, Any]) -> dict[str, Any]:
        """Select fields declared by cls for use as dataclass constructor arguments."""
        return {field.name: values[field.name] for field in fields(cls) if field.name in values}

    llm_connection = LLMBackendConfig(**group_values(LLMBackendConfig, data))
    llm_runtime = LLMRuntimeConfig(**group_values(LLMRuntimeConfig, data))
    llm_vision = LLMVisionConfig(**group_values(LLMVisionConfig, data))
    llm_content = LLMContentConfig(**group_values(LLMContentConfig, data))
    heuristic_scoring = HeuristicScoreConfig(**group_values(HeuristicScoreConfig, data))
    heuristic_category = HeuristicCategoryConfig(**group_values(HeuristicCategoryConfig, data))
    heuristic_skip = HeuristicSkipConfig(**group_values(HeuristicSkipConfig, data))
    heuristic_window = HeuristicWindowConfig(**group_values(HeuristicWindowConfig, data))
    output_naming = OutputNamingConfig(**group_values(OutputNamingConfig, data))
    output_template = OutputTemplateConfig(**group_values(OutputTemplateConfig, data))
    output_paths = OutputPathConfig(**group_values(OutputPathConfig, data))
    output_traversal = OutputTraversalConfig(**group_values(OutputTraversalConfig, data))
    output_mode = OutputModeConfig(**group_values(OutputModeConfig, data))
    output_hooks = OutputHookConfig(**group_values(OutputHookConfig, data))
    output_progress = OutputProgressConfig(**group_values(OutputProgressConfig, data))
    return RenamerConfig(
        llm=LLMConfig(
            backend=llm_connection,
            runtime=llm_runtime,
            vision=llm_vision,
            content=llm_content,
        ),
        heuristic=HeuristicConfig(
            scoring=heuristic_scoring,
            category=heuristic_category,
            skip=heuristic_skip,
            window=heuristic_window,
        ),
        extraction=ExtractionConfig(**group_values(ExtractionConfig, data)),
        output=OutputConfig(
            naming=output_naming,
            template=output_template,
            paths=output_paths,
            traversal=output_traversal,
            mode=output_mode,
            hooks=output_hooks,
            progress_options=output_progress,
        ),
    )
