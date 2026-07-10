"""
Renamer configuration: dataclass and build from flat dict.

Used by CLI and GUI; re-exported from renamer for backward compatibility.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, ClassVar

_VALID_DESIRED_CASES = frozenset({"camelCase", "kebabCase", "snakeCase"})
_VALID_DATE_LOCALES = frozenset({"dmy", "mdy"})
_VALID_CATEGORY_DISPLAY = frozenset({"specific", "with_parent", "parent_only"})


@dataclass(frozen=True)
class LLMBackendConfig:
    llm_backend: str = "http"
    llm_base_url: str | None = None
    llm_model: str | None = None
    llm_timeout_s: float | None = None
    llm_model_path: str | None = None
    llm_preset: str | None = None


@dataclass(frozen=True)
class LLMRuntimeConfig:
    require_https: bool = False
    use_llm: bool = True
    lenient_llm_json: bool = False
    simple_naming_mode: bool = False
    use_single_llm_call: bool = True
    llm_use_chat_api: bool = True
    llm_json_mode: bool = True


@dataclass(frozen=True)
class LLMVisionConfig:
    use_vision_fallback: bool = False
    vision_fallback_min_text_len: int = 50
    vision_model: str | None = None
    vision_first: bool = False


@dataclass(frozen=True)
class LLMContentConfig:
    max_context_chars: int | None = None
    max_content_chars: int | None = None
    max_content_tokens: int | None = None
    use_cache: bool = True
    cache_dir: str | Path | None = None


_LLM_BACKEND_CONFIG_FIELDS = frozenset(f.name for f in fields(LLMBackendConfig))
_LLM_RUNTIME_CONFIG_FIELDS = frozenset(f.name for f in fields(LLMRuntimeConfig))
_LLM_VISION_CONFIG_FIELDS = frozenset(f.name for f in fields(LLMVisionConfig))
_LLM_CONTENT_CONFIG_FIELDS = frozenset(f.name for f in fields(LLMContentConfig))
_LLM_CONFIG_GROUPS = (
    ("backend", LLMBackendConfig, _LLM_BACKEND_CONFIG_FIELDS),
    ("runtime", LLMRuntimeConfig, _LLM_RUNTIME_CONFIG_FIELDS),
    ("vision", LLMVisionConfig, _LLM_VISION_CONFIG_FIELDS),
    ("content", LLMContentConfig, _LLM_CONTENT_CONFIG_FIELDS),
)


def _pop_config_values(legacy: dict[str, Any], keys: frozenset[str]) -> dict[str, Any]:
    return {key: legacy.pop(key) for key in tuple(legacy) if key in keys}


def _coerce_group_config(cls: type, existing: Any, values: dict[str, Any], option_name: str) -> Any:
    if existing is not None:
        if values:
            raise TypeError(f"Pass either {option_name} or its flat keyword options, not both")
        return existing
    return cls(**values)


_ConfigGroupSpec = tuple[str, type, frozenset[str]]


def _flat_fields_for_groups(group_specs: tuple[_ConfigGroupSpec, ...]) -> frozenset[str]:
    fields_set: frozenset[str] = frozenset()
    for _, _, group_fields in group_specs:
        fields_set |= group_fields
    return fields_set


def _init_grouped_config(
    instance: object,
    config_name: str,
    group_specs: tuple[_ConfigGroupSpec, ...],
    kwargs: dict[str, Any],
) -> None:
    group_names = {group_name for group_name, _, _ in group_specs}
    flat_fields = _flat_fields_for_groups(group_specs)
    unknown = sorted(set(kwargs) - group_names - flat_fields)
    if unknown:
        raise TypeError(f"{config_name}() got unexpected keyword argument(s): {', '.join(unknown)}")
    groups = {group_name: kwargs.pop(group_name, None) for group_name in group_names}
    for group_name, group_cls, field_names in group_specs:
        values = _pop_config_values(kwargs, field_names)
        group_config = _coerce_group_config(group_cls, groups[group_name], values, group_name)
        object.__setattr__(instance, group_name, group_config)


def _get_grouped_config_attr(instance: object, name: str) -> Any:
    for group_name, _, field_names in object.__getattribute__(instance, "__config_groups__"):
        if name in field_names:
            return getattr(object.__getattribute__(instance, group_name), name)
    raise AttributeError(f"'{type(instance).__name__}' object has no attribute {name!r}")


@dataclass(frozen=True, init=False)
class LLMConfig:
    """LLM backend and model configuration."""

    __config_groups__: ClassVar[tuple[_ConfigGroupSpec, ...]] = _LLM_CONFIG_GROUPS
    __flat_fields__: ClassVar[frozenset[str]] = _flat_fields_for_groups(_LLM_CONFIG_GROUPS)

    backend: LLMBackendConfig
    runtime: LLMRuntimeConfig
    vision: LLMVisionConfig
    content: LLMContentConfig

    def __init__(self, **kwargs: Any) -> None:
        _init_grouped_config(self, "LLMConfig", self.__config_groups__, kwargs)

    def __getattr__(self, name: str) -> Any:
        return _get_grouped_config_attr(self, name)


@dataclass(frozen=True)
class HeuristicScoreConfig:
    min_heuristic_score_gap: float = 0.0
    min_heuristic_score: float = 0.0
    title_weight_region: int = 2000
    title_weight_factor: float = 1.5
    max_score_per_category: float | None = None
    heuristic_score_weight: float = 0.15


@dataclass(frozen=True)
class HeuristicCategoryConfig:
    use_keyword_overlap_for_category: bool = True
    use_embeddings_for_conflict: bool = False
    category_display: str = "specific"
    use_constrained_llm_category: bool = True
    prefer_llm_category: bool = True


@dataclass(frozen=True)
class HeuristicSkipConfig:
    skip_llm_category_if_heuristic_score_ge: float | None = None
    skip_llm_category_if_heuristic_gap_ge: float | None = None
    heuristic_suggestions_top_n: int = 5
    heuristic_override_min_score: float | None = None
    heuristic_override_min_gap: float | None = None


@dataclass(frozen=True)
class HeuristicWindowConfig:
    heuristic_leading_chars: int = 0
    heuristic_long_doc_chars_threshold: int = 40_000
    heuristic_long_doc_leading_chars: int = 12_000


_HEURISTIC_SCORE_CONFIG_FIELDS = frozenset(f.name for f in fields(HeuristicScoreConfig))
_HEURISTIC_CATEGORY_CONFIG_FIELDS = frozenset(f.name for f in fields(HeuristicCategoryConfig))
_HEURISTIC_SKIP_CONFIG_FIELDS = frozenset(f.name for f in fields(HeuristicSkipConfig))
_HEURISTIC_WINDOW_CONFIG_FIELDS = frozenset(f.name for f in fields(HeuristicWindowConfig))
_HEURISTIC_CONFIG_GROUPS = (
    ("scoring", HeuristicScoreConfig, _HEURISTIC_SCORE_CONFIG_FIELDS),
    ("category", HeuristicCategoryConfig, _HEURISTIC_CATEGORY_CONFIG_FIELDS),
    ("skip", HeuristicSkipConfig, _HEURISTIC_SKIP_CONFIG_FIELDS),
    ("window", HeuristicWindowConfig, _HEURISTIC_WINDOW_CONFIG_FIELDS),
)


@dataclass(frozen=True, init=False)
class HeuristicConfig:
    """Heuristic scoring and category resolution configuration."""

    __config_groups__: ClassVar[tuple[_ConfigGroupSpec, ...]] = _HEURISTIC_CONFIG_GROUPS
    __flat_fields__: ClassVar[frozenset[str]] = _flat_fields_for_groups(_HEURISTIC_CONFIG_GROUPS)

    scoring: HeuristicScoreConfig
    category: HeuristicCategoryConfig
    skip: HeuristicSkipConfig
    window: HeuristicWindowConfig

    def __init__(self, **kwargs: Any) -> None:
        _init_grouped_config(self, "HeuristicConfig", self.__config_groups__, kwargs)

    def __getattr__(self, name: str) -> Any:
        return _get_grouped_config_attr(self, name)


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
    language: str = "de"
    desired_case: str = "kebabCase"
    project: str = ""
    version: str = ""
    date_locale: str = "dmy"
    date_prefer_leading_chars: int = 8000
    max_filename_chars: int | None = None


@dataclass(frozen=True)
class OutputTemplateConfig:
    override_category_map: dict[str, str] | None = None
    filename_template: str | None = None
    use_timestamp_fallback: bool = True
    timestamp_fallback_segment: str = "document"


@dataclass(frozen=True)
class OutputPathConfig:
    backup_dir: str | Path | None = None
    rename_log_path: str | Path | None = None
    export_metadata_path: str | Path | None = None
    summary_json_path: str | Path | None = None
    plan_file_path: str | Path | None = None
    rules_file: str | Path | None = None


@dataclass(frozen=True)
class OutputTraversalConfig:
    workers: int = 1
    recursive: bool = False
    max_depth: int = 0
    include_patterns: list[str] | None = None
    exclude_patterns: list[str] | None = None


@dataclass(frozen=True)
class OutputModeConfig:
    dry_run: bool = False
    skip_if_already_named: bool = False
    interactive: bool = False
    manual_mode: bool = False
    write_pdf_metadata: bool = False


@dataclass(frozen=True)
class OutputHookConfig:
    post_rename_hook: str | None = None
    stop_event: object | None = None


@dataclass(frozen=True)
class OutputProgressConfig:
    progress: bool = False
    quiet_progress: bool = False
    explain: bool = False


_OUTPUT_NAMING_CONFIG_FIELDS = frozenset(f.name for f in fields(OutputNamingConfig))
_OUTPUT_TEMPLATE_CONFIG_FIELDS = frozenset(f.name for f in fields(OutputTemplateConfig))
_OUTPUT_PATH_CONFIG_FIELDS = frozenset(f.name for f in fields(OutputPathConfig))
_OUTPUT_TRAVERSAL_CONFIG_FIELDS = frozenset(f.name for f in fields(OutputTraversalConfig))
_OUTPUT_MODE_CONFIG_FIELDS = frozenset(f.name for f in fields(OutputModeConfig))
_OUTPUT_HOOK_CONFIG_FIELDS = frozenset(f.name for f in fields(OutputHookConfig))
_OUTPUT_PROGRESS_CONFIG_FIELDS = frozenset(f.name for f in fields(OutputProgressConfig))
_OUTPUT_CONFIG_GROUPS = (
    ("naming", OutputNamingConfig, _OUTPUT_NAMING_CONFIG_FIELDS),
    ("template", OutputTemplateConfig, _OUTPUT_TEMPLATE_CONFIG_FIELDS),
    ("paths", OutputPathConfig, _OUTPUT_PATH_CONFIG_FIELDS),
    ("traversal", OutputTraversalConfig, _OUTPUT_TRAVERSAL_CONFIG_FIELDS),
    ("mode", OutputModeConfig, _OUTPUT_MODE_CONFIG_FIELDS),
    ("hooks", OutputHookConfig, _OUTPUT_HOOK_CONFIG_FIELDS),
    ("progress_options", OutputProgressConfig, _OUTPUT_PROGRESS_CONFIG_FIELDS),
)


@dataclass(frozen=True, init=False)
class OutputConfig:
    """Output, naming, and file-handling configuration."""

    __config_groups__: ClassVar[tuple[_ConfigGroupSpec, ...]] = _OUTPUT_CONFIG_GROUPS
    __flat_fields__: ClassVar[frozenset[str]] = _flat_fields_for_groups(_OUTPUT_CONFIG_GROUPS)

    naming: OutputNamingConfig
    template: OutputTemplateConfig
    paths: OutputPathConfig
    traversal: OutputTraversalConfig
    mode: OutputModeConfig
    hooks: OutputHookConfig
    progress_options: OutputProgressConfig

    def __init__(self, **kwargs: Any) -> None:
        _init_grouped_config(self, "OutputConfig", self.__config_groups__, kwargs)

    def __getattr__(self, name: str) -> Any:
        return _get_grouped_config_attr(self, name)


def _config_flat_fields(cls: type) -> frozenset[str]:
    flat_fields = getattr(cls, "__flat_fields__", None)
    if isinstance(flat_fields, frozenset):
        return flat_fields
    return frozenset(f.name for f in fields(cls))


# Build field -> sub-config mapping at import time
_LLM_FIELDS = _config_flat_fields(LLMConfig)
_HEURISTIC_FIELDS = _config_flat_fields(HeuristicConfig)
_EXTRACTION_FIELDS = frozenset(f.name for f in fields(ExtractionConfig))
_OUTPUT_FIELDS = _config_flat_fields(OutputConfig)

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


def _reject_unknown_flat_kwargs(flat_kwargs: dict[str, Any]) -> None:
    unknown = flat_kwargs.keys() - _ALL_FLAT_FIELDS
    if unknown:
        raise TypeError(f"RenamerConfig() got unexpected keyword argument(s): {', '.join(sorted(unknown))}")


def _flat_kwargs_for(fields_set: frozenset[str], flat_kwargs: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in flat_kwargs.items() if key in fields_set}


def _merge_config(cls: type, existing: Any, overrides: dict[str, Any]) -> Any:
    if existing is not None and overrides:
        base = {field_name: getattr(existing, field_name) for field_name in _config_flat_fields(cls)}
        base.update(overrides)
        return cls(**base)
    if existing is not None:
        return existing
    return cls(**overrides)


def _validate_choice(value: str, valid_values: frozenset[str], option_name: str) -> None:
    if value not in valid_values:
        raise ValueError(f"Invalid {option_name} value: {value!r}. Choose from: {', '.join(sorted(valid_values))}")


def _validate_renamer_config(config: RenamerConfig) -> None:
    _validate_choice(config.desired_case, _VALID_DESIRED_CASES, "--case")
    _validate_choice((config.date_locale or "dmy").strip().lower(), _VALID_DATE_LOCALES, "--date-format")
    _validate_choice(
        (config.category_display or "specific").strip().lower(),
        _VALID_CATEGORY_DISPLAY,
        "--category-display",
    )


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
        _reject_unknown_flat_kwargs(flat_kwargs)

        # Split flat kwargs into sub-config buckets
        llm_kw = _flat_kwargs_for(_LLM_FIELDS, flat_kwargs)
        heur_kw = _flat_kwargs_for(_HEURISTIC_FIELDS, flat_kwargs)
        ext_kw = _flat_kwargs_for(_EXTRACTION_FIELDS, flat_kwargs)
        out_kw = _flat_kwargs_for(_OUTPUT_FIELDS, flat_kwargs)

        object.__setattr__(self, "llm", _merge_config(LLMConfig, llm, llm_kw))
        object.__setattr__(self, "heuristic", _merge_config(HeuristicConfig, heuristic, heur_kw))
        object.__setattr__(self, "extraction", _merge_config(ExtractionConfig, extraction, ext_kw))
        object.__setattr__(self, "output", _merge_config(OutputConfig, output, out_kw))

        # Validation
        _validate_renamer_config(self)

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
