"""Option containers and merge helpers for LLM workflows."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, cast

from .cache import ResponseCache
from .llm_parsing import CONTEXT_128K_MAX_CHARS_SINGLE
from .options import merge_options


@dataclass(frozen=True)
class JsonCompletionOptions:
    temperature: float = 0.0
    max_retries: int = 3
    max_tokens: int | None = 1024
    json_mode: bool = False
    cache: ResponseCache | None = None
    cache_key: str | None = None


@dataclass(frozen=True)
class LlmPromptOptions:
    language: str = "de"
    temperature: float = 0.0
    lenient_json: bool = False


@dataclass(frozen=True)
class LlmCacheOptions:
    cache: ResponseCache | None = None
    cache_key_base: str | None = None


@dataclass(frozen=True)
class LlmContentLimits:
    max_content_chars: int | None = None
    max_content_tokens: int | None = None
    max_chars_single: int = CONTEXT_128K_MAX_CHARS_SINGLE


@dataclass(frozen=True)
class AnalysisGuidance:
    suggested_doc_type: str | None = None
    allowed_categories: list[str] | None = None
    suggested_categories: list[str] | None = None


@dataclass(frozen=True)
class PromptRetryOptions:
    language: str
    temperature: float
    max_tokens: int | None = 1024
    lenient: bool = False


@dataclass(frozen=True)
class PromptKeyOptions:
    key: str
    operation: str
    retry: PromptRetryOptions
    cache_options: LlmCacheOptions = field(default_factory=LlmCacheOptions)

    @property
    def language(self) -> str:
        return self.retry.language

    @property
    def temperature(self) -> float:
        return self.retry.temperature

    @property
    def max_tokens(self) -> int | None:
        return self.retry.max_tokens

    @property
    def lenient(self) -> bool:
        return self.retry.lenient

    @property
    def cache(self) -> ResponseCache | None:
        return self.cache_options.cache

    @property
    def cache_key_base(self) -> str | None:
        return self.cache_options.cache_key_base


@dataclass(frozen=True)
class AnalysisOptions:
    prompt: LlmPromptOptions = field(default_factory=LlmPromptOptions)
    limits: LlmContentLimits = field(default_factory=LlmContentLimits)
    guidance: AnalysisGuidance = field(default_factory=AnalysisGuidance)
    json_mode: bool = False
    cache_options: LlmCacheOptions = field(default_factory=LlmCacheOptions)

    @property
    def language(self) -> str:
        return self.prompt.language

    @property
    def temperature(self) -> float:
        return self.prompt.temperature

    @property
    def lenient_json(self) -> bool:
        return self.prompt.lenient_json

    @property
    def max_content_chars(self) -> int | None:
        return self.limits.max_content_chars

    @property
    def max_content_tokens(self) -> int | None:
        return self.limits.max_content_tokens

    @property
    def suggested_doc_type(self) -> str | None:
        return self.guidance.suggested_doc_type

    @property
    def allowed_categories(self) -> list[str] | None:
        return self.guidance.allowed_categories

    @property
    def suggested_categories(self) -> list[str] | None:
        return self.guidance.suggested_categories

    @property
    def cache(self) -> ResponseCache | None:
        return self.cache_options.cache

    @property
    def cache_key_base(self) -> str | None:
        return self.cache_options.cache_key_base


@dataclass(frozen=True)
class SummaryOptions:
    prompt: LlmPromptOptions = field(default_factory=LlmPromptOptions)
    limits: LlmContentLimits = field(default_factory=LlmContentLimits)
    suggested_doc_type: str | None = None
    cache_options: LlmCacheOptions = field(default_factory=LlmCacheOptions)

    @property
    def language(self) -> str:
        return self.prompt.language

    @property
    def temperature(self) -> float:
        return self.prompt.temperature

    @property
    def lenient_json(self) -> bool:
        return self.prompt.lenient_json

    @property
    def max_chars_single(self) -> int:
        return self.limits.max_chars_single

    @property
    def max_content_chars(self) -> int | None:
        return self.limits.max_content_chars

    @property
    def max_content_tokens(self) -> int | None:
        return self.limits.max_content_tokens

    @property
    def cache(self) -> ResponseCache | None:
        return self.cache_options.cache

    @property
    def cache_key_base(self) -> str | None:
        return self.cache_options.cache_key_base


@dataclass(frozen=True)
class KeywordsOptions:
    language: str = "de"
    temperature: float = 0.0
    suggested_category: str | None = None
    lenient_json: bool = False
    cache: ResponseCache | None = None
    cache_key_base: str | None = None


@dataclass(frozen=True)
class CategoryOptions:
    language: str = "de"
    temperature: float = 0.0
    suggested_categories: list[str] | None = None
    allowed_categories: list[str] | None = None
    lenient_json: bool = False
    cache: ResponseCache | None = None
    cache_key_base: str | None = None


@dataclass(frozen=True)
class FinalSummaryOptions:
    language: str = "de"
    temperature: float = 0.0
    lenient_json: bool = False
    cache: ResponseCache | None = None
    cache_key_base: str | None = None


@dataclass(frozen=True)
class SimpleFilenameOptions:
    language: str = "de"
    temperature: float = 0.0
    max_content_chars: int | None = None
    max_content_tokens: int | None = None
    cache: ResponseCache | None = None
    cache_key_base: str | None = None


def _merge_prompt_options(prompt: LlmPromptOptions, overrides: dict[str, object]) -> LlmPromptOptions:
    prompt_keys = ("language", "temperature", "lenient_json")
    prompt_overrides = {key: overrides.pop(key) for key in prompt_keys if key in overrides}
    return replace(prompt, **cast(dict[str, Any], prompt_overrides)) if prompt_overrides else prompt


def _merge_cache_options(cache_options: LlmCacheOptions, overrides: dict[str, object]) -> LlmCacheOptions:
    cache_keys = ("cache", "cache_key_base")
    cache_overrides = {key: overrides.pop(key) for key in cache_keys if key in overrides}
    return replace(cache_options, **cast(dict[str, Any], cache_overrides)) if cache_overrides else cache_options


def _merge_content_limits(limits: LlmContentLimits, overrides: dict[str, object]) -> LlmContentLimits:
    limit_keys = ("max_content_chars", "max_content_tokens", "max_chars_single")
    limit_overrides = {key: overrides.pop(key) for key in limit_keys if key in overrides}
    return replace(limits, **cast(dict[str, Any], limit_overrides)) if limit_overrides else limits


def _merge_analysis_guidance(guidance: AnalysisGuidance, overrides: dict[str, object]) -> AnalysisGuidance:
    guidance_keys = ("suggested_doc_type", "allowed_categories", "suggested_categories")
    guidance_overrides = {key: overrides.pop(key) for key in guidance_keys if key in overrides}
    return replace(guidance, **cast(dict[str, Any], guidance_overrides)) if guidance_overrides else guidance


def _merge_analysis_options(options: AnalysisOptions, overrides: dict[str, object]) -> AnalysisOptions:
    values = dict(overrides)
    merged = replace(
        options,
        prompt=_merge_prompt_options(options.prompt, values),
        limits=_merge_content_limits(options.limits, values),
        guidance=_merge_analysis_guidance(options.guidance, values),
        cache_options=_merge_cache_options(options.cache_options, values),
    )
    return merge_options(merged, values)


def _merge_summary_options(options: SummaryOptions, overrides: dict[str, object]) -> SummaryOptions:
    values = dict(overrides)
    merged = replace(
        options,
        prompt=_merge_prompt_options(options.prompt, values),
        limits=_merge_content_limits(options.limits, values),
        cache_options=_merge_cache_options(options.cache_options, values),
    )
    return merge_options(merged, values)
