"""Option containers for LLM workflows."""

from __future__ import annotations

from dataclasses import dataclass, field

from .cache import ResponseCache
from .parsing import CONTEXT_128K_MAX_CHARS_SINGLE


@dataclass(frozen=True)
class VisionCompletionOptions:
    """Optional controls for image and text completions."""

    model: str | None = None
    image_mime_type: str = "image/jpeg"
    timeout_s: float = 120.0


DEFAULT_LLM_SUMMARY = "na"
DEFAULT_LLM_CATEGORY = "unknown"
DEFAULT_LLM_KEYWORDS: list[str] = []


@dataclass(frozen=True)
class DocumentAnalysisResult:
    """Normalized structured response from document analysis."""

    summary: str = DEFAULT_LLM_SUMMARY
    keywords: tuple[str, ...] = ()
    category: str = DEFAULT_LLM_CATEGORY
    final_summary_tokens: tuple[str, ...] | None = None


def validate_llm_document_result(parsed: dict[str, object]) -> DocumentAnalysisResult:
    """Normalize a parsed document-analysis response and fill safe defaults."""
    return DocumentAnalysisResult(
        summary=_normalized_summary(parsed.get("summary")),
        keywords=_normalized_keywords(parsed.get("keywords")),
        category=_normalized_category(parsed.get("category")),
        final_summary_tokens=_normalized_final_summary_tokens(parsed.get("final_summary_tokens")),
    )


def _normalized_summary(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        return DEFAULT_LLM_SUMMARY
    summary = value.strip()
    return DEFAULT_LLM_SUMMARY if summary.lower() == "na" else summary


def _normalized_keywords(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(str(item).strip() for item in value if item and str(item).strip())


def _normalized_category(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        return DEFAULT_LLM_CATEGORY
    category = value.strip()
    return DEFAULT_LLM_CATEGORY if category.lower() in ("na", "unknown", "document", "") else category


def _normalized_final_summary_tokens(value: object) -> tuple[str, ...] | None:
    if isinstance(value, list):
        cleaned = tuple(str(item).strip() for item in value if item and str(item).strip())
        return cleaned
    if isinstance(value, str) and value.strip():
        return tuple(token.strip() for token in value.split(",") if token.strip())
    return None


@dataclass(frozen=True)
class JsonCompletionOptions:
    """Control JSON completion retries, output mode, and caching."""

    temperature: float = 0.0
    max_retries: int = 3
    max_tokens: int | None = 1024
    json_mode: bool = False
    cache: ResponseCache | None = None
    cache_key: str | None = None


@dataclass(frozen=True)
class LlmPromptOptions:
    """Shared language, temperature, and parsing options for prompts."""

    language: str = "de"
    temperature: float = 0.0
    lenient_json: bool = False


@dataclass(frozen=True)
class LlmCacheOptions:
    """Cache configuration shared by related LLM operations."""

    cache: ResponseCache | None = None
    cache_key_base: str | None = None


@dataclass(frozen=True)
class LlmContentLimits:
    """Character and token limits for content sent to an LLM."""

    max_content_chars: int | None = None
    max_content_tokens: int | None = None
    max_chars_single: int = CONTEXT_128K_MAX_CHARS_SINGLE


@dataclass(frozen=True)
class AnalysisGuidance:
    """Optional category guidance for document analysis."""

    suggested_doc_type: str | None = None
    allowed_categories: list[str] | None = None
    suggested_categories: list[str] | None = None


@dataclass(frozen=True)
class PromptRetryOptions:
    """Retry and parsing settings for one prompt sequence."""

    language: str
    temperature: float
    max_tokens: int | None = 1024
    lenient: bool = False


@dataclass(frozen=True)
class PromptKeyOptions:
    """Options for extracting one JSON field through fallback prompts."""

    key: str
    operation: str
    retry: PromptRetryOptions
    cache_options: LlmCacheOptions = field(default_factory=LlmCacheOptions)

    @property
    def language(self) -> str:
        """Return the language used for analysis prompts."""
        return self.retry.language

    @property
    def temperature(self) -> float:
        """Return the completion temperature."""
        return self.retry.temperature

    @property
    def max_tokens(self) -> int | None:
        """Return the completion token limit."""
        return self.retry.max_tokens

    @property
    def lenient(self) -> bool:
        """Return whether tolerant JSON parsing is allowed."""
        return self.retry.lenient

    @property
    def cache(self) -> ResponseCache | None:
        """Return the configured response cache."""
        return self.cache_options.cache

    @property
    def cache_key_base(self) -> str | None:
        """Return the base key shared by cached LLM responses."""
        return self.cache_options.cache_key_base


class _LlmPromptAccess:
    """Provide stateless accessors backed by nested prompt options."""

    prompt: LlmPromptOptions

    @property
    def language(self) -> str:
        """Return the language used for analysis prompts."""
        return self.prompt.language

    @property
    def temperature(self) -> float:
        """Return the completion temperature."""
        return self.prompt.temperature

    @property
    def lenient_json(self) -> bool:
        """Return whether tolerant JSON parsing is allowed."""
        return self.prompt.lenient_json


class _LlmContentLimitsAccess:
    """Provide stateless accessors backed by nested content limits."""

    limits: LlmContentLimits

    @property
    def max_content_chars(self) -> int | None:
        """Return the maximum document characters sent to the model."""
        return self.limits.max_content_chars

    @property
    def max_content_tokens(self) -> int | None:
        """Return the maximum document tokens sent to the model."""
        return self.limits.max_content_tokens


class _LlmCacheAccess:
    """Provide stateless accessors backed by nested cache options."""

    cache_options: LlmCacheOptions

    @property
    def cache(self) -> ResponseCache | None:
        """Return the configured response cache."""
        return self.cache_options.cache

    @property
    def cache_key_base(self) -> str | None:
        """Return the base key shared by cached LLM responses."""
        return self.cache_options.cache_key_base


@dataclass(frozen=True)
class AnalysisOptions(_LlmPromptAccess, _LlmContentLimitsAccess, _LlmCacheAccess):
    """Options for single-call document analysis."""

    prompt: LlmPromptOptions = field(default_factory=LlmPromptOptions)
    limits: LlmContentLimits = field(default_factory=LlmContentLimits)
    guidance: AnalysisGuidance = field(default_factory=AnalysisGuidance)
    json_mode: bool = False
    cache_options: LlmCacheOptions = field(default_factory=LlmCacheOptions)

    @property
    def suggested_doc_type(self) -> str | None:
        """Return the suggested document type for prompting."""
        return self.guidance.suggested_doc_type

    @property
    def allowed_categories(self) -> list[str] | None:
        """Return the categories permitted by the current rules."""
        return self.guidance.allowed_categories

    @property
    def suggested_categories(self) -> list[str] | None:
        """Return the categories suggested to the model."""
        return self.guidance.suggested_categories


@dataclass(frozen=True)
class SummaryOptions(_LlmPromptAccess, _LlmContentLimitsAccess, _LlmCacheAccess):
    """Options for document summarization."""

    prompt: LlmPromptOptions = field(default_factory=LlmPromptOptions)
    limits: LlmContentLimits = field(default_factory=LlmContentLimits)
    suggested_doc_type: str | None = None
    cache_options: LlmCacheOptions = field(default_factory=LlmCacheOptions)

    @property
    def max_chars_single(self) -> int:
        """Return the single-call summary character limit."""
        return self.limits.max_chars_single


@dataclass(frozen=True)
class KeywordsOptions:
    """Options for keyword extraction from a summary."""

    language: str = "de"
    temperature: float = 0.0
    suggested_category: str | None = None
    lenient_json: bool = False
    cache: ResponseCache | None = None
    cache_key_base: str | None = None


@dataclass(frozen=True)
class CategoryOptions:
    """Options for LLM category classification."""

    language: str = "de"
    temperature: float = 0.0
    suggested_categories: list[str] | None = None
    allowed_categories: list[str] | None = None
    lenient_json: bool = False
    cache: ResponseCache | None = None
    cache_key_base: str | None = None


@dataclass(frozen=True)
class FinalSummaryOptions:
    """Options for deriving compact final summary tokens."""

    language: str = "de"
    temperature: float = 0.0
    lenient_json: bool = False
    cache: ResponseCache | None = None
    cache_key_base: str | None = None


@dataclass(frozen=True)
class SimpleFilenameOptions:
    """Options for one-shot filename generation."""

    language: str = "de"
    temperature: float = 0.0
    max_content_chars: int | None = None
    max_content_tokens: int | None = None
    cache: ResponseCache | None = None
    cache_key_base: str | None = None
