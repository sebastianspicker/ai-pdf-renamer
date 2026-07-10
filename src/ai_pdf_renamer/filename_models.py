"""Small request/result containers for filename generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, cast

from .cache import ResponseCache
from .config import RenamerConfig
from .heuristics import HeuristicScorer
from .llm_backend import LLMClient
from .rules import ProcessingRules
from .text_utils import Stopwords


@dataclass(frozen=True)
class _FilenameCacheContext:
    response_cache: ResponseCache | None = None
    cache_key_base: str | None = None


@dataclass(frozen=True)
class _HeuristicCategorySignal:
    heuristic_text: str
    cat_heur: str
    heuristic_score: float
    heuristic_gap: float
    suggested_doc_type: str | None = None


@dataclass(frozen=True)
class _CategoryResolutionInput:
    heuristic: _HeuristicCategorySignal
    summary: str
    keywords: list[str]
    rules: ProcessingRules | None = None
    precomputed_llm_category: str | None = None
    cache_context: _FilenameCacheContext = field(default_factory=_FilenameCacheContext)

    @property
    def heuristic_text(self) -> str:
        return self.heuristic.heuristic_text

    @property
    def cat_heur(self) -> str:
        return self.heuristic.cat_heur

    @property
    def heuristic_score(self) -> float:
        return self.heuristic.heuristic_score

    @property
    def heuristic_gap(self) -> float:
        return self.heuristic.heuristic_gap

    @property
    def response_cache(self) -> ResponseCache | None:
        return self.cache_context.response_cache

    @property
    def cache_key_base(self) -> str | None:
        return self.cache_context.cache_key_base


@dataclass(frozen=True)
class _LlmSummaryInput:
    pdf_content: str
    heuristic: _HeuristicCategorySignal
    override_category: str | None
    rules: ProcessingRules | None = None
    cache_context: _FilenameCacheContext = field(default_factory=_FilenameCacheContext)

    @property
    def heuristic_text(self) -> str:
        return self.heuristic.heuristic_text

    @property
    def cat_heur(self) -> str:
        return self.heuristic.cat_heur

    @property
    def suggested_doc_type(self) -> str | None:
        return self.heuristic.suggested_doc_type

    @property
    def response_cache(self) -> ResponseCache | None:
        return self.cache_context.response_cache

    @property
    def cache_key_base(self) -> str | None:
        return self.cache_context.cache_key_base


@dataclass(frozen=True)
class _MetadataResolutionInput:
    pdf_content: str
    override_category: str | None
    rules: ProcessingRules | None = None
    response_cache: ResponseCache | None = None
    cache_key_base: str | None = None


@dataclass(frozen=True)
class FilenameGenerationDependencies:
    llm_client: LLMClient | None = None
    heuristic_scorer: HeuristicScorer | None = None
    stopwords: Stopwords | None = None


@dataclass(frozen=True)
class FilenameGenerationContext:
    override_category: str | None = None
    today: date | None = None
    pdf_metadata: dict[str, object] | None = None
    rules: ProcessingRules | None = None
    source_path: Path | None = None


_FILENAME_DEPENDENCY_KEYS = frozenset({"llm_client", "heuristic_scorer", "stopwords"})
_FILENAME_CONTEXT_KEYS = frozenset({"override_category", "today", "pdf_metadata", "rules", "source_path"})


def _pop_filename_request_values(legacy: dict[str, object], keys: frozenset[str]) -> dict[str, object]:
    return {key: legacy.pop(key) for key in tuple(legacy) if key in keys}


def _split_filename_request_legacy(legacy: dict[str, object]) -> tuple[dict[str, object], dict[str, object]]:
    unknown = sorted(set(legacy) - _FILENAME_DEPENDENCY_KEYS - _FILENAME_CONTEXT_KEYS)
    if unknown:
        raise TypeError(f"Unexpected filename request option(s): {', '.join(unknown)}")
    return (
        _pop_filename_request_values(legacy, _FILENAME_DEPENDENCY_KEYS),
        _pop_filename_request_values(legacy, _FILENAME_CONTEXT_KEYS),
    )


def _coerce_filename_dependencies(
    dependencies: FilenameGenerationDependencies | None,
    values: dict[str, object],
) -> FilenameGenerationDependencies:
    if dependencies is not None:
        if values:
            raise TypeError("Pass either dependencies or dependency keyword options, not both")
        return dependencies
    return FilenameGenerationDependencies(**cast(dict[str, Any], values))


def _coerce_filename_context(
    context: FilenameGenerationContext | None,
    values: dict[str, object],
) -> FilenameGenerationContext:
    if context is not None:
        if values:
            raise TypeError("Pass either context or context keyword options, not both")
        return context
    return FilenameGenerationContext(**cast(dict[str, Any], values))


@dataclass(frozen=True, init=False)
class FilenameGenerationRequest:
    config: RenamerConfig
    dependencies: FilenameGenerationDependencies
    context: FilenameGenerationContext

    def __init__(
        self,
        config: RenamerConfig,
        dependencies: FilenameGenerationDependencies | None = None,
        context: FilenameGenerationContext | None = None,
        **legacy: object,
    ) -> None:
        dependency_values, context_values = _split_filename_request_legacy(legacy)
        object.__setattr__(self, "config", config)
        object.__setattr__(self, "dependencies", _coerce_filename_dependencies(dependencies, dependency_values))
        object.__setattr__(self, "context", _coerce_filename_context(context, context_values))

    @property
    def llm_client(self) -> LLMClient | None:
        return self.dependencies.llm_client

    @property
    def heuristic_scorer(self) -> HeuristicScorer | None:
        return self.dependencies.heuristic_scorer

    @property
    def stopwords(self) -> Stopwords | None:
        return self.dependencies.stopwords

    @property
    def override_category(self) -> str | None:
        return self.context.override_category

    @property
    def today(self) -> date | None:
        return self.context.today

    @property
    def pdf_metadata(self) -> dict[str, object] | None:
        return self.context.pdf_metadata

    @property
    def rules(self) -> ProcessingRules | None:
        return self.context.rules

    @property
    def source_path(self) -> Path | None:
        return self.context.source_path


@dataclass(frozen=True)
class _FilenameTemplateTokens:
    category_for_filename: str
    category_clean: list[str]
    keyword_clean: list[str]
    summary_clean: list[str]
    structured_fields: dict[str, str] | None = None


@dataclass(frozen=True)
class _FilenameTemplateInput:
    filename: str
    date_str: str
    project: str
    version: str
    tokens: _FilenameTemplateTokens

    @property
    def category_for_filename(self) -> str:
        return self.tokens.category_for_filename

    @property
    def category_clean(self) -> list[str]:
        return self.tokens.category_clean

    @property
    def keyword_clean(self) -> list[str]:
        return self.tokens.keyword_clean

    @property
    def summary_clean(self) -> list[str]:
        return self.tokens.summary_clean

    @property
    def structured_fields(self) -> dict[str, str] | None:
        return self.tokens.structured_fields


@dataclass(frozen=True)
class _CategoryContext:
    category: str
    category_for_filename: str
    category_source: str
    heuristic: _HeuristicCategorySignal
    skip_llm_by_rule: bool

    @property
    def cat_heur(self) -> str:
        return self.heuristic.cat_heur

    @property
    def heuristic_text(self) -> str:
        return self.heuristic.heuristic_text

    @property
    def heuristic_score(self) -> float:
        return self.heuristic.heuristic_score

    @property
    def heuristic_gap(self) -> float:
        return self.heuristic.heuristic_gap

    @property
    def suggested_doc_type(self) -> str | None:
        return self.heuristic.suggested_doc_type


@dataclass(frozen=True)
class _LlmResult:
    summary: str
    raw_keywords: list[str] | tuple[str, ...]
    precomputed_category: str | None = None
    precomputed_summary_tokens: list[str] | tuple[str, ...] | None = None


@dataclass(frozen=True)
class _FilenameDependencies:
    llm_client: LLMClient
    heuristic_scorer: HeuristicScorer
    stopwords: Stopwords
    response_cache: ResponseCache | None
    cache_key_base: str | None


@dataclass(frozen=True)
class _FilenameMetadataParts:
    category_for_filename: str
    category_clean: list[str]
    keyword_clean: list[str]
    summary_clean: list[str]
    metadata: dict[str, object]
