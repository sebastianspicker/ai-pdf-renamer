"""Small request/result containers for filename generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from .cache import ResponseCache
from .config import RenamerConfig
from .heuristics import HeuristicScorer
from .llm_backend import LLMClient
from .rules import ProcessingRules
from .text_utils import Stopwords


@dataclass(frozen=True)
class _FilenameCacheContext:
    """Cache and base key shared by LLM calls for one source file."""

    response_cache: ResponseCache | None = None
    cache_key_base: str | None = None


@dataclass(frozen=True)
class _HeuristicCategorySignal:
    """Heuristic category candidate and confidence signals."""

    heuristic_text: str
    cat_heur: str
    heuristic_score: float
    heuristic_gap: float
    suggested_doc_type: str | None = None


class _HeuristicCacheInput:
    """Shared heuristic and cache accessors for filename metadata inputs."""

    heuristic: _HeuristicCategorySignal
    cache_context: _FilenameCacheContext

    @property
    def heuristic_text(self) -> str:
        """Return text prepared for heuristic classification."""
        return self.heuristic.heuristic_text

    @property
    def cat_heur(self) -> str:
        """Return the heuristic category candidate."""
        return self.heuristic.cat_heur

    @property
    def response_cache(self) -> ResponseCache | None:
        """Return the optional response cache for this request."""
        return self.cache_context.response_cache

    @property
    def cache_key_base(self) -> str | None:
        """Return the base key shared by cached LLM responses."""
        return self.cache_context.cache_key_base


@dataclass(frozen=True)
class _CategoryResolutionInput(_HeuristicCacheInput):
    """Inputs required to reconcile heuristic and LLM categories."""

    heuristic: _HeuristicCategorySignal
    summary: str
    keywords: list[str]
    rules: ProcessingRules | None = None
    precomputed_llm_category: str | None = None
    cache_context: _FilenameCacheContext = field(default_factory=_FilenameCacheContext)

    @property
    def heuristic_score(self) -> float:
        """Return the heuristic confidence score."""
        return self.heuristic.heuristic_score

    @property
    def heuristic_gap(self) -> float:
        """Return the score gap to the next category."""
        return self.heuristic.heuristic_gap


@dataclass(frozen=True)
class _LlmSummaryInput(_HeuristicCacheInput):
    """Inputs required to obtain LLM-derived filename metadata."""

    pdf_content: str
    heuristic: _HeuristicCategorySignal
    override_category: str | None
    rules: ProcessingRules | None = None
    cache_context: _FilenameCacheContext = field(default_factory=_FilenameCacheContext)

    @property
    def suggested_doc_type(self) -> str | None:
        """Return the suggested document type for prompting."""
        return self.heuristic.suggested_doc_type


@dataclass(frozen=True)
class _MetadataResolutionInput:
    """Document and optional overrides used to resolve filename metadata."""

    pdf_content: str
    override_category: str | None
    rules: ProcessingRules | None = None
    response_cache: ResponseCache | None = None
    cache_key_base: str | None = None


@dataclass(frozen=True)
class FilenameGenerationDependencies:
    """Optional caller-supplied filename-generation dependencies."""

    llm_client: LLMClient | None = None
    heuristic_scorer: HeuristicScorer | None = None
    stopwords: Stopwords | None = None


@dataclass(frozen=True)
class FilenameGenerationContext:
    """Per-document context for filename generation."""

    override_category: str | None = None
    today: date | None = None
    pdf_metadata: dict[str, object] | None = None
    rules: ProcessingRules | None = None
    source_path: Path | None = None


@dataclass(frozen=True)
class FilenameGenerationRequest:
    """Configuration, dependencies, and context for one filename request."""

    config: RenamerConfig
    dependencies: FilenameGenerationDependencies = FilenameGenerationDependencies()
    context: FilenameGenerationContext = FilenameGenerationContext()

    @property
    def llm_client(self) -> LLMClient | None:
        """Return the optional caller-supplied LLM client."""
        return self.dependencies.llm_client

    @property
    def heuristic_scorer(self) -> HeuristicScorer | None:
        """Return the optional caller-supplied heuristic scorer."""
        return self.dependencies.heuristic_scorer

    @property
    def stopwords(self) -> Stopwords | None:
        """Return the optional caller-supplied stopword set."""
        return self.dependencies.stopwords

    @property
    def override_category(self) -> str | None:
        """Return the optional category override."""
        return self.context.override_category

    @property
    def today(self) -> date | None:
        """Return the optional date used for deterministic extraction."""
        return self.context.today

    @property
    def pdf_metadata(self) -> dict[str, object] | None:
        """Return the optional PDF metadata fallback."""
        return self.context.pdf_metadata

    @property
    def rules(self) -> ProcessingRules | None:
        """Return the optional processing rules for this request."""
        return self.context.rules

    @property
    def source_path(self) -> Path | None:
        """Return the source path used for cache identity."""
        return self.context.source_path


@dataclass(frozen=True)
class _FilenameTemplateTokens:
    """Normalized tokens available to filename templates."""

    category_for_filename: str
    category_clean: list[str]
    keyword_clean: list[str]
    summary_clean: list[str]
    structured_fields: dict[str, str] | None = None


@dataclass(frozen=True)
class _FilenameTemplateInput:
    """Inputs used to build or render a filename."""

    filename: str
    date_str: str
    project: str
    version: str
    tokens: _FilenameTemplateTokens

    @property
    def category_for_filename(self) -> str:
        """Return the category token chosen for the filename."""
        return self.tokens.category_for_filename

    @property
    def category_clean(self) -> list[str]:
        """Return the normalized category metadata."""
        return self.tokens.category_clean

    @property
    def keyword_clean(self) -> list[str]:
        """Return the normalized keyword metadata."""
        return self.tokens.keyword_clean

    @property
    def summary_clean(self) -> list[str]:
        """Return the normalized summary metadata."""
        return self.tokens.summary_clean

    @property
    def structured_fields(self) -> dict[str, str] | None:
        """Return the extracted structured metadata fields."""
        return self.tokens.structured_fields


@dataclass(frozen=True)
class _CategoryContext:
    """Current category decision and its heuristic provenance."""

    category: str
    category_for_filename: str
    category_source: str
    heuristic: _HeuristicCategorySignal
    skip_llm_by_rule: bool

    @property
    def cat_heur(self) -> str:
        """Return the heuristic category candidate."""
        return self.heuristic.cat_heur

    @property
    def heuristic_text(self) -> str:
        """Return text prepared for heuristic classification."""
        return self.heuristic.heuristic_text

    @property
    def heuristic_score(self) -> float:
        """Return the heuristic confidence score."""
        return self.heuristic.heuristic_score

    @property
    def heuristic_gap(self) -> float:
        """Return the score gap to the next category."""
        return self.heuristic.heuristic_gap

    @property
    def suggested_doc_type(self) -> str | None:
        """Return the suggested document type for prompting."""
        return self.heuristic.suggested_doc_type


@dataclass(frozen=True)
class _LlmResult:
    """LLM-derived metadata returned by one analysis path."""

    summary: str
    raw_keywords: list[str] | tuple[str, ...]
    precomputed_category: str | None = None
    precomputed_summary_tokens: list[str] | tuple[str, ...] | None = None


@dataclass(frozen=True)
class _FilenameDependencies:
    """Resolved dependencies used during filename generation."""

    llm_client: LLMClient
    heuristic_scorer: HeuristicScorer
    stopwords: Stopwords
    response_cache: ResponseCache | None
    cache_key_base: str | None


@dataclass(frozen=True)
class _FilenameMetadataParts:
    """Cleaned metadata parts used to assemble the final filename."""

    category_for_filename: str
    category_clean: list[str]
    keyword_clean: list[str]
    summary_clean: list[str]
    metadata: dict[str, object]
