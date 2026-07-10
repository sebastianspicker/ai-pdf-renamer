"""Deterministic category scoring and heuristic/LLM category reconciliation."""

from __future__ import annotations

import json
import logging
import re
import sys
import threading
from dataclasses import dataclass
from typing import Any, cast

from .heuristic_scoring import (
    HeuristicRule,
    HeuristicScorer,
    _score_text,
    load_heuristic_rules,
    load_heuristic_rules_for_language,
)
from .options import merge_options

logger = logging.getLogger(__name__)


__all__ = [
    "CategoryCombineParams",
    "HeuristicRule",
    "HeuristicScorer",
    "_score_text",
    "clear_category_aliases_cache",
    "clear_embedding_model_cache",
    "combine_categories",
    "load_heuristic_rules",
    "load_heuristic_rules_for_language",
    "normalize_llm_category",
]

# Global cache for category aliases (loaded once from category_aliases.json).
_CATEGORY_ALIASES: dict[str, str] | None = None
_CATEGORY_ALIASES_LOCK = threading.Lock()
_CATEGORY_ALIASES_MTIME_NS: int = 0


def _load_category_aliases() -> dict[str, str]:
    """Load alias map: LLM output (lowercase, _) -> heuristic category."""
    module = sys.modules[__name__]
    cached_aliases = cast(dict[str, str] | None, module.__dict__["_CATEGORY_ALIASES"])
    path = _category_aliases_path_or_none()
    if path is None:
        if cached_aliases is None:
            cached_aliases = {}
            module.__dict__["_CATEGORY_ALIASES"] = cached_aliases
        return cached_aliases

    current_mtime_ns = _path_mtime_ns(path)
    cached_mtime_ns = int(module.__dict__["_CATEGORY_ALIASES_MTIME_NS"])

    if cached_aliases is not None and current_mtime_ns == cached_mtime_ns:
        return cached_aliases
    with _CATEGORY_ALIASES_LOCK:
        cached_aliases = cast(dict[str, str] | None, module.__dict__["_CATEGORY_ALIASES"])
        cached_mtime_ns = int(module.__dict__["_CATEGORY_ALIASES_MTIME_NS"])
        if cached_aliases is not None and current_mtime_ns == cached_mtime_ns:
            return cached_aliases
        aliases = _read_category_aliases(path)
        module.__dict__["_CATEGORY_ALIASES"] = aliases
        module.__dict__["_CATEGORY_ALIASES_MTIME_NS"] = current_mtime_ns
        return aliases


def clear_category_aliases_cache() -> None:
    """Clear the cached category alias map."""
    with _CATEGORY_ALIASES_LOCK:
        module = sys.modules[__name__]
        module.__dict__["_CATEGORY_ALIASES"] = None
        module.__dict__["_CATEGORY_ALIASES_MTIME_NS"] = 0


def _category_aliases_path_or_none() -> Any | None:
    try:
        from .data_paths import category_aliases_path

        return category_aliases_path()
    except (ImportError, ValueError, FileNotFoundError):
        return None


def _path_mtime_ns(path: Any) -> int:
    try:
        return path.stat().st_mtime_ns if path.exists() else 0
    except OSError:
        return 0


def _read_category_aliases(path: Any) -> dict[str, str]:
    try:
        if not path.exists():
            return {}
        data = json.loads(path.read_text(encoding="utf-8"))
        aliases = data.get("aliases") if isinstance(data, dict) else {}
        return _normalized_alias_map(aliases if isinstance(aliases, dict) else {})
    except (OSError, json.JSONDecodeError, ValueError):
        return {}


def _normalized_alias_map(aliases: dict[Any, Any]) -> dict[str, str]:
    return {str(k).strip().lower().replace(" ", "_"): str(v) for k, v in aliases.items() if k and v}


def normalize_llm_category(cat_llm: str | None, *, _aliases: dict[str, str] | None = None) -> str:
    """Map LLM category to heuristic vocabulary to reduce false conflicts."""
    if not cat_llm or not isinstance(cat_llm, str):
        return ""
    # Preserve hierarchy markers as token separators instead of deleting them.
    cleaned = cat_llm.replace("/", "_")
    key = re.sub(r"[^\w\s-]", "", cleaned).strip().lower().replace(" ", "_")
    if not key or key in {"document", "unknown", "na"}:
        return key if key else "unknown"
    aliases = _aliases if _aliases is not None else _load_category_aliases()
    return aliases.get(key, key)


def _tokenize_for_overlap(text: str) -> set[str]:
    """Lowercase token set from category or context."""
    if not text or not isinstance(text, str):
        return set()
    tokens = re.split(r"[\s_]+", text.lower())
    return {t for t in tokens if t and t.isalnum()}


def _overlap_count(category_tokens: set[str], context_tokens: set[str]) -> int:
    """Number of category tokens that appear in context."""
    return len(category_tokens & context_tokens)


# Global cache for embedding model (loaded lazily when embeddings feature is used).
_embedding_model: Any = None
_embedding_model_lock = threading.Lock()


def _embedding_conflict_pick(
    context: str,
    cat_heur: str,
    cat_llm: str,
) -> str | None:
    """Return the category source with higher embedding similarity when available."""
    if not context or not context.strip():
        return None
    try:
        model = _embedding_model_instance()
        if model is None:
            return None
        embs = model.encode([context.strip(), cat_heur, cat_llm])
        if embs is None or len(embs) < 3:
            return None
        ctx_emb, heur_emb, llm_emb = embs[0], embs[1], embs[2]
        return "heuristic" if _cosine_similarity(ctx_emb, heur_emb) >= _cosine_similarity(ctx_emb, llm_emb) else "llm"
    except (TypeError, ValueError, AttributeError, RuntimeError, ImportError) as exc:
        logger.debug("Embedding conflict resolution failed: %s", exc)
        return None


def _embedding_model_instance() -> Any | None:
    module = sys.modules[__name__]
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        return None
    model = module.__dict__["_embedding_model"]
    if model is None:
        with _embedding_model_lock:
            model = module.__dict__["_embedding_model"]
            if model is None:
                model = SentenceTransformer("all-MiniLM-L6-v2")
                module.__dict__["_embedding_model"] = model
    return model


def clear_embedding_model_cache() -> None:
    """Clear the cached embedding model instance."""
    with _embedding_model_lock:
        sys.modules[__name__].__dict__["_embedding_model"] = None


def _cosine_similarity(a: Any, b: Any) -> float:
    dot = sum(float(x) * float(y) for x, y in zip(a, b, strict=True))
    na = sum(float(x) * float(x) for x in a) ** 0.5
    nb = sum(float(y) * float(y) for y in b) ** 0.5
    if na * nb <= 0:
        return 0.0
    return float(dot / (na * nb))


def _combine_apply_heuristic_override(
    cat_heuristic: str,
    cat_llm_norm: str,
    opts: CategoryCombineOptions,
    params: CategoryCombineParams,
) -> str | None:
    """If high-confidence heuristic override applies, return cat_heuristic; else None."""
    if params.heuristic_override_min_score is None or params.heuristic_override_min_gap is None:
        return None
    if opts.heuristic_score is None or opts.heuristic_gap is None:
        return None
    if (
        opts.heuristic_score < params.heuristic_override_min_score
        or opts.heuristic_gap < params.heuristic_override_min_gap
    ):
        return None
    logger.info(
        "High-confidence heuristic (score=%.2f gap=%.2f). Using %s.",
        opts.heuristic_score,
        opts.heuristic_gap,
        cat_heuristic,
    )
    logger.info(
        "CategoryConflict chosen=heuristic llm=%s heuristic=%s",
        cat_llm_norm,
        cat_heuristic,
    )
    return cat_heuristic


def _combine_agreement_or_parent(
    cat_llm_norm: str,
    cat_heuristic: str,
    category_parent_map: dict[str, str] | None,
) -> str | None:
    """If exact, parent, or sibling agreement, return category; else None."""
    if cat_llm_norm == cat_heuristic:
        return cat_heuristic
    if not category_parent_map:
        return None
    heur_parent = category_parent_map.get(cat_heuristic)
    llm_parent = category_parent_map.get(cat_llm_norm)
    if heur_parent == cat_llm_norm:
        logger.info(
            "LLM %s and heuristic %s agree (parent match). Using more specific: heuristic.",
            cat_llm_norm,
            cat_heuristic,
        )
        return cat_heuristic
    if llm_parent == cat_heuristic:
        logger.info(
            "LLM %s and heuristic %s agree (parent match). Using more specific: LLM.",
            cat_llm_norm,
            cat_heuristic,
        )
        return cat_llm_norm
    if heur_parent is not None and llm_parent is not None and heur_parent == llm_parent:
        logger.info(
            "Sibling categories (parent=%s). Using heuristic %s.",
            heur_parent,
            cat_heuristic,
        )
        logger.info(
            "CategoryConflict chosen=heuristic llm=%s heuristic=%s",
            cat_llm_norm,
            cat_heuristic,
        )
        return cat_heuristic
    return None


def _combine_resolve_conflict(
    cat_llm_norm: str,
    cat_heuristic: str,
    options: ConflictResolutionOptions | None = None,
    **overrides: object,
) -> str:
    """Resolve conflict via embedding, keyword overlap, or prefer_llm; default heuristic."""
    opts = merge_options(options or ConflictResolutionOptions(), overrides)
    context_pick = _context_conflict_pick(cat_llm_norm, cat_heuristic, opts)
    if context_pick is not None:
        return context_pick
    return _preference_conflict_pick(cat_llm_norm, cat_heuristic, opts.prefer_llm)


@dataclass(frozen=True)
class ConflictResolutionOptions:
    prefer_llm: bool = True
    context_for_overlap: str | None = None
    use_embeddings_for_conflict: bool = False
    use_keyword_overlap: bool = False
    heuristic_score: float | None = None
    heuristic_score_weight: float = 1.0


def _context_conflict_pick(
    cat_llm_norm: str,
    cat_heuristic: str,
    options: ConflictResolutionOptions,
) -> str | None:
    if not options.context_for_overlap:
        return None
    if options.use_embeddings_for_conflict:
        pick = _embedding_conflict_pick(options.context_for_overlap, cat_heuristic, cat_llm_norm)
        if pick in {"heuristic", "llm"}:
            chosen = cat_heuristic if pick == "heuristic" else cat_llm_norm
            _log_category_conflict(pick, cat_llm_norm, cat_heuristic, detail="embedding")
            return chosen
    if options.use_keyword_overlap:
        return _overlap_conflict_pick(cat_llm_norm, cat_heuristic, options)
    return None


def _overlap_conflict_pick(
    cat_llm_norm: str,
    cat_heuristic: str,
    options: ConflictResolutionOptions,
) -> str:
    overlap_llm, overlap_heur_weighted = _weighted_overlap_scores(cat_llm_norm, cat_heuristic, options)
    if overlap_llm > overlap_heur_weighted:
        logger.info(
            "Conflict: LLM=%s, Heuristic=%s. Overlap favors LLM (%d vs %.2f).",
            cat_llm_norm,
            cat_heuristic,
            overlap_llm,
            overlap_heur_weighted,
        )
        _log_category_conflict("llm", cat_llm_norm, cat_heuristic)
        return cat_llm_norm
    if overlap_heur_weighted > overlap_llm:
        logger.info(
            "Conflict: LLM=%s, Heur=%s. Overlap favors heuristic (%.2f vs %d).",
            cat_llm_norm,
            cat_heuristic,
            overlap_heur_weighted,
            overlap_llm,
        )
        _log_category_conflict("heuristic", cat_llm_norm, cat_heuristic)
        return cat_heuristic
    logger.info("Conflict: LLM=%s, Heuristic=%s. Overlap tie; using heuristic.", cat_llm_norm, cat_heuristic)
    _log_category_conflict("heuristic", cat_llm_norm, cat_heuristic)
    return cat_heuristic


def _weighted_overlap_scores(
    cat_llm_norm: str,
    cat_heuristic: str,
    options: ConflictResolutionOptions,
) -> tuple[int, float]:
    ctx_tokens = _tokenize_for_overlap(options.context_for_overlap or "")
    overlap_llm = _overlap_count(_tokenize_for_overlap(cat_llm_norm), ctx_tokens)
    overlap_heur = _overlap_count(_tokenize_for_overlap(cat_heuristic), ctx_tokens)
    score_bonus = _heuristic_score_bonus(options.heuristic_score, options.heuristic_score_weight)
    return overlap_llm, overlap_heur + score_bonus


def _heuristic_score_bonus(heuristic_score: float | None, heuristic_score_weight: float) -> float:
    if heuristic_score_weight <= 0 or heuristic_score is None:
        return 0.0
    return heuristic_score_weight * heuristic_score


def _preference_conflict_pick(cat_llm_norm: str, cat_heuristic: str, prefer_llm: bool) -> str:
    chosen = "llm" if prefer_llm else "heuristic"
    if prefer_llm:
        logger.info("Conflict: LLM category=%s, Heuristic=%s. Preferring LLM.", cat_llm_norm, cat_heuristic)
        _log_category_conflict(chosen, cat_llm_norm, cat_heuristic)
        return cat_llm_norm
    logger.info("Conflict: LLM category=%s, Heuristic=%s. Prioritizing heuristic.", cat_llm_norm, cat_heuristic)
    _log_category_conflict(chosen, cat_llm_norm, cat_heuristic)
    return cat_heuristic


def _log_category_conflict(
    chosen: str,
    cat_llm_norm: str,
    cat_heuristic: str,
    *,
    detail: str | None = None,
) -> None:
    chosen_label = f"{chosen} ({detail})" if detail else chosen
    logger.info("CategoryConflict chosen=%s llm=%s heuristic=%s", chosen_label, cat_llm_norm, cat_heuristic)


@dataclass(frozen=True)
class CategoryCombineParams:
    """Parameters that control how heuristic and LLM categories are merged."""

    prefer_llm: bool = True
    min_heuristic_score: float = 0.0
    heuristic_override_min_score: float | None = None
    heuristic_override_min_gap: float | None = None
    heuristic_score_weight: float = 1.0
    use_keyword_overlap: bool = False
    use_embeddings_for_conflict: bool = False


@dataclass(frozen=True)
class CategoryCombineOptions:
    heuristic_score: float | None = None
    heuristic_gap: float | None = None
    params: CategoryCombineParams | None = None
    context_for_overlap: str | None = None
    category_parent_map: dict[str, str] | None = None


def combine_categories(
    cat_llm: str,
    cat_heur: str,
    options: CategoryCombineOptions | None = None,
    **overrides: object,
) -> str:
    """Merge heuristic and LLM categories using configurable conflict resolution, parent matching, and overlap."""
    opts = merge_options(options or CategoryCombineOptions(), overrides)
    params = opts.params or CategoryCombineParams()
    cat_llm_norm = normalize_llm_category(cat_llm)
    early_result = _combine_early_result(cat_llm_norm, cat_heur, opts.heuristic_score, params)
    if early_result is not None:
        return early_result

    override = _combine_apply_heuristic_override(
        cat_heur,
        cat_llm_norm,
        opts,
        params,
    )
    if override is not None:
        return override
    agreed = _combine_agreement_or_parent(cat_llm_norm, cat_heur, opts.category_parent_map)
    if agreed is not None:
        return agreed
    return _combine_resolve_conflict(cat_llm_norm, cat_heur, _conflict_options(params, opts))


def _is_valid_llm_category(cat_llm_norm: str) -> bool:
    return cat_llm_norm not in {"document", "unknown", "na", ""}


def _combine_early_result(
    cat_llm_norm: str,
    cat_heur: str,
    heuristic_score: float | None,
    params: CategoryCombineParams,
) -> str | None:
    if cat_heur == "unknown":
        return cat_llm_norm if _is_valid_llm_category(cat_llm_norm) else cat_heur
    if heuristic_score is not None and heuristic_score < params.min_heuristic_score:
        return _low_score_category_pick(cat_llm_norm, cat_heur, heuristic_score, params.min_heuristic_score)
    if not _is_valid_llm_category(cat_llm_norm):
        return cat_heur
    return None


def _low_score_category_pick(
    cat_llm_norm: str,
    cat_heur: str,
    heuristic_score: float,
    min_heuristic_score: float,
) -> str:
    if _is_valid_llm_category(cat_llm_norm):
        logger.info(
            "Heuristic score %.2f below threshold %.2f. Preferring LLM category %s.",
            heuristic_score,
            min_heuristic_score,
            cat_llm_norm,
        )
        return cat_llm_norm
    logger.info(
        "Heuristic score %.2f below threshold %.2f but LLM category is invalid (%s). Using heuristic.",
        heuristic_score,
        min_heuristic_score,
        cat_llm_norm,
    )
    return cat_heur


def _conflict_options(
    params: CategoryCombineParams,
    options: CategoryCombineOptions,
) -> ConflictResolutionOptions:
    return ConflictResolutionOptions(
        prefer_llm=params.prefer_llm,
        context_for_overlap=options.context_for_overlap,
        use_embeddings_for_conflict=params.use_embeddings_for_conflict,
        use_keyword_overlap=params.use_keyword_overlap,
        heuristic_score=options.heuristic_score,
        heuristic_score_weight=params.heuristic_score_weight,
    )
