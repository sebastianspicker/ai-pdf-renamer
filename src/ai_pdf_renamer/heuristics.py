from __future__ import annotations

import json
import logging
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class HeuristicRule:
    pattern: re.Pattern[str]
    category: str
    score: float
    language: str | None = None
    parent: str | None = None


def load_heuristic_rules(path: str | Path) -> list[HeuristicRule]:
    """Load heuristic scoring rules from a JSON file, compiling regex patterns and validating fields."""
    path_obj = Path(path)
    try:
        raw = path_obj.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"Could not read data file at {path_obj.absolute()}: {exc!s}") from exc
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in data file at {path_obj.absolute()}. {exc!s}") from exc
    rules: list[HeuristicRule] = []
    raw_patterns = data.get("patterns", [])
    if not isinstance(raw_patterns, list):
        raw_patterns = []

    for entry in raw_patterns:
        regex = entry.get("regex")
        category = entry.get("category")
        score = entry.get("score")
        if not isinstance(regex, str) or not isinstance(category, str):
            continue
        try:
            compiled = re.compile(regex)
        except re.error as exc:
            logger.warning("Invalid regex skipped: %r (%s)", regex, exc)
            continue
        try:
            score_f = float(score)
        except (TypeError, ValueError):
            score_f = 0.0
        language = entry.get("language")
        if language is not None and not isinstance(language, str):
            language = None
        elif language is not None:
            language = language.strip().lower() or None
            if language not in ("de", "en"):
                language = None
        parent = entry.get("parent")
        if parent is not None and not isinstance(parent, str):
            parent = None
        elif parent is not None:
            parent = parent.strip() or None
        rules.append(
            HeuristicRule(
                pattern=compiled,
                category=category,
                score=score_f,
                language=language,
                parent=parent,
            )
        )

    return rules


def load_heuristic_rules_for_language(
    base_path: str | Path,
    language: str,
) -> list[HeuristicRule]:
    """
    Load heuristic rules from base file and, if present, from a per-locale file
    (e.g. heuristic_scores_de.json, heuristic_scores_en.json). Base rules come
    first, then locale-specific rules (same structure as heuristic_scores.json).
    """
    path_obj = Path(base_path)
    base_rules = load_heuristic_rules(path_obj)
    lang = (language or "de").strip().lower()
    if lang not in ("de", "en"):
        lang = "de"
    locale_file = path_obj.parent / f"heuristic_scores_{lang}.json"
    if not locale_file.exists():
        return base_rules
    try:
        locale_rules = load_heuristic_rules(locale_file)
        return base_rules + locale_rules
    except (ValueError, OSError) as exc:
        logger.warning(
            "Could not load locale heuristic file %s: %s. Using base rules only.",
            locale_file.name,
            exc,
        )
        return base_rules


def _score_text(
    text: str,
    rules: list[HeuristicRule],
    language: str | None,
    *,
    title_weight_region: int = 0,
    title_weight_factor: float = 1.5,
    max_score_per_category: float | None = None,
    max_text_length: int = 100000,
) -> dict[str, float]:
    """
    Score text against heuristic rules. Returns dict of category -> score.

    Security note: Regex patterns are loaded from heuristic_scores.json. To prevent
    ReDoS (Regular Expression Denial of Service), the text length is capped at
    max_text_length characters. Only trusted users should modify heuristic pattern files.
    Pattern files should be validated for catastrophic backtracking before deployment.
    """
    scores: dict[str, float] = {}
    # Mitigate potential ReDoS by capping the text length searched by regexes.
    search_text = text if len(text) <= max_text_length else text[:max_text_length]
    for rule in rules:
        if language is not None and rule.language is not None and rule.language != language:
            continue
        # P2: Use finditer to count all matches and check title region for each
        matches = list(rule.pattern.finditer(search_text))
        if not matches:
            continue
        delta = 0.0
        for match in matches:
            weight = 1.0
            # P2: Check all matches for title region, not just the first
            if title_weight_region > 0 and match.start() < title_weight_region:
                weight = title_weight_factor
            delta += rule.score * weight
        scores[rule.category] = scores.get(rule.category, 0.0) + delta
    if max_score_per_category is not None:
        for cat in list(scores):
            if scores[cat] > max_score_per_category:
                scores[cat] = max_score_per_category
    return scores


@dataclass(frozen=True)
class HeuristicScorer:
    rules: list[HeuristicRule]
    _parent_map: dict[str, str] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        # Pre-compute parent map once at construction (frozen dataclass: use object.__setattr__).
        cache: dict[str, str] = {}
        for rule in self.rules:
            if rule.parent is not None and rule.parent.strip():
                cache[rule.category] = rule.parent.strip()
        object.__setattr__(self, "_parent_map", cache)

    def best_category(
        self,
        text: str,
        *,
        language: str | None = None,
    ) -> str:
        cat, _best, _runner_cat, _runner_up = self.best_category_with_confidence(
            text,
            language=language,
            min_score_gap=0.0,
        )
        return cat

    def best_category_with_confidence(
        self,
        text: str,
        *,
        language: str | None = None,
        min_score_gap: float = 0.0,
        max_score_per_category: float | None = None,
        title_weight_region: int = 0,
        title_weight_factor: float = 1.5,
    ) -> tuple[str, float, str, float]:
        """
        Returns (category, best_score, runner_up_cat, runner_up_score).
        If no rule matches or min_score_gap not met: 'unknown', 0.0, 'unknown', 0.0.
        """
        if text is None or not isinstance(text, str):
            return ("unknown", 0.0, "unknown", 0.0)
        scores = _score_text(
            text,
            self.rules,
            language,
            title_weight_region=title_weight_region,
            title_weight_factor=title_weight_factor,
            max_score_per_category=max_score_per_category,
        )
        if not scores:
            return ("unknown", 0.0, "unknown", 0.0)
        parents = self._category_to_parent()
        # Tie-break: same score -> prefer category that has a parent (more specific)
        sorted_cats = sorted(
            scores,
            key=lambda c: (scores[c], 1 if c in parents else 0),
            reverse=True,
        )
        best_cat = sorted_cats[0]
        best_score = scores[best_cat]
        runner_up_cat = sorted_cats[1] if len(sorted_cats) > 1 else "unknown"
        runner_up_score = scores.get(runner_up_cat, 0.0)
        if min_score_gap > 0 and (best_score - runner_up_score) < min_score_gap:
            logger.info(
                "Heuristic gap too small (best=%s %.2f, runner_up=%.2f, gap=%.2f). Returning unknown.",
                best_cat,
                best_score,
                runner_up_score,
                best_score - runner_up_score,
            )
            return ("unknown", 0.0, "unknown", 0.0)
        logger.info(
            "Heuristic scoring result: %s. Best: %s (%.2f)",
            scores,
            best_cat,
            best_score,
        )
        if logger.isEnabledFor(logging.DEBUG):
            top3 = sorted_cats[:3]
            logger.debug(
                "Heuristic top-3: %s",
                [(c, scores[c]) for c in top3],
            )
        return (best_cat, best_score, runner_up_cat, runner_up_score)

    def top_n_categories(
        self,
        text: str,
        n: int = 5,
        *,
        language: str | None = None,
        max_score_per_category: float | None = None,
        title_weight_region: int = 0,
        title_weight_factor: float = 1.5,
    ) -> list[str]:
        """Return up to n category names by score (best first). No min_score_gap."""
        if not text or not isinstance(text, str) or n <= 0:
            return []
        scores = _score_text(
            text,
            self.rules,
            language,
            title_weight_region=title_weight_region,
            title_weight_factor=title_weight_factor,
            max_score_per_category=max_score_per_category,
        )
        if not scores:
            return []
        parents = self._category_to_parent()
        sorted_cats = sorted(
            scores,
            key=lambda c: (scores[c], 1 if c in parents else 0),
            reverse=True,
        )
        return sorted_cats[:n]

    def all_categories(self) -> frozenset[str]:
        """Return the set of all category names from rules (for constrained LLM)."""
        return frozenset(rule.category for rule in self.rules)

    def _category_to_parent(self) -> dict[str, str]:
        """Category -> parent from rules (last occurrence wins). Pre-computed at construction."""
        return self._parent_map

    def get_display_category(
        self,
        category: str,
        style: str,
    ) -> str:
        """
        Map category to filename segment using optional parent.
        style: specific (as-is), with_parent (parent_category), parent_only (parent).
        """
        if not category or category in {"unknown", "document", "na", ""}:
            return category
        if style == "specific":
            return category
        parents = self._category_to_parent()
        parent = parents.get(category)
        if style == "parent_only":
            return parent if parent else category
        if style == "with_parent" and parent:
            return f"{parent}_{category}"
        return category


# Global cache for category aliases (loaded once from category_aliases.json).
_CATEGORY_ALIASES: dict[str, str] | None = None
_CATEGORY_ALIASES_LOCK = threading.Lock()
_CATEGORY_ALIASES_MTIME_NS: int = 0  # nanosecond mtime avoids float comparison issues on coarse-grained FSes


def _load_category_aliases() -> dict[str, str]:
    """Load alias map: LLM output (lowercase, _) -> heuristic category. Cached with mtime invalidation."""
    global _CATEGORY_ALIASES, _CATEGORY_ALIASES_MTIME_NS
    try:
        from .data_paths import category_aliases_path

        path = category_aliases_path()
    except (ImportError, ValueError, FileNotFoundError):
        if _CATEGORY_ALIASES is None:
            _CATEGORY_ALIASES = {}
        return _CATEGORY_ALIASES

    try:
        current_mtime_ns = path.stat().st_mtime_ns if path.exists() else 0
    except OSError:
        current_mtime_ns = 0

    if _CATEGORY_ALIASES is not None and current_mtime_ns == _CATEGORY_ALIASES_MTIME_NS:
        return _CATEGORY_ALIASES
    with _CATEGORY_ALIASES_LOCK:
        if _CATEGORY_ALIASES is not None and current_mtime_ns == _CATEGORY_ALIASES_MTIME_NS:
            return _CATEGORY_ALIASES
        try:
            if path.exists():
                raw = path.read_text(encoding="utf-8")
                data = json.loads(raw)
                aliases = data.get("aliases")
                if not isinstance(aliases, dict):
                    aliases = {}
                norm = {str(k).strip().lower().replace(" ", "_"): str(v) for k, v in aliases.items() if k and v}
                _CATEGORY_ALIASES = norm
                _CATEGORY_ALIASES_MTIME_NS = current_mtime_ns
            else:
                _CATEGORY_ALIASES = {}
                _CATEGORY_ALIASES_MTIME_NS = current_mtime_ns
        except (OSError, json.JSONDecodeError, ValueError):
            _CATEGORY_ALIASES = {}
            _CATEGORY_ALIASES_MTIME_NS = current_mtime_ns
    return _CATEGORY_ALIASES


def normalize_llm_category(cat_llm: str | None, *, _aliases: dict[str, str] | None = None) -> str:
    """Map LLM category to heuristic vocabulary to reduce false conflicts."""
    if not cat_llm or not isinstance(cat_llm, str):
        return ""
    # P2: Preserve slashes (split on them) and meaningful punctuation
    # Replace slashes with underscores to preserve hierarchical categories
    cleaned = cat_llm.replace("/", "_")
    key = re.sub(r"[^\w\s-]", "", cleaned).strip().lower().replace(" ", "_")
    if not key or key in {"document", "unknown", "na"}:
        return key if key else "unknown"
    aliases = _aliases if _aliases is not None else _load_category_aliases()
    return aliases.get(key, key)


def _tokenize_for_overlap(text: str) -> set[str]:
    """Lowercase token set from category or context (split on whitespace and _)."""
    if not text or not isinstance(text, str):
        return set()
    tokens = re.split(r"[\s_]+", text.lower())
    return {t for t in tokens if t and t.isalnum()}


def _overlap_count(category_tokens: set[str], context_tokens: set[str]) -> int:
    """Number of category tokens that appear in context (for conflict resolution)."""
    return len(category_tokens & context_tokens)


# Global cache for embedding model (loaded lazily when embeddings feature is used).
_embedding_model: Any = None
_embedding_model_lock = threading.Lock()


def _embedding_conflict_pick(
    context: str,
    cat_heur: str,
    cat_llm: str,
) -> str | None:
    """
    If sentence-transformers is available, return 'heuristic' or 'llm' based on
    which category is more similar to context (cosine similarity). Else return None.
    """
    global _embedding_model
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        return None
    if not context or not context.strip():
        return None
    try:
        if _embedding_model is None:
            with _embedding_model_lock:
                if _embedding_model is None:
                    _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        model = _embedding_model
        embs = model.encode([context.strip(), cat_heur, cat_llm])
        if embs is None or len(embs) < 3:
            return None
        ctx_emb, heur_emb, llm_emb = embs[0], embs[1], embs[2]

        # Cosine similarity (emb can be numpy or list)
        def _cos(a: Any, b: Any) -> float:
            dot = sum(float(x) * float(y) for x, y in zip(a, b, strict=True))
            na = sum(float(x) * float(x) for x in a) ** 0.5
            nb = sum(float(y) * float(y) for y in b) ** 0.5
            if na * nb <= 0:
                return 0.0
            return float(dot / (na * nb))

        sim_heur = _cos(ctx_emb, heur_emb)
        sim_llm = _cos(ctx_emb, llm_emb)
        if sim_heur >= sim_llm:
            return "heuristic"
        return "llm"
    except (TypeError, ValueError, AttributeError, RuntimeError, ImportError) as exc:
        logger.debug("Embedding conflict resolution failed: %s", exc)
        return None


def _combine_apply_heuristic_override(
    cat_heuristic: str,
    cat_llm_norm: str,
    *,
    heuristic_override_min_score: float | None,
    heuristic_override_min_gap: float | None,
    heuristic_score: float | None,
    heuristic_gap: float | None,
) -> str | None:
    """If high-confidence heuristic override applies, return cat_heuristic; else None."""
    if (
        heuristic_override_min_score is None
        or heuristic_override_min_gap is None
        or heuristic_score is None
        or heuristic_gap is None
        or heuristic_score < heuristic_override_min_score
        or heuristic_gap < heuristic_override_min_gap
    ):
        return None
    logger.info(
        "High-confidence heuristic (score=%.2f gap=%.2f). Using %s.",
        heuristic_score,
        heuristic_gap,
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
        # P2: Heuristic is more specific (has cat_llm_norm as parent), return heuristic
        logger.info(
            "LLM %s and heuristic %s agree (parent match). Using more specific: heuristic.",
            cat_llm_norm,
            cat_heuristic,
        )
        return cat_heuristic
    if llm_parent == cat_heuristic:
        # P2: LLM is more specific (has cat_heuristic as parent), return LLM
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
    *,
    prefer_llm: bool,
    context_for_overlap: str | None,
    use_embeddings_for_conflict: bool,
    use_keyword_overlap: bool,
    heuristic_score: float | None,
    heuristic_score_weight: float,
) -> str:
    """Resolve conflict via embedding, keyword overlap, or prefer_llm; default heuristic."""
    if context_for_overlap and (use_embeddings_for_conflict or use_keyword_overlap):
        if use_embeddings_for_conflict:
            pick = _embedding_conflict_pick(context_for_overlap, cat_heuristic, cat_llm_norm)
            if pick == "heuristic":
                logger.info(
                    "CategoryConflict chosen=heuristic (embedding) llm=%s heuristic=%s",
                    cat_llm_norm,
                    cat_heuristic,
                )
                return cat_heuristic
            if pick == "llm":
                logger.info(
                    "CategoryConflict chosen=llm (embedding) llm=%s heuristic=%s",
                    cat_llm_norm,
                    cat_heuristic,
                )
                return cat_llm_norm
        if use_keyword_overlap:
            ctx_tokens = _tokenize_for_overlap(context_for_overlap)
            llm_tokens = _tokenize_for_overlap(cat_llm_norm)
            heur_tokens = _tokenize_for_overlap(cat_heuristic)
            overlap_llm = _overlap_count(llm_tokens, ctx_tokens)
            overlap_heur = _overlap_count(heur_tokens, ctx_tokens)
            score_bonus = (
                heuristic_score_weight * (heuristic_score or 0.0)
                if heuristic_score_weight > 0 and heuristic_score is not None
                else 0.0
            )
            overlap_heur_weighted = overlap_heur + score_bonus
            if overlap_llm > overlap_heur_weighted:
                logger.info(
                    "Conflict: LLM=%s, Heuristic=%s. Overlap favors LLM (%d vs %.2f).",
                    cat_llm_norm,
                    cat_heuristic,
                    overlap_llm,
                    overlap_heur_weighted,
                )
                logger.info(
                    "CategoryConflict chosen=llm llm=%s heuristic=%s",
                    cat_llm_norm,
                    cat_heuristic,
                )
                return cat_llm_norm
            if overlap_heur_weighted > overlap_llm:
                logger.info(
                    "Conflict: LLM=%s, Heur=%s. Overlap favors heuristic (%.2f vs %d).",
                    cat_llm_norm,
                    cat_heuristic,
                    overlap_heur_weighted,
                    overlap_llm,
                )
                logger.info(
                    "CategoryConflict chosen=heuristic llm=%s heuristic=%s",
                    cat_llm_norm,
                    cat_heuristic,
                )
                return cat_heuristic
            logger.info(
                "Conflict: LLM=%s, Heuristic=%s. Overlap tie; using heuristic.",
                cat_llm_norm,
                cat_heuristic,
            )
            logger.info(
                "CategoryConflict chosen=heuristic llm=%s heuristic=%s",
                cat_llm_norm,
                cat_heuristic,
            )
            return cat_heuristic
    if prefer_llm:
        logger.info(
            "Conflict: LLM category=%s, Heuristic=%s. Preferring LLM.",
            cat_llm_norm,
            cat_heuristic,
        )
        logger.info(
            "CategoryConflict chosen=llm llm=%s heuristic=%s",
            cat_llm_norm,
            cat_heuristic,
        )
        return cat_llm_norm
    logger.info(
        "Conflict: LLM category=%s, Heuristic=%s. Prioritizing heuristic.",
        cat_llm_norm,
        cat_heuristic,
    )
    logger.info(
        "CategoryConflict chosen=heuristic llm=%s heuristic=%s",
        cat_llm_norm,
        cat_heuristic,
    )
    return cat_heuristic


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


def combine_categories(
    cat_llm: str,
    cat_heur: str,
    *,
    heuristic_score: float | None = None,
    heuristic_gap: float | None = None,
    params: CategoryCombineParams | None = None,
    context_for_overlap: str | None = None,
    category_parent_map: dict[str, str] | None = None,
) -> str:
    """Merge heuristic and LLM categories using configurable conflict resolution, parent matching, and overlap."""
    if params is None:
        params = CategoryCombineParams()
    cat_llm_norm = normalize_llm_category(cat_llm)
    if cat_heur == "unknown":
        valid = cat_llm_norm not in {"document", "unknown", "na", ""}
        return cat_llm_norm if valid else cat_heur
    if heuristic_score is not None and heuristic_score < params.min_heuristic_score:
        # P1: Only prefer LLM category if it is actually valid
        llm_valid = cat_llm_norm not in {"document", "unknown", "na", ""}
        if llm_valid:
            logger.info(
                "Heuristic score %.2f below threshold %.2f. Preferring LLM category %s.",
                heuristic_score,
                params.min_heuristic_score,
                cat_llm_norm,
            )
            return cat_llm_norm
        logger.info(
            "Heuristic score %.2f below threshold %.2f but LLM category is invalid (%s). Using heuristic.",
            heuristic_score,
            params.min_heuristic_score,
            cat_llm_norm,
        )
        return cat_heur
    if cat_llm_norm in {"document", "unknown", "na", ""}:
        return cat_heur

    r = _combine_apply_heuristic_override(
        cat_heur,
        cat_llm_norm,
        heuristic_override_min_score=params.heuristic_override_min_score,
        heuristic_override_min_gap=params.heuristic_override_min_gap,
        heuristic_score=heuristic_score,
        heuristic_gap=heuristic_gap,
    )
    if r is not None:
        return r
    r = _combine_agreement_or_parent(cat_llm_norm, cat_heur, category_parent_map)
    if r is not None:
        return r
    return _combine_resolve_conflict(
        cat_llm_norm,
        cat_heur,
        prefer_llm=params.prefer_llm,
        context_for_overlap=context_for_overlap,
        use_embeddings_for_conflict=params.use_embeddings_for_conflict,
        use_keyword_overlap=params.use_keyword_overlap,
        heuristic_score=heuristic_score,
        heuristic_score_weight=params.heuristic_score_weight,
    )
