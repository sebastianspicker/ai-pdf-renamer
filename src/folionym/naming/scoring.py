"""Deterministic heuristic rule loading and category scoring."""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..infrastructure.resources import load_json_data

logger = logging.getLogger("folionym.heuristics")


@dataclass(frozen=True)
class HeuristicRule:
    """Represent one weighted rule used during heuristic classification."""

    pattern: re.Pattern[str]
    category: str
    score: float
    negative_pattern: re.Pattern[str] | None = None
    language: str | None = None
    parent: str | None = None


def load_heuristic_rules(path: str | Path) -> list[HeuristicRule]:
    """Load heuristic scoring rules from a JSON file, compiling regex patterns and validating fields."""
    path_obj = Path(path)
    data = _load_heuristic_rule_data(path_obj)
    rules: list[HeuristicRule] = []
    raw_patterns = data.get("patterns", [])
    if not isinstance(raw_patterns, list):
        raw_patterns = []

    for entry in raw_patterns:
        if not isinstance(entry, dict):
            continue
        rule = _build_heuristic_rule(entry, path_obj)
        if rule is not None:
            rules.append(rule)

    return rules


def _load_heuristic_rule_data(path_obj: Path) -> dict[str, Any]:
    """Load heuristic rule data from configured data."""
    data = load_json_data(path_obj)
    if isinstance(data, dict):
        return data
    return {}


def _compile_heuristic_regex(regex: str, path_obj: Path) -> re.Pattern[str]:
    """Compile heuristic regex once for consistent matching."""
    try:
        return re.compile(regex)
    except re.error as exc:
        raise ValueError(f"Invalid regex in data file at {path_obj.absolute()}: {regex!r} ({exc})") from exc


def _compile_negative_regex(regex: Any, category: str, path_obj: Path) -> re.Pattern[str] | None:
    """Compile negative regex once for consistent matching."""
    if not isinstance(regex, str) or not regex.strip():
        return None
    try:
        return re.compile(regex)
    except re.error as exc:
        raise ValueError(
            f"Invalid negative regex in data file at {path_obj.absolute()} for {category!r}: {regex!r} ({exc})"
        ) from exc


def _normalize_rule_language(language: Any) -> str | None:
    """Normalize a supported rule language code, or return None."""
    if not isinstance(language, str):
        return None
    normalized = language.strip().lower() or None
    if normalized not in ("de", "en"):
        return None
    return normalized


def _normalize_rule_parent(parent: Any) -> str | None:
    """Normalize an optional parent-category name."""
    if not isinstance(parent, str):
        return None
    return parent.strip() or None


def _rule_score(score: Any) -> float:
    """Coerce a configured rule score to float, defaulting to zero."""
    try:
        return float(score)
    except (
        TypeError,
        ValueError,
    ):
        return 0.0


def _build_heuristic_rule(entry: dict[str, Any], path_obj: Path) -> HeuristicRule | None:
    """Build a heuristic rule from one validated JSON entry."""
    regex = entry.get("regex")
    category = entry.get("category")
    if not isinstance(regex, str) or not isinstance(category, str):
        return None
    return HeuristicRule(
        pattern=_compile_heuristic_regex(regex, path_obj),
        category=category,
        score=_rule_score(entry.get("score")),
        negative_pattern=_compile_negative_regex(entry.get("negative_regex"), category, path_obj),
        language=_normalize_rule_language(entry.get("language")),
        parent=_normalize_rule_parent(entry.get("parent")),
    )


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


def _group_rules_by_category(rules: list[HeuristicRule]) -> dict[str, list[HeuristicRule]]:
    """Group rules by category to avoid repeated downstream traversal."""
    grouped: dict[str, list[HeuristicRule]] = {}
    for rule in rules:
        grouped.setdefault(rule.category, []).append(rule)
    return grouped


def _rule_matches_language(rule: HeuristicRule, language: str | None) -> bool:
    """Return whether a rule applies to the requested language."""
    return language is None or rule.language is None or rule.language == language


def _match_is_negated(rule: HeuristicRule, search_text: str, match: re.Match[str]) -> bool:
    """Match is negated against the configured heuristic rule."""
    if rule.negative_pattern is None:
        return False
    window_start = max(0, match.start() - 48)
    window_end = min(len(search_text), match.end() + 48)
    return bool(rule.negative_pattern.search(search_text[window_start:window_end]))


def _match_weight(match: re.Match[str], *, title_weight_region: int, title_weight_factor: float) -> float:
    """Match weight against the configured heuristic rule."""
    if title_weight_region > 0 and match.start() < title_weight_region:
        return title_weight_factor
    return 1.0


def _score_rule_matches(
    rule: HeuristicRule,
    search_text: str,
    *,
    title_weight_region: int,
    title_weight_factor: float,
) -> float:
    """Score rule matches for deterministic category selection."""
    delta = 0.0
    for match in rule.pattern.finditer(search_text):
        if _match_is_negated(rule, search_text, match):
            continue
        weight = _match_weight(
            match,
            title_weight_region=title_weight_region,
            title_weight_factor=title_weight_factor,
        )
        delta += rule.score * weight
    return delta


def _score_category_rules(
    category_rules: list[HeuristicRule],
    search_text: str,
    language: str | None,
    options: ScoringOptions,
) -> float:
    """Score category rules for deterministic category selection."""
    category_score = 0.0
    for rule in category_rules:
        if not _rule_matches_language(rule, language):
            continue
        delta = _score_rule_matches(
            rule,
            search_text,
            title_weight_region=options.title_weight_region,
            title_weight_factor=options.title_weight_factor,
        )
        if delta <= 0:
            continue
        category_score += delta
        if options.max_score_per_category is not None and category_score >= options.max_score_per_category:
            return options.max_score_per_category
    return category_score


def _score_text(
    text: str,
    rules: list[HeuristicRule],
    language: str | None,
    options: ScoringOptions | None = None,
) -> dict[str, float]:
    """
    Score text against heuristic rules. Returns dict of category -> score.

    Security note: Regex patterns are loaded from heuristic_scores.json. To prevent
    ReDoS (Regular Expression Denial of Service), the text length is capped at
    max_text_length characters. Only trusted users should modify heuristic pattern files.
    Pattern files should be validated for catastrophic backtracking before deployment.
    """
    opts = options or ScoringOptions()
    scores: dict[str, float] = {}
    started_at = time.perf_counter()
    search_text = _bounded_search_text(text, opts.max_text_length)
    grouped_rules = opts.rules_by_category if opts.rules_by_category is not None else _group_rules_by_category(rules)
    for category, category_rules in grouped_rules.items():
        category_score = _score_category_rules(
            category_rules,
            search_text,
            language,
            opts,
        )
        if category_score > 0:
            scores[category] = category_score
    _cap_scores(scores, opts.max_score_per_category)
    _log_score_timing(started_at, search_text, grouped_rules, rules)
    return scores


@dataclass(frozen=True)
class ScoringOptions:
    """Configure bounded heuristic scoring behavior."""

    rules_by_category: dict[str, list[HeuristicRule]] | None = None
    title_weight_region: int = 0
    title_weight_factor: float = 1.5
    max_score_per_category: float | None = None
    max_text_length: int = 100000


@dataclass(frozen=True)
class ConfidenceOptions:
    """Configure confidence calculation for ranked categories."""

    language: str | None = None
    min_score_gap: float = 0.0
    max_score_per_category: float | None = None
    title_weight_region: int = 0
    title_weight_factor: float = 1.5


@dataclass(frozen=True)
class TopNOptions:
    """Configure the number of ranked category results to retain."""

    n: int = 5
    language: str | None = None
    max_score_per_category: float | None = None
    title_weight_region: int = 0
    title_weight_factor: float = 1.5


def _bounded_search_text(text: str, max_text_length: int) -> str:
    """Bound search text to keep analysis cost predictable."""
    return text if len(text) <= max_text_length else text[:max_text_length]


def _cap_scores(scores: dict[str, float], max_score_per_category: float | None) -> None:
    """Cap scores to preserve the scoring contract."""
    if max_score_per_category is None:
        return
    for cat in list(scores):
        if scores[cat] > max_score_per_category:
            scores[cat] = max_score_per_category


def _log_score_timing(
    started_at: float,
    search_text: str,
    grouped_rules: dict[str, list[HeuristicRule]],
    rules: list[HeuristicRule],
) -> None:
    """Log score timing to make the resolution decision observable."""
    if not logger.isEnabledFor(logging.DEBUG):
        return
    elapsed_ms = (time.perf_counter() - started_at) * 1000.0
    logger.debug(
        "HeuristicMatchTiming chars=%s categories=%s rules=%s elapsed_ms=%.2f",
        len(search_text),
        len(grouped_rules),
        len(rules),
        elapsed_ms,
    )


def _unknown_confidence() -> tuple[str, float, str, float]:
    """Return the canonical no-confidence category result."""
    return ("unknown", 0.0, "unknown", 0.0)


def _rank_scored_categories(scores: dict[str, float], parents: dict[str, str]) -> list[str]:
    """Rank scored categories for deterministic category selection."""
    return sorted(
        scores,
        key=lambda c: (scores[c], 1 if c in parents else 0),
        reverse=True,
    )


def _confidence_from_scores(
    scores: dict[str, float],
    sorted_cats: list[str],
    min_score_gap: float,
) -> tuple[str, float, str, float]:
    """Select the best category when its score gap is sufficient."""
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
        return _unknown_confidence()
    _log_confidence_result(scores, sorted_cats, best_cat, best_score)
    return (best_cat, best_score, runner_up_cat, runner_up_score)


def _log_confidence_result(scores: dict[str, float], sorted_cats: list[str], best_cat: str, best_score: float) -> None:
    """Log confidence result to make the resolution decision observable."""
    logger.info(
        "Heuristic scoring result: %s. Best: %s (%.2f)",
        scores,
        best_cat,
        best_score,
    )
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("Heuristic top-3: %s", [(c, scores[c]) for c in sorted_cats[:3]])


@dataclass(frozen=True)
class HeuristicScorer:
    """Score document text against configured heuristic rules."""

    rules: list[HeuristicRule]
    _parent_map: dict[str, str] = field(default_factory=dict, init=False, repr=False)
    _rules_by_category: dict[str, list[HeuristicRule]] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        """Precompute category groups and parent mappings."""
        cache: dict[str, str] = {}
        grouped: dict[str, list[HeuristicRule]] = {}
        for rule in self.rules:
            if rule.parent is not None and rule.parent.strip():
                cache[rule.category] = rule.parent.strip()
            grouped.setdefault(rule.category, []).append(rule)
        object.__setattr__(self, "_parent_map", cache)
        object.__setattr__(self, "_rules_by_category", grouped)

    def best_category(
        self,
        text: str,
        *,
        language: str | None = None,
    ) -> str:
        """Return the highest-confidence category from the scored rules."""
        cat, _best, _runner_cat, _runner_up = self.best_category_with_confidence(
            text,
            ConfidenceOptions(language=language, min_score_gap=0.0),
        )
        return cat

    def best_category_with_confidence(
        self,
        text: str,
        options: ConfidenceOptions | None = None,
    ) -> tuple[str, float, str, float]:
        """
        Returns (category, best_score, runner_up_cat, runner_up_score).
        If no rule matches or min_score_gap not met: 'unknown', 0.0, 'unknown', 0.0.
        """
        opts = options or ConfidenceOptions()
        if text is None or not isinstance(text, str):
            return _unknown_confidence()
        scores = _score_text(
            text,
            self.rules,
            opts.language,
            ScoringOptions(
                rules_by_category=self._rules_by_category,
                title_weight_region=opts.title_weight_region,
                title_weight_factor=opts.title_weight_factor,
                max_score_per_category=opts.max_score_per_category,
            ),
        )
        if not scores:
            return _unknown_confidence()
        sorted_cats = _rank_scored_categories(scores, self._category_to_parent())
        return _confidence_from_scores(scores, sorted_cats, opts.min_score_gap)

    def top_n_categories(
        self,
        text: str,
        options: TopNOptions | None = None,
    ) -> list[str]:
        """Return up to n category names by score (best first). No min_score_gap."""
        opts = options or TopNOptions()
        if not text or not isinstance(text, str) or opts.n <= 0:
            return []
        scores = _score_text(
            text,
            self.rules,
            opts.language,
            ScoringOptions(
                rules_by_category=self._rules_by_category,
                title_weight_region=opts.title_weight_region,
                title_weight_factor=opts.title_weight_factor,
                max_score_per_category=opts.max_score_per_category,
            ),
        )
        if not scores:
            return []
        sorted_cats = _rank_scored_categories(scores, self._category_to_parent())
        return sorted_cats[: opts.n]

    def all_categories(self) -> frozenset[str]:
        """Return the set of all category names from rules (for constrained LLM)."""
        return frozenset(rule.category for rule in self.rules)

    def category_parent_map(self) -> dict[str, str]:
        """Return category-to-parent mappings from loaded rules."""
        return dict(self._parent_map)

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
