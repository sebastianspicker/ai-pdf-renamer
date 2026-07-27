"""Optional processing rules: skip LLM by heuristic category, force category by pattern, skip files by pattern."""

from __future__ import annotations

import fnmatch
import json
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ProcessingRules:
    """Loaded processing rules. All lists may be empty."""

    skip_llm_if_heuristic_category: list[str]
    force_category_by_pattern: list[dict[str, str]]
    skip_files_by_pattern: list[str]
    allowed_categories: list[str]


def _rules_file_path(path: str | Path, *, raise_on_error: bool) -> Path | None:
    """Resolve an existing rules path, optionally raising when absent."""
    p = Path(path).expanduser().resolve()
    if p.exists():
        return p
    if raise_on_error:
        raise ValueError(f"Rules file not found: {p}")
    logger.debug("Rules file not found: %s", p)
    return None


def _read_rules_json(path: Path, *, raise_on_error: bool) -> dict[str, object] | None:
    """Read a rules JSON object, optionally raising on invalid input."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        if raise_on_error:
            raise ValueError(f"Could not load processing rules from {path}: {e}") from e
        logger.warning("Could not load processing rules from %s: %s. Proceeding with no rules.", path, e)
        return None
    if isinstance(data, dict):
        return data
    if raise_on_error:
        raise ValueError(f"Processing rules at {path} must be a JSON object.")
    logger.warning("Processing rules at %s is not a JSON object. Proceeding with no rules.", path)
    return None


def _string_list(value: object) -> list[str]:
    """Normalize a JSON list to non-empty strings."""
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if item and str(item).strip()]


def _force_category_rules(value: object) -> list[dict[str, str]]:
    """Keep force-category entries containing string pattern and category fields."""
    if not isinstance(value, list):
        return []
    return [
        item
        for item in value
        if isinstance(item, dict) and isinstance(item.get("pattern"), str) and isinstance(item.get("category"), str)
    ]


def _processing_rules_from_data(data: dict[str, object]) -> ProcessingRules:
    """Build ProcessingRules from normalized JSON fields."""
    return ProcessingRules(
        skip_llm_if_heuristic_category=_string_list(data.get("skip_llm_if_heuristic_category")),
        force_category_by_pattern=_force_category_rules(data.get("force_category_by_pattern")),
        skip_files_by_pattern=_string_list(data.get("skip_files_by_pattern")),
        allowed_categories=_string_list(data.get("allowed_categories")),
    )


def load_processing_rules(path: str | Path | None, *, raise_on_error: bool = False) -> ProcessingRules | None:
    """
    Load processing rules from a JSON file. Returns None if path is None, file is missing, or invalid.
    On error, log and return None (caller proceeds with no rules).
    """
    if path is None:
        return None
    p = _rules_file_path(path, raise_on_error=raise_on_error)
    if p is None:
        return None
    data = _read_rules_json(p, raise_on_error=raise_on_error)
    if data is None:
        return None
    return _processing_rules_from_data(data)


def force_category_for_basename(rules: ProcessingRules | None, basename: str) -> str | None:
    """If rules have force_category_by_pattern and basename matches the first matching pattern, return that category."""
    if not rules or not rules.force_category_by_pattern:
        return None
    # Rule globs are operator-authored, so match basename casing leniently.
    basename_lower = basename.lower()
    for entry in rules.force_category_by_pattern:
        pattern = entry.get("pattern") or ""
        category = entry.get("category") or ""
        if fnmatch.fnmatchcase(basename_lower, pattern.lower()):
            return category.strip() or None
    return None


def should_skip_file_by_rules(rules: ProcessingRules | None, basename: str) -> bool:
    """True if rules have skip_files_by_pattern and basename matches any pattern."""
    if not rules or not rules.skip_files_by_pattern:
        return False
    # Rule globs are operator-authored, so match basename casing leniently.
    basename_lower = basename.lower()
    return any(fnmatch.fnmatchcase(basename_lower, p.lower()) for p in rules.skip_files_by_pattern)
