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
    force_category_by_pattern: list[dict]
    skip_files_by_pattern: list[str]
    allowed_categories: list[str]


def load_processing_rules(path: str | Path | None) -> ProcessingRules | None:
    """
    Load processing rules from a JSON file. Returns None if path is None, file is missing, or invalid.
    On error, log and return None (caller proceeds with no rules).
    """
    if path is None:
        return None
    p = Path(path).expanduser().resolve()
    if not p.exists():
        logger.debug("Rules file not found: %s", p)
        return None
    try:
        raw = p.read_text(encoding="utf-8")
        data = json.loads(raw)
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("Could not load processing rules from %s: %s. Proceeding with no rules.", p, e)
        return None
    if not isinstance(data, dict):
        logger.warning("Processing rules at %s is not a JSON object. Proceeding with no rules.", p)
        return None

    skip_llm = data.get("skip_llm_if_heuristic_category")
    skip_llm = [str(x).strip() for x in skip_llm if x and str(x).strip()] if isinstance(skip_llm, list) else []

    force_cat = data.get("force_category_by_pattern")
    if isinstance(force_cat, list):
        force_cat = [
            item
            for item in force_cat
            if isinstance(item, dict) and isinstance(item.get("pattern"), str) and isinstance(item.get("category"), str)
        ]
    else:
        force_cat = []

    skip_files = data.get("skip_files_by_pattern")
    skip_files = [str(x).strip() for x in skip_files if x and str(x).strip()] if isinstance(skip_files, list) else []

    allowed_cats = data.get("allowed_categories")
    if isinstance(allowed_cats, list):
        allowed_cats = [str(x).strip() for x in allowed_cats if x and str(x).strip()]
    else:
        allowed_cats = []

    return ProcessingRules(
        skip_llm_if_heuristic_category=skip_llm,
        force_category_by_pattern=force_cat,
        skip_files_by_pattern=skip_files,
        allowed_categories=allowed_cats,
    )


def force_category_for_basename(rules: ProcessingRules | None, basename: str) -> str | None:
    """If rules have force_category_by_pattern and basename matches the first matching pattern, return that category."""
    if not rules or not rules.force_category_by_pattern:
        return None
    for entry in rules.force_category_by_pattern:
        pattern = entry.get("pattern") or ""
        category = entry.get("category") or ""
        if fnmatch.fnmatch(basename, pattern):
            return category.strip() or None
    return None


def should_skip_file_by_rules(rules: ProcessingRules | None, basename: str) -> bool:
    """True if rules have skip_files_by_pattern and basename matches any pattern."""
    if not rules or not rules.skip_files_by_pattern:
        return False
    return any(fnmatch.fnmatch(basename, p) for p in rules.skip_files_by_pattern)
