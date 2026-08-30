"""
Load stopwords and heuristic scorer from data files. Used by renamer and filename.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from ..infrastructure.resources import data_path, load_json_data
from .heuristics import HeuristicScorer, load_heuristic_rules_for_language
from .tokens import Stopwords


def load_meta_stopwords(path: str | Path) -> Stopwords:
    """Load meta stopwords from configured data."""
    path_obj = Path(path)
    data = load_json_data(path_obj)
    stopword_list = data.get("stopwords", [])
    if not isinstance(stopword_list, list):
        stopword_list = []
    words = {str(w).lower() for w in stopword_list if str(w).strip()}
    return Stopwords(words=words)


def _file_mtime(path_str: str) -> float:
    """Get file modification time; return 0.0 on error."""
    try:
        return os.path.getmtime(path_str)
    except OSError:
        return 0.0


# Include mtime in the cache key so local data-file edits are visible to watch mode.
@lru_cache(maxsize=32)
def _stopwords_cached(path_str: str, _mtime: float = 0.0) -> Stopwords:
    """Load stopwords through the mtime-sensitive cache."""
    return load_meta_stopwords(Path(path_str))


stopwords_cached = _stopwords_cached


def default_stopwords() -> Stopwords:
    """Return stopwords from the packaged default data file."""
    path_str = str(data_path("meta_stopwords.json"))
    return _stopwords_cached(path_str, _file_mtime(path_str))


@lru_cache(maxsize=32)
def _heuristic_scorer_cached(path_str: str, language: str, _mtime: float = 0.0) -> HeuristicScorer:
    """Build a cached heuristic scorer for one file and language."""
    rules = load_heuristic_rules_for_language(Path(path_str), language)
    return HeuristicScorer(rules)


heuristic_scorer_cached = _heuristic_scorer_cached


def default_heuristic_scorer(language: str = "de") -> HeuristicScorer:
    """Return a scorer using packaged heuristic rules for the language."""
    path_str = str(data_path("heuristic_scores.json"))
    return _heuristic_scorer_cached(path_str, language, _file_mtime(path_str))
