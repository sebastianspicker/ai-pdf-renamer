"""
Load stopwords and heuristic scorer from data files. Used by renamer and filename.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

from .data_paths import data_path
from .heuristics import HeuristicScorer, load_heuristic_rules_for_language
from .text_utils import Stopwords


def load_meta_stopwords(path: str | Path) -> Stopwords:
    path_obj = Path(path)
    try:
        text = path_obj.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"Could not read data file at {path_obj.absolute()}: {exc!s}") from exc
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in data file at {path_obj.absolute()}. {exc!s}") from exc
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


# P2: Include mtime in cache key so cache is invalidated when file changes (watch mode)
@lru_cache(maxsize=32)
def _stopwords_cached(path_str: str, _mtime: float = 0.0) -> Stopwords:
    return load_meta_stopwords(Path(path_str))


def default_stopwords() -> Stopwords:
    path_str = str(data_path("meta_stopwords.json"))
    return _stopwords_cached(path_str, _file_mtime(path_str))


@lru_cache(maxsize=32)
def _heuristic_scorer_cached(path_str: str, language: str, _mtime: float = 0.0) -> HeuristicScorer:
    rules = load_heuristic_rules_for_language(Path(path_str), language)
    return HeuristicScorer(rules)


def default_heuristic_scorer(language: str = "de") -> HeuristicScorer:
    path_str = str(data_path("heuristic_scores.json"))
    return _heuristic_scorer_cached(path_str, language, _file_mtime(path_str))
