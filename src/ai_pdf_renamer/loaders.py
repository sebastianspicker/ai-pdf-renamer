"""
Load stopwords and heuristic scorer from data files. Used by renamer and filename.
"""

from __future__ import annotations

import json
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


@lru_cache(maxsize=32)
def _stopwords_cached(path_str: str) -> Stopwords:
    return load_meta_stopwords(Path(path_str))


def default_stopwords() -> Stopwords:
    return _stopwords_cached(str(data_path("meta_stopwords.json")))


@lru_cache(maxsize=32)
def _heuristic_scorer_cached(path_str: str, language: str) -> HeuristicScorer:
    rules = load_heuristic_rules_for_language(Path(path_str), language)
    return HeuristicScorer(rules)


def default_heuristic_scorer(language: str = "de") -> HeuristicScorer:
    return _heuristic_scorer_cached(str(data_path("heuristic_scores.json")), language)
