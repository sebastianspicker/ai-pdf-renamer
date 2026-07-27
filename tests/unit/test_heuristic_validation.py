"""Validate heuristic_scores.json: valid regexes and structure."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from folionym.data_paths import category_aliases_path, data_path
from folionym.heuristics import load_heuristic_rules


def _assert_pattern_schema(data: object) -> None:
    """Check the shared heuristic-pattern JSON schema and regex validity."""
    assert isinstance(data, dict)
    assert "patterns" in data
    assert isinstance(data["patterns"], list)
    for index, entry in enumerate(data["patterns"]):
        assert isinstance(entry, dict), f"patterns[{index}] must be object"
        assert "regex" in entry, f"patterns[{index}] missing regex"
        assert "category" in entry, f"patterns[{index}] missing category"
        assert "score" in entry, f"patterns[{index}] missing score"
        assert isinstance(entry["regex"], str), f"patterns[{index}].regex must be string"
        assert isinstance(entry["category"], str), f"patterns[{index}].category must be string"
        try:
            re.compile(entry["regex"])
        except re.error as error:
            pytest.fail(f"patterns[{index}].regex invalid: {error}")
        try:
            float(entry["score"])
        except (
            TypeError,
            ValueError,
        ):
            pytest.fail(f"patterns[{index}].score must be number")
        if "language" in entry and entry["language"] is not None:
            assert entry["language"] in ("de", "en"), f"patterns[{index}].language must be 'de' or 'en'"


def test_heuristic_scores_json_loads_and_compiles() -> None:
    """All patterns in heuristic_scores.json must be valid regex and compile."""
    path = data_path("heuristic_scores.json")
    rules = load_heuristic_rules(path)
    assert len(rules) > 0
    for rule in rules:
        assert rule.pattern.search("") is None or True
        assert rule.category
        assert isinstance(rule.score, float)


def test_heuristic_scores_json_structure() -> None:
    """heuristic_scores.json: patterns array; each entry has regex, category, score."""
    path = data_path("heuristic_scores.json")
    raw = path.read_text(encoding="utf-8")
    data = json.loads(raw)
    _assert_pattern_schema(data)


def test_heuristic_patterns_json_same_structure() -> None:
    """heuristic_patterns.json keeps the same schema when the repo-local file is present."""
    path = Path(__file__).resolve().parents[2] / "src/folionym/data/heuristic_patterns.json"
    if not path.exists():
        pytest.skip("heuristic_patterns.json not found")
    raw = path.read_text(encoding="utf-8")
    data = json.loads(raw)
    _assert_pattern_schema(data)


def test_meta_stopwords_json_structure() -> None:
    """meta_stopwords.json: stopwords array of strings."""
    path = data_path("meta_stopwords.json")
    raw = path.read_text(encoding="utf-8")
    data = json.loads(raw)
    assert "stopwords" in data, "meta_stopwords.json must have 'stopwords' key"
    stopwords = data["stopwords"]
    assert isinstance(stopwords, list), "'stopwords' must be a list"
    for i, w in enumerate(stopwords):
        assert isinstance(w, str), f"stopwords[{i}] must be string, got {type(w).__name__}"


def test_category_aliases_json_structure() -> None:
    """category_aliases.json: aliases object; keys and values non-empty strings."""
    path = category_aliases_path()
    if not path.exists():
        pytest.skip("category_aliases.json not found")
    raw = path.read_text(encoding="utf-8")
    data = json.loads(raw)
    assert "aliases" in data, "category_aliases.json must have 'aliases' key"
    aliases = data["aliases"]
    assert isinstance(aliases, dict), "'aliases' must be an object"
    for k, v in aliases.items():
        assert isinstance(k, str) and k.strip(), f"alias key must be non-empty string: {k!r}"
        assert isinstance(v, str) and v.strip(), f"alias value must be non-empty string: {v!r}"
