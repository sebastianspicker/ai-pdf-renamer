"""Verify category normalization, scoring, and conflict-resolution policies."""

from __future__ import annotations

import re

import pytest

from folionym.heuristic_scoring import ConfidenceOptions, TopNOptions
from folionym.heuristics import (
    CategoryCombineOptions,
    CategoryCombineParams,
    HeuristicRule,
    HeuristicScorer,
    combine_categories,
    normalize_llm_category,
)


@pytest.fixture
def invoice_receipt_scorer() -> HeuristicScorer:
    """Score the distinct invoice and receipt matches used by confidence tests."""
    rules = [
        HeuristicRule(pattern=re.compile("invoice", re.IGNORECASE), category="invoice", score=2.0),
        HeuristicRule(pattern=re.compile("receipt", re.IGNORECASE), category="receipt", score=5.0),
    ]
    return HeuristicScorer(rules=rules)


def test_heuristic_scorer_best_category(invoice_receipt_scorer: HeuristicScorer) -> None:
    assert invoice_receipt_scorer.best_category("This is an invoice and a receipt") == "receipt"


def test_heuristic_scorer_best_category_with_confidence(invoice_receipt_scorer: HeuristicScorer) -> None:
    assert invoice_receipt_scorer.best_category_with_confidence("This is an invoice and a receipt") == (
        "receipt",
        5.0,
        "invoice",
        2.0,
    )


def test_heuristic_scorer_min_score_gap_returns_unknown() -> None:
    rules = [
        HeuristicRule(pattern=re.compile("invoice", re.IGNORECASE), category="invoice", score=2.0),
        HeuristicRule(pattern=re.compile("receipt", re.IGNORECASE), category="receipt", score=2.0),
    ]
    scorer = HeuristicScorer(rules=rules)
    cat, best, runner_cat, runner_score = scorer.best_category_with_confidence(
        "invoice and receipt", options=ConfidenceOptions(min_score_gap=1.0)
    )
    assert cat == "unknown"
    assert best == 0.0
    assert runner_cat == "unknown"
    assert runner_score == 0.0


def test_heuristic_scorer_language_filter() -> None:
    rules = [
        HeuristicRule(pattern=re.compile("rechnung", re.IGNORECASE), category="invoice", score=4.0, language="de"),
        HeuristicRule(pattern=re.compile("invoice", re.IGNORECASE), category="invoice", score=4.0, language="en"),
    ]
    scorer = HeuristicScorer(rules=rules)
    assert scorer.best_category("Rechnung", language="de") == "invoice"
    assert scorer.best_category("Rechnung", language="en") == "unknown"
    assert scorer.best_category("invoice", language="en") == "invoice"
    assert scorer.best_category("invoice", language="de") == "unknown"


def test_combine_categories_prefers_llm_by_default() -> None:
    assert combine_categories("invoice", "unknown") == "invoice"
    options = CategoryCombineOptions(params=CategoryCombineParams(prefer_llm=True))
    assert combine_categories("invoice", "receipt", options) == "invoice"


def test_combine_categories_prefer_heuristic_when_asked() -> None:
    options = CategoryCombineOptions(params=CategoryCombineParams(prefer_llm=False))
    assert combine_categories("invoice", "receipt", options) == "receipt"


def test_normalize_llm_category() -> None:
    aliases = {"rechnung": "invoice", "lohnabrechnung": "payslip", "invoice": "invoice"}
    assert normalize_llm_category("Rechnung", _aliases=aliases) == "invoice"
    assert normalize_llm_category("invoice", _aliases=aliases) == "invoice"
    assert normalize_llm_category("Lohnabrechnung", _aliases=aliases) == "payslip"
    assert normalize_llm_category("unknown", _aliases=aliases) == "unknown"
    assert normalize_llm_category("something_else", _aliases=aliases) == "something_else"


def test_combine_categories_parent_match_uses_heuristic() -> None:
    parent_map = {"motor_insurance": "insurance", "health_insurance": "insurance"}
    options = CategoryCombineOptions(category_parent_map=parent_map)
    assert combine_categories("insurance", "motor_insurance", options=options) == "motor_insurance"
    assert combine_categories("motor_insurance", "motor_insurance", options=options) == "motor_insurance"


def test_combine_categories_min_heuristic_score_prefers_llm() -> None:
    assert (
        combine_categories(
            "contract",
            "invoice",
            options=CategoryCombineOptions(heuristic_score=1.0, params=CategoryCombineParams(min_heuristic_score=2.0)),
        )
        == "contract"
    )
    assert (
        combine_categories(
            "contract",
            "invoice",
            options=CategoryCombineOptions(
                heuristic_score=3.0, params=CategoryCombineParams(prefer_llm=False, min_heuristic_score=2.0)
            ),
        )
        == "invoice"
    )


def test_combine_categories_keyword_overlap_favors_llm_when_higher() -> None:
    context = "Kfz Versicherung premium motor insurance document"
    p = CategoryCombineParams(use_keyword_overlap=True)
    assert (
        combine_categories(
            "motor_insurance", "invoice", options=CategoryCombineOptions(context_for_overlap=context, params=p)
        )
        == "motor_insurance"
    )


def test_combine_categories_keyword_overlap_favors_heuristic_when_higher() -> None:
    context = "Rechnung Rechnungsnummer invoice total"
    p = CategoryCombineParams(use_keyword_overlap=True)
    assert (
        combine_categories(
            "motor_insurance", "invoice", options=CategoryCombineOptions(context_for_overlap=context, params=p)
        )
        == "invoice"
    )


def test_combine_categories_keyword_overlap_tie_keeps_heuristic() -> None:
    context = "document summary"
    p = CategoryCombineParams(use_keyword_overlap=True)
    assert (
        combine_categories("contract", "invoice", options=CategoryCombineOptions(context_for_overlap=context, params=p))
        == "invoice"
    )


@pytest.fixture
def parented_category_scorer() -> HeuristicScorer:
    """Return the single parented category shared by display-mode scenarios."""
    rules = [
        HeuristicRule(pattern=re.compile("x", re.IGNORECASE), category="motor_insurance", score=1.0, parent="insurance")
    ]
    return HeuristicScorer(rules=rules)


@pytest.mark.parametrize(
    ("display", "motor_result"),
    [("specific", "motor_insurance"), ("with_parent", "insurance_motor_insurance"), ("parent_only", "insurance")],
)
def test_get_display_category_modes(parented_category_scorer: HeuristicScorer, display: str, motor_result: str) -> None:
    assert parented_category_scorer.get_display_category("motor_insurance", display) == motor_result
    assert parented_category_scorer.get_display_category("invoice", display) == "invoice"


def test_top_n_categories() -> None:
    rules = [
        HeuristicRule(pattern=re.compile("a", re.IGNORECASE), category="cat_a", score=3.0),
        HeuristicRule(pattern=re.compile("b", re.IGNORECASE), category="cat_b", score=2.0),
        HeuristicRule(pattern=re.compile("c", re.IGNORECASE), category="cat_c", score=1.0),
    ]
    scorer = HeuristicScorer(rules=rules)
    assert scorer.top_n_categories("a b c", options=TopNOptions(n=3)) == ["cat_a", "cat_b", "cat_c"]
    assert scorer.top_n_categories("a b c", options=TopNOptions(n=2)) == ["cat_a", "cat_b"]
    assert scorer.top_n_categories("", options=TopNOptions(n=5)) == []


def test_all_categories() -> None:
    rules = [
        HeuristicRule(pattern=re.compile("x", re.IGNORECASE), category="invoice", score=1.0),
        HeuristicRule(pattern=re.compile("y", re.IGNORECASE), category="receipt", score=1.0),
    ]
    scorer = HeuristicScorer(rules=rules)
    assert scorer.all_categories() == frozenset({"invoice", "receipt"})


def test_combine_categories_sibling_uses_heuristic() -> None:
    parent_map = {"motor_insurance": "insurance", "health_insurance": "insurance"}
    assert (
        combine_categories(
            "health_insurance", "motor_insurance", options=CategoryCombineOptions(category_parent_map=parent_map)
        )
        == "motor_insurance"
    )


def test_combine_categories_high_confidence_override() -> None:
    p_above = CategoryCombineParams(heuristic_override_min_score=5.0, heuristic_override_min_gap=2.0)
    assert (
        combine_categories(
            "contract",
            "invoice",
            options=CategoryCombineOptions(heuristic_score=6.0, heuristic_gap=3.0, params=p_above),
        )
        == "invoice"
    )
    # Below threshold: no override; prefer_llm=False -> heuristic wins
    p_below = CategoryCombineParams(prefer_llm=False, heuristic_override_min_score=5.0, heuristic_override_min_gap=2.0)
    assert (
        combine_categories(
            "contract",
            "invoice",
            options=CategoryCombineOptions(heuristic_score=4.0, heuristic_gap=3.0, params=p_below),
        )
        == "invoice"
    )


def test_combine_categories_heuristic_score_weight() -> None:
    context = "document text"
    p = CategoryCombineParams(use_keyword_overlap=True, heuristic_score_weight=0.1)
    assert (
        combine_categories(
            "invoice",
            "receipt",
            options=CategoryCombineOptions(context_for_overlap=context, heuristic_score=10.0, params=p),
        )
        == "receipt"
    )
