"""Canonical option-object tests for LLM orchestration."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from folionym.cache import ResponseCache
from folionym.llm import (
    _RETRY_TEMP_INCREMENT,
    complete_json_with_retry,
    get_document_analysis,
    get_document_category,
    get_document_keywords,
    get_document_summary,
)
from folionym.llm_options import (
    AnalysisGuidance,
    AnalysisOptions,
    CategoryOptions,
    JsonCompletionOptions,
    KeywordsOptions,
    LlmCacheOptions,
    LlmPromptOptions,
    SimpleFilenameOptions,
    SummaryOptions,
)
from folionym.llm_schema import DEFAULT_LLM_CATEGORY, DEFAULT_LLM_SUMMARY
from folionym.llm_simple_filename import get_document_filename_simple


def _client() -> MagicMock:
    client = MagicMock()
    client.model = "test-model"
    client.base_url = "http://test:8080"
    return client


def test_json_completion_uses_options_and_retries() -> None:
    client = _client()
    client.complete.side_effect = ["bad", '{"summary":"ok"}']
    result = complete_json_with_retry(client, "prompt", JsonCompletionOptions(temperature=0.1, max_retries=3))
    assert result == '{"summary":"ok"}'
    assert [call.kwargs["temperature"] for call in client.complete.call_args_list] == pytest.approx(
        [0.1, 0.1 + _RETRY_TEMP_INCREMENT]
    )


def test_json_completion_json_mode_limits_attempts() -> None:
    client = _client()
    client.complete.return_value = "bad"
    complete_json_with_retry(client, "prompt", JsonCompletionOptions(json_mode=True, max_retries=5))
    assert client.complete.call_count == 1
    assert client.complete.call_args.kwargs["response_format"] == {"type": "json_object"}


def test_summary_uses_prompt_options_and_rejects_short_content() -> None:
    client = _client()
    client.complete.return_value = '{"summary":"Invoice from Amazon."}'
    options = SummaryOptions(prompt=LlmPromptOptions(language="en"))
    assert get_document_summary(client, "invoice contents " * 10, options) == "Invoice from Amazon."
    assert get_document_summary(client, "tiny", options) == "na"


def test_keywords_and_category_use_canonical_options() -> None:
    client = _client()
    client.complete.return_value = '{"keywords":["invoice","Amazon"]}'
    keywords = get_document_keywords(client, "An invoice from Amazon.", KeywordsOptions(language="en"))
    assert keywords == ("invoice", "Amazon")
    client.complete.return_value = '{"category":"Invoice"}'
    category = get_document_category(
        client,
        summary="An invoice from Amazon.",
        keywords=["invoice", "Amazon"],
        options=CategoryOptions(language="en", allowed_categories=["Invoice", "Contract"]),
    )
    assert category == "Invoice"


def test_analysis_uses_nested_prompt_limits_guidance_and_cache_options() -> None:
    client = _client()
    client.complete.return_value = json.dumps({"summary": "Invoice.", "keywords": ["invoice"], "category": "Invoice"})
    cache = ResponseCache()
    options = AnalysisOptions(
        prompt=LlmPromptOptions(language="en", lenient_json=True),
        guidance=AnalysisGuidance(allowed_categories=["Invoice", "Contract"]),
        cache_options=LlmCacheOptions(cache=cache, cache_key_base="file:abc"),
        json_mode=True,
    )
    first = get_document_analysis(client, "invoice details " * 10, options)
    second = get_document_analysis(client, "invoice details " * 10, options)
    assert first.category == second.category == "Invoice"
    assert client.complete.call_count == 1
    assert client.complete.call_args.kwargs["response_format"] == {"type": "json_object"}


def test_analysis_defaults_for_invalid_or_short_content() -> None:
    client = _client()
    client.complete.return_value = "not json"
    options = AnalysisOptions(prompt=LlmPromptOptions(language="en"))
    assert get_document_analysis(client, "document contents " * 10, options).summary == DEFAULT_LLM_SUMMARY
    result = get_document_analysis(client, "tiny", options)
    assert result.summary == DEFAULT_LLM_SUMMARY
    assert result.category == DEFAULT_LLM_CATEGORY


def test_simple_filename_uses_simple_filename_options() -> None:
    client = _client()
    client.complete.return_value = "INVOICE_AMAZON.pdf"
    result = get_document_filename_simple(client, "An invoice from Amazon.", SimpleFilenameOptions(language="en"))
    assert result == "INVOICE_AMAZON"
