"""Tests for LLM orchestration functions in ai_pdf_renamer.llm.

All tests use MagicMock for LLMClient to avoid real HTTP/LLM calls.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from ai_pdf_renamer.llm import (
    _RETRY_TEMP_INCREMENT,
    complete_json_with_retry,
    get_document_analysis,
    get_document_category,
    get_document_filename_simple,
    get_document_keywords,
    get_document_summary,
    get_final_summary_tokens,
)
from ai_pdf_renamer.llm_schema import (
    DEFAULT_LLM_CATEGORY,
    DEFAULT_LLM_SUMMARY,
    DocumentAnalysisResult,
)


def _make_client() -> MagicMock:
    """Create a MagicMock that satisfies the LLMClient protocol shape."""
    client = MagicMock()
    client.model = "test-model"
    client.base_url = "http://test:8080"
    return client


# ---------------------------------------------------------------------------
# complete_json_with_retry
# ---------------------------------------------------------------------------


class TestCompleteJsonWithRetry:
    def test_success_on_first_call(self) -> None:
        """Valid JSON on first call returns immediately."""
        client = _make_client()
        client.complete.return_value = '{"summary":"A test summary"}'
        result = complete_json_with_retry(client, "prompt")
        assert result == '{"summary":"A test summary"}'
        assert client.complete.call_count == 1

    def test_invalid_then_valid(self) -> None:
        """Invalid JSON first, valid on retry with higher temp."""
        client = _make_client()
        client.complete.side_effect = [
            "not json at all",
            '{"summary":"ok"}',
        ]
        result = complete_json_with_retry(client, "prompt", temperature=0.0, max_retries=3)
        assert result == '{"summary":"ok"}'
        assert client.complete.call_count == 2
        # Second call should have incremented temperature
        second_call = client.complete.call_args_list[1]
        assert second_call.kwargs["temperature"] == pytest.approx(0.0 + _RETRY_TEMP_INCREMENT)

    def test_all_retries_fail(self) -> None:
        """All retries fail: returns last raw response."""
        client = _make_client()
        client.complete.side_effect = [
            "garbage1",
            "garbage2",
            "garbage3",
        ]
        result = complete_json_with_retry(client, "prompt", temperature=0.0, max_retries=3)
        # Returns the last raw response (not empty string)
        assert result == "garbage3"
        assert client.complete.call_count == 3

    def test_all_retries_fail_returns_last_response(self) -> None:
        """When all retries fail the function returns the last raw LLM output."""
        client = _make_client()
        client.complete.return_value = "still not json"
        result = complete_json_with_retry(client, "prompt", max_retries=2)
        assert result == "still not json"
        assert client.complete.call_count == 2

    def test_json_mode_only_one_attempt(self) -> None:
        """When json_mode=True, effective_retries is 1 so only one call is made."""
        client = _make_client()
        client.complete.return_value = "bad"
        complete_json_with_retry(client, "p", max_retries=5, json_mode=True)
        assert client.complete.call_count == 1
        # response_format should be set
        assert client.complete.call_args.kwargs["response_format"] == {"type": "json_object"}

    def test_json_inside_code_fence(self) -> None:
        """JSON wrapped in ```json ... ``` fences is accepted."""
        client = _make_client()
        client.complete.return_value = '```json\n{"key":"val"}\n```'
        result = complete_json_with_retry(client, "prompt")
        # The raw response is returned (extraction happens at parse_json_field level)
        assert result == '```json\n{"key":"val"}\n```'
        assert client.complete.call_count == 1

    def test_temperature_increments_each_retry(self) -> None:
        """Temperature grows by _RETRY_TEMP_INCREMENT per failed attempt."""
        client = _make_client()
        client.complete.side_effect = ["bad", "bad", '{"ok":1}']
        complete_json_with_retry(client, "p", temperature=0.1, max_retries=3)
        temps = [c.kwargs["temperature"] for c in client.complete.call_args_list]
        assert temps == pytest.approx([0.1, 0.1 + _RETRY_TEMP_INCREMENT, 0.1 + 2 * _RETRY_TEMP_INCREMENT])


# ---------------------------------------------------------------------------
# get_document_summary
# ---------------------------------------------------------------------------


class TestGetDocumentSummary:
    def test_short_text_returns_summary(self) -> None:
        """Text under CONTEXT_128K_MAX_CHARS_SINGLE gets short-text path."""
        client = _make_client()
        # _try_prompts_for_key will call complete_json_with_retry which calls client.complete
        client.complete.return_value = '{"summary":"Invoice from Amazon for headphones."}'
        text = "This is a test document content. " * 10  # well above 50 chars
        result = get_document_summary(client, text, language="en")
        assert result == "Invoice from Amazon for headphones."
        assert client.complete.call_count >= 1

    def test_returns_na_for_empty(self) -> None:
        """Empty content returns 'na' without calling LLM."""
        client = _make_client()
        result = get_document_summary(client, "", language="en")
        assert result == "na"
        client.complete.assert_not_called()

    def test_returns_na_for_short_content(self) -> None:
        """Content under 50 chars returns 'na'."""
        client = _make_client()
        result = get_document_summary(client, "tiny", language="en")
        assert result == "na"
        client.complete.assert_not_called()

    def test_returns_na_for_non_string(self) -> None:
        """Non-string input returns 'na'."""
        client = _make_client()
        result = get_document_summary(client, 12345, language="en")  # type: ignore[arg-type]
        assert result == "na"
        client.complete.assert_not_called()

    def test_default_summary_when_llm_returns_garbage(self) -> None:
        """When LLM returns unparseable garbage, summary defaults to DEFAULT_LLM_SUMMARY."""
        client = _make_client()
        client.complete.return_value = "I cannot parse anything"
        text = "A reasonably long document content for testing. " * 5
        result = get_document_summary(client, text, language="en")
        assert result == DEFAULT_LLM_SUMMARY

    def test_german_language_prompts(self) -> None:
        """German prompts are used when language='de'."""
        client = _make_client()
        client.complete.return_value = '{"summary":"Eine Rechnung von Amazon."}'
        text = "Dies ist ein Testdokument mit ausreichend Inhalt. " * 5
        result = get_document_summary(client, text, language="de")
        assert result == "Eine Rechnung von Amazon."


# ---------------------------------------------------------------------------
# get_document_keywords
# ---------------------------------------------------------------------------


class TestGetDocumentKeywords:
    def test_valid_keywords(self) -> None:
        """Returns keyword list from JSON response."""
        client = _make_client()
        client.complete.return_value = '{"keywords":["invoice","Amazon","headphones","order","2024"]}'
        result = get_document_keywords(client, "An invoice from Amazon for headphones.", language="en")
        assert result == ["invoice", "Amazon", "headphones", "order", "2024"]

    def test_empty_response_returns_none(self) -> None:
        """Unparseable response returns None."""
        client = _make_client()
        client.complete.return_value = ""
        result = get_document_keywords(client, "some summary", language="en")
        assert result is None

    def test_garbage_response_returns_none(self) -> None:
        """Garbage LLM output returns None."""
        client = _make_client()
        client.complete.return_value = "no json here"
        result = get_document_keywords(client, "some summary", language="en")
        assert result is None

    def test_german_with_category_hint(self) -> None:
        """German mode with suggested_category produces keywords."""
        client = _make_client()
        client.complete.return_value = '{"keywords":["Rechnung","Amazon","Kopfhörer"]}'
        result = get_document_keywords(
            client,
            "Eine Rechnung von Amazon.",
            language="de",
            suggested_category="Rechnung",
        )
        assert result == ["Rechnung", "Amazon", "Kopfhörer"]

    def test_empty_keyword_list_returns_none(self) -> None:
        """An empty keywords array returns None (via validate_llm_document_result)."""
        client = _make_client()
        client.complete.return_value = '{"keywords":[]}'
        result = get_document_keywords(client, "some summary", language="en")
        assert result is None


# ---------------------------------------------------------------------------
# get_document_category
# ---------------------------------------------------------------------------


class TestGetDocumentCategory:
    def test_with_allowed_categories(self) -> None:
        """When allowed_categories is set, category is constrained."""
        client = _make_client()
        client.complete.return_value = '{"category":"Invoice"}'
        result = get_document_category(
            client,
            summary="An invoice from Amazon.",
            keywords=["invoice", "Amazon"],
            language="en",
            allowed_categories=["Invoice", "Contract", "Letter"],
        )
        assert result == "Invoice"

    def test_unconstrained(self) -> None:
        """No allowed_categories: LLM picks freely."""
        client = _make_client()
        client.complete.return_value = '{"category":"Receipt"}'
        result = get_document_category(
            client,
            summary="A grocery store receipt.",
            keywords=["grocery", "receipt"],
            language="en",
        )
        assert result == "Receipt"

    def test_garbage_returns_default_category(self) -> None:
        """Garbage LLM output returns DEFAULT_LLM_CATEGORY."""
        client = _make_client()
        client.complete.return_value = "random nonsense"
        result = get_document_category(
            client,
            summary="test",
            keywords=["test"],
            language="en",
        )
        assert result == DEFAULT_LLM_CATEGORY

    def test_too_long_category_treated_as_invalid(self) -> None:
        """Category exceeding 80 chars is treated as invalid."""
        client = _make_client()
        long_cat = "A" * 100
        client.complete.return_value = json.dumps({"category": long_cat})
        result = get_document_category(
            client,
            summary="test",
            keywords=["test"],
            language="en",
        )
        assert result == DEFAULT_LLM_CATEGORY

    def test_german_with_suggested_categories(self) -> None:
        """German mode with suggested_categories."""
        client = _make_client()
        client.complete.return_value = '{"category":"Vertrag"}'
        result = get_document_category(
            client,
            summary="Ein Mietvertrag.",
            keywords=["Vertrag", "Miete"],
            language="de",
            suggested_categories=["Rechnung", "Vertrag", "Brief"],
        )
        assert result == "Vertrag"


# ---------------------------------------------------------------------------
# get_document_analysis (single-call)
# ---------------------------------------------------------------------------


class TestGetDocumentAnalysis:
    def test_single_call_full_json(self) -> None:
        """complete() returns full JSON with summary/keywords/category."""
        client = _make_client()
        response = json.dumps(
            {
                "summary": "Invoice from Amazon for electronics.",
                "keywords": ["invoice", "Amazon", "electronics", "order", "2024"],
                "category": "Invoice",
            }
        )
        client.complete.return_value = response
        result = get_document_analysis(
            client,
            "This is a detailed invoice document from Amazon. " * 10,
            language="en",
        )
        assert isinstance(result, DocumentAnalysisResult)
        assert result.summary == "Invoice from Amazon for electronics."
        assert result.keywords == ["invoice", "Amazon", "electronics", "order", "2024"]
        assert result.category == "Invoice"

    def test_bad_json_returns_defaults(self) -> None:
        """Bad JSON from LLM returns default DocumentAnalysisResult."""
        client = _make_client()
        client.complete.return_value = "This is not JSON at all."
        result = get_document_analysis(
            client,
            "A sufficiently long document to pass the 50-char minimum. " * 3,
            language="en",
        )
        assert isinstance(result, DocumentAnalysisResult)
        assert result.summary == DEFAULT_LLM_SUMMARY
        assert result.category == DEFAULT_LLM_CATEGORY
        assert result.keywords == []

    def test_short_content_returns_defaults(self) -> None:
        """Content under 50 chars returns default result without calling LLM."""
        client = _make_client()
        result = get_document_analysis(client, "tiny", language="en")
        assert result.summary == DEFAULT_LLM_SUMMARY
        client.complete.assert_not_called()

    def test_empty_content_returns_defaults(self) -> None:
        """Empty content returns default result."""
        client = _make_client()
        result = get_document_analysis(client, "", language="en")
        assert result.summary == DEFAULT_LLM_SUMMARY
        client.complete.assert_not_called()

    def test_non_string_content_returns_defaults(self) -> None:
        """Non-string content returns default result."""
        client = _make_client()
        result = get_document_analysis(client, None, language="en")  # type: ignore[arg-type]
        assert result.summary == DEFAULT_LLM_SUMMARY
        client.complete.assert_not_called()

    def test_partial_json_fills_defaults(self) -> None:
        """JSON with only summary fills defaults for missing fields."""
        client = _make_client()
        client.complete.return_value = '{"summary":"Only a summary here."}'
        result = get_document_analysis(
            client,
            "A sufficiently long document to pass the 50-char minimum. " * 3,
            language="en",
        )
        assert result.summary == "Only a summary here."
        assert result.category == DEFAULT_LLM_CATEGORY
        assert result.keywords == []

    def test_with_allowed_categories(self) -> None:
        """allowed_categories is passed through to prompt builder."""
        client = _make_client()
        response = json.dumps(
            {
                "summary": "A contract.",
                "keywords": ["contract"],
                "category": "Contract",
            }
        )
        client.complete.return_value = response
        result = get_document_analysis(
            client,
            "This is a legal contract document with sufficient text. " * 5,
            language="en",
            allowed_categories=["Invoice", "Contract", "Letter"],
        )
        assert result.category == "Contract"
        # Verify the prompt contained the allowed categories
        prompt_arg = client.complete.call_args[0][0]
        assert "Invoice" in prompt_arg
        assert "Contract" in prompt_arg

    def test_json_mode_passes_response_format(self) -> None:
        """json_mode=True passes response_format to complete()."""
        client = _make_client()
        client.complete.return_value = '{"summary":"test","keywords":[],"category":"Test"}'
        get_document_analysis(
            client,
            "Enough content to pass the minimum length check easily. " * 3,
            language="en",
            json_mode=True,
        )
        call_kwargs = client.complete.call_args.kwargs
        assert call_kwargs["response_format"] == {"type": "json_object"}


# ---------------------------------------------------------------------------
# get_final_summary_tokens
# ---------------------------------------------------------------------------


class TestGetFinalSummaryTokens:
    def test_valid_tokens(self) -> None:
        """Returns token list from LLM response."""
        client = _make_client()
        client.complete.return_value = '{"final_summary":"invoice,Amazon,electronics,2024"}'
        result = get_final_summary_tokens(
            client,
            summary="An invoice from Amazon.",
            keywords=["invoice", "Amazon"],
            category="Invoice",
            language="en",
        )
        assert result == ["invoice", "Amazon", "electronics", "2024"]

    def test_tokens_capped_at_five(self) -> None:
        """Only first 5 tokens are returned."""
        client = _make_client()
        tokens = ",".join(f"kw{i}" for i in range(10))
        client.complete.return_value = json.dumps({"final_summary": tokens})
        result = get_final_summary_tokens(
            client,
            summary="test",
            keywords=["test"],
            category="Test",
            language="en",
        )
        assert result is not None
        assert len(result) == 5

    def test_garbage_returns_none(self) -> None:
        """Unparseable response returns None."""
        client = _make_client()
        client.complete.return_value = "not json"
        result = get_final_summary_tokens(
            client,
            summary="test",
            keywords=["test"],
            category="Test",
            language="en",
        )
        assert result is None

    def test_german_prompts(self) -> None:
        """German language prompts produce tokens."""
        client = _make_client()
        client.complete.return_value = '{"final_summary":"Rechnung,Amazon,Elektronik"}'
        result = get_final_summary_tokens(
            client,
            summary="Eine Rechnung von Amazon.",
            keywords=["Rechnung", "Amazon"],
            category="Rechnung",
            language="de",
        )
        assert result == ["Rechnung", "Amazon", "Elektronik"]


# ---------------------------------------------------------------------------
# get_document_filename_simple
# ---------------------------------------------------------------------------


class TestGetDocumentFilenameSimple:
    def test_simple_filename(self) -> None:
        """LLM returns a clean filename; it passes through sanitization."""
        client = _make_client()
        client.complete.return_value = "INVOICE_AMAZON_HEADPHONES_2024-01-15"
        result = get_document_filename_simple(
            client,
            "An invoice from Amazon for headphones dated January 15, 2024.",
            language="en",
        )
        assert result == "INVOICE_AMAZON_HEADPHONES_2024-01-15"

    def test_empty_content_returns_document(self) -> None:
        """Empty content returns 'document' without calling LLM."""
        client = _make_client()
        result = get_document_filename_simple(client, "", language="en")
        assert result == "document"
        client.complete.assert_not_called()

    def test_whitespace_only_returns_document(self) -> None:
        """Whitespace-only content returns 'document'."""
        client = _make_client()
        result = get_document_filename_simple(client, "   \n\t  ", language="en")
        assert result == "document"
        client.complete.assert_not_called()

    def test_sanitizes_special_chars(self) -> None:
        """Special characters in LLM output are sanitized."""
        client = _make_client()
        client.complete.return_value = 'Invoice: Amazon <2024> "test"'
        result = get_document_filename_simple(client, "some content here", language="en")
        # sanitize_filename_from_llm replaces special chars with _
        assert ":" not in result
        assert "<" not in result
        assert ">" not in result
        assert '"' not in result

    def test_strips_pdf_extension(self) -> None:
        """LLM output ending in .pdf has extension stripped."""
        client = _make_client()
        client.complete.return_value = "INVOICE_AMAZON.pdf"
        result = get_document_filename_simple(client, "content for testing", language="en")
        assert not result.lower().endswith(".pdf")
        assert "INVOICE_AMAZON" in result

    def test_german_prompt(self) -> None:
        """German language uses German prompt."""
        client = _make_client()
        client.complete.return_value = "RECHNUNG_AMAZON_2024"
        result = get_document_filename_simple(
            client,
            "Eine Rechnung von Amazon.",
            language="de",
        )
        assert result == "RECHNUNG_AMAZON_2024"
        # Verify prompt contains German text
        prompt_arg = client.complete.call_args[0][0]
        assert "Dateinamen" in prompt_arg

    def test_llm_returns_none_gets_document(self) -> None:
        """If LLM returns empty string, sanitize_filename_from_llm returns 'document'."""
        client = _make_client()
        client.complete.return_value = ""
        result = get_document_filename_simple(client, "some valid content", language="en")
        assert result == "document"

    def test_trailing_dot_stripped(self) -> None:
        """Trailing dots are stripped by sanitize_filename_from_llm."""
        client = _make_client()
        client.complete.return_value = "INVOICE_AMAZON."
        result = get_document_filename_simple(client, "some content", language="en")
        assert not result.endswith(".")
