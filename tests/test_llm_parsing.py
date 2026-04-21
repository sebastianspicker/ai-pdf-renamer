from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

from ai_pdf_renamer.llm_parsing import (
    TRUNCATION_SUFFIX,
    _extract_json_from_response,
    _lenient_extract_key_value,
    _sanitize_json_string_value,
    extract_and_validate_json,
    parse_json_field,
    truncate_for_llm,
)

# ---------------------------------------------------------------------------
# _extract_json_from_response
# ---------------------------------------------------------------------------


class TestExtractJsonFromResponse:
    def test_code_fence_json(self) -> None:
        resp = '```json\n{"summary":"test"}\n```'
        assert _extract_json_from_response(resp) == '{"summary":"test"}'

    def test_code_fence_no_json_tag(self) -> None:
        resp = '```\n{"summary":"test"}\n```'
        assert _extract_json_from_response(resp) == '{"summary":"test"}'

    def test_leading_prose(self) -> None:
        resp = 'Here is the result: {"summary":"test value"}'
        assert _extract_json_from_response(resp) == '{"summary":"test value"}'

    def test_nested_braces(self) -> None:
        resp = '{"summary":"value with {braces}"}'
        assert _extract_json_from_response(resp) == '{"summary":"value with {braces}"}'

    def test_no_json(self) -> None:
        resp = "This is just text with no braces"
        assert _extract_json_from_response(resp) == resp

    def test_empty_input(self) -> None:
        assert _extract_json_from_response("") == ""
        assert _extract_json_from_response("   ") == ""

    def test_unclosed_brace(self) -> None:
        resp = '{"summary":"test"'
        result = _extract_json_from_response(resp)
        assert result.startswith("{")

    def test_multiple_json_objects_returns_first(self) -> None:
        resp = '{"a":"1"} {"b":"2"}'
        assert _extract_json_from_response(resp) == '{"a":"1"}'

    def test_code_fence_non_json_content(self) -> None:
        resp = "```\nnot json content\n```"
        result = _extract_json_from_response(resp)
        # Code fence content doesn't start with {, falls through to brace search
        assert result == resp.strip()

    def test_escaped_quotes_in_string(self) -> None:
        resp = r'{"summary":"He said \"hello\""}'
        result = _extract_json_from_response(resp)
        assert result == resp


# ---------------------------------------------------------------------------
# _sanitize_json_string_value
# ---------------------------------------------------------------------------


class TestSanitizeJsonStringValue:
    def test_unescaped_quotes(self) -> None:
        raw = '{"summary":"He said "hello" to me"}'
        result = _sanitize_json_string_value(raw, key="summary")
        # Should be parseable now
        import json

        data = json.loads(result)
        assert data["summary"] == 'He said "hello" to me'

    def test_already_escaped_quotes(self) -> None:
        raw = r'{"summary":"He said \"hello\""}'
        result = _sanitize_json_string_value(raw, key="summary")
        import json

        data = json.loads(result)
        assert "hello" in data["summary"]

    def test_missing_key(self) -> None:
        raw = '{"other":"value"}'
        result = _sanitize_json_string_value(raw, key="summary")
        assert result == raw  # unchanged

    def test_missing_closing_brace(self) -> None:
        raw = '{"summary":"test value"'
        result = _sanitize_json_string_value(raw, key="summary")
        # Best-effort; should not crash
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# _lenient_extract_key_value
# ---------------------------------------------------------------------------


class TestLenientExtractKeyValue:
    def test_valid_json_key_value(self) -> None:
        text = '{"summary":"A good summary"}'
        assert _lenient_extract_key_value(text, "summary") == "A good summary"

    def test_malformed_json_with_key(self) -> None:
        text = 'blah blah "summary":"Found it" blah'
        assert _lenient_extract_key_value(text, "summary") == "Found it"

    def test_key_not_present(self) -> None:
        text = '{"other":"value"}'
        assert _lenient_extract_key_value(text, "summary") is None

    def test_empty_value(self) -> None:
        text = '{"summary":""}'
        assert _lenient_extract_key_value(text, "summary") is None

    def test_value_with_escaped_quotes(self) -> None:
        text = r'{"summary":"He said \"hi\""}'
        result = _lenient_extract_key_value(text, "summary")
        assert result is not None
        assert "hi" in result

    def test_known_array_key_returns_strings(self) -> None:
        text = 'noise "keywords": [" invoice ", "Amazon", " ", "2024"] trailing'
        assert _lenient_extract_key_value(text, "keywords") == ["invoice", "Amazon", "2024"]

    def test_known_final_summary_tokens_array_returns_strings(self) -> None:
        text = '{"final_summary_tokens": ["  invoice", "Amazon  "]'
        assert _lenient_extract_key_value(text, "final_summary_tokens") == ["invoice", "Amazon"]

    def test_unknown_array_key_is_not_salvaged(self) -> None:
        text = 'noise "other_list": ["a", "b"] trailing'
        assert _lenient_extract_key_value(text, "other_list") is None

    def test_known_array_rejects_non_string_items(self) -> None:
        text = 'noise "keywords": ["invoice", 42] trailing'
        assert _lenient_extract_key_value(text, "keywords") is None

    def test_no_quotes_at_all(self) -> None:
        text = "just plain text"
        assert _lenient_extract_key_value(text, "summary") is None


# ---------------------------------------------------------------------------
# parse_json_field
# ---------------------------------------------------------------------------


class TestParseJsonField:
    def test_string_value(self) -> None:
        assert parse_json_field('{"summary":"Hello"}', key="summary") == "Hello"

    def test_list_value(self) -> None:
        assert parse_json_field('{"keywords":["A","B"]}', key="keywords") == ["A", "B"]

    def test_na_returns_none(self) -> None:
        assert parse_json_field('{"summary":"na"}', key="summary") is None
        assert parse_json_field('{"summary":"NA"}', key="summary") is None

    def test_empty_string_returns_none(self) -> None:
        assert parse_json_field('{"summary":""}', key="summary") is None
        assert parse_json_field('{"summary":"   "}', key="summary") is None

    def test_none_response(self) -> None:
        assert parse_json_field(None, key="summary") is None

    def test_empty_response(self) -> None:
        assert parse_json_field("", key="summary") is None
        assert parse_json_field("   ", key="summary") is None

    def test_code_fence_wrapped(self) -> None:
        resp = '```json\n{"summary":"wrapped"}\n```'
        assert parse_json_field(resp, key="summary") == "wrapped"

    def test_leading_prose_with_json(self) -> None:
        resp = 'Here is the analysis: {"summary":"found it"}'
        assert parse_json_field(resp, key="summary") == "found it"

    def test_lenient_without_braces(self) -> None:
        resp = '"summary":"lenient value"'
        assert parse_json_field(resp, key="summary", lenient=True) == "lenient value"
        assert parse_json_field(resp, key="summary", lenient=False) is None

    def test_lenient_no_match(self) -> None:
        assert parse_json_field("No JSON here", key="summary", lenient=True) is None

    def test_lenient_known_array_value(self) -> None:
        resp = '"keywords":[" invoice ","Amazon"," ","2024"]'
        assert parse_json_field(resp, key="keywords", lenient=True) == ["invoice", "Amazon", "2024"]

    def test_lenient_final_summary_tokens_array_value(self) -> None:
        resp = 'prefix "final_summary_tokens":[" invoice ","Amazon"] suffix'
        assert parse_json_field(resp, key="final_summary_tokens", lenient=True) == ["invoice", "Amazon"]

    def test_lenient_unknown_array_key_stays_none(self) -> None:
        resp = '"other_list":["A","B"]'
        assert parse_json_field(resp, key="other_list", lenient=True) is None

    def test_invalid_json_returns_none(self) -> None:
        assert parse_json_field("not json", key="summary") is None

    def test_unescaped_quotes_sanitized(self) -> None:
        raw = '{"summary":"He said "hello" today"}'
        assert parse_json_field(raw, key="summary") == 'He said "hello" today'

    def test_list_with_empty_strings_filtered(self) -> None:
        assert parse_json_field('{"keywords":["A","","B"," "]}', key="keywords") == ["A", "B"]

    def test_list_all_empty_returns_none(self) -> None:
        assert parse_json_field('{"keywords":["","  "]}', key="keywords") is None

    def test_missing_key_returns_none(self) -> None:
        assert parse_json_field('{"other":"value"}', key="summary") is None

    def test_non_string_non_list_value_returns_none(self) -> None:
        assert parse_json_field('{"summary":42}', key="summary") is None
        assert parse_json_field('{"summary":true}', key="summary") is None


class TestExtractAndValidateJson:
    def test_extract_and_validate_json_valid_object(self) -> None:
        result = extract_and_validate_json('```json\n{"summary":"ok"}\n```', expected_keys={"summary"})
        assert result["summary"] == "ok"

    def test_extract_and_validate_json_lenient_fallback(self) -> None:
        result = extract_and_validate_json('"summary":"fallback"', expected_keys={"summary"}, lenient_keys={"summary"})
        assert result["summary"] == "fallback"

    def test_extract_and_validate_json_lenient_array_fallback(self) -> None:
        result = extract_and_validate_json(
            '"keywords":[" invoice ","Amazon","2024"]',
            expected_keys={"keywords"},
            lenient_keys={"keywords"},
        )
        assert result["keywords"] == ["invoice", "Amazon", "2024"]

    def test_extract_and_validate_json_lenient_final_summary_tokens_array(self) -> None:
        result = extract_and_validate_json(
            'prefix "final_summary_tokens":[" invoice ","Amazon"] suffix',
            expected_keys={"final_summary_tokens"},
            lenient_keys={"final_summary_tokens"},
        )
        assert result["final_summary_tokens"] == ["invoice", "Amazon"]

    def test_extract_and_validate_json_reports_context(self) -> None:
        try:
            extract_and_validate_json("not json", expected_keys={"summary", "category"})
        except ValueError as exc:
            message = str(exc)
            assert "summary" in message
            assert "category" in message
            assert "received" in message.lower()
        else:
            raise AssertionError("Expected ValueError")


# ---------------------------------------------------------------------------
# truncate_for_llm
# ---------------------------------------------------------------------------


class TestTruncateForLlm:
    def test_short_text_no_truncation(self) -> None:
        assert truncate_for_llm("short", None) == "short"
        assert truncate_for_llm("short", 100) == "short"

    def test_truncation_with_suffix(self) -> None:
        text = "x" * 200
        out = truncate_for_llm(text, 50)
        assert len(out) == 50
        assert out.endswith("\n[...]")

    def test_max_chars_none_no_truncation(self) -> None:
        text = "x" * 1000
        assert truncate_for_llm(text, None) == text

    def test_max_chars_zero_no_truncation(self) -> None:
        text = "x" * 1000
        assert truncate_for_llm(text, 0) == text

    def test_tiny_max_chars(self) -> None:
        out = truncate_for_llm("abcdefghijklmnopqrstuvwxyz", 3)
        assert out == "abc"

    def test_exact_length_no_truncation(self) -> None:
        text = "x" * 50
        assert truncate_for_llm(text, 50) == text

    def test_one_over_triggers_truncation(self) -> None:
        text = "x" * 51
        result = truncate_for_llm(text, 50)
        assert len(result) == 50
        assert result.endswith("\n[...]")


# ---------------------------------------------------------------------------
# parse_json_field — JSON salvage chain (lines 171-185)
# ---------------------------------------------------------------------------


class TestParseJsonFieldSalvageChain:
    """Cover the multi-stage JSON salvage chain inside parse_json_field."""

    def test_parse_json_field_salvage_via_sanitize(self) -> None:
        """Initial json.loads fails, but _sanitize_json_string_value fixes
        unescaped quotes so the second json.loads succeeds."""
        # The unescaped inner quotes make this invalid JSON, but sanitize can fix it.
        raw = '{"summary":"He said "hello" today"}'
        result = parse_json_field(raw, key="summary")
        assert result == 'He said "hello" today'

    def test_parse_json_field_salvage_via_extract(self) -> None:
        """Sanitize cannot fix the JSON, but _extract_json_from_response isolates
        valid JSON from the original response."""
        # Response starts with {"summary": " so it matches single_key_pattern.
        # The null byte causes json.loads to fail even after sanitize.
        # _extract_json_from_response finds the code-fenced valid JSON.
        response = '{"summary": "broken \x00"}\n```json\n{"summary": "rescued"}\n```'
        result = parse_json_field(response, key="summary")
        assert result == "rescued"

    def test_parse_json_field_all_salvage_fails(self, caplog: object) -> None:
        """Completely malformed response matching single-key pattern but
        neither sanitize nor extract can recover valid JSON → returns None
        with a warning logged."""
        # Must match single_key_pattern: starts with {"summary": "
        # Must fail json.loads after sanitize AND after extract.
        # _extract_json_from_response on this will find { and try brace matching,
        # but the result still won't parse.
        response = '{"summary": "\x00\x01\x02 unclosed'

        with patch("ai_pdf_renamer.llm_parsing.logger") as mock_logger:
            result = parse_json_field(response, key="summary")
            assert result is None
            mock_logger.warning.assert_called()

    def test_parse_json_field_non_matching_key_pattern(self) -> None:
        """Malformed JSON where the key pattern doesn't match single-key format
        → returns None immediately without attempting salvage (line 170-172)."""
        # This starts with { but doesn't match {"summary": " pattern
        # (different key or wrong format)
        response = '{"other_key": "value", broken}'
        with patch("ai_pdf_renamer.llm_parsing.logger") as mock_logger:
            result = parse_json_field(response, key="summary")
            assert result is None
            warning_call = mock_logger.warning.call_args
            assert warning_call is not None
            assert "LLM response could not be parsed as JSON; using fallback" in warning_call.args[0]


# ---------------------------------------------------------------------------
# truncate_for_llm — token-based truncation (lines 226-242)
# ---------------------------------------------------------------------------


def _make_mock_tiktoken(
    tokens_per_word: int = 1,
) -> tuple[MagicMock, MagicMock]:
    """Create a mock tiktoken module and encoding.

    The mock encoding treats each whitespace-separated word as `tokens_per_word`
    token(s). encode() returns a list of ints (one per word * tokens_per_word),
    decode() joins them back with spaces.
    """
    mock_enc = MagicMock()

    def _encode(text: str) -> list[int]:
        words = text.split()
        # Each word becomes tokens_per_word tokens: word_index repeated
        result: list[int] = []
        for idx, _w in enumerate(words):
            result.extend([idx] * tokens_per_word)
        return result

    def _decode(token_ids: list[int]) -> str:
        if not token_ids:
            return ""
        # Return N placeholder words where N = number of tokens / tokens_per_word.
        n_words = len(token_ids) // tokens_per_word if tokens_per_word else len(token_ids)
        return " ".join(f"w{i}" for i in range(n_words))

    mock_enc.encode = MagicMock(side_effect=_encode)
    mock_enc.decode = MagicMock(side_effect=_decode)

    mock_tiktoken = MagicMock()
    mock_tiktoken.get_encoding = MagicMock(return_value=mock_enc)
    mock_tiktoken.encoding_for_model = MagicMock(return_value=mock_enc)

    return mock_tiktoken, mock_enc


class TestTruncateForLlmTokenBased:
    """Cover the token-based truncation path (lines 225-242)."""

    def test_truncate_for_llm_token_based_under_limit(self) -> None:
        """When text is under max_tokens, it is returned unchanged."""
        mock_tiktoken, mock_enc = _make_mock_tiktoken()
        text = "hello world foo"  # 3 tokens (one per word)

        with patch.dict(sys.modules, {"tiktoken": mock_tiktoken}):
            result = truncate_for_llm(text, max_chars=None, max_tokens=10)

        assert result == text
        mock_tiktoken.get_encoding.assert_called_once_with("cl100k_base")
        mock_enc.encode.assert_called_with(text)

    def test_truncate_for_llm_token_based_over_limit(self) -> None:
        """When text exceeds max_tokens, it is truncated with the suffix."""
        mock_tiktoken, _mock_enc = _make_mock_tiktoken()
        text = "word0 word1 word2 word3 word4"  # 5 tokens

        with patch.dict(sys.modules, {"tiktoken": mock_tiktoken}):
            result = truncate_for_llm(text, max_chars=None, max_tokens=3)

        # 5 tokens > 3, so truncation happens.
        # suffix = TRUNCATION_SUFFIX = "\n[...]"
        # suffix tokens: "\n[...]".split() → ["[...]"] → 1 token
        # keep = max(1, 3 - 1) = 2
        # decode(tokens[:2]) → "w0 w1" (from our mock)
        # result = "w0 w1" + "\n[...]"
        assert TRUNCATION_SUFFIX in result
        # The truncated part should be shorter than the original
        assert result != text

    def test_truncate_for_llm_token_based_suffix_exceeds_budget(self) -> None:
        """When max_tokens is smaller than the suffix token count, truncate
        without appending suffix (just take first max_tokens tokens)."""
        mock_tiktoken, _mock_enc = _make_mock_tiktoken()
        # TRUNCATION_SUFFIX = "\n[...]" → split() → ["[...]"] → 1 token
        # So we need max_tokens <= 1 to hit the branch max_tokens <= len(suffix_tokens)
        text = "word0 word1 word2 word3"  # 4 tokens

        with patch.dict(sys.modules, {"tiktoken": mock_tiktoken}):
            result = truncate_for_llm(text, max_chars=None, max_tokens=1)

        # max_tokens=1 <= len(suffix_tokens)=1, so just decode tokens[:1]
        # decode([0]) → "w0"
        assert result == "w0"
        assert TRUNCATION_SUFFIX not in result

    def test_truncate_for_llm_token_based_tiktoken_fails(self) -> None:
        """When tiktoken import fails, fall back to char-based truncation."""
        text = "x" * 200

        # Simulate ImportError when trying to import tiktoken
        with patch.dict(sys.modules, {"tiktoken": None}):
            # Importing a module mapped to None raises ImportError
            result = truncate_for_llm(text, max_chars=50, max_tokens=10)

        # Should fall back to char-based: 50 chars with suffix
        assert len(result) == 50
        assert result.endswith(TRUNCATION_SUFFIX)

    def test_truncate_for_llm_with_model_hint(self) -> None:
        """When model_hint is provided, encoding_for_model is called
        instead of get_encoding."""
        mock_tiktoken, _mock_enc = _make_mock_tiktoken()
        text = "hello world"  # 2 tokens, under any reasonable limit

        with patch.dict(sys.modules, {"tiktoken": mock_tiktoken}):
            result = truncate_for_llm(text, max_chars=None, max_tokens=10, model_hint="gpt-4")

        assert result == text
        mock_tiktoken.encoding_for_model.assert_called_once_with("gpt-4")
        mock_tiktoken.get_encoding.assert_not_called()
