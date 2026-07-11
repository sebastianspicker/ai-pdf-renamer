"""
LLM response parsing and text truncation. Used by llm for JSON extraction and prompt truncation.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

_LENIENT_ARRAY_KEYS = frozenset({"keywords", "final_summary_tokens"})

# Fallback context limits (128K model); overridden by llm_preset.
CONTEXT_128K_MAX_CHARS_SINGLE = 480_000  # ~120K tokens at ~4 chars/token
CONTEXT_128K_CHUNK_SIZE = 100_000
CONTEXT_128K_CHUNK_OVERLAP = 5_000

TRUNCATION_SUFFIX = "\n[...]"


def _build_json_error_context(
    *,
    expected_keys: set[str] | None,
    attempted_paths: list[str],
    errors: list[str],
) -> str:
    """Build a readable error message for JSON parsing failures."""
    expected = ", ".join(sorted(expected_keys or set())) or "<any JSON object>"
    attempted = ", ".join(attempted_paths) or "<none>"
    details = "; ".join(errors) or "no parse candidates were produced"
    return (
        "Expected JSON object with keys "
        f"{expected}; received response redacted. "
        f"Attempted paths: {attempted}. Errors: {details}"
    )


def _extract_code_fence_json(text: str) -> str | None:
    code_fence = re.search(
        r"```(?:json)?\s*\n?(.*?)```",
        text,
        re.DOTALL | re.IGNORECASE,
    )
    if code_fence is None:
        return None
    candidate = code_fence.group(1).strip()
    return candidate if candidate.startswith("{") else None


def _skip_quoted_text(text: str, index: int, quote: str) -> int:
    index += 1
    while index < len(text):
        if text[index] == "\\" and index + 1 < len(text):
            index += 2
            continue
        if text[index] == quote:
            break
        index += 1
    return index


def _balanced_json_object_slice(text: str, start: int) -> str:
    depth = 0
    index = start
    while index < len(text):
        char = text[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
        elif char in ("'", '"'):
            index = _skip_quoted_text(text, index, char)
        index += 1
    return text[start:]


def _extract_json_from_response(response: str) -> str:
    """
    Try to extract a JSON object from LLM response that may contain leading prose
    or code fences (e.g. ```json ... ```). Returns the first plausible JSON slice
    or the original string stripped.
    """
    text = response.strip()
    if not text:
        return text

    # Code fence: ```json ... ``` or ``` ... ```
    if candidate := _extract_code_fence_json(text):
        return candidate

    # Leading prose: skip until first {
    start = text.find("{")
    if start == -1:
        return text

    # Find matching closing brace (simple stack-based).
    return _balanced_json_object_slice(text, start)


def _sanitize_json_string_value(response: str, *, key: str) -> str:
    """
    Attempts to escape unescaped quotes inside a JSON string value for `key`.
    This is a best-effort fix for common LLM formatting issues.
    """
    # Match from the key's opening quote to the last quote before a closing
    # delimiter so embedded quotes inside the value can be escaped.
    pattern = r'("' + re.escape(key) + r'":\s*")(.*?)("\s*[,}\]])'

    def replacer(match: re.Match[str]) -> str:
        prefix, value, suffix = match.groups()
        fixed_value = re.sub(r'(?<!\\)"', r'\\"', value)
        return prefix + fixed_value + suffix

    sanitized = re.sub(pattern, replacer, response, flags=re.DOTALL)
    if sanitized != response:
        logger.debug("JSON sanitization modified value for key %r (possible truncation from embedded quotes)", key)

    return _sanitize_single_key_string_value(sanitized, key)


def _sanitize_single_key_string_value(response: str, key: str) -> str:
    """Escape embedded quotes in a single string field after regex sanitization."""
    key_idx = response.find(f'"{key}"')
    if key_idx == -1:
        return response

    colon_idx = response.find(":", key_idx)
    if colon_idx == -1:
        return response

    first_quote = response.find('"', colon_idx)
    if first_quote == -1:
        return response

    close_brace = response.rfind("}")
    if close_brace == -1:
        return response

    last_quote = _last_unescaped_quote_before(response, first_quote, close_brace)
    if last_quote <= first_quote:
        return response

    raw_value = response[first_quote + 1 : last_quote]
    fixed_value = re.sub(r'(?<!\\)"', r'\\"', raw_value)
    return response[: first_quote + 1] + fixed_value + response[last_quote:]


def _last_unescaped_quote_before(text: str, first_quote: int, close_brace: int) -> int:
    i = first_quote + 1
    last_quote = -1
    while i < close_brace and i < len(text):
        if text[i] == "\\" and i + 1 < len(text):
            i += 2
            continue
        if text[i] == '"':
            last_quote = i
        i += 1
    return last_quote


def _lenient_extract_string_value(text: str, key: str) -> str | None:
    """Best-effort extraction of a string value for key from text that may not be valid JSON."""
    pattern = re.compile(
        r'"' + re.escape(key) + r'"\s*:\s*"((?:[^"\\]|\\.)*)"',
        re.DOTALL,
    )
    m = pattern.search(text)
    if m is None:
        return None
    raw = m.group(1)
    if not raw:
        return None
    unescaped = raw.replace('\\"', '"').replace("\\\\", "\\").strip()
    return unescaped or None


def _lenient_extract_known_string_array(text: str, key: str) -> list[str] | None:
    """Recover a JSON string array for the repo's known array-valued fields only."""
    if key not in _LENIENT_ARRAY_KEYS:
        return None
    array_start = _find_key_array_start(text, key)
    if array_start is None:
        return None
    array_bounds = _balanced_array_bounds(text, array_start)
    if array_bounds is None:
        return None
    return _parsed_string_array(text[array_bounds[0] : array_bounds[1]])


def _find_key_array_start(text: str, key: str) -> int | None:
    key_match = re.search(r'"' + re.escape(key) + r'"\s*:\s*\[', text)
    if key_match is None:
        return None
    array_start = text.find("[", key_match.start())
    return array_start if array_start != -1 else None


def _balanced_array_bounds(text: str, array_start: int) -> tuple[int, int] | None:
    depth = 0
    i = array_start
    while i < len(text):
        char = text[i]
        if char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
            if depth == 0:
                return (array_start, i + 1)
        elif char == '"':
            i = _skip_quoted_text(text, i, char)
        i += 1
    return None


def _parsed_string_array(candidate: str) -> list[str] | None:
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, list) or not all(isinstance(item, str) for item in parsed):
        return None
    cleaned = [item.strip() for item in parsed if item.strip()]
    return cleaned or None


def _lenient_extract_key_value(text: str, key: str) -> str | list[str] | None:
    """Best-effort extraction for repo-defined scalar and string-array fields."""
    array_value = _lenient_extract_known_string_array(text, key)
    if array_value is not None:
        return array_value
    return _lenient_extract_string_value(text, key)


def extract_and_validate_json(
    response: str | None,
    *,
    expected_keys: set[str] | None = None,
    lenient_keys: set[str] | None = None,
) -> dict[str, object]:
    """Extract, sanitize, and parse a JSON object from an LLM response.

    Raises ValueError with parsing context when no valid JSON object can be recovered.
    """
    resp_str = _validated_response_text(response)
    expected = _normalized_key_set(expected_keys)
    lenient = _normalized_key_set(lenient_keys)
    attempted_paths: list[str] = []
    errors: list[str] = []

    parsed = _first_parsed_json_candidate(resp_str, expected, attempted_paths, errors)
    if parsed is not None:
        return parsed

    salvaged = _try_lenient_salvage(resp_str, lenient, attempted_paths, errors)
    if salvaged is not None:
        return salvaged

    raise ValueError(
        _build_json_error_context(
            expected_keys=expected,
            attempted_paths=attempted_paths,
            errors=errors,
        )
    )


def _validated_response_text(response: str | None) -> str:
    if response is None or not isinstance(response, str):
        raise ValueError("Expected JSON response as string; received non-string response.")
    resp_str = response.strip()
    if not resp_str:
        raise ValueError("Expected JSON response as string; received empty response.")
    return resp_str


def _normalized_key_set(keys: set[str] | None) -> set[str]:
    return {key for key in (keys or set()) if key}


def _first_parsed_json_candidate(
    response: str,
    expected: set[str],
    attempted_paths: list[str],
    errors: list[str],
) -> dict[str, object] | None:
    for path_name, candidate in _json_parse_candidates(response, expected):
        attempted_paths.append(path_name)
        if parsed := _parse_json_object_candidate(path_name, candidate, errors):
            return parsed
    return None


def _try_lenient_salvage(
    response: str,
    lenient: set[str],
    attempted_paths: list[str],
    errors: list[str],
) -> dict[str, object] | None:
    if not lenient:
        return None
    attempted_paths.append("lenient")
    salvaged = _salvage_lenient_values(response, lenient)
    if salvaged:
        return salvaged
    errors.append(f"lenient: none of {', '.join(sorted(lenient))} found")
    return None


def _json_parse_candidates(response: str, expected: set[str]) -> list[tuple[str, str]]:
    candidates: list[tuple[str, str]] = []
    seen_candidates: set[str] = set()

    def add_candidate(path_name: str, candidate: str) -> None:
        normalized = candidate.strip()
        if normalized.startswith("{") and normalized not in seen_candidates:
            seen_candidates.add(normalized)
            candidates.append((path_name, normalized))

    add_candidate("raw", response)
    for key in sorted(expected):
        add_candidate(f"sanitized:{key}", _sanitize_json_string_value(response, key=key))
    add_candidate("extracted", _extract_json_from_response(response))
    return candidates


def _parse_json_object_candidate(
    path_name: str,
    candidate: str,
    errors: list[str],
) -> dict[str, object] | None:
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError as exc:
        errors.append(f"{path_name}: {exc.msg} at pos {exc.pos}")
        return None
    if not isinstance(parsed, dict):
        errors.append(f"{path_name}: expected object, got {type(parsed).__name__}")
        return None
    return parsed


def _salvage_lenient_values(response: str, lenient: set[str]) -> dict[str, object]:
    salvaged: dict[str, object] = {}
    for key in sorted(lenient):
        if value := _lenient_extract_key_value(response, key):
            salvaged[key] = value
    return salvaged


def parse_json_field(response: str | None, *, key: str, lenient: bool = False) -> str | list[str] | None:
    """Extract a field value from an LLM JSON response, with code-fence stripping and salvage fallbacks.

    Return the string or list[str] value for key, or None if extraction fails.
    """
    try:
        data = extract_and_validate_json(
            response,
            expected_keys={key},
            lenient_keys={key} if lenient else None,
        )
    except ValueError as exc:
        logger.warning("LLM response could not be parsed as JSON; using fallback. %s", exc)
        return None

    return _clean_json_field_value(data.get(key))


def _clean_json_field_value(value: object) -> str | list[str] | None:
    if isinstance(value, list):
        return _clean_json_string_list(value)
    if isinstance(value, str):
        return _clean_json_string(value)
    return None


def _clean_json_string_list(value: list[object]) -> list[str] | None:
    cleaned: list[str] = []
    for item in value:
        if not isinstance(item, str):
            return None
        if stripped := item.strip():
            cleaned.append(stripped)
    return cleaned or None


def _clean_json_string(value: str) -> str | None:
    stripped = value.strip()
    if not stripped or stripped.lower() == "na":
        return None
    return stripped


def _replace_prompt_placeholders(template: str, replacements: dict[str, str]) -> str:
    """
    Replace placeholders (e.g. %KEY%) in a prompt template with actual values.
    Use %...% style for all placeholder keys. Unknown placeholders are left as-is.
    New prompts should use placeholders for variable bits (language, examples, content).
    """
    result = template
    for key, value in replacements.items():
        result = result.replace(key, value)
    return result


def truncate_for_llm(
    text: str,
    max_chars: int | None,
    max_tokens: int | None = None,
    model_hint: str | None = None,
) -> str:
    """
    Truncate text for LLM input. When max_tokens is set and tiktoken is available,
    truncate by token count (and ignore max_chars for that path). Otherwise truncate by max_chars.
    """
    token_truncated = _truncate_by_tokens(text, max_tokens, model_hint)
    if token_truncated is not None:
        return token_truncated
    return _truncate_by_chars(text, max_chars)


def _encoding_for_model(model_hint: str | None) -> Any | None:
    try:
        import tiktoken

        if model_hint:
            return tiktoken.encoding_for_model(model_hint)
        return tiktoken.get_encoding("cl100k_base")
    except (ImportError, KeyError, RuntimeError, ValueError, LookupError):
        return None


def _truncate_by_tokens(text: str, max_tokens: int | None, model_hint: str | None) -> str | None:
    if max_tokens is not None and max_tokens > 0:
        enc = _encoding_for_model(model_hint)
        if enc is not None:
            tokens = enc.encode(text)
            if len(tokens) <= max_tokens:
                return text
            suffix_tokens = enc.encode(TRUNCATION_SUFFIX)
            if max_tokens <= len(suffix_tokens):
                return str(enc.decode(tokens[:max_tokens]))
            keep = max(1, max_tokens - len(suffix_tokens))
            return str(enc.decode(tokens[:keep])) + TRUNCATION_SUFFIX
    return None


def _truncate_by_chars(text: str, max_chars: int | None) -> str:
    if max_chars is None or max_chars <= 0:
        return text
    if len(text) <= max_chars:
        return text
    if max_chars <= len(TRUNCATION_SUFFIX):
        return text[:max_chars]
    return text[: max_chars - len(TRUNCATION_SUFFIX)] + TRUNCATION_SUFFIX
