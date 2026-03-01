"""
LLM response parsing and text truncation. Used by llm for JSON extraction and prompt truncation.
"""

from __future__ import annotations

import json
import logging
import re

logger = logging.getLogger(__name__)

# Qwen3 8B 128K context: single-shot up to ~120K tokens (~480K chars).
CONTEXT_128K_MAX_CHARS_SINGLE = 480_000  # ~120K tokens at ~4 chars/token
CONTEXT_128K_CHUNK_SIZE = 100_000
CONTEXT_128K_CHUNK_OVERLAP = 5_000

TRUNCATION_SUFFIX = "\n[...]"


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
    code_fence = re.search(
        r"```(?:json)?\s*\n?(.*?)```",
        text,
        re.DOTALL | re.IGNORECASE,
    )
    if code_fence:
        candidate = code_fence.group(1).strip()
        if candidate.startswith("{"):
            return candidate

    # Leading prose: skip until first {
    start = text.find("{")
    if start == -1:
        return text

    # Find matching closing brace (simple stack-based).
    depth = 0
    i = start
    while i < len(text):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
        elif text[i] == '"':
            # Skip string content to avoid counting braces inside strings.
            i += 1
            while i < len(text):
                if text[i] == "\\" and i + 1 < len(text):
                    i += 2
                    continue
                if text[i] == '"':
                    break
                i += 1
        i += 1

    return text[start:]


def _sanitize_json_string_value(response: str, *, key: str) -> str:
    """
    Attempts to escape unescaped quotes inside a JSON string value for `key`.
    This is a best-effort fix for common LLM formatting issues.
    """
    # Fast-path regex replacement for common cases where the JSON is almost valid.
    pattern = r'("' + re.escape(key) + r'":\s*")(.*?)(")'

    def replacer(match: re.Match[str]) -> str:
        prefix, value, suffix = match.groups()
        fixed_value = re.sub(r'(?<!\\)"', r'\\"', value)
        return prefix + fixed_value + suffix

    sanitized = re.sub(pattern, replacer, response, flags=re.DOTALL)

    # If the string is still malformed because unescaped quotes prematurely closed the
    # value, try a best-effort salvage assuming a single-key JSON object.
    #
    # This is intentionally conservative and only aims to support the script's
    # prompts, which ask for JSON objects with a single string field.
    key_idx = sanitized.find(f'"{key}"')
    if key_idx == -1:
        return sanitized

    colon_idx = sanitized.find(":", key_idx)
    if colon_idx == -1:
        return sanitized

    first_quote = sanitized.find('"', colon_idx)
    if first_quote == -1:
        return sanitized

    close_brace = sanitized.rfind("}")
    if close_brace == -1:
        return sanitized

    # Find closing quote: the last unescaped " before } (respects \" in value).
    i = first_quote + 1
    last_quote = -1
    while i < close_brace and i < len(sanitized):
        if sanitized[i] == "\\" and i + 1 < len(sanitized):
            i += 2
            continue
        if sanitized[i] == '"':
            last_quote = i
        i += 1
    if last_quote <= first_quote:
        return sanitized

    raw_value = sanitized[first_quote + 1 : last_quote]
    # Escape only unescaped quotes so existing \" is preserved.
    fixed_value = re.sub(r'(?<!\\)"', r'\\"', raw_value)
    return sanitized[: first_quote + 1] + fixed_value + sanitized[last_quote:]


def _lenient_extract_key_value(text: str, key: str) -> str | None:
    """Best-effort extraction of a string value for key from text that may not be valid JSON."""
    # Match "key":"value" with value possibly containing escaped quotes.
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
    # Unescape only \" inside the value so we don't corrupt normal text.
    unescaped = raw.replace('\\"', '"').replace("\\\\", "\\").strip()
    return unescaped or None


def parse_json_field(response: str | None, *, key: str, lenient: bool = False) -> str | list[str] | None:
    if response is None:
        return None
    if not isinstance(response, str):
        return None
    resp_str = response.strip()
    if not resp_str:
        return None
    # Normalize: extract JSON from code fences or leading prose.
    if not resp_str.startswith("{"):
        extracted = _extract_json_from_response(resp_str)
        if extracted.startswith("{"):
            resp_str = extracted
        else:
            if lenient:
                val = _lenient_extract_key_value(resp_str, key)
                if val is not None and val.strip() and val.strip().lower() != "na":
                    return val.strip()
            return None

    try:
        data = json.loads(resp_str)
    except json.JSONDecodeError:
        # Only salvage when response looks like a single-key string object (avoids corrupting lists/multi-key).
        single_key_pattern = re.compile(r'^\s*\{\s*"' + re.escape(key) + r'"\s*:\s*"', re.DOTALL)
        if not single_key_pattern.match(resp_str):
            logger.warning("LLM response could not be parsed as JSON; using fallback")
            return None
        try:
            data = json.loads(_sanitize_json_string_value(resp_str, key=key))
        except json.JSONDecodeError:
            extracted = _extract_json_from_response(response)
            if extracted.startswith("{"):
                try:
                    data = json.loads(extracted)
                except json.JSONDecodeError:
                    logger.warning("LLM response could not be parsed as JSON; using fallback")
                    return None
            else:
                logger.warning("LLM response could not be parsed as JSON; using fallback")
                return None

    value = data.get(key)
    if isinstance(value, list):
        if all(isinstance(x, str) for x in value):
            cleaned = [x.strip() for x in value if x and x.strip()]
            return cleaned or None
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        if stripped.lower() == "na":
            return None
        return stripped
    return None


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
    if max_tokens is not None and max_tokens > 0:
        try:
            import tiktoken

            enc = tiktoken.encoding_for_model(model_hint) if model_hint else tiktoken.get_encoding("cl100k_base")
        except (ImportError, Exception):
            enc = None
        if enc is not None:
            tokens = enc.encode(text)
            if len(tokens) <= max_tokens:
                return text
            # Reserve space for suffix; decode truncated tokens and append suffix
            keep = max(1, max_tokens - 1)
            truncated = enc.decode(tokens[:keep]) + TRUNCATION_SUFFIX
            return truncated
    if max_chars is None or max_chars <= 0:
        return text
    if len(text) <= max_chars:
        return text
    return text[: max_chars - len(TRUNCATION_SUFFIX)] + TRUNCATION_SUFFIX
