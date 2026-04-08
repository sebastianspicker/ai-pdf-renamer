"""
LLM response parsing and text truncation. Used by llm for JSON extraction and prompt truncation.
"""

from __future__ import annotations

import json
import logging
import re

logger = logging.getLogger(__name__)

# Fallback context limits (128K model); overridden by llm_preset.
CONTEXT_128K_MAX_CHARS_SINGLE = 480_000  # ~120K tokens at ~4 chars/token
CONTEXT_128K_CHUNK_SIZE = 100_000
CONTEXT_128K_CHUNK_OVERLAP = 5_000

TRUNCATION_SUFFIX = "\n[...]"


def _preview_text(text: str, *, limit: int = 160) -> str:
    """Return a compact single-line preview for log and error messages."""
    compact = re.sub(r"\s+", " ", text.strip())
    return compact if len(compact) <= limit else compact[: limit - 3] + "..."


def _build_json_error_context(
    response: str,
    *,
    expected_keys: set[str] | None,
    attempted_paths: list[str],
    errors: list[str],
) -> str:
    """Build a readable error message for JSON parsing failures."""
    expected = ", ".join(sorted(expected_keys or set())) or "<any JSON object>"
    attempted = ", ".join(attempted_paths) or "<none>"
    details = "; ".join(errors) or "no parse candidates were produced"
    received = _preview_text(response)
    return (
        f"Expected JSON object with keys {expected}; received: {received!r}. "
        f"Attempted paths: {attempted}. Errors: {details}"
    )


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
            # Skip double-quoted string content to avoid counting braces inside strings.
            i += 1
            while i < len(text):
                if text[i] == "\\" and i + 1 < len(text):
                    i += 2
                    continue
                if text[i] == '"':
                    break
                i += 1
        elif text[i] == "'":
            # P2: Skip single-quoted string content (non-standard JSON but common in LLM output)
            i += 1
            while i < len(text):
                if text[i] == "\\" and i + 1 < len(text):
                    i += 2
                    continue
                if text[i] == "'":
                    break
                i += 1
        i += 1

    return text[start:]


def _sanitize_json_string_value(response: str, *, key: str) -> str:
    """
    Attempts to escape unescaped quotes inside a JSON string value for `key`.
    This is a best-effort fix for common LLM formatting issues.
    """
    # P2: Use greedy match with proper boundary detection for values with embedded quotes.
    # The non-greedy (.*?) pattern truncates values at the first embedded quote.
    # Instead, match from the key's opening quote to the last quote before a closing brace or comma.
    pattern = r'("' + re.escape(key) + r'":\s*")(.*?)("\s*[,}\]])'

    def replacer(match: re.Match[str]) -> str:
        prefix, value, suffix = match.groups()
        fixed_value = re.sub(r'(?<!\\)"', r'\\"', value)
        return prefix + fixed_value + suffix

    sanitized = re.sub(pattern, replacer, response, flags=re.DOTALL)
    if sanitized != response:
        logger.debug("JSON sanitization modified value for key %r (possible truncation from embedded quotes)", key)

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


def extract_and_validate_json(
    response: str | None,
    *,
    expected_keys: set[str] | None = None,
    lenient_keys: set[str] | None = None,
) -> dict[str, object]:
    """Extract, sanitize, and parse a JSON object from an LLM response.

    Raises ValueError with parsing context when no valid JSON object can be recovered.
    """
    if response is None or not isinstance(response, str):
        raise ValueError("Expected JSON response as string; received non-string response.")
    resp_str = response.strip()
    if not resp_str:
        raise ValueError("Expected JSON response as string; received empty response.")

    expected = {key for key in (expected_keys or set()) if key}
    lenient = {key for key in (lenient_keys or set()) if key}
    attempted_paths: list[str] = []
    errors: list[str] = []
    candidates: list[tuple[str, str]] = []
    seen_candidates: set[str] = set()

    def add_candidate(path_name: str, candidate: str) -> None:
        normalized = candidate.strip()
        if not normalized.startswith("{") or normalized in seen_candidates:
            return
        seen_candidates.add(normalized)
        candidates.append((path_name, normalized))

    add_candidate("raw", resp_str)
    if expected:
        for key in sorted(expected):
            add_candidate(f"sanitized:{key}", _sanitize_json_string_value(resp_str, key=key))
    add_candidate("extracted", _extract_json_from_response(resp_str))

    for path_name, candidate in candidates:
        attempted_paths.append(path_name)
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError as exc:
            errors.append(f"{path_name}: {exc.msg} at pos {exc.pos}")
            continue
        if not isinstance(parsed, dict):
            errors.append(f"{path_name}: expected object, got {type(parsed).__name__}")
            continue
        return parsed

    if lenient:
        attempted_paths.append("lenient")
        salvaged: dict[str, object] = {}
        for key in sorted(lenient):
            if value := _lenient_extract_key_value(resp_str, key):
                salvaged[key] = value
        if salvaged:
            return salvaged
        errors.append(f"lenient: none of {', '.join(sorted(lenient))} found")

    raise ValueError(
        _build_json_error_context(
            response,
            expected_keys=expected,
            attempted_paths=attempted_paths,
            errors=errors,
        )
    )


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
        except (ImportError, KeyError, RuntimeError, ValueError, LookupError):
            enc = None
        if enc is not None:
            tokens = enc.encode(text)
            if len(tokens) <= max_tokens:
                return text
            suffix_tokens = enc.encode(TRUNCATION_SUFFIX)
            if max_tokens <= len(suffix_tokens):
                return str(enc.decode(tokens[:max_tokens]))
            # Reserve suffix budget so the output stays within token cap.
            keep = max(1, max_tokens - len(suffix_tokens))
            truncated = str(enc.decode(tokens[:keep])) + TRUNCATION_SUFFIX
            return truncated
    if max_chars is None or max_chars <= 0:
        return text
    if len(text) <= max_chars:
        return text
    if max_chars <= len(TRUNCATION_SUFFIX):
        return text[:max_chars]
    return text[: max_chars - len(TRUNCATION_SUFFIX)] + TRUNCATION_SUFFIX
