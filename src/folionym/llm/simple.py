"""Simple one-shot LLM filename generation."""

from __future__ import annotations

from ..rename_ops import sanitize_filename_from_llm
from .cache import ResponseCache
from .models import SimpleFilenameOptions
from .parsing import _replace_prompt_placeholders, truncate_for_llm
from .prompts import _escape_doc_content
from .protocol import LLMClient

# Max chars of content to send for simple filename (one-shot prompt).
SIMPLE_FILENAME_MAX_CONTENT_CHARS = 12_000

_PLACEHOLDER_INTRO_LINE = "%INTRO_LINE%"
_PLACEHOLDER_DATE_HINT = "%DATE_HINT%"
_PLACEHOLDER_EXAMPLE = "%EXAMPLE%"
_PLACEHOLDER_CLOSING_LINE = "%CLOSING_LINE%"
_PLACEHOLDER_DOCUMENT_CONTENT = "%DOCUMENT_CONTENT%"
_PLACEHOLDER_EXAMPLE_LABEL = "%EXAMPLE_LABEL%"

_SIMPLE_FILENAME_TEMPLATE = (
    "%INTRO_LINE%\n"
    "- 3–6 words, target language in the requested language.\n"
    "- Use underscores only (no spaces), no special characters except underscores and hyphens.\n"
    "- Optional: date at end (e.g. dd-mm-yyyy).\n"
    "- Uppercase preferred.\n"
    "%DATE_HINT%\n"
    "%EXAMPLE_LABEL%%EXAMPLE%\n\n"
    "%CLOSING_LINE%\n\n"
    "<document_content>\n%DOCUMENT_CONTENT%\n</document_content>"
)


def get_document_filename_simple(
    client: LLMClient,
    content: str,
    options: SimpleFilenameOptions | None = None,
) -> str:
    """
    Ask the LLM for a single short filename (3-6 words, underscores). No JSON.
    Returns a sanitized string suitable as the middle part of a filename.
    """
    opts = options or SimpleFilenameOptions()
    if not content or not content.strip():
        return "document"
    prompt = _simple_filename_prompt(content, opts)
    cache_key = _simple_filename_cache_key(client, prompt, opts)
    if opts.cache is not None and cache_key is not None:
        cached = opts.cache.get(cache_key)
        if cached is not None:
            return sanitize_filename_from_llm(cached)
    raw = _complete_simple_filename(client, prompt, opts)
    _store_simple_filename_cache(opts, cache_key, raw)
    return sanitize_filename_from_llm(raw)


def _complete_simple_filename(client: LLMClient, prompt: str, options: SimpleFilenameOptions) -> str:
    """Request and strip one raw filename completion."""
    raw = client.complete(prompt, temperature=options.temperature, max_tokens=128)
    return raw.strip().rstrip(".") if raw and isinstance(raw, str) else raw


def _store_simple_filename_cache(options: SimpleFilenameOptions, cache_key: str | None, raw: str) -> None:
    """Cache a non-empty raw filename response when configured."""
    if options.cache is not None and cache_key is not None and raw:
        options.cache.set(cache_key, raw)


def _simple_filename_prompt(content: str, options: SimpleFilenameOptions) -> str:
    """Render the localized one-shot filename prompt."""
    text = _simple_filename_text(content, options)
    words = _simple_filename_words(options.language)
    return _replace_prompt_placeholders(
        _SIMPLE_FILENAME_TEMPLATE,
        {
            _PLACEHOLDER_INTRO_LINE: words["intro"],
            _PLACEHOLDER_DATE_HINT: words["date_hint"],
            _PLACEHOLDER_EXAMPLE_LABEL: words["example_label"],
            _PLACEHOLDER_EXAMPLE: words["example"],
            _PLACEHOLDER_CLOSING_LINE: words["closing"],
            _PLACEHOLDER_DOCUMENT_CONTENT: _escape_doc_content(text),
        },
    )


def _simple_filename_text(content: str, options: SimpleFilenameOptions) -> str:
    """Strip and truncate document text for the one-shot prompt."""
    effective_max = SIMPLE_FILENAME_MAX_CONTENT_CHARS
    if options.max_content_chars is not None:
        effective_max = min(SIMPLE_FILENAME_MAX_CONTENT_CHARS, options.max_content_chars)
    return truncate_for_llm(content.strip(), effective_max, max_tokens=options.max_content_tokens)


def _simple_filename_words(language: str) -> dict[str, str]:
    """Return localized strings used by the one-shot prompt."""
    if language == "de":
        return {
            "intro": "Erzeuge einen kurzen Dateinamen (ohne Endung) für dieses Dokument.",
            "date_hint": "Optional: Datum am Ende (z. B. 15-11-2023).",
            "example_label": "Beispiel: ",
            "example": "RECHNUNG_AMAZON_MAX_2023-11-15",
            "closing": "Antworte mit NUR dem Dateinamen, sonst nichts.",
        }
    return {
        "intro": "Generate a short filename (without extension) for this document.",
        "date_hint": "Optional: date at end (e.g. 15-11-2023).",
        "example_label": "Example: ",
        "example": "INVOICE_AMAZON_JOHN_2023-11-15",
        "closing": "Respond with ONLY the filename, nothing else.",
    }


def _simple_filename_cache_key(
    client: LLMClient,
    prompt: str,
    options: SimpleFilenameOptions,
) -> str | None:
    """Build the cache key for a one-shot filename request."""
    if not options.cache_key_base:
        return None
    return ResponseCache.derive_response_key(
        options.cache_key_base,
        operation="simple_filename",
        model=client.model,
        language=options.language,
        extra=prompt,
    )
