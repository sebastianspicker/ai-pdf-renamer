from __future__ import annotations

import logging

from .cache import ResponseCache
from .llm_backend import HttpLLMBackend, LLMClient, LocalLLMClient  # noqa: F401 (re-exports)
from .llm_parsing import (
    CONTEXT_128K_CHUNK_OVERLAP,
    CONTEXT_128K_CHUNK_SIZE,
    CONTEXT_128K_MAX_CHARS_SINGLE,
    _replace_prompt_placeholders,
    extract_and_validate_json,
    parse_json_field,
    truncate_for_llm,
)
from .llm_prompts import (
    _PLACEHOLDER_ALLOWED_CATEGORIES,
    _build_allowed_categories_instruction,
    _escape_doc_content,
    _summary_doc_type_hint,
    _summary_prompt_chunk,
    _summary_prompt_combine,
    _summary_prompts_short,
    build_analysis_prompt,
)
from .llm_schema import DocumentAnalysisResult, validate_llm_document_result
from .rename_ops import sanitize_filename_from_llm
from .text_utils import chunk_text

logger = logging.getLogger(__name__)

# Temperature increment per retry attempt when LLM JSON parsing fails.
_RETRY_TEMP_INCREMENT = 0.2


def _build_cache_key(
    cache_key_base: str | None,
    *,
    operation: str,
    model: str,
    language: str,
    payload: str,
) -> str | None:
    """Build a stable response-cache key for one LLM operation."""
    if not cache_key_base:
        return None
    return ResponseCache.derive_response_key(
        cache_key_base,
        operation=operation,
        model=model,
        language=language,
        extra=payload,
    )


def complete_json_with_retry(
    client: LLMClient,
    prompt: str,
    *,
    temperature: float = 0.0,
    max_retries: int = 3,
    max_tokens: int | None = 1024,
    json_mode: bool = False,
    cache: ResponseCache | None = None,
    cache_key: str | None = None,
) -> str:
    """Retry LLM completion up to max_retries times, increasing temperature, until valid JSON."""
    if cache is not None and cache_key is not None:
        cached = cache.get(cache_key)
        if cached is not None:
            logger.debug("LLM cache hit for %s", cache_key)
            return cached

    response_format = {"type": "json_object"} if json_mode else None
    effective_retries = 1 if json_mode else max_retries
    temp = temperature
    last = ""
    for i in range(effective_retries):
        last = client.complete(prompt, temperature=temp, max_tokens=max_tokens, response_format=response_format)
        try:
            extract_and_validate_json(last)
            if cache is not None and cache_key is not None:
                cache.set(cache_key, last)
            return last
        except ValueError:
            pass
        temp += _RETRY_TEMP_INCREMENT
        logger.info("Retry %s: New temperature=%s", i + 1, temp)
    logger.error(
        "LLM returned no valid JSON after %s retries. Using heuristic or 'na' for this document.",
        effective_retries,
    )
    return last


def _try_prompts_for_key(
    client: LLMClient,
    prompts: list[str],
    *,
    key: str,
    language: str,
    operation: str,
    temperature: float,
    max_tokens: int | None = 1024,
    lenient: bool = False,
    cache: ResponseCache | None = None,
    cache_key_base: str | None = None,
) -> str | list[str] | None:
    for i, prompt in enumerate(prompts):
        cache_key = _build_cache_key(
            cache_key_base,
            operation=f"{operation}:{i}",
            model=client.model,
            language=language,
            payload=prompt,
        )
        r = complete_json_with_retry(
            client,
            prompt,
            temperature=temperature + i * _RETRY_TEMP_INCREMENT,
            max_tokens=max_tokens,
            cache=cache,
            cache_key=cache_key,
        )
        v = parse_json_field(r, key=key, lenient=lenient)
        if v is not None:
            return v
    return None


def get_document_analysis(
    client: LLMClient,
    pdf_content: str,
    *,
    language: str = "de",
    temperature: float = 0.0,
    max_content_chars: int | None = None,
    max_content_tokens: int | None = None,
    suggested_doc_type: str | None = None,
    allowed_categories: list[str] | None = None,
    suggested_categories: list[str] | None = None,
    lenient_json: bool = False,
    json_mode: bool = False,
    cache: ResponseCache | None = None,
    cache_key_base: str | None = None,
) -> DocumentAnalysisResult:
    """Extract summary, keywords, and category from document text in a single LLM call.

    Return a DocumentAnalysisResult with empty defaults when content is too short or parsing fails.
    """
    if not isinstance(pdf_content, str) or len(pdf_content.strip()) < 50:
        return DocumentAnalysisResult()

    text = pdf_content.strip()
    effective_max = CONTEXT_128K_MAX_CHARS_SINGLE
    if max_content_chars is not None:
        effective_max = min(CONTEXT_128K_MAX_CHARS_SINGLE, max_content_chars)
    text = truncate_for_llm(text, effective_max, max_tokens=max_content_tokens)

    prompt = build_analysis_prompt(
        language,
        text,
        suggested_doc_type=suggested_doc_type,
        allowed_categories=allowed_categories,
        suggested_categories=suggested_categories,
    )
    raw = complete_json_with_retry(
        client,
        prompt,
        temperature=temperature,
        max_retries=2,
        max_tokens=1024,
        json_mode=json_mode,
        cache=cache,
        cache_key=_build_cache_key(
            cache_key_base,
            operation="analysis",
            model=client.model,
            language=language,
            payload=prompt,
        ),
    )
    try:
        data = extract_and_validate_json(
            raw,
            expected_keys={"summary", "keywords", "category"},
            lenient_keys={"summary", "keywords", "category"} if lenient_json else None,
        )
    except ValueError:
        data = {}

    return validate_llm_document_result(data)


def get_document_summary(
    client: LLMClient,
    pdf_content: str,
    *,
    language: str = "de",
    temperature: float = 0.0,
    max_chars_single: int = CONTEXT_128K_MAX_CHARS_SINGLE,
    max_content_chars: int | None = None,
    max_content_tokens: int | None = None,
    suggested_doc_type: str | None = None,
    lenient_json: bool = False,
    cache: ResponseCache | None = None,
    cache_key_base: str | None = None,
) -> str:
    """Generate a short document summary via LLM, chunking long content and combining partial summaries.

    Return 'na' when content is missing or too short.
    """
    if not isinstance(pdf_content, str):
        return "na"
    text = pdf_content.strip()
    if len(text) < 50:
        return "na"

    # P1: Compare original text length before truncation to decide chunking
    original_text_length = len(text)
    effective_max = max_chars_single
    if max_content_chars is not None:
        effective_max = min(max_chars_single, max_content_chars)
    text = truncate_for_llm(text, effective_max, max_tokens=max_content_tokens)

    doc_type_hint = _summary_doc_type_hint(language, suggested_doc_type)

    if original_text_length < max_chars_single:
        prompts = _summary_prompts_short(language, doc_type_hint, text)
        val = _try_prompts_for_key(
            client,
            prompts,
            key="summary",
            language=language,
            operation="summary_short",
            temperature=temperature,
            max_tokens=1024,
            lenient=lenient_json,
            cache=cache,
            cache_key_base=cache_key_base,
        )
        result = validate_llm_document_result({"summary": val if isinstance(val, str) else ""})
        return result.summary

    chunks = chunk_text(
        text,
        chunk_size=CONTEXT_128K_CHUNK_SIZE,
        overlap=CONTEXT_128K_CHUNK_OVERLAP,
    )
    partial: list[str] = []
    for chunk in chunks:
        chunk_prompt = _summary_prompt_chunk(language, doc_type_hint, chunk)
        r = complete_json_with_retry(
            client,
            chunk_prompt,
            temperature=temperature,
            max_retries=3,
            max_tokens=1024,
            cache=cache,
            cache_key=_build_cache_key(
                cache_key_base,
                operation=f"summary_chunk:{len(partial)}",
                model=client.model,
                language=language,
                payload=chunk_prompt,
            ),
        )
        v = parse_json_field(r, key="summary", lenient=lenient_json)
        partial.append(v if isinstance(v, str) else "")

    combined = " ".join(p for p in partial if p)
    if not combined:
        return "na"

    final_prompt = _summary_prompt_combine(language, doc_type_hint, combined)
    r_final = complete_json_with_retry(
        client,
        final_prompt,
        temperature=temperature + _RETRY_TEMP_INCREMENT,
        max_retries=3,
        max_tokens=1024,
        cache=cache,
        cache_key=_build_cache_key(
            cache_key_base,
            operation="summary_combine",
            model=client.model,
            language=language,
            payload=final_prompt,
        ),
    )
    v_final = parse_json_field(r_final, key="summary", lenient=lenient_json)
    result = validate_llm_document_result({"summary": v_final if isinstance(v_final, str) else ""})
    return result.summary


def get_document_keywords(
    client: LLMClient,
    summary: str,
    *,
    language: str = "de",
    temperature: float = 0.0,
    suggested_category: str | None = None,
    lenient_json: bool = False,
    cache: ResponseCache | None = None,
    cache_key_base: str | None = None,
) -> tuple[str, ...] | None:
    """Extract 5-7 keywords from a document summary via LLM. Return None on failure."""
    cat_hint = ""
    if suggested_category and suggested_category.strip():
        c = suggested_category.strip()
        cat_hint = f"Das Dokument ist voraussichtlich: {c}. " if language == "de" else f"The document is likely: {c}. "

    if language == "de":
        prompts = [
            (
                cat_hint + "Extrahiere bitte 5–7 Schlüsselwörter aus dieser Zusammenfassung.\n"
                "Gib ausschließlich eine Ausgabe in der Form:\n"
                '{"keywords":["KW1","KW2","KW3"]}\n\n'
                "Jetzt bitte NUR reines JSON, sonst nichts.\n"
                "Zusammenfassung:\n" + summary
            ),
            (
                cat_hint + "Bitte NUR reines JSON in der Form:\n"
                '{"keywords":["KW1","KW2"]}\n\n'
                "Hier die Zusammenfassung:\n" + summary
            ),
        ]
    else:
        prompts = [
            (
                cat_hint + "Extract 5–7 keywords from this summary. Return ONLY JSON:\n"
                '{"keywords":["KW1","KW2"]}\n\n'
                "Summary:\n" + summary
            )
        ]

    val = _try_prompts_for_key(
        client,
        prompts,
        key="keywords",
        language=language,
        operation="keywords",
        temperature=temperature,
        max_tokens=512,
        lenient=lenient_json,
        cache=cache,
        cache_key_base=cache_key_base,
    )
    result = validate_llm_document_result({"keywords": val if isinstance(val, list) else []})
    return result.keywords if result.keywords else None


def get_document_category(
    client: LLMClient,
    *,
    summary: str,
    keywords: list[str],
    language: str = "de",
    temperature: float = 0.0,
    suggested_categories: list[str] | None = None,
    allowed_categories: list[str] | None = None,
    lenient_json: bool = False,
    cache: ResponseCache | None = None,
    cache_key_base: str | None = None,
) -> str:
    """Classify document category via LLM, optionally constrained to allowed/suggested categories."""
    keywords_joined = ", ".join(keywords)
    if language == "de":
        base_text = f"Zusammenfassung:\n{summary}\nKeywords:{keywords_joined}"
    else:
        base_text = f"Summary:\n{summary}\nKeywords:{keywords_joined}"
    category_instruction = _build_allowed_categories_instruction(
        allowed_categories=allowed_categories,
        suggested_categories=suggested_categories,
        language=language,
    )
    content = _replace_prompt_placeholders(
        base_text + "\n\n" + _PLACEHOLDER_ALLOWED_CATEGORIES,
        {_PLACEHOLDER_ALLOWED_CATEGORIES: category_instruction},
    )
    if language == "de":
        prompt_templates = [
            "Bestimme eine sinnvolle Kategorie als reines JSON.\n"
            'Gib nur: {"category":"..."}\n\nKeine weiteren Erklärungen. Text:\n',
            'Bitte nur {"category":"..."} - ohne Zusätze:\n',
        ]
        prompts = [t + content for t in prompt_templates]
    else:
        prompts = [f'Determine a suitable category. Return ONLY JSON: {{"category":"..."}}\n\nText:\n{content}']

    val = _try_prompts_for_key(
        client,
        prompts,
        key="category",
        language=language,
        operation="category",
        temperature=temperature,
        max_tokens=256,
        lenient=lenient_json,
        cache=cache,
        cache_key_base=cache_key_base,
    )
    raw = val if isinstance(val, str) else ""
    if len(raw.strip()) > 80:
        logger.info("LLM category too long (%d chars); treating as invalid.", len(raw))
        raw = ""
    result = validate_llm_document_result({"category": raw})
    return result.category


def get_final_summary_tokens(
    client: LLMClient,
    *,
    summary: str,
    keywords: list[str],
    category: str,
    language: str = "de",
    temperature: float = 0.0,
    lenient_json: bool = False,
    cache: ResponseCache | None = None,
    cache_key_base: str | None = None,
) -> list[str] | None:
    """Extract 3-5 short summary tokens from document text via LLM. Return None on failure."""
    kw_str = ", ".join(keywords)
    base_text = f"Zusammenfassung: {summary}\nSchlagworte: {kw_str}\nKategorie: {category}"

    if language == "de":
        prompts = [
            (
                "Erstelle bitte bis zu 5 Stichworte (kurz! 1–2 Wörter pro Stichwort) "
                "als reines JSON.\n"
                '{"final_summary":"stichwort1,stichwort2"}\n\n'
                "WICHTIG: Keine Sätze, nur Stichworte. Nur JSON.\n\n" + base_text
            ),
            (
                'Bitte nur reines JSON {"final_summary":"stichwort1,stichwort2"}. '
                "Max. 5 Stichworte, keine Sätze!\n\n" + base_text
            ),
        ]
    else:
        prompts = [
            (
                "Return up to 5 short keywords (1–2 words each) as JSON:\n"
                '{"final_summary":"kw1,kw2"}\n\n'
                "Only JSON.\n\n" + base_text
            )
        ]

    val = _try_prompts_for_key(
        client,
        prompts,
        key="final_summary",
        language=language,
        operation="final_summary",
        temperature=temperature,
        max_tokens=256,
        lenient=lenient_json,
        cache=cache,
        cache_key_base=cache_key_base,
    )
    if not isinstance(val, str):
        return None

    tokens = [t.strip() for t in val.split(",") if t.strip()]
    return tokens[:5] if tokens else None


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
    *,
    language: str = "de",
    temperature: float = 0.0,
    max_content_chars: int | None = None,
    max_content_tokens: int | None = None,
    cache: ResponseCache | None = None,
    cache_key_base: str | None = None,
) -> str:
    """
    Ask the LLM for a single short filename (3-6 words, underscores). No JSON.
    Returns a sanitized string suitable as the middle part of a filename.
    """
    if not content or not content.strip():
        return "document"
    text = content.strip()
    effective_max = (
        min(SIMPLE_FILENAME_MAX_CONTENT_CHARS, max_content_chars)
        if max_content_chars is not None
        else SIMPLE_FILENAME_MAX_CONTENT_CHARS
    )
    text = truncate_for_llm(text, effective_max, max_tokens=max_content_tokens)
    safe_text = _escape_doc_content(text)
    if language == "de":
        intro = "Erzeuge einen kurzen Dateinamen (ohne Endung) für dieses Dokument."
        date_hint = "Optional: Datum am Ende (z. B. 15-11-2023)."
        example_label = "Beispiel: "
        example = "RECHNUNG_AMAZON_MAX_2023-11-15"
        closing = "Antworte mit NUR dem Dateinamen, sonst nichts."
    else:
        intro = "Generate a short filename (without extension) for this document."
        date_hint = "Optional: date at end (e.g. 15-11-2023)."
        example_label = "Example: "
        example = "INVOICE_AMAZON_JOHN_2023-11-15"
        closing = "Respond with ONLY the filename, nothing else."
    prompt = _replace_prompt_placeholders(
        _SIMPLE_FILENAME_TEMPLATE,
        {
            _PLACEHOLDER_INTRO_LINE: intro,
            _PLACEHOLDER_DATE_HINT: date_hint,
            _PLACEHOLDER_EXAMPLE_LABEL: example_label,
            _PLACEHOLDER_EXAMPLE: example,
            _PLACEHOLDER_CLOSING_LINE: closing,
            _PLACEHOLDER_DOCUMENT_CONTENT: safe_text,
        },
    )
    cache_key = _build_cache_key(
        cache_key_base,
        operation="simple_filename",
        model=client.model,
        language=language,
        payload=prompt,
    )
    if cache is not None and cache_key is not None:
        cached = cache.get(cache_key)
        if cached is not None:
            return sanitize_filename_from_llm(cached)
    raw = client.complete(prompt, temperature=temperature, max_tokens=128)
    if raw and isinstance(raw, str):
        raw = raw.strip().rstrip(".")
    if cache is not None and cache_key is not None and raw:
        cache.set(cache_key, raw)
    return sanitize_filename_from_llm(raw)
