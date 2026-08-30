"""LLM and filename-generation helpers."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from .cache import ResponseCache
from .models import (
    AnalysisOptions,
    CategoryOptions,
    DocumentAnalysisResult,
    FinalSummaryOptions,
    JsonCompletionOptions,
    KeywordsOptions,
    LlmCacheOptions,
    PromptKeyOptions,
    PromptRetryOptions,
    SummaryOptions,
    validate_llm_document_result,
)
from .parsing import (
    CONTEXT_128K_CHUNK_OVERLAP,
    CONTEXT_128K_CHUNK_SIZE,
    CONTEXT_128K_MAX_CHARS_SINGLE,
    _replace_prompt_placeholders,
    extract_and_validate_json,
    parse_json_field,
    truncate_for_llm,
)
from .prompts import (
    _PLACEHOLDER_ALLOWED_CATEGORIES,
    _build_allowed_categories_instruction,
    _summary_doc_type_hint,
    _summary_prompt_chunk,
    _summary_prompt_combine,
    _summary_prompts_short,
    build_analysis_prompt,
)
from .protocol import LLMClient

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
    options: JsonCompletionOptions | None = None,
) -> str:
    """Retry LLM completion up to max_retries times, increasing temperature, until valid JSON."""
    opts = options or JsonCompletionOptions()
    if cached := _cached_llm_response(opts):
        return cached
    result = _complete_json_attempts(client, prompt, opts)
    if result.valid:
        _store_cached_llm_response(opts, result.response)
        return result.response
    logger.error(
        "LLM returned no valid JSON after %s retries. Using heuristic or 'na' for this document.",
        _effective_retries(opts),
    )
    return result.response


@dataclass(frozen=True)
class JsonCompletionResult:
    """Hold a completion response and whether it parsed as JSON."""

    response: str
    valid: bool


def _cached_llm_response(options: JsonCompletionOptions) -> str | None:
    """Return a cached completion when both cache and key are available."""
    if options.cache is None or options.cache_key is None:
        return None
    cached = options.cache.get(options.cache_key)
    if cached is not None:
        logger.debug("LLM cache hit for %s", options.cache_key)
    return cached


def _store_cached_llm_response(options: JsonCompletionOptions, response: str) -> None:
    """Store a completion when caching is configured."""
    if options.cache is not None and options.cache_key is not None:
        options.cache.set(options.cache_key, response)


def _effective_retries(options: JsonCompletionOptions) -> int:
    """Use one attempt in server-enforced JSON mode; otherwise use the retry limit."""
    return 1 if options.json_mode else options.max_retries


def _response_format(options: JsonCompletionOptions) -> dict[str, str] | None:
    """Return the OpenAI-compatible JSON response-format request when enabled."""
    return {"type": "json_object"} if options.json_mode else None


def _complete_json_attempts(client: LLMClient, prompt: str, options: JsonCompletionOptions) -> JsonCompletionResult:
    """Attempt completion until valid JSON is received or retries are exhausted."""
    temp = options.temperature
    last = ""
    for attempt in range(_effective_retries(options)):
        last = client.complete(
            prompt,
            temperature=temp,
            max_tokens=options.max_tokens,
            response_format=_response_format(options),
        )
        if _is_valid_json_response(last):
            return JsonCompletionResult(last, valid=True)
        temp += _RETRY_TEMP_INCREMENT
        logger.info("Retry %s: New temperature=%s", attempt + 1, temp)
    return JsonCompletionResult(last, valid=False)


def _is_valid_json_response(response: str) -> bool:
    """Return whether a response contains a parseable JSON object."""
    try:
        extract_and_validate_json(response)
    except ValueError:
        return False
    return True


def _try_prompts_for_key(
    client: LLMClient,
    prompts: list[str],
    options: PromptKeyOptions,
) -> str | list[str] | None:
    """Try prompts for key in the defined fallback order."""
    for i, prompt in enumerate(prompts):
        cache_key = _build_cache_key(
            options.cache_key_base,
            operation=f"{options.operation}:{i}",
            model=client.model,
            language=options.language,
            payload=prompt,
        )
        r = complete_json_with_retry(
            client,
            prompt,
            JsonCompletionOptions(
                temperature=options.temperature + i * _RETRY_TEMP_INCREMENT,
                max_tokens=options.max_tokens,
                cache=options.cache,
                cache_key=cache_key,
            ),
        )
        v = parse_json_field(r, key=options.key, lenient=options.lenient)
        if v is not None:
            return v
    return None


def get_document_analysis(
    client: LLMClient,
    pdf_content: str,
    options: AnalysisOptions | None = None,
) -> DocumentAnalysisResult:
    """Extract summary, keywords, and category from document text in a single LLM call.

    Return a DocumentAnalysisResult with empty defaults when content is too short or parsing fails.
    """
    opts = options or AnalysisOptions()
    text = _usable_document_text(pdf_content)
    if text is None:
        return DocumentAnalysisResult()

    text = _truncated_analysis_text(text, opts)

    prompt = build_analysis_prompt(
        opts.language,
        text,
        suggested_doc_type=opts.suggested_doc_type,
        allowed_categories=opts.allowed_categories,
        suggested_categories=opts.suggested_categories,
    )
    raw = complete_json_with_retry(
        client,
        prompt,
        JsonCompletionOptions(
            temperature=opts.temperature,
            max_retries=2,
            max_tokens=1024,
            json_mode=opts.json_mode,
            cache=opts.cache,
            cache_key=_analysis_cache_key(client, prompt, opts),
        ),
    )
    return _validated_analysis_result(raw, opts.lenient_json)


def _usable_document_text(pdf_content: object) -> str | None:
    """Return stripped document text only when it meets the minimum length."""
    if not isinstance(pdf_content, str):
        return None
    text = pdf_content.strip()
    return text if len(text) >= 50 else None


def _truncated_analysis_text(text: str, options: AnalysisOptions) -> str:
    """Truncate analysis input to configured character or token limits."""
    effective_max = CONTEXT_128K_MAX_CHARS_SINGLE
    if options.max_content_chars is not None:
        effective_max = min(CONTEXT_128K_MAX_CHARS_SINGLE, options.max_content_chars)
    return truncate_for_llm(text, effective_max, max_tokens=options.max_content_tokens)


def _analysis_cache_key(client: LLMClient, prompt: str, options: AnalysisOptions) -> str | None:
    """Build the cache key for a single-call document analysis."""
    return _build_cache_key(
        options.cache_key_base,
        operation="analysis",
        model=client.model,
        language=options.language,
        payload=prompt,
    )


def _validated_analysis_result(raw: str, lenient_json: bool) -> DocumentAnalysisResult:
    """Parse and normalize a document-analysis response."""
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
    options: SummaryOptions | None = None,
) -> str:
    """Generate a short document summary via LLM, chunking long content and combining partial summaries.

    Return 'na' when content is missing or too short.
    """
    opts = options or SummaryOptions()
    text = _usable_document_text(pdf_content)
    if text is None:
        return "na"

    original_text_length = len(text)
    text = _truncated_summary_text(text, opts)
    doc_type_hint = _summary_doc_type_hint(opts.language, opts.suggested_doc_type)

    if original_text_length < opts.max_chars_single:
        return _short_document_summary(client, text, doc_type_hint, opts)
    return _chunked_document_summary(client, text, doc_type_hint, opts)


def _truncated_summary_text(text: str, options: SummaryOptions) -> str:
    """Truncate summary input to configured character or token limits."""
    effective_max = options.max_chars_single
    if options.max_content_chars is not None:
        effective_max = min(options.max_chars_single, options.max_content_chars)
    return truncate_for_llm(text, effective_max, max_tokens=options.max_content_tokens)


def _short_document_summary(client: LLMClient, text: str, doc_type_hint: str, options: SummaryOptions) -> str:
    """Summarize a short document using fallback prompt variants."""
    prompts = _summary_prompts_short(options.language, doc_type_hint, text)
    val = _try_prompts_for_key(
        client,
        prompts,
        PromptKeyOptions(
            key="summary",
            operation="summary_short",
            retry=PromptRetryOptions(
                language=options.language,
                temperature=options.temperature,
                lenient=options.lenient_json,
            ),
            cache_options=LlmCacheOptions(options.cache, options.cache_key_base),
        ),
    )
    result = validate_llm_document_result({"summary": val if isinstance(val, str) else ""})
    return result.summary


def _chunked_document_summary(client: LLMClient, text: str, doc_type_hint: str, options: SummaryOptions) -> str:
    """Summarize chunks and combine their partial summaries."""
    combined = _combined_chunk_summaries(client, text, doc_type_hint, options)
    if not combined:
        return "na"
    return _combined_summary_result(client, combined, doc_type_hint, options)


def _chunk_text(text: str, *, chunk_size: int, overlap: int) -> list[str]:
    """Split document content into overlapping LLM context windows."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    if overlap >= chunk_size:
        raise ValueError("overlap must be < chunk_size")
    if not (text and text.strip()):
        return []

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks


def _combined_chunk_summaries(client: LLMClient, text: str, doc_type_hint: str, options: SummaryOptions) -> str:
    """Summarize each text chunk and concatenate successful results."""
    chunks = _chunk_text(
        text,
        chunk_size=CONTEXT_128K_CHUNK_SIZE,
        overlap=CONTEXT_128K_CHUNK_OVERLAP,
    )
    partial: list[str] = []
    for chunk in chunks:
        chunk_prompt = _summary_prompt_chunk(options.language, doc_type_hint, chunk)
        r = complete_json_with_retry(
            client,
            chunk_prompt,
            JsonCompletionOptions(
                temperature=options.temperature,
                max_retries=3,
                max_tokens=1024,
                cache=options.cache,
                cache_key=_build_cache_key(
                    options.cache_key_base,
                    operation=f"summary_chunk:{len(partial)}",
                    model=client.model,
                    language=options.language,
                    payload=chunk_prompt,
                ),
            ),
        )
        v = parse_json_field(r, key="summary", lenient=options.lenient_json)
        partial.append(v if isinstance(v, str) else "")
    return " ".join(p for p in partial if p)


def _combined_summary_result(client: LLMClient, combined: str, doc_type_hint: str, options: SummaryOptions) -> str:
    """Combine partial summaries into the normalized final summary."""
    final_prompt = _summary_prompt_combine(options.language, doc_type_hint, combined)
    r_final = complete_json_with_retry(
        client,
        final_prompt,
        JsonCompletionOptions(
            temperature=options.temperature + _RETRY_TEMP_INCREMENT,
            max_retries=3,
            max_tokens=1024,
            cache=options.cache,
            cache_key=_build_cache_key(
                options.cache_key_base,
                operation="summary_combine",
                model=client.model,
                language=options.language,
                payload=final_prompt,
            ),
        ),
    )
    v_final = parse_json_field(r_final, key="summary", lenient=options.lenient_json)
    result = validate_llm_document_result({"summary": v_final if isinstance(v_final, str) else ""})
    return result.summary


def get_document_keywords(
    client: LLMClient,
    summary: str,
    options: KeywordsOptions | None = None,
) -> tuple[str, ...] | None:
    """Extract 5-7 keywords from a document summary via LLM. Return None on failure."""
    opts = options or KeywordsOptions()
    prompts = _keyword_prompts(summary, opts)
    val = _try_prompts_for_key(
        client,
        prompts,
        PromptKeyOptions(
            key="keywords",
            operation="keywords",
            retry=PromptRetryOptions(
                language=opts.language,
                temperature=opts.temperature,
                max_tokens=512,
                lenient=opts.lenient_json,
            ),
            cache_options=LlmCacheOptions(opts.cache, opts.cache_key_base),
        ),
    )
    result = validate_llm_document_result({"keywords": val if isinstance(val, list) else []})
    return result.keywords if result.keywords else None


def _keyword_prompts(summary: str, options: KeywordsOptions) -> list[str]:
    """Build localized keyword-extraction prompt variants."""
    cat_hint = ""
    if options.suggested_category and options.suggested_category.strip():
        category = options.suggested_category.strip()
        cat_hint = (
            f"Das Dokument ist voraussichtlich: {category}. "
            if options.language == "de"
            else f"The document is likely: {category}. "
        )

    if options.language == "de":
        return [
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
    return [
        (
            cat_hint + "Extract 5–7 keywords from this summary. Return ONLY JSON:\n"
            '{"keywords":["KW1","KW2"]}\n\n'
            "Summary:\n" + summary
        )
    ]


def get_document_category(
    client: LLMClient,
    *,
    summary: str,
    keywords: list[str],
    options: CategoryOptions | None = None,
) -> str:
    """Classify document category via LLM, optionally constrained to allowed/suggested categories."""
    opts = options or CategoryOptions()
    prompts = _category_prompts(summary, keywords, opts)
    val = _try_prompts_for_key(
        client,
        prompts,
        PromptKeyOptions(
            key="category",
            operation="category",
            retry=PromptRetryOptions(
                language=opts.language,
                temperature=opts.temperature,
                max_tokens=256,
                lenient=opts.lenient_json,
            ),
            cache_options=LlmCacheOptions(opts.cache, opts.cache_key_base),
        ),
    )
    result = validate_llm_document_result({"category": _validated_raw_category(val)})
    return result.category


def _category_prompts(summary: str, keywords: list[str], options: CategoryOptions) -> list[str]:
    """Build localized category-classification prompt variants."""
    keywords_joined = ", ".join(keywords)
    if options.language == "de":
        base_text = f"Zusammenfassung:\n{summary}\nKeywords:{keywords_joined}"
    else:
        base_text = f"Summary:\n{summary}\nKeywords:{keywords_joined}"
    category_instruction = _build_allowed_categories_instruction(
        allowed_categories=options.allowed_categories,
        suggested_categories=options.suggested_categories,
        language=options.language,
    )
    content = _replace_prompt_placeholders(
        base_text + "\n\n" + _PLACEHOLDER_ALLOWED_CATEGORIES,
        {_PLACEHOLDER_ALLOWED_CATEGORIES: category_instruction},
    )
    if options.language == "de":
        prompt_templates = [
            "Bestimme eine sinnvolle Kategorie als reines JSON.\n"
            'Gib nur: {"category":"..."}\n\nKeine weiteren Erklärungen. Text:\n',
            'Bitte nur {"category":"..."} - ohne Zusätze:\n',
        ]
        return [t + content for t in prompt_templates]
    return [f'Determine a suitable category. Return ONLY JSON: {{"category":"..."}}\n\nText:\n{content}']


def _validated_raw_category(val: str | list[str] | None) -> str:
    """Reject non-string or implausibly long category responses."""
    raw = val if isinstance(val, str) else ""
    if len(raw.strip()) > 80:
        logger.info("LLM category too long (%d chars); treating as invalid.", len(raw))
        return ""
    return raw


def get_final_summary_tokens(
    client: LLMClient,
    *,
    summary: str,
    keywords: list[str],
    category: str,
    options: FinalSummaryOptions | None = None,
) -> list[str] | None:
    """Derive up to five short tokens from summary metadata via LLM."""
    opts = options or FinalSummaryOptions()
    prompts = _final_summary_prompts(summary, keywords, category, opts.language)
    val = _try_prompts_for_key(
        client,
        prompts,
        PromptKeyOptions(
            key="final_summary",
            operation="final_summary",
            retry=PromptRetryOptions(
                language=opts.language,
                temperature=opts.temperature,
                max_tokens=256,
                lenient=opts.lenient_json,
            ),
            cache_options=LlmCacheOptions(opts.cache, opts.cache_key_base),
        ),
    )
    return _split_final_summary_tokens(val)


def _final_summary_prompts(summary: str, keywords: list[str], category: str, language: str) -> list[str]:
    """Build localized prompts for compact final summary tokens."""
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

    return prompts


def _split_final_summary_tokens(val: str | list[str] | None) -> list[str] | None:
    """Split a comma-separated summary response into at most five tokens."""
    if not isinstance(val, str):
        return None
    tokens = [t.strip() for t in val.split(",") if t.strip()]
    return tokens[:5] if tokens else None
