from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass

import requests

from .llm_parsing import (
    CONTEXT_128K_CHUNK_OVERLAP,
    CONTEXT_128K_CHUNK_SIZE,
    CONTEXT_128K_MAX_CHARS_SINGLE,
    _extract_json_from_response,
    _replace_prompt_placeholders,
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
)
from .llm_schema import validate_llm_document_result
from .rename_ops import sanitize_filename_from_llm
from .text_utils import chunk_text

logger = logging.getLogger(__name__)

# Session per base_url for connection reuse across multiple complete() calls.
# Thread-safe: all accesses are protected by _llm_sessions_lock.
# Note: For multi-process deployments, consider using a connection pool manager
# or passing sessions explicitly via dependency injection.
_llm_sessions: dict[str, requests.Session] = {}
_llm_sessions_lock = threading.Lock()


def close_all_sessions() -> None:
    """Close all global requests sessions."""
    with _llm_sessions_lock:
        for session in _llm_sessions.values():
            try:
                session.close()
            except Exception:
                pass
        _llm_sessions.clear()


@dataclass(frozen=True)
class LocalLLMClient:
    base_url: str = "http://127.0.0.1:11434/v1/completions"
    model: str = "qwen3:8b"
    timeout_s: float = 60.0

    def complete(
        self,
        prompt: str,
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> str:
        payload: dict[str, object] = {
            "model": self.model,
            "prompt": prompt,
            "temperature": temperature,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        try:
            with _llm_sessions_lock:
                session = _llm_sessions.get(self.base_url)
                if session is None:
                    session = requests.Session()
                    session.trust_env = False
                    _llm_sessions[self.base_url] = session

            resp = session.post(
                self.base_url,
                json=payload,
                timeout=self.timeout_s,
            )
            resp.raise_for_status()
            data = resp.json()
            if not isinstance(data, dict):
                logger.warning("LLM response is not a dict: %s", type(data).__name__)
                return ""
            choices = data.get("choices")
            if not isinstance(choices, list) or len(choices) == 0:
                logger.warning(
                    "LLM response missing or empty 'choices' (status=%s)",
                    getattr(resp, "status_code", None),
                )
                return ""
            first_choice = choices[0]
            if not isinstance(first_choice, dict):
                logger.warning(
                    "LLM response 'choices[0]' is not a dict: %s",
                    type(first_choice).__name__,
                )
                return ""
            text = first_choice.get("text", "")
            return str(text).strip() if text is not None else ""
        except (IndexError, AttributeError, TypeError, KeyError) as exc:
            logger.warning("LLM response structure unexpected: %s", exc)
            return ""
        except requests.HTTPError as exc:
            logger.warning(
                "LLM HTTP error: %s (status=%s, body=%s)",
                exc,
                getattr(exc.response, "status_code", None),
                (getattr(exc.response, "text", None) or "")[:500],
            )
            return ""
        except requests.RequestException as exc:
            logger.error(
                "LLM unreachable (%s). Category/summary will use heuristic or 'na' for this document.",
                exc,
            )
            return ""
        except json.JSONDecodeError as exc:
            logger.error(
                "LLM response not valid JSON: %s. Using fallback for this document.",
                exc,
            )
            return ""


def _ollama_chat_url_from_completions_url(completions_url: str) -> str:
    """Derive Ollama /api/chat URL from /v1/completions URL."""
    base = (completions_url or "").strip().rstrip("/")
    if "/v1/completions" in base:
        return base.replace("/v1/completions", "") + "/api/chat"
    if base.endswith("/api/chat"):
        return base
    return base + "/api/chat" if base else "http://127.0.0.1:11434/api/chat"


def complete_vision(
    base_url: str,
    model: str,
    image_b64: str,
    prompt: str,
    *,
    timeout_s: float = 120.0,
) -> str:
    """
    Send image + prompt to Ollama chat API (vision). Returns assistant message text.
    Used when text extraction is empty/short and use_vision_fallback is True.
    """
    chat_url = _ollama_chat_url_from_completions_url(base_url)
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt, "images": [image_b64]},
        ],
        "stream": False,
    }
    try:
        with _llm_sessions_lock:
            session = _llm_sessions.get(base_url)
            if session is None:
                session = requests.Session()
                session.trust_env = False
                _llm_sessions[base_url] = session
        resp = session.post(chat_url, json=payload, timeout=timeout_s)
        resp.raise_for_status()
        data = resp.json()
        if not isinstance(data, dict):
            return ""
        msg = data.get("message")
        if not isinstance(msg, dict):
            return ""
        content = msg.get("content")
        return str(content).strip() if content is not None else ""
    except Exception as exc:
        logger.warning("Vision API failed: %s", exc)
        return ""


def complete_json_with_retry(
    client: LocalLLMClient,
    prompt: str,
    *,
    temperature: float = 0.0,
    max_retries: int = 3,
    max_tokens: int | None = 1024,
) -> str:
    temp = temperature
    last = ""
    for i in range(max_retries):
        last = client.complete(prompt, temperature=temp, max_tokens=max_tokens)
        candidate = last.strip()
        if candidate.startswith("{"):
            try:
                json.loads(candidate)
                return last
            except json.JSONDecodeError:
                pass
        extracted = _extract_json_from_response(last)
        if extracted.startswith("{"):
            try:
                json.loads(extracted)
                return last
            except json.JSONDecodeError:
                pass
        temp += 0.2
        logger.info("Retry %s: New temperature=%s", i + 1, temp)
    logger.error(
        "LLM returned no valid JSON after %s retries. Using heuristic or 'na' for this document.",
        max_retries,
    )
    return last


def _try_prompts_for_key(
    client: LocalLLMClient,
    prompts: list[str],
    *,
    key: str,
    temperature: float,
    max_tokens: int | None = 1024,
    lenient: bool = False,
) -> str | list[str] | None:
    for i, prompt in enumerate(prompts):
        r = complete_json_with_retry(
            client,
            prompt,
            temperature=temperature + i * 0.2,
            max_tokens=max_tokens,
        )
        v = parse_json_field(r, key=key, lenient=lenient)
        if v is not None:
            return v
    return None


def get_document_summary(
    client: LocalLLMClient,
    pdf_content: str,
    *,
    language: str = "de",
    temperature: float = 0.0,
    max_chars_single: int = CONTEXT_128K_MAX_CHARS_SINGLE,
    max_content_chars: int | None = None,
    max_content_tokens: int | None = None,
    suggested_doc_type: str | None = None,
    lenient_json: bool = False,
) -> str:
    if pdf_content is None or not isinstance(pdf_content, str):
        return "na"
    text = pdf_content.strip()
    if len(text) < 50:
        return "na"

    effective_max = max_chars_single
    if max_content_chars is not None:
        effective_max = min(max_chars_single, max_content_chars)
    text = truncate_for_llm(text, effective_max, max_tokens=max_content_tokens)

    doc_type_hint = _summary_doc_type_hint(language, suggested_doc_type)

    if len(text) < max_chars_single:
        prompts = _summary_prompts_short(language, doc_type_hint, text)
        val = _try_prompts_for_key(
            client,
            prompts,
            key="summary",
            temperature=temperature,
            max_tokens=1024,
            lenient=lenient_json,
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
        temperature=temperature + 0.2,
        max_retries=3,
        max_tokens=1024,
    )
    v_final = parse_json_field(r_final, key="summary", lenient=lenient_json)
    result = validate_llm_document_result({"summary": v_final if isinstance(v_final, str) else ""})
    return result.summary


def get_document_keywords(
    client: LocalLLMClient,
    summary: str,
    *,
    language: str = "de",
    temperature: float = 0.0,
    suggested_category: str | None = None,
    lenient_json: bool = False,
) -> list[str] | None:
    cat_hint = ""
    if suggested_category and suggested_category.strip():
        c = suggested_category.strip()
        if language == "de":
            cat_hint = f"Das Dokument ist voraussichtlich: {c}. "
        else:
            cat_hint = f"The document is likely: {c}. "

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
        temperature=temperature,
        max_tokens=512,
        lenient=lenient_json,
    )
    result = validate_llm_document_result({"keywords": val if isinstance(val, list) else []})
    return result.keywords if result.keywords else None


def get_document_category(
    client: LocalLLMClient,
    *,
    summary: str,
    keywords: list[str],
    language: str = "de",
    temperature: float = 0.0,
    suggested_categories: list[str] | None = None,
    allowed_categories: list[str] | None = None,
    lenient_json: bool = False,
) -> str:
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
        temperature=temperature,
        max_tokens=256,
        lenient=lenient_json,
    )
    raw = val if isinstance(val, str) else ""
    if len(raw.strip()) > 80:
        logger.info("LLM category too long (%d chars); treating as invalid.", len(raw))
        raw = ""
    result = validate_llm_document_result({"category": raw})
    return result.category


def get_final_summary_tokens(
    client: LocalLLMClient,
    *,
    summary: str,
    keywords: list[str],
    category: str,
    language: str = "de",
    temperature: float = 0.0,
    lenient_json: bool = False,
) -> list[str] | None:
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
        temperature=temperature,
        max_tokens=256,
        lenient=lenient_json,
    )
    if not isinstance(val, str):
        return None

    tokens = [t.strip() for t in val.split(",") if t.strip()]
    return tokens[:5] if tokens else None


# Max chars of content to send for simple filename (one-shot prompt).
SIMPLE_FILENAME_MAX_CONTENT_CHARS = 12_000

# Placeholders for simple-naming prompt (Montscan-style: single template, one example, strict closing).
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
    client: LocalLLMClient,
    content: str,
    *,
    language: str = "de",
    temperature: float = 0.0,
    max_content_chars: int | None = None,
    max_content_tokens: int | None = None,
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
    raw = client.complete(prompt, temperature=temperature, max_tokens=128)
    if raw and isinstance(raw, str):
        raw = raw.strip().rstrip(".")
    return sanitize_filename_from_llm(raw)
