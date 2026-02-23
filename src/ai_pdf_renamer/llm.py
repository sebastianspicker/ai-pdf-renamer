from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

import requests
import threading

from .text_utils import chunk_text

logger = logging.getLogger(__name__)

# Session per base_url for connection reuse across multiple complete() calls.
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


# Qwen3 8B 128K context: single-shot up to ~120K tokens (~480K chars).
CONTEXT_128K_MAX_CHARS_SINGLE = 480_000  # ~120K tokens at ~4 chars/token
CONTEXT_128K_CHUNK_SIZE = 100_000
CONTEXT_128K_CHUNK_OVERLAP = 5_000


def _summary_doc_type_hint(language: str, suggested_doc_type: str | None) -> str:
    """Build doc-type hint prefix for summary prompts."""
    if not suggested_doc_type or not suggested_doc_type.strip():
        return ""
    hint = suggested_doc_type.strip()
    if language == "de":
        return (
            f'Kontext: Das Dokument wurde heuristisch als Typ "{hint}" eingestuft. '
            "Betone in der Zusammenfassung den Dokumenttyp (z. B. Rechnung, Vertrag). "
        )
    return (
        f'Context: The document was heuristically classified as type "{hint}". '
        "Emphasize in the summary what type of document this is (e.g. invoice, contract). "
    )


def _escape_doc_content(text: str) -> str:
    """Escape closing tags to prevent prompt injection."""
    return text.replace("</document_content>", "<\\/document_content>")


def _summary_prompts_short(language: str, doc_type_hint: str, text: str) -> list[str]:
    """Build prompt for one chunk in long-document summary."""
    safe_text = _escape_doc_content(text)
    if language == "de":
        return [
            doc_type_hint
            + "Fasse den folgenden Text in 1–2 präzisen Sätzen zusammen. "
            + 'Nur reines JSON: {"summary":"..."}\n\n'
            + "<document_content>\n"
            + safe_text
            + "\n</document_content>",
            doc_type_hint
            + "Extrahiere die wichtigsten Informationen des Dokuments. "
            + 'Nur reines JSON: {"summary":"..."}\n\n'
            + "<document_content>\n"
            + safe_text
            + "\n</document_content>",
        ]
    return [
        doc_type_hint
        + "Summarize the following text in 1-2 precise sentences. "
        + 'Return only valid JSON: {"summary":"..."}\n\n'
        + "<document_content>\n"
        + safe_text
        + "\n</document_content>",
        doc_type_hint
        + "Extract the core description of this document. "
        + 'Return only valid JSON: {"summary":"..."}\n\n'
        + "<document_content>\n"
        + safe_text
        + "\n</document_content>",
    ]


def _summary_prompt_chunk(language: str, doc_type_hint: str, chunk: str) -> str:
    """Build prompt for one chunk in long-document summary."""
    if language == "de":
        return (
            doc_type_hint + "Fasse den folgenden Text in 1–2 kurzen Sätzen zusammen. "
            'NUR reines JSON {"summary":"..."}, keine Erklärungen.\n\n'
            f"<document_content>\n{chunk}\n</document_content>"
        )
    return (
        doc_type_hint + "Summarize the following text in 1–2 short sentences. "
        'Return ONLY {"summary":"..."} in JSON, no explanations.\n\n'
        f"<document_content>\n{chunk}\n</document_content>"
    )


def _summary_prompt_combine(language: str, doc_type_hint: str, combined: str) -> str:
    """Build prompt to combine partial summaries into one."""
    if language == "de":
        return (
            doc_type_hint
            + "Hier mehrere Teilzusammenfassungen eines langen Dokuments:\n"
            + combined
            + "\n\nFasse sie in 1–2 prägnanten Sätzen zusammen. "
            "Stelle sicher, dass der Dokumenttyp erkennbar bleibt. "
            'Nur reines JSON {"summary":"..."}.\n'
        )
    return (
        doc_type_hint
        + "Here are multiple partial summaries of a large document:\n"
        + combined
        + "\n\nCombine them into 1–2 concise sentences. "
        "Ensure the document type remains clear. "
        'Return ONLY {"summary":"..."} in JSON.\n'
    )


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
    suggested_doc_type: str | None = None,
    lenient_json: bool = False,
) -> str:
    if pdf_content is None or not isinstance(pdf_content, str):
        return "na"
    text = pdf_content.strip()
    if len(text) < 50:
        return "na"

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
        return val if isinstance(val, str) else "na"

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
    return v_final if isinstance(v_final, str) else "na"


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
    return val if isinstance(val, list) else None


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
        if allowed_categories:
            cats = ", ".join(sorted(allowed_categories))
            base_text += f"\n\nGib genau eine dieser Kategorien oder 'unknown': {cats}"
        elif suggested_categories:
            base_text += "\n\nVorschläge nutzen falls passend, sonst andere Kategorie."
            base_text += " Vorschläge: " + ", ".join(suggested_categories) + "."
        prompts = [
            (
                "Bestimme eine sinnvolle Kategorie als reines JSON.\n"
                'Gib nur: {"category":"..."}\n\n'
                "Keine weiteren Erklärungen, nur JSON. Text:\n" + base_text
            ),
            ('Bitte nur {"category":"..."} - ohne Zusätze:\n' + base_text),
        ]
    else:
        base_text = f"Summary:\n{summary}\nKeywords:{keywords_joined}"
        if allowed_categories:
            cats = ", ".join(sorted(allowed_categories))
            base_text += f"\n\nReturn exactly one of these or 'unknown': {cats}"
        elif suggested_categories:
            base_text += "\n\nUse one suggestion if appropriate, else another category."
            base_text += " Suggestions: " + ", ".join(suggested_categories) + "."
        prompts = [('Determine a suitable category. Return ONLY JSON: {"category":"..."}\n\nText:\n' + base_text)]

    val = _try_prompts_for_key(
        client,
        prompts,
        key="category",
        temperature=temperature,
        max_tokens=256,
        lenient=lenient_json,
    )
    if not isinstance(val, str):
        return "na"
    if len(val.strip()) > 80:
        logger.info("LLM category too long (%d chars); treating as invalid.", len(val))
        return "na"
    return val


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
