"""
LLM prompt builders for summary, category, and filename. Pure text-in/text-out helpers.
"""

from __future__ import annotations

_PLACEHOLDER_ALLOWED_CATEGORIES = "%ALLOWED_CATEGORIES%"


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


def build_vision_filename_prompt(language: str) -> str:
    """Build Montscan-style strict prompt for vision API: filename only, 3–6 words, underscores, optional date."""
    if language == "de":
        return (
            "Erzeuge einen kurzen Dateinamen (ohne Endung) für dieses gescannte Dokument. "
            "In Deutsch.\n"
            "- 3–6 Wörter, nur Unterstriche (keine Leerzeichen), keine Sonderzeichen außer _ und -.\n"
            "- Optional: Datum am Ende (z. B. 15-11-2023).\n"
            "- Großschreibung bevorzugt.\n"
            "Beispiel: RECHNUNG_AMAZON_MAX_2023-11-15\n\n"
            "Antworte mit NUR dem Dateinamen, sonst nichts."
        )
    return (
        "Generate a short filename (without extension) for this scanned document. "
        "In English.\n"
        "- 3–6 words, underscores only (no spaces), no special characters except _ and -.\n"
        "- Optional: date at end (e.g. 15-11-2023).\n"
        "- Uppercase preferred.\n"
        "Example: INVOICE_AMAZON_JOHN_2023-11-15\n\n"
        "Respond with ONLY the filename, nothing else."
    )


def build_analysis_prompt(
    language: str,
    text: str,
    *,
    suggested_doc_type: str | None = None,
    allowed_categories: list[str] | None = None,
    suggested_categories: list[str] | None = None,
) -> str:
    """Build a single prompt that asks for summary, keywords, and category in one JSON response."""
    doc_type_hint = _summary_doc_type_hint(language, suggested_doc_type)
    safe_text = _escape_doc_content(text)
    category_instruction = _build_allowed_categories_instruction(
        allowed_categories=allowed_categories,
        suggested_categories=suggested_categories,
        language=language,
    )
    if language == "de":
        return (
            doc_type_hint + "Analysiere das folgende Dokument und gib das Ergebnis als reines JSON zurück.\n"
            "Antworte NUR mit einem JSON-Objekt in genau dieser Struktur:\n"
            '{"summary":"1-2 präzise Sätze","keywords":["KW1","KW2","KW3","KW4","KW5"],"category":"Kategorie"}\n\n'
            "Regeln:\n"
            "- summary: 1-2 präzise Sätze, die den Dokumentinhalt und -typ beschreiben\n"
            "- keywords: 5-7 relevante Schlüsselwörter\n"
            f"- category: {category_instruction}\n"
            "- Keine weiteren Erklärungen, nur JSON\n\n"
            f"<document_content>\n{safe_text}\n</document_content>"
        )
    return (
        doc_type_hint + "Analyze the following document and return the result as pure JSON.\n"
        "Respond ONLY with a JSON object in exactly this structure:\n"
        '{"summary":"1-2 precise sentences","keywords":["KW1","KW2","KW3","KW4","KW5"],"category":"Category"}\n\n'
        "Rules:\n"
        "- summary: 1-2 precise sentences describing the document content and type\n"
        "- keywords: 5-7 relevant keywords\n"
        f"- category: {category_instruction}\n"
        "- No additional explanations, only JSON\n\n"
        f"<document_content>\n{safe_text}\n</document_content>"
    )


def _build_allowed_categories_instruction(
    *,
    allowed_categories: list[str] | None = None,
    suggested_categories: list[str] | None = None,
    language: str = "de",
) -> str:
    """Build the instruction string for allowed/suggested categories (replaces %ALLOWED_CATEGORIES% in prompts)."""
    if allowed_categories:
        cats = ", ".join(sorted(allowed_categories))
        if language == "de":
            return f"Gib genau eine dieser Kategorien oder 'unknown': {cats}"
        return f"Return exactly one of these or 'unknown': {cats}"
    if suggested_categories:
        cats = ", ".join(suggested_categories)
        if language == "de":
            return f"Vorschläge nutzen falls passend, sonst andere Kategorie. Vorschläge: {cats}."
        return f"Use one suggestion if appropriate, else another category. Suggestions: {cats}."
    if language == "de":
        return "Gib eine passende Kategorie."
    return "Return any suitable category."
