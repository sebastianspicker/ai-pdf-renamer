"""
LLM prompt builders for summary, category, and filename. Pure text-in/text-out helpers.
"""

from __future__ import annotations

from typing import Any, cast

_PLACEHOLDER_ALLOWED_CATEGORIES = "%ALLOWED_CATEGORIES%"
_DOCUMENT_CONTENT_TEMPLATE = "<document_content>\n{document_content}\n</document_content>"
_SUMMARY_COMBINE_CONTENT_TEMPLATE = "<partial_summaries>\n{combined_summaries}\n</partial_summaries>"
_SUMMARY_SHORT_TEMPLATE = "{doc_type_hint}{instruction} {json_rule}\n\n{document_block}"
_SUMMARY_CHUNK_TEMPLATE = "{doc_type_hint}{instruction} {json_rule}\n\n{document_block}"
_SUMMARY_COMBINE_TEMPLATE = "{doc_type_hint}{intro}{combined}\n\n{instruction} {json_rule}\n"
_ANALYSIS_TEMPLATE = (
    "{doc_type_hint}{analysis_intro}\n"
    "{analysis_schema_intro}\n"
    "{analysis_schema}\n\n"
    "{analysis_examples}"
    "{analysis_rules_heading}\n"
    "- {analysis_summary_rule}\n"
    "- {analysis_keywords_rule}\n"
    "- category: {category_instruction}\n"
    "- {analysis_specificity_rule}\n"
    "- {analysis_generic_rule}\n"
    "- {analysis_failure_mode_rule}\n"
    "- {analysis_json_only_rule}\n\n"
    "{document_block}"
)
_VISION_FILENAME_TEMPLATE = (
    "{intro} {language_line}\n"
    "- 3–6 words, underscores only (no spaces), no special characters except _ and -.\n"
    "- {date_hint}\n"
    "- {uppercase_hint}\n"
    "{example_label}{example}\n\n"
    "{closing}"
)

PROMPT_STRINGS: dict[str, dict[str, Any]] = {
    "de": {
        "doc_type_hint": (
            'Kontext: Das Dokument wurde heuristisch als Typ "{hint}" eingestuft. '
            "Betone in der Zusammenfassung den Dokumenttyp (z. B. Rechnung, Vertrag). "
        ),
        "analysis_intro": "Analysiere das folgende Dokument und gib das Ergebnis als reines JSON zurück.",
        "analysis_schema_intro": "Antworte NUR mit einem JSON-Objekt in genau dieser Struktur:",
        "analysis_schema": (
            '{"summary":"1-2 präzise Sätze","keywords":["KW1","KW2","KW3","KW4","KW5","KW6"],"category":"Kategorie"}'
        ),
        "analysis_examples_heading": "Beispiele:",
        "analysis_examples": [
            '1. Text: "Rechnung Nr. INV-2025-0042 der Muster GmbH über 249,90 EUR vom 15.03.2025."\n'
            '   JSON: {"summary":"Rechnung der Muster GmbH über 249,90 EUR vom 15.03.2025.",'
            '"keywords":["Rechnung","Muster GmbH","INV-2025-0042","249,90 EUR","15.03.2025"],'
            '"category":"invoice"}',
            '2. Text: "Gehaltsabrechnung März 2025 für Erika Mustermann, Nettoauszahlung 2.845,12 EUR."\n'
            '   JSON: {"summary":"Gehaltsabrechnung für März 2025 mit einer Nettoauszahlung von 2.845,12 EUR.",'
            '"keywords":["Gehaltsabrechnung","März 2025","Nettoauszahlung","2.845,12 EUR","Erika Mustermann"],'
            '"category":"payslip"}',
        ],
        "analysis_rules_heading": "Regeln:",
        "analysis_summary_rule": "summary: 1-2 präzise Sätze, die den Dokumentinhalt und -typ beschreiben",
        "analysis_keywords_rule": "keywords: 5-7 relevante Schlüsselwörter",
        "analysis_specificity_rule": (
            "Wenn mehrere Dokumenttypen möglich sind, wähle die spezifischste passende Kategorie."
        ),
        "analysis_generic_rule": (
            'Generische Kategorien wie "document" oder "letter" nur verwenden, '
            "wenn wirklich keine spezifischere Kategorie passt."
        ),
        "analysis_failure_mode_rule": (
            "Vermeide generische oder ausweichende Klassifikationen bei klar erkennbaren Dokumenttypen."
        ),
        "analysis_json_only_rule": "Keine weiteren Erklärungen, nur JSON",
        "allowed_categories_exact": "Gib genau eine dieser Kategorien oder 'unknown': {categories}",
        "allowed_categories_suggested": (
            "Vorschläge nutzen falls passend, sonst andere Kategorie. Vorschläge: {categories}."
        ),
        "allowed_categories_any": "Gib eine passende Kategorie.",
        "summary_short_variants": [
            {
                "instruction": "Fasse den folgenden Text in 1–2 präzisen Sätzen zusammen.",
                "json_rule": 'Nur reines JSON: {"summary":"..."}',
            },
            {
                "instruction": "Extrahiere die wichtigsten Informationen des Dokuments.",
                "json_rule": 'Nur reines JSON: {"summary":"..."}',
            },
        ],
        "summary_chunk_instruction": "Fasse den folgenden Text in 1–2 kurzen Sätzen zusammen.",
        "summary_chunk_json_rule": 'NUR reines JSON {"summary":"..."}, keine Erklärungen.',
        "summary_combine_intro": "Hier mehrere Teilzusammenfassungen eines langen Dokuments:\n",
        "summary_combine_instruction": (
            "Fasse sie in 1–2 prägnanten Sätzen zusammen. Stelle sicher, dass der Dokumenttyp erkennbar bleibt."
        ),
        "summary_combine_json_rule": 'Nur reines JSON {"summary":"..."}.',
        "vision_intro": "Erzeuge einen kurzen Dateinamen (ohne Endung) für dieses gescannte Dokument.",
        "vision_language_line": "In Deutsch.",
        "vision_date_hint": "Optional: Datum am Ende (z. B. 2023-11-15).",
        "vision_uppercase_hint": "Großschreibung bevorzugt.",
        "vision_example_label": "Beispiel: ",
        "vision_example": "RECHNUNG_AMAZON_MAX_2023-11-15",
        "vision_closing": "Antworte mit NUR dem Dateinamen, sonst nichts.",
    },
    "en": {
        "doc_type_hint": (
            'Context: The document was heuristically classified as type "{hint}". '
            "Emphasize in the summary what type of document this is (e.g. invoice, contract). "
        ),
        "analysis_intro": "Analyze the following document and return the result as pure JSON.",
        "analysis_schema_intro": "Respond ONLY with a JSON object in exactly this structure:",
        "analysis_schema": (
            '{"summary":"1-2 precise sentences","keywords":["KW1","KW2","KW3","KW4","KW5","KW6"],"category":"Category"}'
        ),
        "analysis_examples_heading": "Examples:",
        "analysis_examples": [
            '1. Text: "Invoice INV-2025-0042 from Sample GmbH for EUR 249.90 dated 2025-03-15."\n'
            '   JSON: {"summary":"Invoice from Sample GmbH for EUR 249.90 dated 2025-03-15.",'
            '"keywords":["invoice","Sample GmbH","INV-2025-0042","EUR 249.90","2025-03-15"],'
            '"category":"invoice"}',
            '2. Text: "Payslip for March 2025 for Erika Mustermann, net pay EUR 2,845.12."\n'
            '   JSON: {"summary":"Payslip for March 2025 with net pay of EUR 2,845.12.",'
            '"keywords":["payslip","March 2025","net pay","EUR 2,845.12","Erika Mustermann"],'
            '"category":"payslip"}',
        ],
        "analysis_rules_heading": "Rules:",
        "analysis_summary_rule": "summary: 1-2 precise sentences describing the document content and type",
        "analysis_keywords_rule": "keywords: 5-7 relevant keywords",
        "analysis_specificity_rule": (
            "If multiple document types seem possible, choose the most specific applicable category."
        ),
        "analysis_generic_rule": (
            'Do not return generic categories like "document" or "letter" when a more specific category applies.'
        ),
        "analysis_failure_mode_rule": "Avoid vague fallback labels when the document type is identifiable.",
        "analysis_json_only_rule": "No additional explanations, only JSON",
        "allowed_categories_exact": "Return exactly one of these or 'unknown': {categories}",
        "allowed_categories_suggested": (
            "Use one suggestion if appropriate, else another category. Suggestions: {categories}."
        ),
        "allowed_categories_any": "Return any suitable category.",
        "summary_short_variants": [
            {
                "instruction": "Summarize the following text in 1-2 precise sentences.",
                "json_rule": 'Return only valid JSON: {"summary":"..."}',
            },
            {
                "instruction": "Extract the core description of this document.",
                "json_rule": 'Return only valid JSON: {"summary":"..."}',
            },
        ],
        "summary_chunk_instruction": "Summarize the following text in 1–2 short sentences.",
        "summary_chunk_json_rule": 'Return ONLY {"summary":"..."} in JSON, no explanations.',
        "summary_combine_intro": "Here are multiple partial summaries of a large document:\n",
        "summary_combine_instruction": (
            "Combine them into 1–2 concise sentences. Ensure the document type remains clear."
        ),
        "summary_combine_json_rule": 'Return ONLY {"summary":"..."} in JSON.',
        "vision_intro": "Generate a short filename (without extension) for this scanned document.",
        "vision_language_line": "In English.",
        "vision_date_hint": "Optional: date at end (e.g. 2023-11-15).",
        "vision_uppercase_hint": "Uppercase preferred.",
        "vision_example_label": "Example: ",
        "vision_example": "INVOICE_AMAZON_JOHN_2023-11-15",
        "vision_closing": "Respond with ONLY the filename, nothing else.",
    },
}


def _language_code(language: str) -> str:
    """Normalize prompt language to a supported dictionary key."""
    normalized = language.strip().lower()
    primary = normalized.replace("_", "-").split("-", 1)[0]
    return "de" if primary == "de" else "en"


def _prompt_strings(language: str) -> dict[str, Any]:
    """Return language strings for prompt rendering."""
    return PROMPT_STRINGS[_language_code(language)]


def _render_prompt(template: str, /, **replacements: str) -> str:
    """Render a prompt template with explicit placeholders."""
    return template.format(**replacements)


def _document_block(text: str) -> str:
    """Wrap document content in the shared prompt block."""
    return _render_prompt(_DOCUMENT_CONTENT_TEMPLATE, document_content=text)


def _combine_summary_block(text: str) -> str:
    """Wrap partial summaries in a shared block so they stay data, not instructions."""
    return _render_prompt(_SUMMARY_COMBINE_CONTENT_TEMPLATE, combined_summaries=text)


def _analysis_examples(language: str) -> str:
    """Render few-shot examples for the analysis prompt."""
    strings = _prompt_strings(language)
    heading = cast(str, strings["analysis_examples_heading"])
    examples = cast(list[str], strings["analysis_examples"])
    return heading + "\n" + "\n".join(examples) + "\n\n"


def _summary_doc_type_hint(language: str, suggested_doc_type: str | None) -> str:
    """Build doc-type hint prefix for summary prompts."""
    if not suggested_doc_type or not suggested_doc_type.strip():
        return ""
    strings = _prompt_strings(language)
    return cast(str, strings["doc_type_hint"]).format(hint=suggested_doc_type.strip())


def _escape_doc_content(text: str) -> str:
    """Escape prompt block closing tags to prevent prompt injection."""
    import re as _re

    # Escape every closing tag used as a prompt delimiter, case-insensitively.
    return _re.sub(
        r"</(document_content|partial_summaries)>",
        lambda match: f"<\\/{match.group(1)}>",
        text,
        flags=_re.IGNORECASE,
    )


def _summary_prompts_short(language: str, doc_type_hint: str, text: str) -> list[str]:
    """Build prompt for one chunk in long-document summary."""
    strings = _prompt_strings(language)
    safe_text = _escape_doc_content(text)
    variants = cast(list[dict[str, str]], strings["summary_short_variants"])
    return [
        _render_prompt(
            _SUMMARY_SHORT_TEMPLATE,
            doc_type_hint=doc_type_hint,
            instruction=variant["instruction"],
            json_rule=variant["json_rule"],
            document_block=_document_block(safe_text),
        )
        for variant in variants
    ]


def _summary_prompt_chunk(language: str, doc_type_hint: str, chunk: str) -> str:
    """Build prompt for one chunk in long-document summary."""
    safe_chunk = _escape_doc_content(chunk)
    strings = _prompt_strings(language)
    return _render_prompt(
        _SUMMARY_CHUNK_TEMPLATE,
        doc_type_hint=doc_type_hint,
        instruction=cast(str, strings["summary_chunk_instruction"]),
        json_rule=cast(str, strings["summary_chunk_json_rule"]),
        document_block=_document_block(safe_chunk),
    )


def _summary_prompt_combine(language: str, doc_type_hint: str, combined: str) -> str:
    """Build prompt to combine partial summaries into one."""
    strings = _prompt_strings(language)
    safe_combined = _escape_doc_content(combined)
    return _render_prompt(
        _SUMMARY_COMBINE_TEMPLATE,
        doc_type_hint=doc_type_hint,
        intro=cast(str, strings["summary_combine_intro"]),
        combined=_combine_summary_block(safe_combined),
        instruction=cast(str, strings["summary_combine_instruction"]),
        json_rule=cast(str, strings["summary_combine_json_rule"]),
    )


def build_vision_filename_prompt(language: str) -> str:
    """Build Montscan-style strict prompt for vision API: filename only, 3–6 words, underscores, optional date."""
    strings = _prompt_strings(language)
    return _render_prompt(
        _VISION_FILENAME_TEMPLATE,
        intro=cast(str, strings["vision_intro"]),
        language_line=cast(str, strings["vision_language_line"]),
        date_hint=cast(str, strings["vision_date_hint"]),
        uppercase_hint=cast(str, strings["vision_uppercase_hint"]),
        example_label=cast(str, strings["vision_example_label"]),
        example=cast(str, strings["vision_example"]),
        closing=cast(str, strings["vision_closing"]),
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
    strings = _prompt_strings(language)
    category_instruction = _build_allowed_categories_instruction(
        allowed_categories=allowed_categories,
        suggested_categories=suggested_categories,
        language=language,
    )
    return _render_prompt(
        _ANALYSIS_TEMPLATE,
        doc_type_hint=doc_type_hint,
        analysis_intro=cast(str, strings["analysis_intro"]),
        analysis_schema_intro=cast(str, strings["analysis_schema_intro"]),
        analysis_schema=cast(str, strings["analysis_schema"]),
        analysis_examples=_analysis_examples(language),
        analysis_rules_heading=cast(str, strings["analysis_rules_heading"]),
        analysis_summary_rule=cast(str, strings["analysis_summary_rule"]),
        analysis_keywords_rule=cast(str, strings["analysis_keywords_rule"]),
        category_instruction=category_instruction,
        analysis_specificity_rule=cast(str, strings["analysis_specificity_rule"]),
        analysis_generic_rule=cast(str, strings["analysis_generic_rule"]),
        analysis_failure_mode_rule=cast(str, strings["analysis_failure_mode_rule"]),
        analysis_json_only_rule=cast(str, strings["analysis_json_only_rule"]),
        document_block=_document_block(safe_text),
    )


def _build_allowed_categories_instruction(
    *,
    allowed_categories: list[str] | None = None,
    suggested_categories: list[str] | None = None,
    language: str = "de",
) -> str:
    """Build the instruction string for allowed/suggested categories (replaces %ALLOWED_CATEGORIES% in prompts)."""
    strings = _prompt_strings(language)
    if allowed_categories:
        cats = ", ".join(sorted(allowed_categories))
        return cast(str, strings["allowed_categories_exact"]).format(categories=cats)
    if suggested_categories:
        cats = ", ".join(suggested_categories)
        return cast(str, strings["allowed_categories_suggested"]).format(categories=cats)
    return cast(str, strings["allowed_categories_any"])
