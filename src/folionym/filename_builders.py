"""Final filename string assembly helpers."""

from __future__ import annotations

import logging
from datetime import datetime

from .config import RenamerConfig
from .filename_models import (
    FilenameGenerationRequest,
    _FilenameDependencies,
    _FilenameMetadataParts,
    _FilenameTemplateInput,
    _FilenameTemplateTokens,
)
from .llm_options import SimpleFilenameOptions
from .llm_simple_filename import get_document_filename_simple
from .rename_ops import sanitize_filename_base
from .text_utils import convert_case, extract_structured_fields, split_to_tokens

logger = logging.getLogger(__name__)


def _filename_sep(config: RenamerConfig) -> str:
    """Return filename part separator: '_' for snakeCase, '-' otherwise."""
    return "_" if config.output.naming.desired_case == "snakeCase" else "-"


def _should_use_timestamp_fallback(
    category_for_filename: str,
    category_clean: list[str],
    keyword_clean: list[str],
    summary_clean: list[str],
) -> bool:
    """True when category and all content-derived tokens are empty/unknown (full fallback)."""
    empty_cat = (category_for_filename or "").strip().lower() in {"unknown", "document", "na", ""}
    no_tokens = not category_clean and not keyword_clean and not summary_clean
    return bool(empty_cat and no_tokens)


def _build_timestamp_fallback_filename(
    date_str: str,
    config: RenamerConfig,
    *,
    now: datetime | None = None,
) -> str:
    """Build minimal filename: date + segment + HHMMSS when heuristic+LLM both fail."""
    if now is None:
        now = datetime.now()
    time_str = f"{now.hour:02d}{now.minute:02d}{now.second:02d}"
    segment = (config.output.template.timestamp_fallback_segment or "document").strip() or "document"
    sep = _filename_sep(config)
    filename = sep.join([date_str, segment, time_str])
    filename = _apply_configured_template(
        filename,
        date_str,
        segment,
        [segment, time_str],
        config,
    )
    return _truncate_filename_to_max_chars(filename, config)


def _apply_configured_template(
    filename: str,
    date_str: str,
    category_for_filename: str,
    category_clean: list[str],
    config: RenamerConfig,
) -> str:
    """Apply the configured template to a filename with category-derived tokens."""
    return _apply_filename_template(
        _FilenameTemplateInput(
            filename=filename,
            date_str=date_str,
            project=(config.output.naming.project or "").strip(),
            version=(config.output.naming.version or "").strip(),
            tokens=_FilenameTemplateTokens(
                category_for_filename=category_for_filename,
                category_clean=category_clean,
                keyword_clean=[],
                summary_clean=[],
            ),
        ),
        config,
    )


def _apply_filename_template(template_input: _FilenameTemplateInput, config: RenamerConfig) -> str:
    """If config has a filename template, format and return; else return filename unchanged."""
    filename_template = config.output.template.filename_template
    if not filename_template or not isinstance(filename_template, str):
        return template_input.filename
    filename = _format_filename_template(template_input, config)
    if filename is None:
        return template_input.filename
    if filename.lower().endswith(".pdf"):
        filename = filename[:-4]
    return sanitize_filename_base(filename)


def _template_replacements(template_input: _FilenameTemplateInput, config: RenamerConfig) -> dict[str, str]:
    """Build the values available to the configured filename template."""
    cat_str = (
        convert_case(template_input.category_clean, config.output.naming.desired_case)
        if template_input.category_clean
        else (template_input.category_for_filename or "")
    )
    desired_case = config.output.naming.desired_case
    kw_str = convert_case(template_input.keyword_clean, desired_case) if template_input.keyword_clean else ""
    sum_str = convert_case(template_input.summary_clean, desired_case) if template_input.summary_clean else ""
    sf = template_input.structured_fields or {}
    repl = {
        "date": template_input.date_str,
        "project": template_input.project or "",
        "category": cat_str,
        "keywords": kw_str,
        "summary": sum_str,
        "version": template_input.version or "",
        "invoice_id": sf.get("invoice_id", ""),
        "amount": sf.get("amount", ""),
        "company": sf.get("company", ""),
    }
    return repl


def _format_filename_template(template_input: _FilenameTemplateInput, config: RenamerConfig) -> str | None:
    """Format filename template using the configured naming template."""
    repl = _template_replacements(template_input, config)
    try:
        return str(config.output.template.filename_template).format(**repl).strip()
    except (KeyError, AttributeError, TypeError) as e:
        logger.warning("Template failed (%s); using default filename", e)
        return None


def _truncate_filename_to_max_chars(filename: str, config: RenamerConfig) -> str:
    """Truncate to the configured filename limit, preferably at a separator."""
    max_filename_chars = config.output.naming.max_filename_chars
    if not max_filename_chars or max_filename_chars <= 0 or len(filename) <= max_filename_chars:
        return filename
    _MIN_FILENAME_LENGTH = 8
    effective_max = max(max_filename_chars, _MIN_FILENAME_LENGTH)
    sep = _filename_sep(config)
    while len(filename) > effective_max and sep in filename:
        filename = filename.rsplit(sep, 1)[0]
    if len(filename) > effective_max:
        filename = filename[:effective_max]
    return filename


def _build_filename_str(
    template_input: _FilenameTemplateInput,
    config: RenamerConfig,
) -> str:
    """Build final filename from date, cleaned tokens, and config."""
    if config.output.naming.desired_case == "camelCase":
        filename = _build_camel_case_filename(template_input)
    else:
        filename = _build_separated_filename(template_input, config)

    filename = _apply_filename_template(
        _FilenameTemplateInput(
            filename=filename,
            date_str=template_input.date_str,
            project=template_input.project,
            version=template_input.version,
            tokens=template_input.tokens,
        ),
        config,
    )
    return _truncate_filename_to_max_chars(filename, config)


def _project_version_parts(config: RenamerConfig) -> tuple[str, str]:
    """Return configured project and version, omitting placeholder defaults."""
    project = (config.output.naming.project or "").strip()
    version = (config.output.naming.version or "").strip()
    return ("" if project.lower() == "default" else project, "" if version.lower() == "default" else version)


def _build_camel_case_filename(
    template_input: _FilenameTemplateInput,
) -> str:
    """Build camel case filename from normalized filename inputs."""
    tokens: list[str] = [template_input.date_str]
    tokens += split_to_tokens(template_input.project) if template_input.project else []
    tokens += template_input.category_clean
    tokens += template_input.keyword_clean
    tokens += template_input.summary_clean
    tokens += split_to_tokens(template_input.version) if template_input.version else []
    return convert_case(tokens, "camelCase")


def _build_separated_filename(
    template_input: _FilenameTemplateInput,
    config: RenamerConfig,
) -> str:
    """Build separated filename from normalized filename inputs."""
    parts: list[str] = [template_input.date_str]
    if template_input.project:
        parts.append(convert_case(split_to_tokens(template_input.project), config.output.naming.desired_case))
    parts.append(convert_case(template_input.category_clean, config.output.naming.desired_case))
    if template_input.keyword_clean:
        parts.append(convert_case(template_input.keyword_clean, config.output.naming.desired_case))
    if template_input.summary_clean:
        parts.append(convert_case(template_input.summary_clean, config.output.naming.desired_case))
    if template_input.version:
        parts.append(convert_case(split_to_tokens(template_input.version), config.output.naming.desired_case))
    return _normalize_filename_separators(parts, _filename_sep(config))


def _normalize_filename_separators(parts: list[str], sep: str) -> str:
    """Join non-empty parts and collapse mixed or repeated separators."""
    filename = sep.join(part for part in parts if part)
    alt_sep = "-" if sep == "_" else "_"
    filename = filename.replace(alt_sep, sep)
    return sep.join(part for part in filename.split(sep) if part)


def _with_structured_metadata(metadata: dict[str, object], pdf_content: str, config: RenamerConfig) -> dict[str, str]:
    """Extract structured fields and add them to the result metadata."""
    if not config.extraction.use_structured_fields:
        return {}
    structured_fields = extract_structured_fields(pdf_content)
    metadata["invoice_id"] = structured_fields.get("invoice_id", "")
    metadata["amount"] = structured_fields.get("amount", "")
    metadata["company"] = structured_fields.get("company", "")
    return structured_fields


def _generate_simple_filename(
    pdf_content: str,
    request: FilenameGenerationRequest,
    date_str: str,
    dependencies: _FilenameDependencies,
) -> tuple[str, dict[str, object]]:
    """Generate and sanitize a filename through the one-shot LLM path."""
    config = request.config
    simple_part = get_document_filename_simple(
        dependencies.llm_client,
        pdf_content,
        SimpleFilenameOptions(
            language=config.output.naming.language,
            max_content_chars=config.llm.content.max_content_chars or config.llm.content.max_context_chars,
            max_content_tokens=config.llm.content.max_content_tokens,
            cache=dependencies.response_cache,
            cache_key_base=dependencies.cache_key_base,
        ),
    )
    sep = _filename_sep(config)
    filename = sep.join([date_str, simple_part])
    filename = _apply_configured_template(filename, date_str, simple_part, [simple_part], config)
    filename = _truncate_filename_to_max_chars(filename, config)
    metadata: dict[str, object] = {"category": simple_part, "summary": "", "keywords": ""}
    _with_structured_metadata(metadata, pdf_content, config)
    return sanitize_filename_base(filename), metadata


def _final_generated_filename(
    date_str: str,
    parts: _FilenameMetadataParts,
    structured_fields: dict[str, str],
    config: RenamerConfig,
) -> str:
    """Choose timestamp fallback or assemble the configured final filename."""
    if config.output.template.use_timestamp_fallback and _should_use_timestamp_fallback(
        parts.category_for_filename,
        parts.category_clean,
        parts.keyword_clean,
        parts.summary_clean,
    ):
        return sanitize_filename_base(_build_timestamp_fallback_filename(date_str, config))
    project, version = _project_version_parts(config)
    return _build_filename_str(
        _FilenameTemplateInput(
            filename="",
            date_str=date_str,
            project=project,
            version=version,
            tokens=_FilenameTemplateTokens(
                category_for_filename=parts.category_for_filename,
                category_clean=parts.category_clean,
                keyword_clean=parts.keyword_clean,
                summary_clean=parts.summary_clean,
                structured_fields=structured_fields,
            ),
        ),
        config,
    )
