"""Public composition facade for document filename generation."""

from __future__ import annotations

from dataclasses import replace

from .naming.models import FilenameGenerationRequest
from .naming.service import generate_filename as _generate_filename

__all__ = ["FilenameGenerationRequest", "generate_filename"]


def generate_filename(pdf_content: str, request: FilenameGenerationRequest) -> tuple[str, dict[str, object]]:
    """Generate a filename, owning an HTTP client only for direct LLM-enabled calls."""
    if request.llm_client is not None or not request.config.llm.runtime.use_llm:
        return _generate_filename(pdf_content, request)

    # Keep transport construction at the public composition boundary. Internal
    # naming remains deterministic with respect to its supplied dependencies.
    from .llm.http import create_llm_client_from_config

    client = create_llm_client_from_config(request.config)
    composed_request = replace(
        request,
        dependencies=replace(request.dependencies, llm_client=client),
    )
    try:
        return _generate_filename(pdf_content, composed_request)
    finally:
        client.close()
