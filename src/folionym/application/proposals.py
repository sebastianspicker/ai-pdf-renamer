"""Extraction-to-naming proposal production for a single application run."""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Callable, Iterator
from concurrent.futures import CancelledError, ThreadPoolExecutor, wait
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from ..extraction.pdf import get_pdf_metadata
from ..extraction.pipeline import extract_pdf_content as _extract_pdf_content_default
from ..extraction.pipeline import extract_pdf_content_with
from ..infrastructure.errors import COMMON_RECOVERABLE_EXCEPTIONS
from ..infrastructure.files import reject_source_symlink
from ..llm.http import create_llm_client_from_config
from ..llm.protocol import LLMClient, SerializedLLMClient
from ..naming.models import FilenameGenerationContext, FilenameGenerationDependencies, FilenameGenerationRequest
from ..naming.rules import ProcessingRules, force_category_for_basename, load_processing_rules
from ..naming.service import generate_filename
from ..rename_ops import sanitize_filename_base
from ..settings import RenamerConfig
from .models import Proposal
from .scheduling import ProposalProductionDependencies, ProposalProductionRequest, produce_proposals_with

logger = logging.getLogger(__name__)
_RECOVERABLE_PROPOSAL_EXCEPTIONS = (AttributeError, CancelledError, *COMMON_RECOVERABLE_EXCEPTIONS)
ProgressCallback = Callable[[int, int, Path], None]


def _lookup_override_category(file_path: Path, override_map: dict[str, str] | None) -> str | None:
    """Prefer exact path keys, then slash-containing suffix matches for category overrides."""
    if not override_map:
        return None
    keys = [file_path.name, str(file_path), file_path.as_posix()]
    with contextlib.suppress(OSError):
        resolved = file_path.resolve()
        keys.extend([str(resolved), resolved.as_posix()])
    for key in keys:
        if category := override_map.get(key):
            return category
    normalized_candidates = [key.replace("\\", "/") for key in keys]
    for raw_key, category in override_map.items():
        normalized_key = raw_key.replace("\\", "/")
        if "/" in normalized_key and any(
            candidate == normalized_key or candidate.endswith(f"/{normalized_key}")
            for candidate in normalized_candidates
        ):
            return category
    return None


def stop_requested(config: RenamerConfig) -> bool:
    """Return whether the configured stop event exists and is set."""
    stop_event = config.output.hooks.stop_event
    return bool(stop_event is not None and hasattr(stop_event, "is_set") and stop_event.is_set())


def _run_uses_llm_client(config: RenamerConfig) -> bool:
    """Return whether this run can make an LLM or vision request."""
    return bool(config.llm.runtime.use_llm or config.llm.vision.vision_first or config.llm.vision.use_vision_fallback)


def _close_llm_client(client: SerializedLLMClient | None) -> None:
    """Close a run-scoped client while keeping backend cleanup best-effort."""
    if client is None:
        return
    try:
        client.close()
    except (AttributeError, OSError, RuntimeError) as exc:
        logger.warning("Could not close LLM backend cleanly: %s", exc)


@contextmanager
def run_scoped_llm_client(config: RenamerConfig) -> Iterator[LLMClient | None]:
    """Create at most one serialized LLM client and close it when the run exits."""
    raw_client = create_llm_client_from_config(config) if _run_uses_llm_client(config) else None
    client = SerializedLLMClient(raw_client) if raw_client is not None else None
    try:
        yield client
    finally:
        _close_llm_client(client)


@dataclass(frozen=True)
class ContentProcessingRequest:
    """Inputs for filename generation from previously extracted content."""

    file_path: Path
    content: str
    config: RenamerConfig
    rules: ProcessingRules | None = None
    used_vision: bool = False
    llm_client: LLMClient | None = None


def _extract_pdf_content(path: Path, config: RenamerConfig) -> tuple[str, bool]:
    """Extract through the configured default strategy."""
    return _extract_pdf_content_default(path, config)


def _extract_pdf_content_with_client(
    path: Path, config: RenamerConfig, llm_client: LLMClient | None
) -> tuple[str, bool]:
    """Reject source symlinks, then extract with the run-scoped client when supplied."""
    reject_source_symlink(path)
    if llm_client is None:
        return _extract_pdf_content(path, config)
    return extract_pdf_content_with(path, config, llm_client=llm_client)


def process_content_to_proposal(request: ContentProcessingRequest) -> Proposal:
    """Generate a typed proposal from already extracted content."""
    try:
        override_category = _lookup_override_category(
            request.file_path, request.config.output.template.override_category_map
        ) or force_category_for_basename(request.rules, request.file_path.name)
        pdf_metadata = (
            get_pdf_metadata(request.file_path) if request.config.extraction.use_pdf_metadata_for_date else None
        )
        filename, metadata = generate_filename(
            request.content,
            FilenameGenerationRequest(
                config=request.config,
                dependencies=FilenameGenerationDependencies(llm_client=request.llm_client),
                context=FilenameGenerationContext(
                    override_category=override_category,
                    pdf_metadata=pdf_metadata,
                    rules=request.rules,
                    source_path=request.file_path,
                ),
            ),
        )
        metadata = metadata or {}
        metadata["used_vision_fallback"] = request.used_vision
        return Proposal(request.file_path, sanitize_filename_base(filename), metadata)
    except _RECOVERABLE_PROPOSAL_EXCEPTIONS as exc:
        return Proposal(request.file_path, None, None, exc)


def process_one_file(
    file_path: Path,
    config: RenamerConfig,
    rules: ProcessingRules | None = None,
    llm_client: LLMClient | None = None,
) -> Proposal:
    """Extract one PDF and produce a typed filename proposal without renaming it."""
    if stop_requested(config):
        return Proposal(file_path, None, None)
    try:
        content, used_vision = _extract_pdf_content_with_client(file_path, config, llm_client)
    except _RECOVERABLE_PROPOSAL_EXCEPTIONS as exc:
        return Proposal(file_path, None, None, exc)
    if not content.strip():
        return Proposal(file_path, None, None)
    return process_content_to_proposal(
        ContentProcessingRequest(file_path, content, config, rules, used_vision, llm_client)
    )


def suggest_rename_for_file(
    file_path: Path, config: RenamerConfig
) -> tuple[str | None, dict[str, object] | None, BaseException | None]:
    """Produce a single tuple-shaped suggestion without renaming or prompting."""
    rules_file = config.output.paths.rules_file
    try:
        rules = load_processing_rules(rules_file, raise_on_error=bool(rules_file))
        with run_scoped_llm_client(config) as llm_client:
            proposal = process_one_file(file_path, config, rules, llm_client)
    except _RECOVERABLE_PROPOSAL_EXCEPTIONS as exc:
        return (None, None, exc)
    return (proposal.proposed_base, proposal.metadata, proposal.error)


def produce_proposals(
    files: list[Path],
    config: RenamerConfig,
    rules: ProcessingRules | None = None,
    progress_callback: ProgressCallback | None = None,
) -> list[Proposal]:
    """Produce typed results in input order, with bounded parallelism when configured."""
    if not files:
        return []
    workers = 1 if config.output.mode.interactive else max(1, config.output.traversal.workers or 1)
    try:
        with run_scoped_llm_client(config) as llm_client:
            return produce_proposals_with(
                ProposalProductionRequest(files, config, rules, progress_callback, workers, llm_client),
                ProposalProductionDependencies(
                    process_one_file=process_one_file,
                    stop_requested=stop_requested,
                    recoverable_exceptions=_RECOVERABLE_PROPOSAL_EXCEPTIONS,
                    logger=logger,
                    executor_factory=ThreadPoolExecutor,
                    wait_for_futures=wait,
                ),
            )
    except _RECOVERABLE_PROPOSAL_EXCEPTIONS as exc:
        return [Proposal(path, None, None, exc) for path in files]
