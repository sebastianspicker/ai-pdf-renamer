"""Structured preview and reviewed-plan application for interactive frontends."""

from __future__ import annotations

import os
import stat
from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from threading import Event
from uuid import uuid4

from .config import RenamerConfig
from .rename_ops import RenameApplyOptions, apply_single_rename, sanitize_filename_base
from .renamer import produce_rename_results
from .renamer_discovery import (
    _collect_sorted_pdf_files_with,
    _load_effective_rules,
    _resolve_rename_directory,
)
from .renamer_files import collect_pdf_files, reject_source_symlink
from .renamer_hooks import _make_post_rename_success_callback
from .renamer_output import RenameOutputData, RenameSummaryData, _write_export_metadata, _write_summary_json

ProgressCallback = Callable[[int, int, Path], None]


class PreviewStatus(StrEnum):
    """Operator-facing result classification for one preview item."""

    READY = "ready"
    REVIEW = "review"
    SKIPPED = "skipped"
    FAILED = "failed"


class ApplyStatus(StrEnum):
    """Outcome of one item in a reviewed apply run."""

    RENAMED = "renamed"
    SKIPPED = "skipped"
    UNCHANGED = "unchanged"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class FileFingerprint:
    """Filesystem identity used to reject stale reviewed plans."""

    device: int
    inode: int
    size: int
    modified_ns: int

    @classmethod
    def capture(cls, path: Path) -> FileFingerprint:
        """Capture a regular, non-symlink file identity."""
        current = path.stat(follow_symlinks=False)
        if not stat.S_ISREG(current.st_mode):
            raise OSError(f"Source is not a regular file: {path}")
        return cls(current.st_dev, current.st_ino, current.st_size, current.st_mtime_ns)

    def matches(self, path: Path) -> bool:
        """Return whether ``path`` still identifies the reviewed file bytes."""
        try:
            return self == self.capture(path)
        except OSError:
            return False


@dataclass(frozen=True)
class PreviewItem:
    """One structured source-to-proposal result."""

    id: str
    source: Path
    proposed_base: str | None
    metadata: dict[str, object]
    status: PreviewStatus
    included: bool
    fingerprint: FileFingerprint | None
    reason: str | None = None

    @property
    def proposed_name(self) -> str | None:
        """Return the complete proposed filename when a proposal exists."""
        if self.proposed_base is None:
            return None
        return self.proposed_base + self.source.suffix


@dataclass(frozen=True)
class PreviewPlan:
    """Immutable reviewed-plan input retained by the local process."""

    id: str
    source: Path
    source_kind: str
    config: RenamerConfig
    items: tuple[PreviewItem, ...]
    created_at: datetime
    revision: int = 1


@dataclass(frozen=True)
class ApplyItemResult:
    """One reviewed-plan apply outcome."""

    item_id: str
    source_name: str
    target_name: str | None
    status: ApplyStatus
    reason: str | None = None


@dataclass(frozen=True)
class ApplyReport:
    """Complete result of applying a selected subset of a preview plan."""

    id: str
    plan_id: str
    source: Path
    started_at: datetime
    completed_at: datetime
    items: tuple[ApplyItemResult, ...]

    def count(self, status: ApplyStatus) -> int:
        """Count outcomes with ``status``."""
        return sum(item.status == status for item in self.items)


def _result_status(
    proposed_base: str | None,
    metadata: dict[str, object],
    error: BaseException | None,
) -> tuple[PreviewStatus, str | None]:
    """Classify a core rename result without inventing confidence."""
    if error is not None:
        return (PreviewStatus.FAILED, str(error))
    if proposed_base is None:
        return (PreviewStatus.SKIPPED, "No extractable text or filename proposal.")
    category = str(metadata.get("category", "")).strip().lower()
    if bool(metadata.get("llm_failed")) or category in {"", "document", "unknown", "na"}:
        return (PreviewStatus.REVIEW, "The category or model result needs review.")
    return (PreviewStatus.READY, None)


def _source_scope(source: Path) -> tuple[Path, str, list[Path] | None]:
    """Resolve a directory or one-PDF source into core discovery inputs."""
    expanded = source.expanduser()
    if expanded.is_file():
        if expanded.suffix.lower() != ".pdf":
            raise ValueError("Single-file sources must be PDFs.")
        resolved_file = expanded.resolve()
        return (resolved_file.parent, "file", [resolved_file])
    return (_resolve_rename_directory(expanded, files_override=None), "directory", None)


def _capture_fingerprints(files: Iterable[Path]) -> dict[Path, FileFingerprint | None]:
    """Capture source identities before expensive preview processing begins."""
    fingerprints: dict[Path, FileFingerprint | None] = {}
    for path in files:
        try:
            fingerprints[path] = FileFingerprint.capture(path)
        except OSError:
            fingerprints[path] = None
    return fingerprints


def _preview_items(
    results: Iterable[tuple[Path, str | None, dict[str, object] | None, BaseException | None]],
    fingerprints: dict[Path, FileFingerprint | None],
) -> tuple[PreviewItem, ...]:
    """Translate core rename results into reviewed items with stale-source protection."""
    items: list[PreviewItem] = []
    for index, (path, proposed_base, metadata, error) in enumerate(results, start=1):
        current_meta: dict[str, object] = metadata or {}
        fingerprint = fingerprints.get(path)
        status: PreviewStatus
        reason: str | None
        if fingerprint is None or not fingerprint.matches(path):
            status, reason, fingerprint = (
                PreviewStatus.FAILED,
                "The source changed while Preview was running.",
                None,
            )
        else:
            status, reason = _result_status(proposed_base, current_meta, error)
        items.append(
            PreviewItem(
                id=f"item-{index}",
                source=path,
                proposed_base=proposed_base,
                metadata=current_meta,
                status=status,
                included=status is PreviewStatus.READY,
                fingerprint=fingerprint,
                reason=reason,
            )
        )
    return tuple(items)


def _plan_source(directory: Path, source_kind: str, files_override: list[Path] | None) -> Path:
    """Return the plan's canonical source while validating the file-source contract."""
    if source_kind != "file":
        return directory
    if not files_override:
        raise ValueError("Single-file preview source is unavailable.")
    return files_override[0]


def create_preview_plan(
    source: str | Path,
    config: RenamerConfig,
    *,
    progress_callback: ProgressCallback | None = None,
) -> PreviewPlan:
    """Process a local source into an immutable structured preview plan."""
    directory, source_kind, files_override = _source_scope(Path(source))
    rules = _load_effective_rules(config, None)
    files = _collect_sorted_pdf_files_with(
        directory,
        config,
        files_override=files_override,
        rules=rules,
        collect_pdf_files_fn=collect_pdf_files,
    )
    fingerprints = _capture_fingerprints(files)
    results = produce_rename_results(files, config, rules=rules, progress_callback=progress_callback)
    return PreviewPlan(
        id=uuid4().hex,
        source=_plan_source(directory, source_kind, files_override),
        source_kind=source_kind,
        config=config,
        items=_preview_items(results, fingerprints),
        created_at=datetime.now(UTC),
    )


def _apply_config(plan: PreviewPlan, stop_event: Event) -> RenamerConfig:
    """Return the immutable Preview configuration in apply mode."""
    output = replace(
        plan.config.output,
        mode=replace(plan.config.output.mode, dry_run=False, interactive=False, manual_mode=False),
        hooks=replace(plan.config.output.hooks, stop_event=stop_event),
    )
    return replace(plan.config, output=output)


def _duplicate_target_ids(items: Iterable[PreviewItem]) -> set[str]:
    """Return item IDs whose exact reviewed targets collide within the selection."""
    by_target: dict[tuple[str, str], list[str]] = {}
    for item in items:
        proposed_name = item.proposed_name
        if proposed_name is None:
            continue
        key = (str(item.source.parent.resolve()).casefold(), proposed_name.casefold())
        by_target.setdefault(key, []).append(item.id)
    return {item_id for ids in by_target.values() if len(ids) > 1 for item_id in ids}


def _write_web_outputs(config: RenamerConfig, source: Path, output: RenameOutputData) -> None:
    """Write configured machine-readable outputs without terminal summary rendering."""
    _write_export_metadata(config, output)
    _write_summary_json(
        config.output.paths.summary_json_path,
        RenameSummaryData(
            directory=source if source.is_dir() else source.parent,
            processed=output.processed_count,
            renamed=output.renamed_count,
            skipped=output.skipped_count,
            failed=output.failed_count,
            dry_run=False,
            failures=output.failure_details,
        ),
    )


def _failed_apply_item(item: PreviewItem, reason: str) -> ApplyItemResult:
    """Build a failed exact-plan outcome without leaking implementation details."""
    return ApplyItemResult(item.id, item.source.name, item.proposed_name, ApplyStatus.FAILED, reason)


def _preflight_selected_item(item: PreviewItem, *, duplicate_target: bool) -> ApplyItemResult | None:
    """Return an immediate non-rename outcome when exact-plan preflight fails."""
    if duplicate_target:
        return _failed_apply_item(item, "Another selected item has the same reviewed target.")
    if item.fingerprint is None or not item.fingerprint.matches(item.source):
        return _failed_apply_item(item, "The source changed after Preview.")
    if item.proposed_base is None:
        return _failed_apply_item(item, "No reviewed filename is available.")
    return None


def _perform_exact_rename(item: PreviewItem, config: RenamerConfig, output: RenameOutputData) -> ApplyItemResult:
    """Apply one preflighted item without silently changing its reviewed name."""
    if item.proposed_base is None:
        return _failed_apply_item(item, "No reviewed filename is available.")
    try:
        reject_source_symlink(item.source)
        base = sanitize_filename_base(item.proposed_base)
        exact_target = item.source.with_name(base + item.source.suffix)
        if exact_target == item.source:
            return ApplyItemResult(
                item.id,
                item.source.name,
                exact_target.name,
                ApplyStatus.UNCHANGED,
                "The reviewed filename already matches the source.",
            )
        if os.path.lexists(exact_target):
            return _failed_apply_item(item, "The reviewed target already exists.")
        on_success = _make_post_rename_success_callback(config, item.metadata, output.export_rows)
        success, target = apply_single_rename(
            item.source,
            base,
            RenameApplyOptions(
                dry_run=False,
                backup_dir=config.output.paths.backup_dir,
                on_success=on_success,
                max_filename_chars=config.output.naming.max_filename_chars,
                exact_target=True,
            ),
        )
        if success:
            return ApplyItemResult(item.id, item.source.name, target.name, ApplyStatus.RENAMED)
        return _failed_apply_item(item, "The reviewed target became unavailable.")
    except (OSError, ValueError) as exc:
        return _failed_apply_item(item, str(exc))


def _apply_selected_item(
    item: PreviewItem,
    config: RenamerConfig,
    output: RenameOutputData,
    *,
    duplicate_target: bool,
) -> ApplyItemResult:
    """Revalidate and apply one selected item without changing its reviewed target."""
    preflight = _preflight_selected_item(item, duplicate_target=duplicate_target)
    return preflight if preflight is not None else _perform_exact_rename(item, config, output)


def _record_apply_output(output: RenameOutputData, item: PreviewItem, result: ApplyItemResult) -> None:
    """Update configured output counters from one structured apply result."""
    output.processed_count += 1
    if result.status is ApplyStatus.RENAMED:
        output.renamed_count += 1
    elif result.status is ApplyStatus.UNCHANGED:
        output.skipped_count += 1
    else:
        output.failed_count += 1
        output.failure_details.append({"file": str(item.source), "error": result.reason or "Rename failed."})


def _selected_items(plan: PreviewPlan, selected: frozenset[str]) -> list[PreviewItem]:
    """Validate the requested item IDs and return their reviewed items."""
    selectable = {item.id: item for item in plan.items if item.status in {PreviewStatus.READY, PreviewStatus.REVIEW}}
    unknown = selected - selectable.keys()
    if unknown:
        raise ValueError(f"Selected items are not applicable: {', '.join(sorted(unknown))}")
    return [selectable[item_id] for item_id in selected]


def _apply_plan_items(
    items: tuple[PreviewItem, ...],
    selected: frozenset[str],
    config: RenamerConfig,
    state: tuple[RenameOutputData, set[str], Event],
    progress_callback: ProgressCallback | None,
) -> list[ApplyItemResult]:
    """Apply selected items in preview order while preserving every unselected outcome."""
    output, duplicate_ids, stop_event = state
    results: list[ApplyItemResult] = []
    completed = 0
    for item in items:
        if item.id not in selected:
            results.append(
                ApplyItemResult(item.id, item.source.name, item.proposed_name, ApplyStatus.UNCHANGED, "Not selected.")
            )
            continue
        if stop_event.is_set():
            results.append(
                ApplyItemResult(item.id, item.source.name, item.proposed_name, ApplyStatus.CANCELLED, "Run cancelled.")
            )
            continue
        result = _apply_selected_item(item, config, output, duplicate_target=item.id in duplicate_ids)
        _record_apply_output(output, item, result)
        results.append(result)
        completed += 1
        if progress_callback is not None:
            progress_callback(completed, len(selected), item.source)
    return results


def apply_reviewed_plan(
    plan: PreviewPlan,
    selected_ids: Iterable[str],
    *,
    stop_event: Event | None = None,
    progress_callback: ProgressCallback | None = None,
) -> ApplyReport:
    """Apply only selected reviewed names after per-file identity and collision validation."""
    started_at = datetime.now(UTC)
    selected = frozenset(selected_ids)
    selected_items = _selected_items(plan, selected)
    duplicate_ids = _duplicate_target_ids(selected_items)
    active_stop_event = stop_event or Event()
    active_stop_event.clear()
    config = _apply_config(plan, active_stop_event)
    output = RenameOutputData()
    results = _apply_plan_items(
        plan.items,
        selected,
        config,
        (output, duplicate_ids, active_stop_event),
        progress_callback,
    )
    _write_web_outputs(config, plan.source, output)
    return ApplyReport(
        id=uuid4().hex,
        plan_id=plan.id,
        source=plan.source,
        started_at=started_at,
        completed_at=datetime.now(UTC),
        items=tuple(results),
    )
