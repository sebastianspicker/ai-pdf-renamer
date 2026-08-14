"""Unit tests for framework-neutral TUI rename operations."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from folionym.config import RenamerConfig
from folionym.rename_ops import RenameApplyOptions
from folionym.tui_operations import process_single_file, run_directory_rename


def _success_callback(
    _config: RenamerConfig,
    _metadata: dict[str, object],
    _rows: list[dict[str, object]],
) -> Any:
    """Return a callback acceptable to the rename boundary."""
    return lambda *_args, **_kwargs: None


def test_process_single_file_returns_ordered_success_output(tmp_path: Path) -> None:
    """A successful operation keeps suggestion, rename, and metadata order."""
    source = tmp_path / "scan.pdf"
    source.touch()
    captured: list[RenameApplyOptions] = []

    def apply(_source: Path, base: str, options: RenameApplyOptions) -> tuple[bool, Path]:
        captured.append(options)
        return True, tmp_path / f"{base}.pdf"

    result = process_single_file(
        source,
        RenamerConfig(),
        suggest=lambda *_args: ("invoice", {"category": "finance", "summary": "paid"}, None),
        apply=apply,
        sanitize=lambda value: value,
        success_callback=_success_callback,
    )

    assert result.ok is True
    assert result.message == "Completed"
    assert [
        marker for marker in ("Suggested", "Renamed", "category:", "summary:") if marker in "".join(result.log_lines)
    ] == ["Suggested", "Renamed", "category:", "summary:"]
    assert captured[0].dry_run is False


def test_process_single_file_preserves_error_skip_and_apply_failure(tmp_path: Path) -> None:
    """Each non-success path returns the established terminal state and copy."""
    source = tmp_path / "scan.pdf"
    config = RenamerConfig()
    common = {
        "config": config,
        "apply": lambda *_args: (False, source),
        "sanitize": lambda value: value,
        "success_callback": _success_callback,
    }

    errored = process_single_file(source, suggest=lambda *_args: (None, None, ValueError("bad PDF")), **common)
    skipped = process_single_file(source, suggest=lambda *_args: (None, None, None), **common)
    failed = process_single_file(source, suggest=lambda *_args: ("invoice", {}, None), **common)

    assert (errored.ok, errored.message) == (False, "bad PDF")
    assert (skipped.ok, skipped.message) == (True, "Skipped")
    assert (failed.ok, failed.message) == (False, "Could not rename file")


def test_run_directory_rename_removes_handler_on_success_and_failure(monkeypatch: Any) -> None:
    """The logging bridge is always detached and terminal outcomes are reported."""
    root_logger = logging.getLogger()
    handler = logging.NullHandler()
    outcomes: list[tuple[bool, str]] = []
    monkeypatch.setattr("folionym.renamer.rename_pdfs_in_directory", lambda *_args, **_kwargs: None)

    run_directory_rename(".", RenamerConfig(), handler=handler, on_finished=lambda *outcome: outcomes.append(outcome))

    assert outcomes == [(True, "Completed")]
    assert handler not in root_logger.handlers

    def fail(*_args: object, **_kwargs: object) -> None:
        raise OSError("unavailable")

    monkeypatch.setattr("folionym.renamer.rename_pdfs_in_directory", fail)
    run_directory_rename(".", RenamerConfig(), handler=handler, on_finished=lambda *outcome: outcomes.append(outcome))

    assert outcomes[-1] == (False, "unavailable")
    assert handler not in root_logger.handlers
