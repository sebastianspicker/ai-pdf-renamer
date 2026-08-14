"""Framework-free single-file rename operation for the Textual controller."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rich.markup import escape as escape_markup

from .config import RenamerConfig
from .rename_ops import RenameApplyOptions
from .tui_assets import ERROR_COLOR, SUCCESS_COLOR, WARNING_COLOR


@dataclass(frozen=True)
class SingleFileOperationResult:
    """Worker-ready logs and terminal state for one suggested rename."""

    log_lines: tuple[str, ...]
    ok: bool
    message: str


def process_single_file(  # noqa: PLR0913
    fp: Path,
    config: RenamerConfig,
    *,
    suggest: Callable[[Path, RenamerConfig], tuple[str | None, dict[str, object] | None, BaseException | None]],
    apply: Callable[[Path, str, RenameApplyOptions], tuple[bool, Path]],
    sanitize: Callable[[str], str],
    success_callback: Callable[[RenamerConfig, dict[str, object], list[dict[str, object]]], Callable[..., Any]],
) -> SingleFileOperationResult:
    """Suggest and apply one rename while returning UI-neutral output events."""
    new_base, meta, err = suggest(fp, config)
    if err is not None:
        return SingleFileOperationResult(
            (f"[bold {ERROR_COLOR}]Error:[/bold {ERROR_COLOR}] {escape_markup(str(err))}\n",), False, str(err)
        )
    if new_base is None:
        return SingleFileOperationResult(
            (f"[{WARNING_COLOR}]Skipped -- no extractable content.[/{WARNING_COLOR}]\n",), True, "Skipped"
        )
    metadata = meta or {}
    suggested = new_base + fp.suffix
    export_rows: list[dict[str, object]] = []
    on_success = success_callback(config, metadata, export_rows)
    success, target = apply(
        fp,
        sanitize(new_base),
        RenameApplyOptions(
            plan_file_path=None,
            plan_entries=[],
            dry_run=False,
            backup_dir=config.output.paths.backup_dir,
            on_success=on_success,
            max_filename_chars=config.output.naming.max_filename_chars,
        ),
    )
    if not success:
        return SingleFileOperationResult(
            (
                f"[dim]Suggested:[/dim] [bold]{escape_markup(suggested)}[/bold]\n",
                f"[bold {ERROR_COLOR}]Could not rename file.[/bold {ERROR_COLOR}]\n",
            ),
            False,
            "Could not rename file",
        )
    rename_line = (
        f"[{SUCCESS_COLOR}]Renamed[/{SUCCESS_COLOR}] [dim]{escape_markup(fp.name)}[/dim]"
        f" [{SUCCESS_COLOR} bold]->[/{SUCCESS_COLOR} bold] [bold]{escape_markup(target.name)}[/bold]\n"
    )
    metadata_line = _format_metadata(metadata)
    lines = [f"[dim]Suggested:[/dim] [bold]{escape_markup(suggested)}[/bold]\n", rename_line]
    if metadata_line:
        lines.append(metadata_line)
    return SingleFileOperationResult(tuple(lines), True, "Completed")


def _format_metadata(meta: dict[str, object]) -> str | None:
    """Format inspectable metadata with the exact established field order."""
    parts = [
        f"[dim]{key}:[/dim] {escape_markup(str(value))}"
        for key in ("category", "summary", "keywords", "category_source")
        if (value := meta.get(key))
    ]
    return "  " + "  |  ".join(parts) + "\n" if parts else None


def run_directory_rename(
    directory: str,
    config: RenamerConfig,
    *,
    handler: logging.Handler,
    on_finished: Callable[[bool, str], object],
) -> None:
    """Run the directory operation while keeping Textual worker ownership with the caller."""
    from .renamer import rename_pdfs_in_directory

    root_logger = logging.getLogger()
    root_logger.addHandler(handler)
    try:
        rename_pdfs_in_directory(directory, config=config)
        on_finished(True, "Completed")
    except (AttributeError, KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        on_finished(False, str(exc))
    finally:
        root_logger.removeHandler(handler)
