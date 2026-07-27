"""Rename-run output collection, serialization, and summary rendering."""

from __future__ import annotations

import csv
import io
import json
import logging
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import RenamerConfig
from .private_io import atomic_write_private_text

_CSV_FORMULA_TRIGGERS = frozenset("=+-@|")
_CSV_CONTROL_CHARS_RE = re.compile(r"[\t\r\n]")
logger = logging.getLogger(__name__)


@dataclass
class RenameOutputData:
    """Mutable output rows and counters accumulated during one rename run."""

    export_rows: list[dict[str, object]] = field(default_factory=list)
    plan_entries: list[dict[str, str]] = field(default_factory=list)
    processed_count: int = 0
    renamed_count: int = 0
    skipped_count: int = 0
    failed_count: int = 0
    failure_details: list[dict[str, str]] = field(default_factory=list)


@dataclass(frozen=True)
class RenameSummaryData:
    """Immutable values used to serialize the final run summary."""

    directory: Path
    processed: int
    renamed: int
    skipped: int
    failed: int
    dry_run: bool
    failures: list[dict[str, str]]


def _sanitize_csv_cell(value: object) -> object:
    """Prevent CSV formula injection in spreadsheet apps (OWASP)."""
    if not isinstance(value, str) or not value:
        return value
    cleaned = _CSV_CONTROL_CHARS_RE.sub(" ", value)
    stripped = cleaned.lstrip()
    if stripped and stripped[0] in _CSV_FORMULA_TRIGGERS:
        return "'" + cleaned
    return cleaned


def _write_json_or_csv(path: Path, rows: list[Any], csv_fieldnames: list[str] | None) -> None:
    """Write rows to path as CSV (if csv_fieldnames and .csv suffix) or JSON."""
    path = Path(path)
    output = io.StringIO(newline="")
    if path.suffix.lower() == ".csv" and csv_fieldnames:
        writer = csv.DictWriter(output, fieldnames=csv_fieldnames)
        writer.writeheader()
        writer.writerows({k: _sanitize_csv_cell(v) for k, v in row.items()} for row in rows)
    else:
        json.dump(rows, output, ensure_ascii=False, indent=2)
    atomic_write_private_text(path, output.getvalue())


def _write_summary_json(
    summary_path: str | Path | None,
    summary_data: RenameSummaryData,
) -> None:
    """Write summary JSON; best-effort logging on I/O errors."""
    if not summary_path:
        return
    summary = {
        "directory": str(summary_data.directory),
        "processed": summary_data.processed,
        "renamed": summary_data.renamed,
        "skipped": summary_data.skipped,
        "failed": summary_data.failed,
        "dry_run": summary_data.dry_run,
        "failures": summary_data.failures,
    }
    target = Path(summary_path)
    try:
        atomic_write_private_text(target, json.dumps(summary, ensure_ascii=False, indent=2))
    except OSError as exc:
        logger.warning("Could not write summary JSON %s: %s", target, exc)


def _append_export_row(
    export_rows: list[dict[str, object]],
    *,
    file_path: Path,
    target: Path,
    meta: dict[str, object],
) -> None:
    """Append one file's rename and extraction metadata to the export rows."""
    export_rows.append(
        {
            "path": str(file_path),
            "new_name": target.name,
            "category": meta.get("category", ""),
            "summary": meta.get("summary", ""),
            "keywords": meta.get("keywords", ""),
            "category_source": meta.get("category_source", ""),
            "llm_failed": meta.get("llm_failed", False),
            "used_vision_fallback": meta.get("used_vision_fallback", False),
            "invoice_id": meta.get("invoice_id", ""),
            "amount": meta.get("amount", ""),
            "company": meta.get("company", ""),
        }
    )


def _write_export_metadata(config: RenamerConfig, output: RenameOutputData) -> None:
    """Write non-empty export rows as private JSON or CSV when configured."""
    export_metadata_path = config.output.paths.export_metadata_path
    if export_metadata_path and output.export_rows:
        _write_json_or_csv(
            Path(export_metadata_path),
            output.export_rows,
            [
                "path",
                "new_name",
                "category",
                "summary",
                "keywords",
                "category_source",
                "llm_failed",
                "used_vision_fallback",
                "invoice_id",
                "amount",
                "company",
            ],
        )


def _write_plan_file(config: RenamerConfig, output: RenameOutputData) -> None:
    """Write accumulated plan entries as private JSON or CSV and log the destination."""
    plan_file_path = config.output.paths.plan_file_path
    if plan_file_path and output.plan_entries:
        plan_path = Path(plan_file_path)
        _write_json_or_csv(plan_path, output.plan_entries, ["old", "new"])
        logger.info("Wrote rename plan (%s entries) to %s", len(output.plan_entries), plan_path)


def _log_summary(output: RenameOutputData) -> None:
    """Log the final rename-run counters."""
    logger.info(
        "Summary: %s file(s) processed, %s renamed, %s skipped, %s failed",
        output.processed_count,
        output.renamed_count,
        output.skipped_count,
        output.failed_count,
    )


def _print_rich_summary(output: RenameOutputData) -> None:
    """Print a colorized run summary to stderr with Rich."""
    from rich.console import Console

    _con = Console(stderr=True)
    _con.print()
    parts = [f"[bold]{output.processed_count}[/bold] processed"]
    if output.renamed_count:
        parts.append(f"[green]{output.renamed_count} renamed[/green]")
    else:
        parts.append(f"{output.renamed_count} renamed")
    if output.skipped_count:
        parts.append(f"[yellow]{output.skipped_count} skipped[/yellow]")
    else:
        parts.append(f"{output.skipped_count} skipped")
    if output.failed_count:
        parts.append(f"[red]{output.failed_count} failed[/red]")
    else:
        parts.append(f"{output.failed_count} failed")
    _con.print("[bold]Summary:[/bold] " + ", ".join(parts))


def _print_plain_summary(output: RenameOutputData) -> None:
    """Print a plain-text run summary to stderr."""
    print(
        f"Summary: {output.processed_count} processed, {output.renamed_count} renamed, "
        f"{output.skipped_count} skipped, {output.failed_count} failed.",
        file=sys.stderr,
    )


def _print_summary(output: RenameOutputData) -> None:
    """Print a Rich summary when available, otherwise use plain text."""
    try:
        _print_rich_summary(output)
    except ImportError:
        _print_plain_summary(output)


def _write_rename_outputs(config: RenamerConfig, directory: Path, output: RenameOutputData) -> None:
    """Write export metadata, plan file, and summary JSON after rename loop completes."""
    _write_export_metadata(config, output)
    _write_plan_file(config, output)
    _write_summary_json(
        config.output.paths.summary_json_path,
        RenameSummaryData(
            directory=directory,
            processed=output.processed_count,
            renamed=output.renamed_count,
            skipped=output.skipped_count,
            failed=output.failed_count,
            dry_run=config.output.mode.dry_run,
            failures=output.failure_details,
        ),
    )
    _log_summary(output)
    _print_summary(output)
