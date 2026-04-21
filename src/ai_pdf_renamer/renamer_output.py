from __future__ import annotations

import csv
import json
import logging
import re
from pathlib import Path
from typing import Any

_CSV_FORMULA_TRIGGERS = frozenset("=+-@|")
_CSV_CONTROL_CHARS_RE = re.compile(r"[\t\r\n]")
logger = logging.getLogger(__name__)


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
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".csv" and csv_fieldnames:
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=csv_fieldnames)
            writer.writeheader()
            writer.writerows({k: _sanitize_csv_cell(v) for k, v in row.items()} for row in rows)
    else:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)


def _write_summary_json(
    summary_path: str | Path | None,
    *,
    directory: Path,
    processed: int,
    renamed: int,
    skipped: int,
    failed: int,
    dry_run: bool,
    failures: list[dict[str, str]],
) -> None:
    """Write summary JSON; best-effort logging on I/O errors."""
    if not summary_path:
        return
    summary = {
        "directory": str(directory),
        "processed": processed,
        "renamed": renamed,
        "skipped": skipped,
        "failed": failed,
        "dry_run": dry_run,
        "failures": failures,
    }
    target = Path(summary_path)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError as exc:
        logger.warning("Could not write summary JSON %s: %s", target, exc)


def _append_export_row(
    export_rows: list[dict[str, object]],
    *,
    file_path: Path,
    target: Path,
    meta: dict[str, object],
) -> None:
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
