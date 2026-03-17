"""CLI for reverting renames from a rename log (--rename-log). Entry point: ai-pdf-renamer-undo.

Log format: one line per rename, 'old_path\\tnew_path'. Filenames containing tab or newline
characters in the original path are not supported; those entries are skipped at write time
with a WARNING. The renamer sanitizes generated names so they never contain these characters.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def run_undo(log_path: Path, dry_run: bool) -> None:
    """Read rename log and revert renames (LIFO). Caller must ensure log_path.exists()."""
    if not log_path.is_file():
        print(
            f"Error: rename log path is not a file (e.g. directory): {log_path}",
            file=sys.stderr,
        )
        return
    pairs: list[tuple[str, str]] = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s:
            continue
        parts = s.split("\t", 1)
        if len(parts) != 2:
            continue
        old_path, new_path = parts[0].strip(), parts[1].strip()
        if old_path and new_path:
            pairs.append((old_path, new_path))
    if not pairs:
        print("No entries in rename log.", file=sys.stderr)
        return
    pairs.reverse()
    for old_path, new_path in pairs:
        old_p, new_p = Path(old_path), Path(new_path)
        if not new_p.exists():
            print(f"Skip (new path missing): {new_p}", file=sys.stderr)
            continue
        if old_p.exists() and old_p.resolve() != new_p.resolve():
            print(f"Skip (old path already exists): {old_p}", file=sys.stderr)
            continue
        if dry_run:
            print(f"Would revert: {new_p} -> {old_p}")
            continue
        try:
            new_p.rename(old_p)
            print(f"Reverted: {new_p} -> {old_p}")
        except OSError as e:
            print(f"Error reverting {new_p}: {e}", file=sys.stderr)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Revert renames from a rename log file.")
    ap.add_argument(
        "--rename-log",
        "-l",
        required=True,
        metavar="FILE",
        help="Log file with old_path\\tnew_path per line",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print what would be done",
    )
    args = ap.parse_args(argv)
    log_path = Path(args.rename_log)
    if not log_path.exists():
        print(f"Error: log file not found: {log_path}", file=sys.stderr)
        sys.exit(1)
    if not log_path.is_file():
        print(f"Error: log path is not a file: {log_path}", file=sys.stderr)
        sys.exit(1)
    run_undo(log_path, args.dry_run)
