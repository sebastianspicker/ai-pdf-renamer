"""CLI for reverting renames from a rename log (--rename-log). Entry point: ai-pdf-renamer-undo.

Log format: one line per rename, 'old_path\\tnew_path'. Filenames containing tab or newline
characters in the original path are not supported; those entries are skipped at write time
with a WARNING. The renamer sanitizes generated names so they never contain these characters.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from .rename_ops import is_path_within


def run_undo(log_path: Path, dry_run: bool) -> None:
    """Read rename log and revert renames (LIFO). Caller must ensure log_path.exists()."""
    if not log_path.is_file():
        print(
            f"Error: rename log path is not a file (e.g. directory): {log_path}",
            file=sys.stderr,
        )
        return
    # Use the log file's directory as the trusted root: both source and target of every
    # undo operation must resolve within this tree to prevent path-traversal attacks.
    trusted_root = log_path.resolve().parent
    pairs: list[tuple[str, str]] = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        if line == "" or line.isspace():
            continue
        parts = line.split("\t", 1)
        if len(parts) != 2:
            continue
        old_path, new_path = parts
        if old_path and new_path:
            pairs.append((old_path, new_path))
    if not pairs:
        print("No entries in rename log.", file=sys.stderr)
        return
    pairs.reverse()
    for old_path, new_path in pairs:
        old_p, new_p = Path(old_path), Path(new_path)
        if not is_path_within(old_p, trusted_root):
            print(f"Skip (path traversal detected): {old_p}", file=sys.stderr)
            continue
        if not is_path_within(new_p, trusted_root):
            print(f"Skip (path traversal detected): {new_p}", file=sys.stderr)
            continue
        try:
            if old_p.parent.resolve() != new_p.parent.resolve():
                print(f"Skip (cross-directory undo denied): {new_p} -> {old_p}", file=sys.stderr)
                continue
        except OSError:
            print(f"Skip (path traversal detected): {new_p}", file=sys.stderr)
            continue
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
            # P2: Use shutil.move instead of Path.rename to handle cross-filesystem
            shutil.move(str(new_p), str(old_p))
            print(f"Reverted: {new_p} -> {old_p}")
        except OSError as e:
            print(f"Error reverting {new_p}: {e}", file=sys.stderr)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        prog="ai-pdf-renamer-undo",
        description="Revert renames using a rename log file (produced by --rename-log).",
    )
    ap.add_argument(
        "--rename-log",
        "-l",
        required=True,
        metavar="FILE",
        help="Path to the rename log (tab-separated old/new paths, one per line).",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be reverted without making changes.",
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
