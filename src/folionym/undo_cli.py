"""CLI for reverting renames from a rename log (--rename-log). Entry point: folionym-undo.

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


def _read_rename_log_pairs(log_path: Path) -> list[tuple[str, str]]:
    """Parse valid tab-separated rename pairs and return them in reverse order for LIFO undo."""
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
    pairs.reverse()
    return pairs


def _same_parent(old_p: Path, new_p: Path) -> bool | None:
    """Compare parent defensively because path resolution can fail."""
    try:
        return old_p.parent.resolve() == new_p.parent.resolve()
    except OSError:
        return None


def _undo_paths(old_path: str, new_path: str) -> tuple[Path, Path]:
    """Convert paths into the path representation used by undo safeguards."""
    return (Path(old_path), Path(new_path))


def _can_undo_pair(old_p: Path, new_p: Path, trusted_root: Path) -> bool:
    """Allow undo only when both paths stay inside the trusted root, share a parent, and avoid overwriting a file."""
    for candidate in (old_p, new_p):
        if not is_path_within(candidate, trusted_root):
            print(f"Skip (path traversal detected): {candidate}", file=sys.stderr)
            return False
    same_parent = _same_parent(old_p, new_p)
    if same_parent is None:
        print(f"Skip (path traversal detected): {new_p}", file=sys.stderr)
        return False
    if not same_parent:
        print(f"Skip (cross-directory undo denied): {new_p} -> {old_p}", file=sys.stderr)
        return False
    if not new_p.exists():
        print(f"Skip (new path missing): {new_p}", file=sys.stderr)
        return False
    if old_p.exists() and old_p.resolve() != new_p.resolve():
        print(f"Skip (old path already exists): {old_p}", file=sys.stderr)
        return False
    return True


def _move_undo_pair(old_p: Path, new_p: Path) -> None:
    """Move the renamed path back, reporting filesystem errors without aborting later undo entries."""
    try:
        shutil.move(str(new_p), str(old_p))
        print(f"Reverted: {new_p} -> {old_p}")
    except OSError as e:
        print(f"Error reverting {new_p}: {e}", file=sys.stderr)


def _run_undo_pair(old_path: str, new_path: str, *, trusted_root: Path, dry_run: bool) -> None:
    """Validate one logged rename pair, then preview or perform its reversal."""
    old_p, new_p = _undo_paths(old_path, new_path)
    if not _can_undo_pair(old_p, new_p, trusted_root):
        return
    if dry_run:
        print(f"Would revert: {new_p} -> {old_p}")
        return
    _move_undo_pair(old_p, new_p)


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
    pairs = _read_rename_log_pairs(log_path)
    if not pairs:
        print("No entries in rename log.", file=sys.stderr)
        return
    for old_path, new_path in pairs:
        _run_undo_pair(old_path, new_path, trusted_root=trusted_root, dry_run=dry_run)


def main(argv: list[str] | None = None) -> None:
    """Parse command-line input and hand off to the command lifecycle."""
    ap = argparse.ArgumentParser(
        prog="folionym-undo",
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
