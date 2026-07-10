"""Runtime path resolution helpers for the CLI."""

from __future__ import annotations

import logging
import sys
from collections.abc import Callable
from pathlib import Path
from typing import NoReturn

from rich.console import Console


def resolve_dirs(
    args: object,
    *,
    is_interactive: Callable[[], bool],
    console: Console,
    logger: logging.Logger,
) -> tuple[list[str], str | None]:
    """Resolve directory list and optional single-file path from parsed CLI args."""
    _validate_single_file_mode(args, console)
    single_file = getattr(args, "single_file", None) or getattr(args, "manual_file", None)
    dirs = _initial_dirs(args, console=console, logger=logger)
    if single_file:
        dirs = [_single_file_parent(single_file, console)]
    if not dirs:
        dirs = [_prompt_for_default_dir()] if is_interactive() else _raise_no_input(console)
    return (_resolved_dirs(dirs, single_file=single_file, console=console), single_file)


def _validate_single_file_mode(args: object, console: Console) -> None:
    has_file = bool(getattr(args, "single_file", None))
    has_manual = bool(getattr(args, "manual_file", None))
    if not (has_file and has_manual):
        return
    console.print("[red]Error:[/red] --file and --manual are mutually exclusive.")
    console.print("[dim]Use --file for non-interactive single-file, or --manual for interactive mode.[/dim]")
    raise SystemExit(1)


def _initial_dirs(args: object, *, console: Console, logger: logging.Logger) -> list[str]:
    dirs: list[str] = list(getattr(args, "dirs", None) or [])
    dirs_from_file = getattr(args, "dirs_from_file", None)
    if dirs_from_file:
        dirs.extend(_read_dirs_from_file(dirs_from_file, console=console, logger=logger))
    return dirs


def _single_file_parent(single_file: str, console: Console) -> str:
    single_path = Path(single_file).resolve()
    if not single_path.is_file():
        console.print(f"[red]File not found:[/red] {single_path}")
        console.print("[dim]Provide a valid path to an existing PDF file.[/dim]")
        raise SystemExit(1)
    return str(single_path.parent)


def _resolved_dirs(dirs: list[str], *, single_file: str | None, console: Console) -> list[str]:
    resolved_dirs = [str(Path(d).resolve()) for d in dirs if d.strip()]
    if resolved_dirs:
        return resolved_dirs
    _raise_empty_dirs(console, dirs=dirs, single_file=single_file)


def _read_dirs_from_file(
    dirs_from_file: str,
    *,
    console: Console,
    logger: logging.Logger,
) -> list[str]:
    try:
        lines = Path(dirs_from_file).read_text(encoding="utf-8").splitlines()
    except OSError as e:
        console.print(f"[red]Cannot read --dirs-from-file:[/red] {e}")
        raise SystemExit(1) from e

    _MAX_DIRS_FROM_FILE_LINES = 10_000
    if len(lines) > _MAX_DIRS_FROM_FILE_LINES:
        logger.warning(
            "--dirs-from-file has %s lines; using first %s only.",
            len(lines),
            _MAX_DIRS_FROM_FILE_LINES,
        )
        lines = lines[:_MAX_DIRS_FROM_FILE_LINES]
    return [line.strip() for line in lines if line.strip()]


def _prompt_for_default_dir() -> str:
    return read_prompt_value("Path to the directory with PDFs (default: ./input_files): ") or "./input_files"


def read_prompt_value(prompt: str) -> str:
    """Read stripped input, mapping closed stdin and interrupts to CLI exits."""
    try:
        return input(prompt).strip()
    except EOFError:
        print("Error: stdin closed.", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        sys.exit(130)


def _raise_no_input(console: Console) -> NoReturn:
    console.print("[red]No input specified.[/red] Provide --dir or --file in non-interactive mode.")
    raise SystemExit(1)


def _raise_empty_dirs(console: Console, *, dirs: list[str], single_file: str | None) -> NoReturn:
    if dirs and not single_file:
        console.print("[red]Error:[/red] --dir path is empty. Provide a valid directory path.")
    else:
        console.print("[red]No input specified.[/red] Use --dir or --file.")
    raise SystemExit(1)
