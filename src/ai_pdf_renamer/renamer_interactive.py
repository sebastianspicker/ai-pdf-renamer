"""Interactive rename prompt helpers."""

from __future__ import annotations

from pathlib import Path

from .rename_ops import sanitize_filename_base


def _interactive_rename_prompt(
    file_path: Path,
    target: Path,
    default_base: str,
    edit_default_base: str | None = None,
) -> tuple[str, str, Path]:
    """Prompt for y/n/e=edit and return the selected action, basename, and target."""
    base = default_base
    current_target = target
    while True:
        reply = _read_rename_prompt_reply(file_path, current_target)
        if reply == "n":
            return ("n", base, current_target)
        if reply == "e":
            edited = _read_edited_rename_target(file_path, edit_default_base)
            if edited is not None:
                base, current_target = edited
                return ("y", base, current_target)
            continue
        return ("y", base, current_target)


def _read_rename_prompt_reply(file_path: Path, current_target: Path) -> str:
    try:
        return (
            input(f"Rename '{file_path.name}' to '{current_target.name}'? (y/n/e=edit, default y): ").strip().lower()
            or "y"
        )
    except (EOFError, KeyboardInterrupt):
        return "n"


def _read_edited_rename_target(file_path: Path, edit_default_base: str | None) -> tuple[str, Path] | None:
    try:
        custom = input(_edit_prompt_text(edit_default_base)).strip()
    except (EOFError, KeyboardInterrupt):
        return None
    if not custom and edit_default_base:
        custom = edit_default_base
    if not custom:
        return None
    custom_base = sanitize_filename_base(_strip_pdf_suffix(custom, file_path.suffix))
    return (custom_base, file_path.with_name(custom_base + file_path.suffix))


def _edit_prompt_text(edit_default_base: str | None) -> str:
    prompt = "New filename (without path)"
    if edit_default_base:
        prompt += f" [default: {edit_default_base}]"
    return f"{prompt}: "


def _strip_pdf_suffix(value: str, suffix: str) -> str:
    return value.removesuffix(suffix) if value.lower().endswith(suffix.lower()) else value
