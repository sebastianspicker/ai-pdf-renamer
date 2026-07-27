"""Shared private settings persistence for interactive Folionym frontends."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from .private_io import atomic_write_private_text

logger = logging.getLogger(__name__)

SETTINGS_PATH = Path.home() / ".folionym_ui.json"
LEGACY_TUI_SETTINGS_PATH = Path.home() / ".folionym_tui.json"

DEFAULT_UI_SETTINGS: dict[str, object] = {
    "directory": "",
    "single_file": "",
    "language": "de",
    "case": "kebabCase",
    "date_format": "dmy",
    "preset": "",
    "project": "",
    "version": "",
    "template": "",
    "backup_dir": "",
    "rename_log": "",
    "export_metadata": "",
    "summary_json": "",
    "rules_file": "",
    "post_rename_hook": "",
    "llm_url": "",
    "llm_model": "",
    "llm_timeout": "",
    "max_tokens": "",
    "max_content_chars": "",
    "max_content_tokens": "",
    "workers": "1",
    "max_filename_chars": "",
    "dry_run": True,
    "use_llm": True,
    "use_ocr": False,
    "recursive": False,
    "skip_already_named": False,
    "use_pdf_metadata_date": True,
    "use_structured_fields": True,
    "write_pdf_metadata": False,
    "use_vision_fallback": False,
    "simple_naming_mode": False,
    "vision_first": False,
    "acknowledged_external_endpoint": "",
}


def _read_settings(path: Path) -> dict[str, object]:
    """Read a JSON object from ``path`` or return an empty mapping."""
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except OSError, json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def load_ui_settings(
    settings_path: Path = SETTINGS_PATH,
    *,
    legacy_path: Path = LEGACY_TUI_SETTINGS_PATH,
) -> dict[str, object]:
    """Load shared UI settings, migrating the legacy TUI file once when needed."""
    current = _read_settings(settings_path)
    if current or settings_path.exists():
        return current

    legacy = _read_settings(legacy_path)
    if not legacy:
        return {}
    save_ui_settings(legacy, settings_path=settings_path)
    return legacy


def merged_ui_settings(data: dict[str, object] | None = None) -> dict[str, object]:
    """Overlay persisted values on the complete interactive settings defaults."""
    return {**DEFAULT_UI_SETTINGS, **(data or {})}


def save_ui_settings(data: dict[str, object], *, settings_path: Path = SETTINGS_PATH) -> None:
    """Atomically persist owner-only settings without making the UI unusable on I/O failure."""
    try:
        atomic_write_private_text(settings_path, json.dumps(data, indent=2, ensure_ascii=False))
    except OSError:
        logger.debug("Could not save shared UI settings", exc_info=True)
