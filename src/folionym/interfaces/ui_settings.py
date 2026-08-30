"""Shared settings persistence and config assembly for interactive frontends."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from threading import Event

from ..infrastructure.private_io import atomic_write_private_text
from ..settings import RenamerConfig, build_config

logger = logging.getLogger(__name__)

type UiSettingsSnapshot = dict[str, object]
SETTINGS_PATH = Path.home() / ".folionym_ui.json"
LEGACY_TUI_SETTINGS_PATH = Path.home() / ".folionym_tui.json"

DEFAULT_UI_SETTINGS: UiSettingsSnapshot = {
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
_VALID_UI_LANGUAGES = frozenset({"de", "en"})
_VALID_UI_CASES = frozenset({"camelCase", "kebabCase", "snakeCase"})
_VALID_UI_DATE_FORMATS = frozenset({"dmy", "mdy"})


def _read_settings(path: Path) -> UiSettingsSnapshot:
    """Read a JSON object from ``path`` or return an empty snapshot."""
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
) -> UiSettingsSnapshot:
    """Load shared settings, migrating the legacy Textual file once when needed."""
    current = _read_settings(settings_path)
    if current or settings_path.exists():
        return current
    legacy = _read_settings(legacy_path)
    if not legacy:
        return {}
    save_ui_settings(legacy, settings_path=settings_path)
    return legacy


def merged_ui_settings(data: UiSettingsSnapshot | None = None) -> UiSettingsSnapshot:
    """Overlay persisted values on the complete interactive settings defaults."""
    merged = {**DEFAULT_UI_SETTINGS, **(data or {})}
    if merged["language"] not in _VALID_UI_LANGUAGES:
        merged["language"] = DEFAULT_UI_SETTINGS["language"]
    if merged["case"] == "snake_case":
        merged["case"] = "snakeCase"
    if merged["case"] not in _VALID_UI_CASES:
        merged["case"] = DEFAULT_UI_SETTINGS["case"]
    if merged["date_format"] not in _VALID_UI_DATE_FORMATS:
        merged["date_format"] = DEFAULT_UI_SETTINGS["date_format"]
    return merged


def save_ui_settings(data: UiSettingsSnapshot, *, settings_path: Path = SETTINGS_PATH) -> None:
    """Atomically persist owner-only settings without disabling the interactive UI on I/O failure."""
    try:
        atomic_write_private_text(settings_path, json.dumps(data, indent=2, ensure_ascii=False))
    except OSError:
        logger.debug("Could not save shared UI settings", exc_info=True)


def build_config_from_ui_settings(
    snapshot: UiSettingsSnapshot,
    stop_event: Event,
    *,
    dry_run: bool,
    manual_mode: bool = False,
) -> RenamerConfig:
    """Translate one persisted UI snapshot into the canonical rename configuration."""
    return build_config(
        {
            "language": snapshot["language"],
            "desired_case": snapshot["case"],
            "project": snapshot["project"],
            "version": snapshot["version"],
            "date_locale": snapshot["date_format"],
            "dry_run": dry_run,
            "use_llm": snapshot["use_llm"],
            "use_ocr": snapshot["use_ocr"],
            "use_pdf_metadata_for_date": snapshot["use_pdf_metadata_date"],
            "use_structured_fields": snapshot["use_structured_fields"],
            "skip_if_already_named": snapshot["skip_already_named"],
            "recursive": snapshot["recursive"],
            "backup_dir": snapshot["backup_dir"],
            "rename_log_path": snapshot["rename_log"],
            "export_metadata_path": snapshot["export_metadata"],
            "summary_json_path": snapshot["summary_json"],
            "rules_file": snapshot["rules_file"],
            "post_rename_hook": snapshot["post_rename_hook"],
            "llm_base_url": snapshot["llm_url"],
            "llm_model": snapshot["llm_model"],
            "llm_timeout_s": snapshot["llm_timeout"],
            "max_tokens_for_extraction": snapshot["max_tokens"],
            "max_content_chars": snapshot["max_content_chars"],
            "max_content_tokens": snapshot["max_content_tokens"],
            "workers": snapshot["workers"],
            "max_filename_chars": snapshot["max_filename_chars"],
            "write_pdf_metadata": snapshot["write_pdf_metadata"],
            "filename_template": snapshot["template"],
            "use_vision_fallback": snapshot["use_vision_fallback"],
            "simple_naming_mode": snapshot["simple_naming_mode"],
            "vision_first": snapshot["vision_first"],
            "preset": snapshot["preset"],
            "manual_mode": manual_mode,
            "interactive": manual_mode,
            "stop_event": stop_event,
        }
    )
