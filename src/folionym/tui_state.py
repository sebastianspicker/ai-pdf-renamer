"""State and config helpers for the Textual TUI."""

from __future__ import annotations

from threading import Event

from .config import RenamerConfig
from .config_resolver import build_config
from .ui_settings import (
    LEGACY_TUI_SETTINGS_PATH,
    load_ui_settings,
    save_ui_settings,
)
from .ui_settings import (
    SETTINGS_PATH as SHARED_SETTINGS_PATH,
)

SETTINGS_PATH = SHARED_SETTINGS_PATH


def _load_settings() -> dict[str, object]:
    """Load shared UI settings and migrate the legacy TUI file when appropriate."""
    return load_ui_settings(SETTINGS_PATH, legacy_path=LEGACY_TUI_SETTINGS_PATH)


def _save_settings(data: dict[str, object]) -> None:
    """Persist shared private UI settings."""
    save_ui_settings(data, settings_path=SETTINGS_PATH)


def build_config_from_snapshot(
    snap: dict[str, object],
    stop_event: Event,
    *,
    dry_run: bool,
    manual_mode: bool = False,
) -> RenamerConfig:
    """Build a rename configuration from a TUI form snapshot."""
    return build_config(
        {
            "language": snap["language"],
            "desired_case": snap["case"],
            "project": snap["project"],
            "version": snap["version"],
            "date_locale": snap["date_format"],
            "dry_run": dry_run,
            "use_llm": snap["use_llm"],
            "use_ocr": snap["use_ocr"],
            "use_pdf_metadata_for_date": snap["use_pdf_metadata_date"],
            "use_structured_fields": snap["use_structured_fields"],
            "skip_if_already_named": snap["skip_already_named"],
            "recursive": snap["recursive"],
            "backup_dir": snap["backup_dir"],
            "rename_log_path": snap["rename_log"],
            "export_metadata_path": snap["export_metadata"],
            "summary_json_path": snap["summary_json"],
            "rules_file": snap["rules_file"],
            "post_rename_hook": snap["post_rename_hook"],
            "llm_base_url": snap["llm_url"],
            "llm_model": snap["llm_model"],
            "llm_timeout_s": snap["llm_timeout"],
            "max_tokens_for_extraction": snap["max_tokens"],
            "max_content_chars": snap["max_content_chars"],
            "max_content_tokens": snap["max_content_tokens"],
            "workers": snap["workers"],
            "max_filename_chars": snap["max_filename_chars"],
            "write_pdf_metadata": snap["write_pdf_metadata"],
            "filename_template": snap["template"],
            "use_vision_fallback": snap["use_vision_fallback"],
            "simple_naming_mode": snap["simple_naming_mode"],
            "vision_first": snap["vision_first"],
            "preset": snap["preset"],
            "manual_mode": manual_mode,
            "interactive": manual_mode,
            "stop_event": stop_event,
        }
    )
