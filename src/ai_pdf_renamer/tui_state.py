"""State and config helpers for the Textual TUI."""

from __future__ import annotations

import contextlib
import json
import logging
import queue
import sys
from pathlib import Path
from threading import Event
from typing import cast

from rich.markup import escape as _escape_markup

from .config import RenamerConfig
from .config_resolver import build_config

logger = logging.getLogger(__name__)
SETTINGS_PATH = Path.home() / ".ai_pdf_renamer_gui.json"
_TUI_LOGGING_EXCEPTIONS = (AttributeError, KeyError, RuntimeError, TypeError, ValueError, queue.Full)


def _compat_tui_attr(name: str, default: object) -> object:
    tui_mod = sys.modules.get("ai_pdf_renamer.tui")
    return getattr(tui_mod, name, default) if tui_mod is not None else default


def _settings_path() -> Path:
    return Path(cast(str | Path, _compat_tui_attr("SETTINGS_PATH", SETTINGS_PATH)))


def _escape_log_markup(text: str) -> str:
    escape_func = _compat_tui_attr("_escape_markup", _escape_markup)
    return escape_func(text) if callable(escape_func) else _escape_markup(text)


class _QueueHandler(logging.Handler):
    def __init__(self, q: queue.Queue[str | None]) -> None:
        super().__init__()
        self._queue = q

    def emit(self, record: logging.LogRecord) -> None:
        try:
            # Escape Rich markup so filenames/paths in log messages are not interpreted as tags.
            self._queue.put(_escape_log_markup(self.format(record)) + "\n")
        except _TUI_LOGGING_EXCEPTIONS:
            self.handleError(record)


def _load_settings() -> dict[str, object]:
    settings_path = _settings_path()
    if not settings_path.exists():
        return {}
    try:
        raw = settings_path.read_text(encoding="utf-8")
        data = json.loads(raw)
        if isinstance(data, dict):
            return data
    except (OSError, json.JSONDecodeError):
        pass
    return {}


def _save_settings(data: dict[str, object]) -> None:
    settings_path = _settings_path()
    try:
        settings_path.parent.mkdir(parents=True, exist_ok=True)
        settings_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        with contextlib.suppress(OSError):
            settings_path.chmod(0o600)
    except OSError:
        logger.debug("Could not save TUI settings", exc_info=True)


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
            "llm_backend": snap["llm_backend"],
            "llm_base_url": snap["llm_url"],
            "llm_model": snap["llm_model"],
            "llm_model_path": snap["llm_model_path"],
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
