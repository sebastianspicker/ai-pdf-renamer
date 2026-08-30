"""Neutral shared interfaces for Folionym's interactive entry points."""

from .ui_settings import UiSettingsSnapshot, build_config_from_ui_settings

__all__ = ["UiSettingsSnapshot", "build_config_from_ui_settings"]
