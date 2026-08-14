"""Tests for settings shared by browser and Textual frontends."""

from __future__ import annotations

import json
import stat
from pathlib import Path

import pytest

from folionym.tui_state import SETTINGS_PATH, _load_settings, _save_settings
from folionym.ui_settings import DEFAULT_UI_SETTINGS, load_ui_settings, merged_ui_settings, save_ui_settings
from folionym.web_schema import UISettingsPayload


def test_load_ui_settings_migrates_legacy_file(tmp_path: Path) -> None:
    settings_path = tmp_path / ".folionym_ui.json"
    legacy_path = tmp_path / ".folionym_tui.json"
    legacy_path.write_text(json.dumps({"language": "en", "dry_run": False}), encoding="utf-8")

    loaded = load_ui_settings(settings_path, legacy_path=legacy_path)

    assert loaded == {"language": "en", "dry_run": False}
    assert json.loads(settings_path.read_text(encoding="utf-8")) == loaded
    assert legacy_path.exists()


def test_save_ui_settings_is_owner_only(tmp_path: Path) -> None:
    settings_path = tmp_path / "settings.json"

    save_ui_settings({"directory": "/tmp/pdfs"}, settings_path=settings_path)

    assert stat.S_IMODE(settings_path.stat().st_mode) == 0o600


def test_merged_ui_settings_fills_defaults_without_overwriting_saved_values() -> None:
    merged = merged_ui_settings({"language": "en", "workers": "4"})

    assert merged["language"] == "en"
    assert merged["workers"] == "4"
    assert merged["use_structured_fields"] is True


def test_persisted_and_http_settings_defaults_stay_identical() -> None:
    """Prevent the persistence and HTTP schema copies from drifting silently."""
    assert UISettingsPayload().model_dump() == DEFAULT_UI_SETTINGS


def test_tui_settings_path_uses_shared_filename() -> None:
    assert isinstance(SETTINGS_PATH, Path)
    assert SETTINGS_PATH.name == ".folionym_ui.json"


def test_load_tui_settings_missing_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("folionym.tui_state.SETTINGS_PATH", tmp_path / "nonexistent.json")

    assert _load_settings() == {}


def test_load_tui_settings_valid_mapping(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings_file = tmp_path / "settings.json"
    data = {"language": "en", "dry_run": True}
    settings_file.write_text(json.dumps(data), encoding="utf-8")
    monkeypatch.setattr("folionym.tui_state.SETTINGS_PATH", settings_file)

    assert _load_settings() == data


@pytest.mark.parametrize("contents", ["not valid json{{{", "[1, 2, 3]"])
def test_load_tui_settings_rejects_invalid_content(
    contents: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings_file = tmp_path / "settings.json"
    settings_file.write_text(contents, encoding="utf-8")
    monkeypatch.setattr("folionym.tui_state.SETTINGS_PATH", settings_file)

    assert _load_settings() == {}


def test_save_tui_settings_creates_parent_and_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings_file = tmp_path / "subdir" / "settings.json"
    monkeypatch.setattr("folionym.tui_state.SETTINGS_PATH", settings_file)
    data = {"directory": "/tmp/pdfs", "use_llm": False}

    _save_settings(data)

    assert json.loads(settings_file.read_text(encoding="utf-8")) == data


def test_tui_settings_roundtrip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings_file = tmp_path / "settings.json"
    monkeypatch.setattr("folionym.tui_state.SETTINGS_PATH", settings_file)
    data: dict[str, object] = {"language": "de", "case": "kebabCase", "dry_run": True}

    _save_settings(data)

    assert _load_settings() == data
