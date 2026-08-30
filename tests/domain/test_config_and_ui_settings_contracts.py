"""Configuration and interactive-settings behavior locks."""

from __future__ import annotations

import json
from itertools import product
from threading import Event

import pytest
from pydantic import ValidationError

from folionym.config import build_config
from folionym.interfaces.ui_settings import (
    DEFAULT_UI_SETTINGS,
    build_config_from_ui_settings,
    load_ui_settings,
    merged_ui_settings,
)
from folionym.interfaces.web.schema import UISettingsPayload


def test_config_precedence_is_raw_then_environment_then_file_then_builtin() -> None:
    file_defaults = {
        "llm_base_url": "http://file.example.test/v1/completions",
        "llm_model": "file-model",
        "max_content_chars": 123,
    }
    environment = {
        "FOLIONYM_LLM_URL": "http://environment.example.test/v1/completions",
        "FOLIONYM_LLM_MODEL": "environment-model",
        "FOLIONYM_MAX_CONTENT_CHARS": "456",
    }

    from_environment = build_config({}, file_defaults=file_defaults, env=environment)
    assert from_environment.llm.backend.llm_base_url == environment["FOLIONYM_LLM_URL"]
    assert from_environment.llm.backend.llm_model == environment["FOLIONYM_LLM_MODEL"]
    assert from_environment.llm.content.max_content_chars == 456

    from_raw = build_config(
        {"llm_base_url": "http://raw.example.test/v1/completions", "llm_model": "raw-model", "max_content_chars": 789},
        file_defaults=file_defaults,
        env=environment,
    )
    assert from_raw.llm.backend.llm_base_url == "http://raw.example.test/v1/completions"
    assert from_raw.llm.backend.llm_model == "raw-model"
    assert from_raw.llm.content.max_content_chars == 789


def test_vision_environment_flags_are_additive_even_when_raw_values_are_false() -> None:
    config = build_config(
        {"use_vision_fallback": False, "vision_first": False},
        env={"FOLIONYM_USE_VISION_FALLBACK": "true", "FOLIONYM_VISION_FIRST": "yes"},
    )
    assert config.llm.vision.use_vision_fallback is True
    assert config.llm.vision.vision_first is True


def test_ui_settings_defaults_schema_and_legacy_migration_share_one_contract(tmp_path) -> None:
    assert UISettingsPayload().model_dump() == DEFAULT_UI_SETTINGS
    assert merged_ui_settings({"language": "en", "workers": "4"})["use_structured_fields"] is True

    settings_path = tmp_path / ".folionym_ui.json"
    legacy_path = tmp_path / ".folionym_tui.json"
    legacy = {"language": "en", "dry_run": False}
    legacy_path.write_text(json.dumps(legacy), encoding="utf-8")
    assert load_ui_settings(settings_path, legacy_path=legacy_path) == legacy
    assert json.loads(settings_path.read_text(encoding="utf-8")) == legacy


def test_existing_shared_settings_prevent_legacy_overwrite_and_schema_rejects_unknown_keys(tmp_path) -> None:
    settings_path = tmp_path / ".folionym_ui.json"
    legacy_path = tmp_path / ".folionym_tui.json"
    settings_path.write_text("{}", encoding="utf-8")
    legacy_path.write_text(json.dumps({"language": "en"}), encoding="utf-8")

    assert load_ui_settings(settings_path, legacy_path=legacy_path) == {}
    with pytest.raises(ValidationError):
        UISettingsPayload.model_validate({"unknown": "value"})


@pytest.mark.parametrize(
    ("language", "case_value", "date_format"),
    product(("de", "en"), ("camelCase", "kebabCase", "snakeCase"), ("dmy", "mdy")),
)
def test_every_browser_naming_option_round_trips_to_canonical_config(
    language: str, case_value: str, date_format: str
) -> None:
    payload = UISettingsPayload(language=language, case=case_value, date_format=date_format, use_llm=False)

    config = build_config_from_ui_settings(payload.model_dump(), Event(), dry_run=True)

    assert config.output.naming.language == language
    assert config.output.naming.desired_case == case_value
    assert config.output.naming.date_locale == date_format


def test_legacy_or_invalid_persisted_browser_choices_are_normalized() -> None:
    assert merged_ui_settings({"case": "snake_case"})["case"] == "snakeCase"
    normalized = merged_ui_settings({"language": "fr", "case": "Title Case", "date_format": "ymd"})
    assert normalized["language"] == DEFAULT_UI_SETTINGS["language"]
    assert normalized["case"] == DEFAULT_UI_SETTINGS["case"]
    assert normalized["date_format"] == DEFAULT_UI_SETTINGS["date_format"]
