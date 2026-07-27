"""Tests for the loopback browser boundary and payload normalization."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import HTTPException

from folionym.web_app import (
    _directory_listing,
    _external_endpoint_requirement,
    _host_name,
    _prepare_preview_settings,
)
from folionym.web_schema import PreviewRequest, UISettingsPayload


def test_host_name_accepts_loopback_header_shapes() -> None:
    assert _host_name("127.0.0.1:8765") == "127.0.0.1"
    assert _host_name("localhost") == "localhost"
    assert _host_name("[::1]:8765") == "::1"


def test_directory_listing_exposes_only_visible_directories(tmp_path: Path) -> None:
    visible = tmp_path / "Visible"
    visible.mkdir()
    (visible / "one.pdf").write_bytes(b"%PDF")
    (tmp_path / ".hidden").mkdir()
    (tmp_path / "direct.pdf").write_bytes(b"%PDF")

    listing = _directory_listing(str(tmp_path))

    assert listing.pdf_count == 1
    assert [(entry.name, entry.pdf_count) for entry in listing.entries] == [("Visible", 1)]


def test_external_endpoint_requires_exact_acknowledgement(tmp_path: Path) -> None:
    payload = PreviewRequest(
        source_kind="directory",
        path=str(tmp_path),
        settings=UISettingsPayload(use_llm=True, llm_url="https://models.example.test/v1"),
    )

    assert _external_endpoint_requirement(payload) == "https://models.example.test/v1"
    with pytest.raises(HTTPException) as exc_info:
        _prepare_preview_settings(payload)
    assert exc_info.value.status_code == 409


def test_loopback_endpoint_does_not_require_acknowledgement(tmp_path: Path) -> None:
    payload = PreviewRequest(
        source_kind="file",
        path=str(tmp_path / "one.pdf"),
        settings=UISettingsPayload(use_llm=True, llm_url="http://127.0.0.1:11434/v1"),
    )

    settings = _prepare_preview_settings(payload)

    assert settings["single_file"] == str(tmp_path / "one.pdf")
    assert settings["directory"] == str(tmp_path)
