"""Round 7 coverage tests: TUI app lifecycle, widget getters, snapshot, settings, and cancel.

Uses Textual's built-in ``App.run_test()`` async testing harness.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from ai_pdf_renamer.tui import _CSS, AIRenamerTUI

# Textual 8.x dropped the ``flex-wrap`` CSS property.  Remove it from the
# app's stylesheet so that ``run_test()`` can compose widgets successfully.
_PATCHED_CSS = _CSS.replace("flex-wrap: wrap;", "")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_app(settings: dict[str, object] | None = None) -> AIRenamerTUI:
    """Create an AIRenamerTUI with patched CSS and optional pre-loaded settings."""
    if settings is not None:
        with patch("ai_pdf_renamer.tui._load_settings", return_value=settings):
            app = AIRenamerTUI()
    else:
        with patch("ai_pdf_renamer.tui._load_settings", return_value={}):
            app = AIRenamerTUI()
    app.CSS = _PATCHED_CSS  # type: ignore[assignment]
    return app


# ---------------------------------------------------------------------------
# 1. Basic app lifecycle
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_app_starts_and_has_title() -> None:
    """App starts headlessly and exposes the expected title."""
    app = _make_app()
    async with app.run_test(size=(120, 40)) as _pilot:
        assert app.title == "AI-PDF-Renamer"


@pytest.mark.asyncio
async def test_app_has_tabs() -> None:
    """The three tab panes (Basic, Advanced, Run) are composed."""
    from textual.widgets import TabPane

    app = _make_app()
    async with app.run_test(size=(120, 40)) as _pilot:
        panes = app.query(TabPane)
        ids = {p.id for p in panes}
        assert "basic" in ids
        assert "advanced" in ids
        assert "run" in ids
        assert len(panes) == 3


@pytest.mark.asyncio
async def test_app_has_buttons() -> None:
    """Preview, Apply, One, and Cancel buttons exist with their IDs."""
    from textual.widgets import Button

    app = _make_app()
    async with app.run_test(size=(120, 40)) as _pilot:
        buttons = app.query(Button)
        ids = {b.id for b in buttons}
        assert "btn-preview" in ids
        assert "btn-apply" in ids
        assert "btn-one" in ids
        assert "btn-cancel" in ids


@pytest.mark.asyncio
async def test_app_default_widget_values() -> None:
    """Default widget values match expected defaults when no settings are loaded."""
    from textual.widgets import Checkbox, Input, Select

    app = _make_app()
    async with app.run_test(size=(120, 40)) as _pilot:
        # Inputs default to empty
        assert app.query_one("#directory", Input).value == ""
        assert app.query_one("#single_file", Input).value == ""
        assert app.query_one("#project", Input).value == ""

        # Selects default to their first-option values
        assert app.query_one("#language", Select).value == "de"
        assert app.query_one("#case", Select).value == "kebabCase"
        assert app.query_one("#date_format", Select).value == "dmy"

        # Checkboxes
        assert app.query_one("#dry_run", Checkbox).value is True
        assert app.query_one("#use_llm", Checkbox).value is True
        assert app.query_one("#use_ocr", Checkbox).value is False
        assert app.query_one("#recursive", Checkbox).value is False


# ---------------------------------------------------------------------------
# 2. Widget getters
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_str_returns_value() -> None:
    """_get_str reads the current value of an Input widget."""
    from textual.widgets import Input

    app = _make_app()
    async with app.run_test(size=(120, 40)) as _pilot:
        app.query_one("#directory", Input).value = "/tmp/pdfs"
        assert app._get_str("directory") == "/tmp/pdfs"


@pytest.mark.asyncio
async def test_get_bool_returns_value() -> None:
    """_get_bool reads the current value of a Checkbox widget."""
    from textual.widgets import Checkbox

    app = _make_app()
    async with app.run_test(size=(120, 40)) as _pilot:
        cb = app.query_one("#dry_run", Checkbox)
        cb.value = False
        assert app._get_bool("dry_run") is False
        cb.value = True
        assert app._get_bool("dry_run") is True


@pytest.mark.asyncio
async def test_get_select_returns_value() -> None:
    """_get_select reads the current value of a Select widget."""
    app = _make_app()
    async with app.run_test(size=(120, 40)) as _pilot:
        val = app._get_select("language")
        assert val == "de"

        val = app._get_select("case")
        assert val == "kebabCase"


@pytest.mark.asyncio
async def test_get_str_missing_widget() -> None:
    """_get_str returns the default when the widget ID doesn't exist."""
    app = _make_app()
    async with app.run_test(size=(120, 40)) as _pilot:
        assert app._get_str("nonexistent_widget", "fallback") == "fallback"
        assert app._get_str("nonexistent_widget") == ""


# ---------------------------------------------------------------------------
# 3. Snapshot and config
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_snapshot_returns_dict() -> None:
    """_snapshot returns a dict containing all expected top-level keys."""
    app = _make_app()
    async with app.run_test(size=(120, 40)) as _pilot:
        snap = app._snapshot()
        assert isinstance(snap, dict)
        expected_keys = {
            "directory",
            "single_file",
            "language",
            "case",
            "date_format",
            "preset",
            "project",
            "version",
            "template",
            "backup_dir",
            "rename_log",
            "export_metadata",
            "summary_json",
            "rules_file",
            "post_rename_hook",
            "llm_backend",
            "llm_url",
            "llm_model",
            "llm_model_path",
            "llm_timeout",
            "max_tokens",
            "max_content_chars",
            "max_content_tokens",
            "workers",
            "max_filename_chars",
            "dry_run",
            "use_llm",
            "use_ocr",
            "recursive",
            "skip_already_named",
            "use_pdf_metadata_date",
            "use_structured_fields",
            "write_pdf_metadata",
            "use_vision_fallback",
            "simple_naming_mode",
            "vision_first",
        }
        assert expected_keys.issubset(snap.keys())


@pytest.mark.asyncio
async def test_snapshot_has_language_field() -> None:
    """_snapshot includes 'language' with the widget's current value."""
    app = _make_app()
    async with app.run_test(size=(120, 40)) as _pilot:
        snap = app._snapshot()
        assert "language" in snap
        assert snap["language"] == "de"


# ---------------------------------------------------------------------------
# 4. Settings persistence
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_load_settings_populates_widgets() -> None:
    """Pre-existing settings JSON populates widget values on startup."""
    from textual.widgets import Checkbox, Input, Select

    settings: dict[str, object] = {
        "directory": "/home/user/docs",
        "language": "en",
        "case": "snakeCase",
        "dry_run": False,
        "use_ocr": True,
        "project": "my-project",
    }
    # Write settings to a temp file and patch the path
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tf:
        json.dump(settings, tf)
        tf_path = Path(tf.name)

    try:
        with patch("ai_pdf_renamer.tui.SETTINGS_PATH", tf_path):
            app = AIRenamerTUI()
            app.CSS = _PATCHED_CSS  # type: ignore[assignment]
            async with app.run_test(size=(120, 40)) as _pilot:
                assert app.query_one("#directory", Input).value == "/home/user/docs"
                assert app.query_one("#language", Select).value == "en"
                assert app.query_one("#case", Select).value == "snakeCase"
                assert app.query_one("#dry_run", Checkbox).value is False
                assert app.query_one("#use_ocr", Checkbox).value is True
                assert app.query_one("#project", Input).value == "my-project"
    finally:
        tf_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# 5. Cancel action
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cancel_sets_stop_event() -> None:
    """Pressing Cancel while a run is in progress sets _stop_event."""
    app = _make_app()
    async with app.run_test(size=(120, 40)) as _pilot:
        # Simulate an in-progress run
        app._running = True
        app._stop_event.clear()

        # Invoke the cancel handler directly (button may not be visible on active tab)
        app.on_cancel()

        assert app._stop_event.is_set()


@pytest.mark.asyncio
async def test_cancel_noop_when_not_running() -> None:
    """Cancel does nothing when no run is in progress."""
    app = _make_app()
    async with app.run_test(size=(120, 40)) as _pilot:
        app._running = False
        app._stop_event.clear()

        app._cancel()

        assert not app._stop_event.is_set()
